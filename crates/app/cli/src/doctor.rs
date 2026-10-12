//! `mx doctor`: whether this build's MLX can use the GPU's Neural
//! Accelerators (NAX) on this machine, and why not when it cannot.
//!
//! MLX compiles its NAX kernels only when built for macOS 26.2 or later, and
//! runs them only on a GPU of generation 17 or later (18 for phone GPUs) on
//! macOS 26.2 or later. A build for an older deployment target drops them
//! without an error, so the kernel count here is the check.

use std::{
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::ExitCode,
};

use crate::gpu;

/// What MLX offers for NAX on this machine.
#[derive(Debug)]
pub(crate) struct MetalReport {
    pub(crate) nax_available: bool,
    pub(crate) architecture: String,
    pub(crate) macos: String,
    pub(crate) metallib: PathBuf,
    pub(crate) kernels: usize,
    pub(crate) nax_kernels: usize,
}

impl MetalReport {
    pub(crate) fn read() -> Result<Self, String> {
        let override_path =
            mlx_rs::metal::metallib_path().map_err(|error| format!("metallib path: {error}"))?;
        let executable_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf));
        let metallib = loaded_metallib(
            &override_path,
            executable_dir.as_deref(),
            mlx_sys::DEFAULT_METALLIB_PATH,
            Path::is_file,
        )
        .ok_or("no metallib found where MLX looks for one")?;
        let names = metallib_function_names(&metallib)?;
        Ok(Self {
            nax_available: gpu::nax_available().ok_or("MLX could not report NAX availability")?,
            architecture: gpu::gpu_architecture()
                .ok_or("MLX could not report the GPU architecture")?,
            macos: macos_version(),
            kernels: names.len(),
            nax_kernels: names.iter().filter(|name| name.contains("_nax")).count(),
            metallib,
        })
    }

    /// `available`, or the first reason MLX will not run NAX kernels.
    pub(crate) fn nax_status(&self) -> String {
        if self.nax_available {
            return String::from("available");
        }
        if self.nax_kernels == 0 {
            return String::from("built without NAX kernels");
        }
        match gpu_generation(&self.architecture) {
            Some((generation, class)) if generation < if class == 'p' { 18 } else { 17 } => {
                format!("hardware absent (g{generation})")
            }
            _ => String::from("unavailable (macOS below 26.2?)"),
        }
    }
}

impl fmt::Display for MetalReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "nax={}; arch={}; macos={}; nax_kernels={} of {}; metallib={}",
            self.nax_status(),
            self.architecture,
            self.macos,
            self.nax_kernels,
            self.kernels,
            self.metallib.display()
        )
    }
}

/// Prints the report. Fails when the linked metallib has no NAX kernels,
/// which means MLX was built for a deployment target below 26.2.
pub(crate) fn run() -> ExitCode {
    match MetalReport::read() {
        Ok(report) => {
            println!("metal: {report}");
            if report.nax_kernels == 0 {
                eprintln!(
                    "mx doctor: {} has no NAX kernels; rebuild with MACOSX_DEPLOYMENT_TARGET >= 26.2",
                    report.metallib.display()
                );
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("mx doctor: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The metallib MLX loads, in MLX's own order (`load_default_library` in
/// mlx/backend/metal/device.cpp): a path set through the API, then
/// `mlx.metallib` and `Resources/mlx.metallib` next to the executable, then
/// the path compiled in at build time.
fn loaded_metallib(
    override_path: &str,
    executable_dir: Option<&Path>,
    compiled_in: &str,
    exists: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    if !override_path.is_empty() {
        return Some(PathBuf::from(override_path));
    }
    let colocated = executable_dir
        .into_iter()
        .flat_map(|dir| [dir.join("mlx.metallib"), dir.join("Resources/mlx.metallib")]);
    colocated
        .chain((!compiled_in.is_empty()).then(|| PathBuf::from(compiled_in)))
        .find(|path| exists(path))
}

/// The GPU generation and class letter from an MLX architecture string:
/// `applegpu_g15s` is generation 15, class `s`.
fn gpu_generation(architecture: &str) -> Option<(u32, char)> {
    let class = architecture.chars().last()?;
    let digits = architecture.get(architecture.len().checked_sub(3)?..architecture.len() - 1)?;
    Some((digits.parse().ok()?, class))
}

fn macos_version() -> String {
    std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || String::from("unknown"),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}

/// Reads only a metallib's header and function list, not its code.
fn metallib_function_names(path: &Path) -> Result<Vec<String>, String> {
    let error = |what: &str| format!("{}: {what}", path.display());
    let mut file = File::open(path).map_err(|e| error(&e.to_string()))?;
    let mut header = [0_u8; 0x28];
    file.read_exact(&mut header)
        .map_err(|_| error("too short for a metallib header"))?;
    let (offset, size) = function_list_span(&header).ok_or_else(|| error("not a metallib"))?;
    let mut list = vec![0_u8; usize::try_from(size).map_err(|_| error("function list too large"))?];
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(&mut list))
        .map_err(|_| error("function list runs past the end of the file"))?;
    function_names(&list).ok_or_else(|| error("malformed function list"))
}

/// Offset and size of the function list in a metallib header (`MTLB`),
/// including its leading `u32` entry count. The header's size field counts
/// only the entries after it.
fn function_list_span(header: &[u8]) -> Option<(u64, u64)> {
    if header.get(..4)? != b"MTLB" {
        return None;
    }
    let field = |at: usize| Some(u64::from_le_bytes(header.get(at..at + 8)?.try_into().ok()?));
    Some((field(0x18)?, field(0x20)?.checked_add(4)?))
}

/// The `NAME` tag of every entry in a metallib function list: a `u32` entry
/// count, then per entry a `u32` byte size and tags of a 4-byte name, a
/// `u16` length and that many bytes, ending at `ENDT`.
fn function_names(list: &[u8]) -> Option<Vec<String>> {
    let read_u32 = |at: usize| Some(u32::from_le_bytes(list.get(at..at + 4)?.try_into().ok()?));
    let count = read_u32(0)? as usize;
    let mut names = Vec::with_capacity(count);
    let mut entry = 4_usize;
    for _ in 0..count {
        let end = entry.checked_add(read_u32(entry)? as usize)?;
        let mut tag = entry + 4;
        while tag + 4 <= end {
            let kind = list.get(tag..tag + 4)?;
            if kind == b"ENDT" {
                break;
            }
            let length = usize::from(u16::from_le_bytes(
                list.get(tag + 4..tag + 6)?.try_into().ok()?,
            ));
            let data = list.get(tag + 6..tag + 6 + length)?;
            if kind == b"NAME" {
                let name = data.split(|&byte| byte == 0).next()?;
                names.push(String::from_utf8_lossy(name).into_owned());
            }
            tag += 6 + length;
        }
        entry = end;
    }
    Some(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> Vec<u8> {
        let mut tags = Vec::new();
        let mut text = name.as_bytes().to_vec();
        text.push(0);
        tags.extend_from_slice(b"NAME");
        tags.extend_from_slice(&u16::try_from(text.len()).expect("short").to_le_bytes());
        tags.extend_from_slice(&text);
        tags.extend_from_slice(b"TYPE");
        tags.extend_from_slice(&1_u16.to_le_bytes());
        tags.push(2);
        tags.extend_from_slice(b"ENDT");
        let mut out = u32::try_from(tags.len() + 4)
            .expect("short")
            .to_le_bytes()
            .to_vec();
        out.extend(tags);
        out
    }

    #[test]
    fn function_names_come_from_name_tags_only() {
        let mut list = 3_u32.to_le_bytes().to_vec();
        for name in [
            "gemv_float32",
            "steel_gemm_fused_nax_nn_bfloat16",
            "copy_v_float16",
        ] {
            list.extend(entry(name));
        }
        let names = function_names(&list).expect("well formed");
        assert_eq!(
            names,
            [
                "gemv_float32",
                "steel_gemm_fused_nax_nn_bfloat16",
                "copy_v_float16"
            ]
        );
        // A list cut short is malformed rather than silently counted.
        assert!(function_names(&list[..list.len() - 10]).is_none());
    }

    #[test]
    fn the_loaded_metallib_follows_mlx_search_order() {
        let exe = Path::new("/opt/mx/bin");
        let compiled = "/home/u/.mlx/lib/key-macos26.2/mlx.metallib";
        let colocated = "/opt/mx/bin/mlx.metallib";
        let resources = "/opt/mx/bin/Resources/mlx.metallib";
        let present = |files: Vec<&'static str>| {
            move |path: &Path| files.iter().any(|f| path == Path::new(f))
        };
        // An override wins even when other files exist.
        assert_eq!(
            loaded_metallib(
                "/tmp/x.metallib",
                Some(exe),
                compiled,
                present(vec![colocated])
            ),
            Some(PathBuf::from("/tmp/x.metallib"))
        );
        // A colocated metallib beats the compiled-in path.
        assert_eq!(
            loaded_metallib("", Some(exe), compiled, present(vec![resources, compiled])),
            Some(PathBuf::from(resources))
        );
        assert_eq!(
            loaded_metallib("", Some(exe), compiled, present(vec![compiled])),
            Some(PathBuf::from(compiled))
        );
        assert_eq!(
            loaded_metallib("", Some(exe), compiled, present(vec![])),
            None
        );
    }

    #[test]
    fn only_metallibs_have_a_function_list() {
        let mut header = [0_u8; 0x28];
        header[..4].copy_from_slice(b"MTLB");
        header[0x18..0x20].copy_from_slice(&0x58_u64.to_le_bytes());
        header[0x20..0x28].copy_from_slice(&9_u64.to_le_bytes());
        // The span adds the 4-byte entry count the size field leaves out.
        assert_eq!(function_list_span(&header), Some((0x58, 13)));
        header[..4].copy_from_slice(b"MZMZ");
        assert_eq!(function_list_span(&header), None);
    }

    #[test]
    fn the_linked_metallib_lists_its_kernels() {
        // Reads the real metallib this build installed, so a header or list
        // layout that disagrees with the Metal toolchain fails here.
        let report = MetalReport::read().expect("the linked metallib is readable");
        assert!(report.kernels > 1000, "{report}");
    }

    #[test]
    fn nax_status_names_the_first_missing_piece() {
        let report = |architecture: &str, nax_available, nax_kernels| MetalReport {
            nax_available,
            architecture: architecture.into(),
            macos: "26.6.2".into(),
            metallib: PathBuf::from("mlx.metallib"),
            kernels: 10,
            nax_kernels,
        };
        assert_eq!(
            report("applegpu_g15s", false, 4).nax_status(),
            "hardware absent (g15)"
        );
        assert_eq!(report("applegpu_g17s", true, 4).nax_status(), "available");
        assert_eq!(
            report("applegpu_g17s", false, 0).nax_status(),
            "built without NAX kernels"
        );
        // Phone GPUs need one generation more.
        assert_eq!(
            report("applegpu_g17p", false, 4).nax_status(),
            "hardware absent (g17)"
        );
    }
}
