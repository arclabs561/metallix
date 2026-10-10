//! Safe wrapper over MLX's custom Metal kernel API.
//!
//! The pinned `mlx-rs` does not wrap MLX-C's `mlx_fast_metal_kernel_*`
//! functions, so this crate calls them directly and is the only workspace
//! crate allowed `unsafe` for custom kernels. Model crates describe a kernel
//! with [`KernelSpec`] and launch it with [`Launch`]; every MLX-C handle is
//! owned by an RAII guard, and MLX's error handler is installed before the
//! first raw call, so a rejected launch returns [`KernelError::Mlx`] instead
//! of ending the process.
//!
//! MLX generates the kernel signature from the input and output names (plus
//! `<name>_shape` and friends when the body uses them) and inserts the body.
//! Outputs are fresh arrays in the lazy graph: a kernel cannot write into an
//! existing buffer.
#![allow(unsafe_code)]
#![deny(missing_docs)]

use std::ffi::CString;
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, StreamOrDevice};
use mlx_sys as sys;
use thiserror::Error;

/// A kernel that could not be built or launched.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelError {
    /// A name or the source contains an interior NUL byte.
    #[error("custom kernel {field} contains a NUL byte")]
    Nul {
        /// Which string.
        field: &'static str,
    },
    /// The launch passed a different number of inputs or outputs than the
    /// kernel declares.
    #[error("custom kernel {kernel}: {role} count is {actual}, kernel declares {declared}")]
    Arity {
        /// Kernel name.
        kernel: String,
        /// `"input"` or `"output"`.
        role: &'static str,
        /// Declared count.
        declared: usize,
        /// Supplied count.
        actual: usize,
    },
    /// A grid, threadgroup or shape extent is zero or exceeds MLX's 32-bit
    /// arguments.
    #[error("custom kernel {field} extent {value} is outside 1..=i32::MAX")]
    Extent {
        /// Which extent.
        field: &'static str,
        /// Supplied value.
        value: usize,
    },
    /// MLX rejected kernel construction or application.
    #[error("MLX custom kernel failed: {0}")]
    Mlx(String),
}

/// What a kernel is: its name, argument names and Metal body.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpec<'a> {
    /// Kernel name; MLX adds template and dtype suffixes for its cache key.
    pub name: &'a str,
    /// Input buffer names, in launch order.
    pub inputs: &'a [&'a str],
    /// Output buffer names, in launch order.
    pub outputs: &'a [&'a str],
    /// The kernel body (MLX writes the signature).
    pub source: &'a str,
    /// Code placed before the generated kernel (helpers, includes).
    pub header: &'a str,
    /// Copy non-row-contiguous inputs before the launch.
    pub ensure_row_contiguous: bool,
    /// Declare outputs as atomic.
    pub atomic_outputs: bool,
}

impl<'a> KernelSpec<'a> {
    /// A spec with no header, row-contiguous inputs and plain outputs.
    #[must_use]
    pub const fn new(
        name: &'a str,
        inputs: &'a [&'a str],
        outputs: &'a [&'a str],
        source: &'a str,
    ) -> Self {
        Self {
            name,
            inputs,
            outputs,
            source,
            header: "",
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
    }
}

/// A compile-time template parameter, substituted into the body by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateArg {
    /// An integer constant.
    Int(i32),
    /// A boolean constant.
    Bool(bool),
    /// A type parameter. Also useful only to put a dtype into MLX's kernel
    /// cache key.
    Dtype(Dtype),
}

/// Three positive extents, as MLX's 32-bit launch arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent3([i32; 3]);

impl Extent3 {
    fn new(field: &'static str, extents: [usize; 3]) -> Result<Self, KernelError> {
        let mut out = [0_i32; 3];
        for (slot, value) in out.iter_mut().zip(extents) {
            *slot = positive_i32(field, value)?;
        }
        Ok(Self(out))
    }
}

/// The total number of threads to dispatch on each axis (Metal
/// `dispatchThreads`, not a threadgroup count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid(Extent3);

impl Grid {
    /// # Errors
    ///
    /// [`KernelError::Extent`] when an extent is zero or exceeds `i32::MAX`.
    pub fn new(x: usize, y: usize, z: usize) -> Result<Self, KernelError> {
        Extent3::new("grid", [x, y, z]).map(Self)
    }
}

/// Threads per threadgroup on each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadGroup(Extent3);

impl ThreadGroup {
    /// # Errors
    ///
    /// [`KernelError::Extent`] when an extent is zero or exceeds `i32::MAX`.
    pub fn new(x: usize, y: usize, z: usize) -> Result<Self, KernelError> {
        Extent3::new("threadgroup", [x, y, z]).map(Self)
    }
}

/// One output buffer's shape and dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    shape: Vec<i32>,
    dtype: Dtype,
}

impl Output {
    /// # Errors
    ///
    /// [`KernelError::Extent`] when a dimension is zero or exceeds `i32::MAX`.
    pub fn new(shape: &[usize], dtype: Dtype) -> Result<Self, KernelError> {
        let shape = shape
            .iter()
            .map(|&dim| positive_i32("output shape", dim))
            .collect::<Result<_, _>>()?;
        Ok(Self { shape, dtype })
    }
}

/// How to launch a kernel once.
#[derive(Clone, Debug)]
pub struct Launch<'a> {
    /// Output shapes and dtypes, one per declared output.
    pub outputs: &'a [Output],
    /// Template parameters.
    pub template: &'a [(&'a str, TemplateArg)],
    /// Threads to dispatch.
    pub grid: Grid,
    /// Threads per threadgroup.
    pub threadgroup: ThreadGroup,
    /// Fill every output with this value before the kernel runs.
    pub init_value: Option<f32>,
}

/// A custom Metal kernel. Build once and reuse; MLX compiles each template
/// and dtype combination on first application and caches it.
pub struct MetalKernel {
    handle: KernelHandle,
    name: String,
    inputs: usize,
    outputs: usize,
}

impl MetalKernel {
    /// Creates the kernel object.
    ///
    /// # Errors
    ///
    /// [`KernelError::Nul`] for a NUL byte in a string, or
    /// [`KernelError::Mlx`] if MLX cannot create the kernel.
    pub fn new(spec: &KernelSpec<'_>) -> Result<Self, KernelError> {
        ensure_mlx_error_handler()?;
        let name = c_string("name", spec.name)?;
        let source = c_string("source", spec.source)?;
        let header = c_string("header", spec.header)?;
        let input_names = StringVector::new(spec.inputs)?;
        let output_names = StringVector::new(spec.outputs)?;
        // SAFETY: all pointers are valid NUL-terminated strings and live handles for the call;
        // MLX copies the names and source. The returned handle is owned by `KernelHandle`.
        let kernel = unsafe {
            sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.0,
                output_names.0,
                source.as_ptr(),
                header.as_ptr(),
                spec.ensure_row_contiguous,
                spec.atomic_outputs,
            )
        };
        if kernel.ctx.is_null() {
            return Err(KernelError::Mlx(format!(
                "kernel {} construction returned null",
                spec.name
            )));
        }
        Ok(Self {
            handle: KernelHandle(kernel),
            name: spec.name.to_owned(),
            inputs: spec.inputs.len(),
            outputs: spec.outputs.len(),
        })
    }

    /// The kernel's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds one launch to the lazy graph on the GPU stream and returns its
    /// outputs in declared order.
    ///
    /// # Errors
    ///
    /// [`KernelError::Arity`] when the input or output count differs from the
    /// spec, [`KernelError::Nul`] for a NUL byte in a template name, or
    /// [`KernelError::Mlx`] if MLX rejects the launch.
    pub fn apply(&self, inputs: &[&Array], launch: &Launch<'_>) -> Result<Vec<Array>, KernelError> {
        self.check_arity("input", self.inputs, inputs.len())?;
        self.check_arity("output", self.outputs, launch.outputs.len())?;
        self.launch(inputs, launch)
    }

    /// The MLX-C calls of [`Self::apply`], without the arity checks (tests use
    /// it to make MLX itself reject a launch).
    fn launch(&self, inputs: &[&Array], launch: &Launch<'_>) -> Result<Vec<Array>, KernelError> {
        ensure_mlx_error_handler()?;
        // SAFETY: `mlx_fast_metal_kernel_config_new` has no preconditions.
        let config = ConfigHandle(unsafe { sys::mlx_fast_metal_kernel_config_new() });
        for output in launch.outputs {
            // SAFETY: `config.0` is live; the shape slice outlives the call and MLX copies it.
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config.0,
                    output.shape.as_ptr(),
                    output.shape.len(),
                    output.dtype.into(),
                )
            })?;
        }
        for &(name, value) in launch.template {
            let name = c_string("template name", name)?;
            // SAFETY: `config.0` is live and `name` outlives each call; MLX copies it.
            check(unsafe {
                match value {
                    TemplateArg::Int(value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config.0,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Bool(value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config.0,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Dtype(value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config.0,
                            name.as_ptr(),
                            value.into(),
                        )
                    }
                }
            })?;
        }
        let [gx, gy, gz] = launch.grid.0.0;
        // SAFETY: `config.0` is live; the extents were validated as positive i32.
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_grid(config.0, gx, gy, gz) })?;
        let [tx, ty, tz] = launch.threadgroup.0.0;
        // SAFETY: as above.
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_thread_group(config.0, tx, ty, tz) })?;
        if let Some(value) = launch.init_value {
            // SAFETY: `config.0` is live.
            check(unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(config.0, value) })?;
        }
        // SAFETY: `mlx_vector_array_new` has no preconditions.
        let input_vector = ArrayVector(unsafe { sys::mlx_vector_array_new() });
        for input in inputs {
            // SAFETY: both handles are live; MLX retains its own reference to the array.
            check(unsafe { sys::mlx_vector_array_append_value(input_vector.0, input.as_ptr()) })?;
        }
        let stream = StreamOrDevice::gpu();
        let stream: &Stream = stream.as_ref();
        // SAFETY: `mlx_vector_array_new` has no preconditions; ownership moves into the guard.
        let mut outputs = ArrayVector(unsafe { sys::mlx_vector_array_new() });
        // SAFETY: every handle is live for the call; `outputs.0` receives a new owned vector.
        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                &raw mut outputs.0,
                self.handle.0,
                input_vector.0,
                config.0,
                stream.as_ptr(),
            )
        })?;
        (0..launch.outputs.len())
            .map(|index| outputs.get(index))
            .collect()
    }

    fn check_arity(
        &self,
        role: &'static str,
        declared: usize,
        actual: usize,
    ) -> Result<(), KernelError> {
        if declared == actual {
            Ok(())
        } else {
            Err(KernelError::Arity {
                kernel: self.name.clone(),
                role,
                declared,
                actual,
            })
        }
    }
}

/// The GPU's core count from the `IORegistry` (`gpu-core-count` on the
/// `AGXAccelerator` service), read once per process; `None` when `ioreg` is
/// missing or prints no count. MLX exposes no core count, and kernels that
/// size their grids to the GPU need one.
#[must_use]
pub fn gpu_core_count() -> Option<u32> {
    static CORES: OnceLock<Option<u32>> = OnceLock::new();
    *CORES.get_or_init(|| {
        let output = std::process::Command::new("/usr/sbin/ioreg")
            .args(["-r", "-c", "AGXAccelerator", "-d", "1"])
            .output()
            .ok()?;
        parse_gpu_core_count(&String::from_utf8_lossy(&output.stdout))
    })
}

/// MLX's architecture string for the default GPU (`architecture` in its
/// device info, for example `applegpu_g15s`), read once per process; `None`
/// when MLX reports none. Cost tables keyed by GPU generation use it.
#[must_use]
pub fn gpu_architecture() -> Option<String> {
    static ARCHITECTURE: OnceLock<Option<String>> = OnceLock::new();
    ARCHITECTURE
        .get_or_init(|| {
            ensure_mlx_error_handler().ok()?;
            let key = CString::new("architecture").ok()?;
            // SAFETY: `mlx_device_new_type` and `mlx_device_info_new` take no
            // pointers and return owned handles, each freed below exactly once.
            let (device, mut info) = unsafe {
                (
                    sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0),
                    sys::mlx_device_info_new(),
                )
            };
            let mut value: *const std::os::raw::c_char = std::ptr::null();
            // SAFETY: `info` and `device` are live handles; `key` is a valid
            // NUL-terminated string for the call. The returned string is owned
            // by `info` and copied before `info` is freed.
            let architecture = unsafe {
                (sys::mlx_device_info_get(&raw mut info, device) == 0
                    && sys::mlx_device_info_get_string(&raw mut value, info, key.as_ptr()) == 0
                    && !value.is_null())
                .then(|| {
                    std::ffi::CStr::from_ptr(value)
                        .to_string_lossy()
                        .into_owned()
                })
            };
            // SAFETY: both handles were created above and are freed once.
            unsafe {
                sys::mlx_device_info_free(info);
                sys::mlx_device_free(device);
            }
            architecture
        })
        .clone()
}

/// The first positive `"gpu-core-count" = N` value in `ioreg` output.
fn parse_gpu_core_count(text: &str) -> Option<u32> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        if key.trim() != "\"gpu-core-count\"" {
            return None;
        }
        value.trim().parse::<u32>().ok().filter(|&cores| cores > 0)
    })
}

fn positive_i32(field: &'static str, value: usize) -> Result<i32, KernelError> {
    match i32::try_from(value) {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(KernelError::Extent { field, value }),
    }
}

fn c_string(field: &'static str, text: &str) -> Result<CString, KernelError> {
    CString::new(text).map_err(|_| KernelError::Nul { field })
}

fn check(status: std::os::raw::c_int) -> Result<(), KernelError> {
    if status == 0 {
        Ok(())
    } else {
        Err(KernelError::Mlx(format!("MLX-C returned status {status}")))
    }
}

/// Installs mlx-rs's MLX error handler before any raw MLX-C call.
///
/// MLX-C's default handler prints the message and calls `exit(-1)`, so a
/// rejected kernel launch would end the process instead of returning a status.
/// mlx-rs replaces that handler the first time it runs a guarded op, which
/// this crate's raw calls never do; one tiny guarded op forces the install.
fn ensure_mlx_error_handler() -> Result<(), KernelError> {
    static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
    INSTALLED
        .get_or_init(|| {
            Array::zeros::<f32>(&[1])
                .map(drop)
                .map_err(|error| error.to_string())
        })
        .clone()
        .map_err(KernelError::Mlx)
}

/// Owns an `mlx_fast_metal_kernel` and frees it on drop.
struct KernelHandle(sys::mlx_fast_metal_kernel);

impl Drop for KernelHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_fast_metal_kernel_new` and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) }
    }
}

/// Owns an `mlx_fast_metal_kernel_config` and frees it on drop.
struct ConfigHandle(sys::mlx_fast_metal_kernel_config);

impl Drop for ConfigHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_fast_metal_kernel_config_new` and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) }
    }
}

/// Owns an `mlx_vector_string` and frees it on drop.
struct StringVector(sys::mlx_vector_string);

impl StringVector {
    fn new(items: &[&str]) -> Result<Self, KernelError> {
        // SAFETY: `mlx_vector_string_new` has no preconditions and returns an owned handle.
        let vector = Self(unsafe { sys::mlx_vector_string_new() });
        for item in items {
            let text = c_string("argument name", item)?;
            // SAFETY: `vector.0` is a live handle and `text` outlives the call; MLX copies the string.
            check(unsafe { sys::mlx_vector_string_append_value(vector.0, text.as_ptr()) })?;
        }
        Ok(vector)
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_vector_string_new` and is freed once.
        unsafe {
            sys::mlx_vector_string_free(self.0);
        }
    }
}

/// Owns an `mlx_vector_array` and frees it on drop.
struct ArrayVector(sys::mlx_vector_array);

impl ArrayVector {
    fn get(&self, index: usize) -> Result<Array, KernelError> {
        // SAFETY: `mlx_array_new` returns an empty owned array that `vector_array_get` fills.
        let mut result = unsafe { sys::mlx_array_new() };
        // SAFETY: `self.0` is live; MLX checks the index and returns a status.
        let status = unsafe { sys::mlx_vector_array_get(&raw mut result, self.0, index) };
        // SAFETY: `result` is an owned MLX array handle; `Array` takes ownership and frees it.
        let array = unsafe { Array::from_ptr(result) };
        check(status)?;
        Ok(array)
    }
}

impl Drop for ArrayVector {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_vector_array_new` and is freed once.
        unsafe {
            sys::mlx_vector_array_free(self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::{Array, Dtype};

    use super::{
        Grid, KernelError, KernelSpec, Launch, MetalKernel, Output, TemplateArg, ThreadGroup,
    };

    const AXPY: &str = "
        uint i = thread_position_in_grid.x;
        out[i] = A * x[i] + y[i];
        if (DOUBLE) { twice[i] = 2.0f * out[i]; } else { twice[i] = out[i]; }
    ";

    fn axpy() -> MetalKernel {
        MetalKernel::new(&KernelSpec::new(
            "mlx_kernels_test_axpy",
            &["x", "y"],
            &["out", "twice"],
            AXPY,
        ))
        .expect("kernel")
    }

    #[test]
    fn core_count_parses_the_recorded_ioreg_line() {
        // Recorded from `ioreg -r -c AGXAccelerator -d 1` on an M3 Max.
        let recorded = "+-o AGXAcceleratorG15X  <class AGXAcceleratorG15X, id 0x1000005f5>\n\
                        {\n      \"gpu-core-count\" = 40\n      \"num-cores\" = 7\n    }\n";
        assert_eq!(super::parse_gpu_core_count(recorded), Some(40));
        assert_eq!(super::parse_gpu_core_count("\"gpu-core-count\" = 0"), None);
        assert_eq!(
            super::parse_gpu_core_count("\"gpu-core-count-x\" = 12"),
            None
        );
        assert_eq!(super::parse_gpu_core_count(""), None);
    }

    #[test]
    fn gpu_architecture_names_an_apple_gpu() {
        let architecture = super::gpu_architecture().expect("MLX reports an architecture");
        assert!(
            architecture.starts_with("applegpu_"),
            "unexpected architecture {architecture}"
        );
    }

    #[test]
    fn two_outputs_with_int_bool_and_dtype_templates() {
        let kernel = axpy();
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let y = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]);
        let outputs = [
            Output::new(&[3], Dtype::Float32).expect("output"),
            Output::new(&[3], Dtype::Float32).expect("output"),
        ];
        let template = [
            ("A", TemplateArg::Int(3)),
            ("DOUBLE", TemplateArg::Bool(true)),
            ("T", TemplateArg::Dtype(Dtype::Float32)),
        ];
        let launch = Launch {
            outputs: &outputs,
            template: &template,
            grid: Grid::new(3, 1, 1).expect("grid"),
            threadgroup: ThreadGroup::new(3, 1, 1).expect("threadgroup"),
            init_value: None,
        };
        let result = kernel.apply(&[&x, &y], &launch).expect("launch");
        assert_eq!(result.len(), 2);
        mlx_rs::transforms::eval(result.iter()).expect("eval");
        assert_eq!(result[0].as_slice::<f32>(), [13.0, 26.0, 39.0]);
        assert_eq!(result[1].as_slice::<f32>(), [26.0, 52.0, 78.0]);
    }

    #[test]
    fn arity_and_extent_errors_are_typed() {
        let kernel = axpy();
        let x = Array::from_slice(&[1.0_f32], &[1]);
        let outputs = [Output::new(&[1], Dtype::Float32).expect("output")];
        let launch = Launch {
            outputs: &outputs,
            template: &[],
            grid: Grid::new(1, 1, 1).expect("grid"),
            threadgroup: ThreadGroup::new(1, 1, 1).expect("threadgroup"),
            init_value: None,
        };
        assert!(matches!(
            kernel.apply(&[&x], &launch),
            Err(KernelError::Arity {
                role: "input",
                declared: 2,
                actual: 1,
                ..
            })
        ));
        assert!(matches!(
            kernel.apply(&[&x, &x], &launch),
            Err(KernelError::Arity {
                role: "output",
                declared: 2,
                actual: 1,
                ..
            })
        ));
        assert!(matches!(
            Grid::new(0, 1, 1),
            Err(KernelError::Extent { field: "grid", .. })
        ));
        assert!(matches!(
            Output::new(&[1 << 31], Dtype::Float32),
            Err(KernelError::Extent { .. })
        ));
    }

    // Run alone with `--exact` to exercise a process where nothing else has
    // installed the handler yet; without the install this exits with status 255.
    #[test]
    fn launch_mlx_rejects_returns_an_error_instead_of_exiting() {
        let kernel = axpy();
        let x = Array::from_slice(&[1.0_f32], &[1]);
        let outputs = [
            Output::new(&[1], Dtype::Float32).expect("output"),
            Output::new(&[1], Dtype::Float32).expect("output"),
        ];
        let launch = Launch {
            outputs: &outputs,
            template: &[],
            grid: Grid::new(1, 1, 1).expect("grid"),
            threadgroup: ThreadGroup::new(1, 1, 1).expect("threadgroup"),
            init_value: None,
        };
        // One input where the kernel declares two: MLX throws inside apply.
        let launched = kernel.launch(&[&x], &launch);
        assert!(matches!(launched, Err(KernelError::Mlx(_))), "{launched:?}");
    }

    #[test]
    fn a_body_that_does_not_compile_returns_an_error() {
        let kernel = MetalKernel::new(&KernelSpec::new(
            "mlx_kernels_test_broken",
            &["x"],
            &["out"],
            "this is not metal;",
        ))
        .expect("construction is lazy");
        let x = Array::from_slice(&[1.0_f32], &[1]);
        let outputs = [Output::new(&[1], Dtype::Float32).expect("output")];
        let launch = Launch {
            outputs: &outputs,
            template: &[],
            grid: Grid::new(1, 1, 1).expect("grid"),
            threadgroup: ThreadGroup::new(1, 1, 1).expect("threadgroup"),
            init_value: None,
        };
        let result = kernel.apply(&[&x], &launch).and_then(|outputs| {
            mlx_rs::transforms::eval(outputs.iter())
                .map_err(|error| KernelError::Mlx(error.to_string()))
        });
        assert!(matches!(result, Err(KernelError::Mlx(_))), "{result:?}");
    }
}
