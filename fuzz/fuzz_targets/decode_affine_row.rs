#![no_main]

use deepseek::checkpoint::mlx::decode_affine_row;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let bits = data[0];
    let logical_width = usize::from(data[1]);
    let group_size = usize::from(data[2]);
    let packed_len = usize::from(data[3] % 16);
    let packed_bytes = packed_len.saturating_mul(4);
    let packed_start = 4_usize;
    let packed_end = packed_start.saturating_add(packed_bytes);
    if packed_end > data.len() {
        return;
    }
    let packed = data[packed_start..packed_end]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();

    let groups = if group_size == 0 {
        0
    } else {
        logical_width.div_ceil(group_size)
    };
    let scale_start = packed_end;
    let scale_end = scale_start.saturating_add(groups.saturating_mul(2));
    let bias_end = scale_end.saturating_add(groups.saturating_mul(2));
    if bias_end > data.len() {
        return;
    }
    let scales = data[scale_start..scale_end]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    let biases = data[scale_end..bias_end]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();

    let _ = decode_affine_row(&packed, &scales, &biases, logical_width, bits, group_size);
});
