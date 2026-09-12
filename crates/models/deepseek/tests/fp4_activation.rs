//! Test-only scalar model of the pinned FP4 activation staging.
//!
//! This uses software round-to-nearest-even encoders as an explicit reference
//! assumption. It is not CUDA/TileLang instruction-level parity or a packed
//! FP4 storage implementation.

use deepseek::attention::{SparseAttentionLayout, sparse_attention_reference};
use deepseek::precision::{decode_e2m1, decode_e4m3fn, decode_e8m0};
use deepseek::rotary::{RotaryDirection, RotaryFrequency, RotaryTailLayout, rotate_tail};
use std::num::NonZeroUsize;
use thiserror::Error;

const MAX_ELEMENTS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp4Mode {
    CompressedKv16E4m3,
    Index32E8m0,
}

impl Fp4Mode {
    const fn block_size(self) -> usize {
        match self {
            Self::CompressedKv16E4m3 => 16,
            Self::Index32E8m0 => 32,
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
enum Fp4ActivationError {
    #[error("rows and width must be nonzero")]
    EmptyShape,
    #[error("FP4 width {width} is not divisible by block size {block_size}")]
    PartialBlock { width: usize, block_size: usize },
    #[error("FP4 element count overflow")]
    ShapeOverflow,
    #[error("FP4 element count {elements} exceeds {MAX_ELEMENTS}")]
    ElementLimit { elements: usize },
    #[error("FP4 {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("FP4 BF16 input at {index} is non-finite")]
    NonFiniteInput { index: usize },
    #[error("FP4 E4M3 scale exceeds finite supported range")]
    E4m3ScaleOverflow,
    #[error("FP4 E8M0 scale exceeds finite supported range")]
    E8m0ScaleOverflow,
    #[error("FP4 reconstruction overflowed at {index}")]
    ReconstructionOverflow { index: usize },
}

struct Fp4Trace {
    fp4_codes: Vec<u8>,
    scale_codes: Vec<u8>,
    scales: Vec<f32>,
}

/// Quantizes finite BF16 rows through logical E2M1 and reconstructs BF16.
///
/// Software RNE for E2M1/E4M3 is a scalar test assumption; hardware cast
/// tie behavior is not qualified. The caller output remains unchanged unless
/// every input, scale, code, and reconstructed BF16 is finite.
fn fp4_activation_reference(
    input: &[u16],
    rows: usize,
    width: usize,
    mode: Fp4Mode,
    output: &mut [u16],
) -> Result<Fp4Trace, Fp4ActivationError> {
    if rows == 0 || width == 0 {
        return Err(Fp4ActivationError::EmptyShape);
    }
    let block_size = mode.block_size();
    if !width.is_multiple_of(block_size) {
        return Err(Fp4ActivationError::PartialBlock { width, block_size });
    }
    let elements = rows
        .checked_mul(width)
        .ok_or(Fp4ActivationError::ShapeOverflow)?;
    if elements > MAX_ELEMENTS {
        return Err(Fp4ActivationError::ElementLimit { elements });
    }
    check_length("input", input.len(), elements)?;
    check_length("output", output.len(), elements)?;
    for (index, &bits) in input.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(Fp4ActivationError::NonFiniteInput { index });
        }
    }

    let groups = elements / block_size;
    let mut values = Vec::with_capacity(elements);
    let mut fp4_codes = Vec::with_capacity(elements);
    let mut scale_codes = Vec::with_capacity(groups);
    let mut scales = Vec::with_capacity(groups);
    for group in 0..groups {
        let start = group * block_size;
        let end = start + block_size;
        let scale = scale_for(&input[start..end], mode)?;
        scale_codes.push(scale.0);
        scales.push(scale.1);
        for (offset, &bits) in input[start..end].iter().enumerate() {
            let code = encode_e2m1_rne((bf16_to_f32(bits) / scale.1).clamp(-6.0, 6.0));
            let reconstructed = decode_e2m1(code).expect("logical E2M1 nibble") * scale.1;
            let result = f32_to_bf16_rne(reconstructed);
            if !bf16_to_f32(result).is_finite() {
                return Err(Fp4ActivationError::ReconstructionOverflow {
                    index: start + offset,
                });
            }
            fp4_codes.push(code);
            values.push(result);
        }
    }
    output.copy_from_slice(&values);
    Ok(Fp4Trace {
        fp4_codes,
        scale_codes,
        scales,
    })
}

fn scale_for(group: &[u16], mode: Fp4Mode) -> Result<(u8, f32), Fp4ActivationError> {
    let amax = group
        .iter()
        .map(|&bits| bf16_to_f32(bits).abs())
        .fold(0.0_f32, f32::max);
    match mode {
        Fp4Mode::CompressedKv16E4m3 => {
            let raw = amax.max(6.0 * 2.0_f32.powi(-9)) / 6.0;
            if raw > 448.0 {
                return Err(Fp4ActivationError::E4m3ScaleOverflow);
            }
            let code = encode_e4m3_rne(raw).ok_or(Fp4ActivationError::E4m3ScaleOverflow)?;
            let scale = decode_e4m3fn(code);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Fp4ActivationError::E4m3ScaleOverflow);
            }
            Ok((code, scale))
        }
        Fp4Mode::Index32E8m0 => {
            // Pinned `fast_round_scale` receives `amax * fp4_max_inv`; keep
            // its FP32 reciprocal multiply rather than algebraically dividing.
            let raw = amax.max(6.0 * 2.0_f32.powi(-126)) * (1.0_f32 / 6.0);
            let exponent = ceil_log2(raw).ok_or(Fp4ActivationError::E8m0ScaleOverflow)?;
            if exponent > 127 {
                return Err(Fp4ActivationError::E8m0ScaleOverflow);
            }
            let code = u8::try_from(exponent + 127).expect("validated E8M0 exponent");
            let scale = decode_e8m0(code);
            if !scale.is_finite() {
                return Err(Fp4ActivationError::E8m0ScaleOverflow);
            }
            Ok((code, scale))
        }
    }
}

fn ceil_log2(value: f32) -> Option<i32> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 23) & 0xff).ok()? - 127;
    Some(exponent + i32::from((bits & 0x7f_ffff) != 0))
}

fn encode_e2m1_rne(value: f32) -> u8 {
    nearest_code(value, 0_u8..=15, |code| {
        decode_e2m1(code).expect("E2M1 range")
    })
}

fn encode_e4m3_rne(value: f32) -> Option<u8> {
    if !value.is_finite() || value < 0.0 || value > 448.0 {
        return None;
    }
    Some(nearest_code(value, 0_u8..=126, decode_e4m3fn))
}

fn nearest_code(value: f32, codes: impl Iterator<Item = u8>, decode: impl Fn(u8) -> f32) -> u8 {
    codes
        .filter(|&code| {
            let decoded = decode(code);
            decoded.is_finite() && value.is_sign_negative() == decoded.is_sign_negative()
        })
        .min_by(|&left, &right| {
            let left_distance = (f64::from(decode(left)) - f64::from(value)).abs();
            let right_distance = (f64::from(decode(right)) - f64::from(value)).abs();
            left_distance
                .total_cmp(&right_distance)
                .then_with(|| (left & 1).cmp(&(right & 1)))
        })
        .expect("finite codebook")
}

fn check_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Fp4ActivationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(Fp4ActivationError::Length {
            field,
            actual,
            expected,
        })
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16)
        .expect("high FP32 half fits")
}

fn fixture_words(value: &serde_json::Value, field: &str) -> Vec<u16> {
    value[field]
        .as_array()
        .expect("fixture words")
        .iter()
        .map(|word| u16::try_from(word.as_u64().expect("word")).expect("u16 word"))
        .collect()
}

#[test]
fn independent_fixture_cases_match_software_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/fp4-activation-reference.json"
    ))
    .expect("checked-in FP4 fixture");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["kernel_sha256"],
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(fixture["receipt"]["device"], "cpu");
    let cases = fixture["cases"].as_array().expect("fixture cases");
    assert_eq!(cases.len(), 12);
    for mode in ["compressed_kv", "index"] {
        for category in [
            "zero",
            "ties",
            "nonpower_scale",
            "subnormal",
            "scale_boundary",
            "upper_finite_scale",
        ] {
            let name = format!("{mode}_{category}");
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case["name"] == name && case["mode"] == mode)
                    .count(),
                1,
                "required fixture case {name} must occur exactly once"
            );
        }
    }
    for case in cases {
        let mode = match case["mode"].as_str().expect("mode") {
            "compressed_kv" => Fp4Mode::CompressedKv16E4m3,
            "index" => Fp4Mode::Index32E8m0,
            other => panic!("unexpected fixture mode {other}"),
        };
        let input = fixture_words(case, "input_bf16");
        let expected = fixture_words(case, "output_bf16");
        let mut output = vec![0_u16; input.len()];
        let trace = fp4_activation_reference(&input, 1, input.len(), mode, &mut output)
            .expect("finite fixture case");
        assert_eq!(output, expected, "{}", case["name"]);
        assert_eq!(trace.scales.len(), 1);
        assert_eq!(
            trace.scales[0].to_bits(),
            u32::try_from(case["scale_f32_bits"].as_u64().expect("scale bits")).expect("FP32 bits"),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn known_groups_zero_signs_and_distinct_scale_modes() {
    let mut compressed = [0_u16; 16];
    compressed[..4].copy_from_slice(&[0x40c0, 0xc0c0, 0x3f80, 0x8000]); // 6,-6,1,-0
    let mut output = [0xdead_u16; 16];
    let trace =
        fp4_activation_reference(&compressed, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output)
            .expect("finite E4M3 group");
    assert_eq!(trace.scale_codes, vec![0x38]); // E4M3 scale 1
    assert_eq!(&trace.fp4_codes[..4], &[7, 15, 2, 8]);
    assert_eq!(&output[..4], &[0x40c0, 0xc0c0, 0x3f80, 0x8000]);

    let mut zeros = [0_u16; 32];
    for bits in zeros.iter_mut().skip(1).step_by(2) {
        *bits = 0x8000;
    }
    let mut index_output = [0xdead_u16; 32];
    let index = fp4_activation_reference(&zeros, 1, 32, Fp4Mode::Index32E8m0, &mut index_output)
        .expect("zero E8M0 group");
    assert_eq!(index.scale_codes, vec![1]); // 2^-126 after FP32 reciprocal
    assert_eq!(index_output, zeros);
    assert_ne!(trace.scale_codes[0], index.scale_codes[0]);
}

#[test]
fn multigroup_modes_keep_distinct_scales_and_late_failure_is_atomic() {
    let mut compressed = [0_u16; 32];
    compressed[..16].fill(0x40c0); // 6 -> E4M3 scale 1
    compressed[16..].fill(0x4140); // 12 -> E4M3 scale 2
    let mut compressed_output = [0_u16; 32];
    let compressed_trace = fp4_activation_reference(
        &compressed,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut compressed_output,
    )
    .expect("two compressed groups");
    assert_eq!(compressed_trace.scales, vec![1.0, 2.0]);

    let mut index = [0_u16; 64];
    index[..32].fill(0x4110); // 9 -> power-of-two scale 2
    index[32..].fill(0x4190); // 18 -> power-of-two scale 4
    let mut index_output = [0_u16; 64];
    let index_trace =
        fp4_activation_reference(&index, 2, 32, Fp4Mode::Index32E8m0, &mut index_output)
            .expect("two index groups");
    assert_eq!(index_trace.scales, vec![2.0, 4.0]);

    let mut late = compressed;
    late[16..].fill(0x7f7f); // second group exceeds E4M3 finite scale range
    let mut output = [0xdead_u16; 32];
    assert!(matches!(
        fp4_activation_reference(&late, 2, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::E4m3ScaleOverflow)
    ));
    assert_eq!(output, [0xdead; 32]);
}

#[test]
fn e2m1_software_rne_covers_ties_and_adjacent_fp32_inputs() {
    for code in 0_u8..=15 {
        let value = decode_e2m1(code).expect("code");
        assert_eq!(encode_e2m1_rne(value), code);
    }
    for low_code in 0_u8..7 {
        let low = decode_e2m1(low_code).expect("low code");
        let high = decode_e2m1(low_code + 1).expect("high code");
        let midpoint = low.midpoint(high);
        let expected = if low_code & 1 == 0 {
            low_code
        } else {
            low_code + 1
        };
        assert_eq!(encode_e2m1_rne(midpoint), expected);
        assert_eq!(encode_e2m1_rne(midpoint.next_down()), low_code);
        assert_eq!(encode_e2m1_rne(midpoint.next_up()), low_code + 1);

        let negative_low = low_code + 8;
        let negative_high = negative_low + 1;
        let negative_midpoint = -midpoint;
        let negative_expected = if negative_low & 1 == 0 {
            negative_low
        } else {
            negative_high
        };
        assert_eq!(encode_e2m1_rne(negative_midpoint), negative_expected);
        assert_eq!(encode_e2m1_rne(negative_midpoint.next_up()), negative_low);
        assert_eq!(
            encode_e2m1_rne(negative_midpoint.next_down()),
            negative_high
        );
    }
}

#[test]
fn failures_are_atomic_for_shape_nonfinite_and_scale_overflow() {
    let mut output = [0xdead_u16; 16];
    assert!(matches!(
        fp4_activation_reference(&[0; 15], 1, 15, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::PartialBlock { .. })
    ));
    assert_eq!(output, [0xdead; 16]);
    let mut nonfinite = [0_u16; 16];
    nonfinite[9] = 0x7f80;
    assert!(matches!(
        fp4_activation_reference(&nonfinite, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::NonFiniteInput { index: 9 })
    ));
    assert_eq!(output, [0xdead; 16]);
    let huge = [0x7f7f_u16; 16];
    assert!(matches!(
        fp4_activation_reference(&huge, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::E4m3ScaleOverflow)
    ));
    assert_eq!(output, [0xdead; 16]);

    let mut index_output = [0xdead_u16; 32];
    let index_huge = [0x7f7f_u16; 32];
    assert!(matches!(
        fp4_activation_reference(&index_huge, 1, 32, Fp4Mode::Index32E8m0, &mut index_output,),
        Err(Fp4ActivationError::ReconstructionOverflow { .. })
    ));
    assert_eq!(index_output, [0xdead; 32]);
}

#[test]
fn e4m3_scale_encoder_covers_every_positive_code_and_rounding_boundary() {
    for code in 0_u8..=126 {
        assert_eq!(encode_e4m3_rne(decode_e4m3fn(code)), Some(code));
    }
    for lower in 0_u8..126 {
        let midpoint = decode_e4m3fn(lower).midpoint(decode_e4m3fn(lower + 1));
        let even = if lower & 1 == 0 { lower } else { lower + 1 };
        assert_eq!(encode_e4m3_rne(midpoint), Some(even));
        assert_eq!(encode_e4m3_rne(midpoint.next_down()), Some(lower));
        assert_eq!(encode_e4m3_rne(midpoint.next_up()), Some(lower + 1));
    }
    for unsupported in [448.0_f32.next_up(), f32::INFINITY, f32::NAN, -1.0] {
        assert_eq!(encode_e4m3_rne(unsupported), None);
    }
}

#[test]
fn malformed_shapes_and_buffers_leave_output_untouched_without_large_allocations() {
    let input = [0_u16; 16];
    for (rows, width, input_len, expected) in [
        (0, 16, 16, Fp4ActivationError::EmptyShape),
        (1, 0, 16, Fp4ActivationError::EmptyShape),
        (usize::MAX, 16, 16, Fp4ActivationError::ShapeOverflow),
        (
            MAX_ELEMENTS,
            16,
            16,
            Fp4ActivationError::ElementLimit {
                elements: MAX_ELEMENTS * 16,
            },
        ),
        (
            1,
            16,
            15,
            Fp4ActivationError::Length {
                field: "input",
                actual: 15,
                expected: 16,
            },
        ),
        (
            1,
            32,
            16,
            Fp4ActivationError::Length {
                field: "input",
                actual: 16,
                expected: 32,
            },
        ),
    ] {
        let mut output = [0xdead_u16; 16];
        let error = fp4_activation_reference(
            &input[..input_len],
            rows,
            width,
            Fp4Mode::CompressedKv16E4m3,
            &mut output,
        )
        .err()
        .expect("invalid shape or input buffer");
        assert_eq!(error, expected);
        assert_eq!(output, [0xdead; 16]);
    }
    let mut short_output = [0xdead_u16; 15];
    let error = fp4_activation_reference(
        &input,
        1,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut short_output,
    )
    .err()
    .expect("short output buffer");
    assert_eq!(
        error,
        Fp4ActivationError::Length {
            field: "output",
            actual: 15,
            expected: 16
        }
    );
    assert_eq!(short_output, [0xdead; 15]);
}

fn fixture_floats(fixture: &serde_json::Value, field: &str) -> Vec<f32> {
    serde_json::from_value(fixture[field].clone()).expect("FP32 fixture array")
}

// Fixed two-key, width-16 composition. This is not a runtime cache API.
fn rotate_compressed_tails(input: &[u16], frequencies: &[f32]) -> Vec<u16> {
    let one = NonZeroUsize::new(1).expect("one");
    let two = NonZeroUsize::new(2).expect("two");
    assert_eq!(input.len(), 32);
    assert_eq!(frequencies.len(), 8);
    let mut tail: Vec<f32> = input
        .chunks_exact(16)
        .flat_map(|row| row[12..].iter().copied().map(bf16_to_f32))
        .collect();
    let frequencies: Vec<_> = frequencies
        .chunks_exact(2)
        .map(|pair| RotaryFrequency::new(pair[0], pair[1]).expect("finite frequency"))
        .collect();
    rotate_tail(
        &mut tail,
        RotaryTailLayout::new(one, two, one, two).expect("tail layout"),
        &frequencies,
        RotaryDirection::Forward,
    )
    .expect("rotated supplied latent");
    let mut result = input.to_vec();
    for (row, rotated) in result.chunks_exact_mut(16).zip(tail.chunks_exact(4)) {
        for (output, &value) in row[12..].iter_mut().zip(rotated) {
            *output = f32_to_bf16_rne(value);
        }
    }
    result
}

fn compressed_attention_output(kv: &[u16], fixture: &serde_json::Value) -> Vec<f32> {
    let nz = |value| NonZeroUsize::new(value).expect("fixed nonzero dimension");
    let keys: Vec<_> = kv.iter().copied().map(bf16_to_f32).collect();
    let indices: Vec<i32> =
        serde_json::from_value(fixture["indices_i32"].clone()).expect("indices");
    let scale: f32 = serde_json::from_value(fixture["softmax_scale"].clone()).expect("scale");
    sparse_attention_reference(
        &fixture_floats(fixture, "query_f32"),
        &keys,
        &fixture_floats(fixture, "sink_f32"),
        &indices,
        scale,
        SparseAttentionLayout::new(nz(1), nz(2), nz(1), nz(16), nz(2), nz(4))
            .expect("attention layout"),
    )
    .expect("compressed-only mathematical attention")
}

#[test]
fn rotated_fp4_compressed_keys_feed_sparse_attention_in_source_order() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-attention-reference.json"
    ))
    .expect("independent composition oracle");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["indices_i32"],
        serde_json::json!([1, 0, 1, -1, -1, -1, -1, -1])
    );
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["model_sha256"],
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture["source"]["kernel_sha256"],
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(fixture["receipt"]["device"], "cpu");
    let latent = fixture_words(&fixture, "latent_bf16");
    let frequencies = fixture_floats(&fixture, "frequencies_f32");
    let rotated = rotate_compressed_tails(&latent, &frequencies);
    assert_eq!(rotated, fixture_words(&fixture, "rotated_bf16"));
    let mut reconstructed = vec![0; 32];
    fp4_activation_reference(
        &rotated,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut reconstructed,
    )
    .expect("post-rotation quantization");
    assert_eq!(reconstructed, fixture_words(&fixture, "reconstructed_bf16"));
    let actual = compressed_attention_output(&reconstructed, &fixture);
    let expected = fixture_floats(&fixture, "output_f32");
    assert_eq!(expected.len(), 32);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "attention element {index}: {actual} != {expected}"
        );
    }
    assert!(actual[16..].iter().all(|value| value.to_bits() == 0));
    let mut early_quantized = vec![0; 32];
    fp4_activation_reference(
        &latent,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut early_quantized,
    )
    .expect("wrong-order distinguisher");
    let wrong_order = rotate_compressed_tails(&early_quantized, &frequencies);
    assert_ne!(wrong_order, reconstructed);
    for wrong_keys in [&wrong_order, &rotated] {
        let wrong = compressed_attention_output(wrong_keys, &fixture);
        assert!(
            wrong
                .iter()
                .zip(&actual)
                .any(|(wrong, actual)| (wrong - actual).abs() > 1e-3),
            "oracle must distinguish reordered or omitted quantization"
        );
    }
}

/// Joins the actual GPU score core and CPU selection with attention. The index
/// operands are supplied post-projection/post-RoPE, not produced by a full indexer.
#[cfg(feature = "metal")]
#[test]
fn fp4_index_scores_select_the_compressed_vector_consumed_by_attention() {
    use deepseek::indexer::index_scores_f32;
    use deepseek::selection::select_indices;

    let mut query = [0x3f80_u16; 64]; // two heads: +1, -1
    query[32..].fill(0xbf80);
    let mut keys = [0x4110_u16; 64]; // two keys: +9, -3
    keys[32..].fill(0xc040);
    let mut quantized_query = [0; 64];
    let mut quantized_keys = [0; 64];
    fp4_activation_reference(&query, 2, 32, Fp4Mode::Index32E8m0, &mut quantized_query)
        .expect("index query preparation");
    fp4_activation_reference(&keys, 2, 32, Fp4Mode::Index32E8m0, &mut quantized_keys)
        .expect("index key preparation");
    assert_eq!(&quantized_keys[..32], &[0x4100; 32]); // 9 -> 8 at scale 2
    assert_eq!(&quantized_keys[32..], &[0xc040; 32]); // -3 preserved
    let q: Vec<_> = quantized_query.iter().copied().map(bf16_to_f32).collect();
    let k: Vec<_> = quantized_keys.iter().copied().map(bf16_to_f32).collect();
    let scores = index_scores_f32(
        &q,
        &k,
        &[-4.0, 1.0],
        NonZeroUsize::new(32).expect("index width"),
    )
    .expect("Metal score reduction");
    // Dot rows [256,-96],[-256,96]; ReLU BEFORE signed head weighting.
    assert_eq!(
        scores
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [-1024.0_f32, 96.0].map(f32::to_bits)
    );
    let selected = select_indices(&scores, 2, 1, 0).expect("strict score cutoff");
    assert_eq!(selected, [1]);

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-attention-reference.json"
    ))
    .expect("supplied compressed latent fixture");
    let rotated = rotate_compressed_tails(
        &fixture_words(&fixture, "latent_bf16"),
        &fixture_floats(&fixture, "frequencies_f32"),
    );
    let mut cache = vec![0; 32];
    fp4_activation_reference(&rotated, 2, 16, Fp4Mode::CompressedKv16E4m3, &mut cache)
        .expect("compressed KV preparation");
    let cache: Vec<_> = cache.into_iter().map(bf16_to_f32).collect();
    let nz = |value| NonZeroUsize::new(value).expect("fixed nonzero dimension");
    let layout = SparseAttentionLayout::new(nz(1), nz(1), nz(1), nz(16), nz(2), nz(1))
        .expect("one selected compressed position");
    // Zero attention query and zero sink yield exactly half the selected KV:
    // one key contributes exp(0), the sink contributes exp(0) only to denominator.
    let output = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &selected, 0.25, layout)
        .expect("attention consumes calculated indices");
    let expected: Vec<_> = cache[16..]
        .iter()
        .map(|value| (value * 0.5).to_bits())
        .collect();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    let earlier = select_indices(&[scores[0], f32::NEG_INFINITY], 1, 1, 0)
        .expect("future compressed key is masked before selection");
    assert_eq!(earlier, [0]);
    let wrong = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &earlier, 0.25, layout)
        .expect("earlier causal prefix selects the other vector");
    assert!(
        output
            .iter()
            .zip(wrong)
            .any(|(actual, wrong)| (actual - wrong).abs() > 0.1)
    );
    let none =
        select_indices(&[f32::NEG_INFINITY; 2], 0, 1, 0).expect("no completed compressed group");
    assert_eq!(none, [-1]);
    let empty = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &none, 0.25, layout)
        .expect("unreachable keys do not contribute");
    assert!(empty.iter().all(|value| value.to_bits() == 0));
}
