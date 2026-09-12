//! Test-only scalar model of the pinned FP4 activation staging.
//!
//! This uses software round-to-nearest-even encoders as an explicit reference
//! assumption. It is not CUDA/TileLang instruction-level parity or a packed
//! FP4 storage implementation.

use deepseek::precision::{decode_e2m1, decode_e4m3fn, decode_e8m0};
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
