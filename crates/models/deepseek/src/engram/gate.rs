//! Bounded scalar BF16 residual gate for already-projected V4.1 Engram values.
//!
//! This mirrors only the residual portion of pinned `Engram.forward`: supplied
//! BF16 stream/key/value tensors, per-copy q/k weights, normalization, signed
//! square-root sigmoid gate, and BF16 output narrowing. It does not hash IDs,
//! decode FP8 rows, project `wkv`, load a checkpoint, or run Metal.

use crate::precision::{bf16_to_f32, f32_to_bf16_rne};
use thiserror::Error;

const MAX_ENGRAM_GATE_ELEMENTS: usize = 1 << 20;

/// Explicit shape of one bounded Engram residual-gate call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngramGateLayout {
    batches: usize,
    positions: usize,
    copies: usize,
    width: usize,
}

impl EngramGateLayout {
    /// Creates a bounded source-shaped residual-gate layout.
    ///
    /// # Errors
    ///
    /// Returns [`EngramGateError`] when an explicit dimension is zero, derived
    /// shape arithmetic overflows, or a scalar buffer cap would be exceeded.
    pub fn new(
        batches: usize,
        positions: usize,
        copies: usize,
        width: usize,
    ) -> Result<Self, EngramGateError> {
        if [batches, positions, copies, width].contains(&0) {
            return Err(EngramGateError::EmptyDimension);
        }
        let layout = Self {
            batches,
            positions,
            copies,
            width,
        };
        for (field, elements) in [
            ("stream", layout.stream_elements()?),
            ("key", layout.stream_elements()?),
            ("value", layout.value_elements()?),
            ("weights", layout.weight_elements()?),
            ("gate", layout.gate_elements()?),
        ] {
            if elements > MAX_ENGRAM_GATE_ELEMENTS {
                return Err(EngramGateError::ElementLimit { field, elements });
            }
        }
        Ok(layout)
    }

    fn rows(self) -> Result<usize, EngramGateError> {
        product(&[self.batches, self.positions], "rows")
    }

    fn stream_elements(self) -> Result<usize, EngramGateError> {
        product(&[self.rows()?, self.copies, self.width], "stream")
    }

    fn value_elements(self) -> Result<usize, EngramGateError> {
        product(&[self.rows()?, self.width], "value")
    }

    fn weight_elements(self) -> Result<usize, EngramGateError> {
        product(&[self.copies, self.width], "weights")
    }

    fn gate_elements(self) -> Result<usize, EngramGateError> {
        product(&[self.rows()?, self.copies], "gate")
    }
}

/// Borrowed, source-shaped tensors for one Engram residual-gate call.
///
/// `stream` and `key` are `[batch, position, copy, width]`; `value` is
/// `[batch, position, width]`; q/k weights are `[copy, width]`; and `mask`,
/// when supplied, is `[batch, position]`.
#[derive(Clone, Copy, Debug)]
pub struct EngramGateInputs<'a> {
    pub stream: &'a [u16],
    pub key: &'a [u16],
    pub value: &'a [u16],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub mask: Option<&'a [bool]>,
}

/// Finite scalar controls validated separately from tensor layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngramGateParams {
    epsilon: f32,
    clamp_value: f32,
}

impl EngramGateParams {
    /// Creates finite, positive source controls for normalization and gating.
    ///
    /// # Errors
    ///
    /// Returns [`EngramGateError`] when either scalar is non-finite or not
    /// strictly positive.
    pub fn new(epsilon: f32, clamp_value: f32) -> Result<Self, EngramGateError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(EngramGateError::InvalidEpsilon);
        }
        if !clamp_value.is_finite() || clamp_value <= 0.0 {
            return Err(EngramGateError::InvalidClamp);
        }
        Ok(Self {
            epsilon,
            clamp_value,
        })
    }
}

/// Errors from bounded scalar Engram residual-gate staging.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngramGateError {
    /// Every source-shaped dimension must be nonzero.
    #[error("Engram gate dimensions must all be nonzero")]
    EmptyDimension,
    /// A derived shape could not be represented by `usize`.
    #[error("Engram gate shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    /// One bounded scalar buffer exceeds its fixed cap.
    #[error("Engram gate {field} has {elements} elements, maximum is 1048576")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    /// An input has an unexpected exact length.
    #[error("Engram gate {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Normalization epsilon must be finite and strictly positive.
    #[error("Engram gate epsilon must be finite and strictly positive")]
    InvalidEpsilon,
    /// The signed-square-root clamp must be finite and strictly positive.
    #[error("Engram gate clamp value must be finite and strictly positive")]
    InvalidClamp,
    /// A BF16 tensor input denotes NaN or infinity.
    #[error("Engram gate {field} BF16 value at element {element} is non-finite")]
    NonFiniteBf16 { field: &'static str, element: usize },
    /// A q/k weight input denotes NaN or infinity.
    #[error("Engram gate {field} weight at element {element} is non-finite")]
    NonFiniteWeight { field: &'static str, element: usize },
    /// A scalar intermediate or final BF16 narrowing did not remain finite.
    #[error("Engram gate overflowed at {stage}, row {row}, copy {copy}, feature {feature}")]
    ValueOverflow {
        stage: &'static str,
        row: usize,
        copy: usize,
        feature: usize,
    },
    /// A bounded private result could not be allocated.
    #[error("could not allocate {elements} Engram gate elements")]
    AllocationFailed { elements: usize },
}

/// Applies the source Engram residual gate to BF16 stream copies.
///
/// False mask rows zero the gate after the source calculation; their residual
/// still executes as `h + (0 * value)`. Results are BF16
/// `[batch, position, copy, width]`. This reference accepts only finite
/// supplied tensors, including masked rows: upstream non-finite behavior is
/// intentionally outside its parity contract. All checks and calculation
/// finish before `output` changes.
///
/// # Errors
///
/// Returns [`EngramGateError`] for shape, finite-value, arithmetic, or
/// allocation failures and leaves `output` unchanged.
pub fn engram_residual_gate_bf16_reference(
    inputs: EngramGateInputs<'_>,
    layout: EngramGateLayout,
    params: EngramGateParams,
    output: &mut [u16],
) -> Result<(), EngramGateError> {
    validate_inputs(inputs, layout, output)?;
    let (result, _) = calculate(inputs, layout, params)?;
    output.copy_from_slice(&result);
    Ok(())
}

fn calculate(
    inputs: EngramGateInputs<'_>,
    layout: EngramGateLayout,
    params: EngramGateParams,
) -> Result<(Vec<u16>, Vec<f32>), EngramGateError> {
    let result_elements = layout.stream_elements()?;
    let gate_elements = layout.gate_elements()?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(result_elements)
        .map_err(|_| EngramGateError::AllocationFailed {
            elements: result_elements,
        })?;
    let mut gates = Vec::new();
    gates
        .try_reserve_exact(gate_elements)
        .map_err(|_| EngramGateError::AllocationFailed {
            elements: gate_elements,
        })?;

    for row in 0..layout.rows()? {
        for copy in 0..layout.copies {
            let stream_start = (row * layout.copies + copy) * layout.width;
            let mut stream_square_sum = 0.0_f32;
            let mut key_square_sum = 0.0_f32;
            let mut dot_sum = 0.0_f32;
            for feature in 0..layout.width {
                let index = stream_start + feature;
                let h = bf16_to_f32(inputs.stream[index]);
                let k = bf16_to_f32(inputs.key[index]);
                let weight_index = copy * layout.width + feature;
                let weight = inputs.q_weight[weight_index] * inputs.k_weight[weight_index];
                finite(weight, "qk_product", row, copy, feature)?;
                let weighted_h = h * weight;
                finite(weighted_h, "weighted_stream", row, copy, feature)?;
                let dot_term = weighted_h * k;
                finite(dot_term, "dot_product", row, copy, feature)?;
                dot_sum += dot_term;
                finite(dot_sum, "dot_sum", row, copy, feature)?;
                let h_square = h * h;
                finite(h_square, "stream_square", row, copy, feature)?;
                stream_square_sum += h_square;
                finite(stream_square_sum, "stream_square_sum", row, copy, feature)?;
                let k_square = k * k;
                finite(k_square, "key_square", row, copy, feature)?;
                key_square_sum += k_square;
                finite(key_square_sum, "key_square_sum", row, copy, feature)?;
            }
            let width = exact_width_as_f32(layout.width);
            let stream_variance = stream_square_sum / width + params.epsilon;
            finite(stream_variance, "stream_variance", row, copy, 0)?;
            let key_variance = key_square_sum / width + params.epsilon;
            finite(key_variance, "key_variance", row, copy, 0)?;
            let rstd = stream_variance.sqrt().recip() * key_variance.sqrt().recip();
            finite(rstd, "rstd", row, copy, 0)?;
            let dot = dot_sum * rstd * width.sqrt().recip();
            finite(dot, "scaled_dot", row, copy, 0)?;
            let signed_root = signed_sqrt_clamp(dot, params.clamp_value);
            finite(signed_root, "signed_sqrt", row, copy, 0)?;
            let mut gate = sigmoid(signed_root);
            if inputs.mask.is_some_and(|values| !values[row]) {
                gate = 0.0;
            }
            finite(gate, "gate", row, copy, 0)?;
            gates.push(gate);
            let value_start = row * layout.width;
            for feature in 0..layout.width {
                let h = bf16_to_f32(inputs.stream[stream_start + feature]);
                let scaled_value = gate * bf16_to_f32(inputs.value[value_start + feature]);
                finite(scaled_value, "value_product", row, copy, feature)?;
                let mixed = h + scaled_value;
                finite(mixed, "residual_sum", row, copy, feature)?;
                result.push(round_bf16(mixed, "bf16_output", row, copy, feature)?);
            }
        }
    }
    Ok((result, gates))
}

fn validate_inputs(
    inputs: EngramGateInputs<'_>,
    layout: EngramGateLayout,
    output: &[u16],
) -> Result<(), EngramGateError> {
    for (field, actual, expected) in [
        ("stream", inputs.stream.len(), layout.stream_elements()?),
        ("key", inputs.key.len(), layout.stream_elements()?),
        ("value", inputs.value.len(), layout.value_elements()?),
        ("q", inputs.q_weight.len(), layout.weight_elements()?),
        ("k", inputs.k_weight.len(), layout.weight_elements()?),
        ("output", output.len(), layout.stream_elements()?),
    ] {
        check_length(field, actual, expected)?;
    }
    if let Some(mask) = inputs.mask {
        check_length("mask", mask.len(), layout.rows()?)?;
    }
    for (field, values) in [
        ("stream", inputs.stream),
        ("key", inputs.key),
        ("value", inputs.value),
    ] {
        for (element, &bits) in values.iter().enumerate() {
            if !bf16_to_f32(bits).is_finite() {
                return Err(EngramGateError::NonFiniteBf16 { field, element });
            }
        }
    }
    for (field, values) in [("q", inputs.q_weight), ("k", inputs.k_weight)] {
        for (element, &weight) in values.iter().enumerate() {
            if !weight.is_finite() {
                return Err(EngramGateError::NonFiniteWeight { field, element });
            }
        }
    }
    Ok(())
}

fn product(values: &[usize], field: &'static str) -> Result<usize, EngramGateError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(EngramGateError::ShapeOverflow { field })
    })
}

fn check_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EngramGateError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EngramGateError::Length {
            field,
            actual,
            expected,
        })
    }
}

fn finite(
    value: f32,
    stage: &'static str,
    row: usize,
    copy: usize,
    feature: usize,
) -> Result<(), EngramGateError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EngramGateError::ValueOverflow {
            stage,
            row,
            copy,
            feature,
        })
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn signed_sqrt_clamp(dot: f32, clamp_value: f32) -> f32 {
    dot.abs().max(clamp_value).sqrt().copysign(dot)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "EngramGateLayout caps every dimension at 2^20, below the FP32 exact-integer limit"
)]
fn exact_width_as_f32(width: usize) -> f32 {
    width as f32
}

fn round_bf16(
    value: f32,
    stage: &'static str,
    row: usize,
    copy: usize,
    feature: usize,
) -> Result<u16, EngramGateError> {
    let bf16 = f32_to_bf16_rne(value);
    if bf16_to_f32(bf16).is_finite() {
        Ok(bf16)
    } else {
        Err(EngramGateError::ValueOverflow {
            stage,
            row,
            copy,
            feature,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        EngramGateError, EngramGateInputs, EngramGateLayout, EngramGateParams, calculate,
        engram_residual_gate_bf16_reference, signed_sqrt_clamp,
    };

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/engram-gate-reference.json"
        ))
        .expect("checked-in Engram gate fixture")
    }

    fn tensor<'a>(fixture: &'a Value, name: &str) -> &'a Value {
        fixture["tensors"]
            .as_array()
            .expect("fixture tensors")
            .iter()
            .find(|candidate| candidate["name"] == name)
            .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
    }

    fn u16_values(tensor: &Value) -> Vec<u16> {
        tensor["values"]
            .as_array()
            .expect("flat BF16 values")
            .iter()
            .map(|value| u16::try_from(value.as_u64().expect("u16 value")).expect("u16 range"))
            .collect()
    }

    fn f32_values(tensor: &Value) -> Vec<f32> {
        tensor["values"]
            .as_array()
            .expect("flat f32 words")
            .iter()
            .map(|value| {
                f32::from_bits(
                    u32::try_from(value.as_u64().expect("u32 value")).expect("u32 range"),
                )
            })
            .collect()
    }

    fn bool_values(tensor: &Value) -> Vec<bool> {
        tensor["values"]
            .as_array()
            .expect("flat bool values")
            .iter()
            .map(|value| value.as_bool().expect("bool value"))
            .collect()
    }

    fn fixture_parameter(fixture: &Value, name: &str) -> f32 {
        let bits = u32::try_from(
            fixture["parameters"][name]
                .as_u64()
                .expect("fixture parameter bits"),
        )
        .expect("fixture parameter must fit FP32 bits");
        f32::from_bits(bits)
    }

    #[test]
    fn layout_rejects_zero_overflow_and_cap_excess() {
        for dimensions in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            assert_eq!(
                EngramGateLayout::new(dimensions.0, dimensions.1, dimensions.2, dimensions.3),
                Err(EngramGateError::EmptyDimension)
            );
        }
        assert_eq!(
            EngramGateLayout::new(usize::MAX, 2, 1, 1),
            Err(EngramGateError::ShapeOverflow { field: "rows" })
        );
        assert_eq!(
            EngramGateLayout::new(1, 1, 1, (1 << 20) + 1),
            Err(EngramGateError::ElementLimit {
                field: "stream",
                elements: (1 << 20) + 1,
            })
        );
    }

    #[test]
    fn params_require_finite_positive_controls() {
        assert!(EngramGateParams::new(f32::MIN_POSITIVE, f32::MIN_POSITIVE).is_ok());
        for epsilon in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert_eq!(
                EngramGateParams::new(epsilon, 1.0),
                Err(EngramGateError::InvalidEpsilon)
            );
        }
        for clamp_value in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert_eq!(
                EngramGateParams::new(1.0, clamp_value),
                Err(EngramGateError::InvalidClamp)
            );
        }
    }

    #[test]
    fn rejects_tensor_mask_and_output_length_mismatches_without_writing() {
        let layout = EngramGateLayout::new(1, 2, 1, 1).expect("small layout");
        let params = EngramGateParams::new(1.0e-6, 1.0e-6).expect("small controls");
        let stream = [0x3f80_u16, 0x3f80];
        let key = [0x3f80_u16, 0x3f80];
        let value = [0x3f80_u16, 0x3f80];
        let weights = [1.0_f32];
        let mut output = [0xdead_u16, 0xdead];

        let wrong_stream = EngramGateInputs {
            stream: &stream[..1],
            key: &key,
            value: &value,
            q_weight: &weights,
            k_weight: &weights,
            mask: None,
        };
        assert!(matches!(
            engram_residual_gate_bf16_reference(wrong_stream, layout, params, &mut output),
            Err(EngramGateError::Length {
                field: "stream",
                ..
            })
        ));
        assert_eq!(output, [0xdead, 0xdead]);

        let wrong_mask = EngramGateInputs {
            stream: &stream,
            key: &key,
            value: &value,
            q_weight: &weights,
            k_weight: &weights,
            mask: Some(&[true]),
        };
        assert!(matches!(
            engram_residual_gate_bf16_reference(wrong_mask, layout, params, &mut output),
            Err(EngramGateError::Length { field: "mask", .. })
        ));
        assert_eq!(output, [0xdead, 0xdead]);

        let inputs = EngramGateInputs {
            stream: &stream,
            key: &key,
            value: &value,
            q_weight: &weights,
            k_weight: &weights,
            mask: None,
        };
        assert!(matches!(
            engram_residual_gate_bf16_reference(inputs, layout, params, &mut output[..1]),
            Err(EngramGateError::Length {
                field: "output",
                ..
            })
        ));
        assert_eq!(output, [0xdead, 0xdead]);
    }

    #[test]
    fn captured_source_gate_matches_bf16_output_and_close_fp32_gate() {
        let fixture = fixture();
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["source"]["symbol"], "Engram.forward");
        assert_eq!(fixture["receipt"]["device"], "cpu");
        let layout = EngramGateLayout::new(1, 3, 2, 4).expect("captured shape");
        let stream = u16_values(tensor(&fixture, "x"));
        let key = u16_values(tensor(&fixture, "stubbed_wkv_key"));
        let value = u16_values(tensor(&fixture, "stubbed_wkv_value"));
        let q = f32_values(tensor(&fixture, "q_weight"));
        let k = f32_values(tensor(&fixture, "k_weight"));
        let mask = bool_values(tensor(&fixture, "token_mask"));
        let epsilon = fixture_parameter(&fixture, "eps_f32_bits");
        let clamp = fixture_parameter(&fixture, "clamp_value_f32_bits");
        let params = EngramGateParams::new(epsilon, clamp).expect("captured controls");
        let inputs = EngramGateInputs {
            stream: &stream,
            key: &key,
            value: &value,
            q_weight: &q,
            k_weight: &k,
            mask: Some(&mask),
        };
        let mut output = vec![0_u16; stream.len()];
        engram_residual_gate_bf16_reference(inputs, layout, params, &mut output)
            .expect("captured gate must remain finite");
        assert_eq!(output, u16_values(tensor(&fixture, "expected_output")));

        let (_, gates) = calculate(inputs, layout, params).expect("captured scalar gate");
        let expected_gates = f32_values(tensor(&fixture, "expected_gate"));
        // Torch reductions can select a different valid FP32 reduction tree;
        // the captured BF16 output is exact, while gate checks use this narrow
        // scalar tolerance instead of claiming bitwise FP32 parity.
        for (actual, expected) in gates.iter().zip(expected_gates) {
            assert!(
                (actual - expected).abs() <= 2.0e-6,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn unmasked_zero_dot_uses_positive_clamp_floor_and_exact_bf16_output() {
        let fixture = fixture();
        let layout = EngramGateLayout::new(1, 3, 2, 4).expect("captured shape");
        let stream = u16_values(tensor(&fixture, "x"));
        let key = u16_values(tensor(&fixture, "stubbed_wkv_key"));
        let value = u16_values(tensor(&fixture, "stubbed_wkv_value"));
        let q = f32_values(tensor(&fixture, "q_weight"));
        let k = f32_values(tensor(&fixture, "k_weight"));
        let epsilon = fixture_parameter(&fixture, "eps_f32_bits");
        let clamp = fixture_parameter(&fixture, "clamp_value_f32_bits");
        let params = EngramGateParams::new(epsilon, clamp).expect("captured controls");
        let inputs = EngramGateInputs {
            stream: &stream,
            key: &key,
            value: &value,
            q_weight: &q,
            k_weight: &k,
            mask: None,
        };
        let mut output = vec![0_u16; stream.len()];
        engram_residual_gate_bf16_reference(inputs, layout, params, &mut output)
            .expect("unmasked captured gate");
        assert_eq!(
            output,
            u16_values(tensor(&fixture, "unmasked_zero_dot_output"))
        );
        let (_, gates) = calculate(inputs, layout, params).expect("unmasked scalar gate");
        let floor = (clamp.sqrt()).exp() / (1.0 + (clamp.sqrt()).exp());
        assert!(gates[4] > 0.5);
        assert!((gates[4] - floor).abs() <= 1.0e-7);
        assert!((gates[5] - floor).abs() <= 1.0e-7);
    }

    #[test]
    fn rejects_nonfinite_and_late_private_overflow_without_writing_output() {
        let layout = EngramGateLayout::new(1, 1, 1, 1).expect("small layout");
        let params = EngramGateParams::new(1.0e-6, 1.0e-6).expect("small controls");
        let mut output = [0xdead_u16];
        assert!(matches!(
            engram_residual_gate_bf16_reference(
                EngramGateInputs {
                    stream: &[0x7fc0],
                    key: &[0x3f80],
                    value: &[0x3f80],
                    q_weight: &[1.0],
                    k_weight: &[1.0],
                    mask: None,
                },
                layout,
                params,
                &mut output,
            ),
            Err(EngramGateError::NonFiniteBf16 {
                field: "stream",
                ..
            })
        ));
        assert_eq!(output, [0xdead]);
        let late_layout = EngramGateLayout::new(1, 2, 1, 1).expect("two-row layout");
        let mut late_output = [0xdead_u16, 0xdead];
        assert!(matches!(
            engram_residual_gate_bf16_reference(
                EngramGateInputs {
                    // The first row produces a private result. The second is
                    // finite input but overflows only in its square reduction.
                    stream: &[0x3f80, 0x7f7f],
                    key: &[0x3f80, 0x3f80],
                    value: &[0x3f80, 0x3f80],
                    q_weight: &[1.0],
                    k_weight: &[1.0],
                    mask: None,
                },
                late_layout,
                params,
                &mut late_output,
            ),
            Err(EngramGateError::ValueOverflow {
                stage: "stream_square",
                row: 1,
                copy: 0,
                feature: 0,
            })
        ));
        assert_eq!(late_output, [0xdead, 0xdead]);
    }

    #[test]
    fn captured_masked_signed_zero_residual_matches_source() {
        let fixture = fixture();
        let layout = EngramGateLayout::new(1, 3, 2, 4).expect("captured shape");
        let stream = u16_values(tensor(&fixture, "masked_signed_zero_stream"));
        let key = u16_values(tensor(&fixture, "stubbed_wkv_key"));
        let value = u16_values(tensor(&fixture, "stubbed_wkv_value"));
        let q = f32_values(tensor(&fixture, "q_weight"));
        let k = f32_values(tensor(&fixture, "k_weight"));
        let mask = bool_values(tensor(&fixture, "token_mask"));
        let epsilon = fixture_parameter(&fixture, "eps_f32_bits");
        let clamp = fixture_parameter(&fixture, "clamp_value_f32_bits");
        let params = EngramGateParams::new(epsilon, clamp).expect("captured controls");
        let inputs = EngramGateInputs {
            stream: &stream,
            key: &key,
            value: &value,
            q_weight: &q,
            k_weight: &k,
            mask: Some(&mask),
        };
        let mut output = vec![0_u16; stream.len()];
        engram_residual_gate_bf16_reference(inputs, layout, params, &mut output)
            .expect("finite signed-zero capture");
        assert_eq!(
            output,
            u16_values(tensor(&fixture, "masked_signed_zero_output"))
        );
    }

    #[test]
    fn qk_weight_is_formed_before_the_stream_product() {
        let layout = EngramGateLayout::new(1, 1, 1, 1).expect("scalar layout");
        let params = EngramGateParams::new(1.0e-6, 1.0e-6).expect("scalar controls");
        let inputs = EngramGateInputs {
            // 2^50 BF16; squaring remains finite, while a premature h*q would
            // overflow before the cancelling k weight is applied.
            stream: &[0x5880],
            key: &[0x3f80],
            value: &[0x3f80],
            q_weight: &[f32::from_bits(0x7180_0000)], // 2^100
            k_weight: &[f32::from_bits(0x0d80_0000)], // 2^-100
            mask: None,
        };
        let mut output = [0_u16];
        engram_residual_gate_bf16_reference(inputs, layout, params, &mut output)
            .expect("source q*k staging remains finite");
        assert_eq!(output, [0x5880]);
    }

    #[test]
    fn signed_sqrt_clamp_retains_both_zero_signs() {
        let clamp = 1.0e-6;
        assert!(signed_sqrt_clamp(0.0, clamp).is_sign_positive());
        assert!(signed_sqrt_clamp(-0.0, clamp).is_sign_negative());
    }
}
