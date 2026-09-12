//! Bounded scalar BF16 Hyper-Connections pre/post mixing references.
//!
//! These primitives mirror only the pinned `Block.hc_pre` and `Block.hc_post`
//! tensor equations. They do not derive Sinkhorn coefficients, run a block,
//! normalize a sublayer, or project a checkpoint.

use thiserror::Error;

/// Maximum Hyper-Connections copies accepted by one scalar mixing call.
pub const MAX_HC_COPIES: usize = 16;

/// Maximum feature width accepted by one scalar mixing call.
pub const MAX_HC_MIX_WIDTH: usize = 16_384;

/// An invalid bounded Hyper-Connections scalar mixing request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum HcMixError {
    /// At least one Hyper-Connections copy is required.
    #[error("Hyper-Connections mixing requires at least one copy")]
    EmptyCopies,
    /// The explicit pre-mix feature width is zero.
    #[error("Hyper-Connections pre-mix requires a nonzero feature width")]
    EmptyWidth,
    /// The requested number of copies exceeds this reference's bound.
    #[error("Hyper-Connections copy count {copies} exceeds maximum {max_copies}")]
    CopyCountTooLarge {
        /// Requested copy count.
        copies: usize,
        /// Maximum accepted copy count.
        max_copies: usize,
    },
    /// The requested feature width exceeds this reference's bound.
    #[error("Hyper-Connections feature width {width} exceeds maximum {max_width}")]
    WidthTooLarge {
        /// Requested feature width.
        width: usize,
        /// Maximum accepted feature width.
        max_width: usize,
    },
    /// A required derived buffer length overflowed `usize`.
    #[error("Hyper-Connections {field} shape overflowed")]
    ShapeOverflow {
        /// Buffer role whose derived length overflowed.
        field: &'static str,
    },
    /// A supplied direct-runtime buffer has an unexpected exact length.
    #[error("Hyper-Connections {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Required element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// A BF16 residual value is non-finite.
    #[error(
        "Hyper-Connections residual BF16 value at copy {copy}, feature {feature} is non-finite"
    )]
    NonFiniteResidual {
        /// Input copy index.
        copy: usize,
        /// Feature index within the copy.
        feature: usize,
    },
    /// A BF16 collapsed sublayer value is non-finite.
    #[error("Hyper-Connections sublayer BF16 value at feature {feature} is non-finite")]
    NonFiniteSublayer {
        /// Feature index.
        feature: usize,
    },
    /// A pre-collapse coefficient is non-finite.
    #[error("Hyper-Connections pre coefficient at copy {copy} is non-finite")]
    NonFinitePre {
        /// Copy index.
        copy: usize,
    },
    /// A post-expansion coefficient is non-finite.
    #[error("Hyper-Connections post coefficient at copy {copy} is non-finite")]
    NonFinitePost {
        /// Output-copy index.
        copy: usize,
    },
    /// A residual-combination coefficient is non-finite.
    #[error(
        "Hyper-Connections comb coefficient from copy {input_copy} to copy {output_copy} is non-finite"
    )]
    NonFiniteComb {
        /// Residual input-copy index.
        input_copy: usize,
        /// Expanded output-copy index.
        output_copy: usize,
    },
    /// An FP32 intermediate or final BF16 conversion is non-finite.
    #[error("Hyper-Connections mixing overflowed at {stage}, copy {copy}, feature {feature}")]
    ValueOverflow {
        /// Named scalar stage.
        stage: &'static str,
        /// Output or input copy relevant to the stage.
        copy: usize,
        /// Feature index.
        feature: usize,
    },
}

/// Collapses BF16 residual copies using finite FP32 pre-mix coefficients.
///
/// `residual` is copy-major `[copies, width]`; `pre` is `[copies]`; and
/// `output` is `[width]`. This is the source `sum(pre.unsqueeze(-1) *
/// residual.float(), dim=2).to(residual.dtype)` equation. All finite FP32
/// coefficients are accepted; this primitive does not impose a Sinkhorn
/// domain. The output remains unchanged on every error.
///
/// # Errors
///
/// Returns [`HcMixError`] for invalid exact shapes, non-finite inputs, or an
/// intermediate that cannot remain finite in this scalar FP32 reference.
pub fn hc_pre_bf16_reference(
    residual: &[u16],
    pre: &[f32],
    width: usize,
    output: &mut [u16],
) -> Result<(), HcMixError> {
    let copies = validate_pre_shape(residual, pre, width, output)?;
    validate_residual(residual, copies, width)?;
    for (copy, &coefficient) in pre.iter().enumerate() {
        if !coefficient.is_finite() {
            return Err(HcMixError::NonFinitePre { copy });
        }
    }

    let mut result = Vec::with_capacity(width);
    for feature in 0..width {
        let mut sum = 0.0_f32;
        for (copy, &coefficient) in pre.iter().enumerate() {
            let term = coefficient * bf16_to_f32(residual[copy * width + feature]);
            if !term.is_finite() {
                return Err(HcMixError::ValueOverflow {
                    stage: "pre_product",
                    copy,
                    feature,
                });
            }
            sum += term;
            if !sum.is_finite() {
                return Err(HcMixError::ValueOverflow {
                    stage: "pre_sum",
                    copy,
                    feature,
                });
            }
        }
        result.push(round_finite_bf16(sum, "pre_bf16", 0, feature)?);
    }
    output.copy_from_slice(&result);
    Ok(())
}

/// Expands a BF16 sublayer result and combines each residual input copy.
///
/// `sublayer` is `[width]`, `residual` and `output` are copy-major
/// `[copies, width]`, `post` is `[copies]`, and `comb` is row-major
/// `[input_copy, output_copy]`. The exact source orientation is
/// `output[j, d] = post[j] * sublayer[d] + sum_i(comb[i, j] * residual[i, d])`.
/// All finite FP32 coefficients are accepted; this primitive does not impose
/// a Sinkhorn domain. The output remains unchanged on every error.
///
/// # Errors
///
/// Returns [`HcMixError`] for invalid exact shapes, non-finite inputs, or an
/// intermediate that cannot remain finite in this scalar FP32 reference.
pub fn hc_post_bf16_reference(
    sublayer: &[u16],
    residual: &[u16],
    post: &[f32],
    comb: &[f32],
    output: &mut [u16],
) -> Result<(), HcMixError> {
    let width = sublayer.len();
    let copies = validate_post_shape(sublayer, residual, post, comb, output)?;
    for (feature, &bits) in sublayer.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(HcMixError::NonFiniteSublayer { feature });
        }
    }
    validate_residual(residual, copies, width)?;
    for (copy, &coefficient) in post.iter().enumerate() {
        if !coefficient.is_finite() {
            return Err(HcMixError::NonFinitePost { copy });
        }
    }
    for input_copy in 0..copies {
        for output_copy in 0..copies {
            if !comb[input_copy * copies + output_copy].is_finite() {
                return Err(HcMixError::NonFiniteComb {
                    input_copy,
                    output_copy,
                });
            }
        }
    }

    let mut result = Vec::with_capacity(output.len());
    for output_copy in 0..copies {
        for feature in 0..width {
            let post_product = post[output_copy] * bf16_to_f32(sublayer[feature]);
            if !post_product.is_finite() {
                return Err(HcMixError::ValueOverflow {
                    stage: "post_product",
                    copy: output_copy,
                    feature,
                });
            }
            // Match `post * x + torch.sum(comb * residual, dim=2)`: reduce
            // the residual branch first, before the final addition.
            let mut residual_sum = 0.0_f32;
            for input_copy in 0..copies {
                let term = comb[input_copy * copies + output_copy]
                    * bf16_to_f32(residual[input_copy * width + feature]);
                if !term.is_finite() {
                    return Err(HcMixError::ValueOverflow {
                        stage: "comb_product",
                        copy: output_copy,
                        feature,
                    });
                }
                residual_sum += term;
                if !residual_sum.is_finite() {
                    return Err(HcMixError::ValueOverflow {
                        stage: "residual_sum",
                        copy: output_copy,
                        feature,
                    });
                }
            }
            let value = post_product + residual_sum;
            if !value.is_finite() {
                return Err(HcMixError::ValueOverflow {
                    stage: "post_sum",
                    copy: output_copy,
                    feature,
                });
            }
            result.push(round_finite_bf16(value, "post_bf16", output_copy, feature)?);
        }
    }
    output.copy_from_slice(&result);
    Ok(())
}

fn validate_pre_shape(
    residual: &[u16],
    pre: &[f32],
    width: usize,
    output: &[u16],
) -> Result<usize, HcMixError> {
    validate_width(width)?;
    let copies = validate_copies(pre.len())?;
    let expected_residual = copies
        .checked_mul(width)
        .ok_or(HcMixError::ShapeOverflow { field: "residual" })?;
    check_length("residual", expected_residual, residual.len())?;
    check_length("output", width, output.len())?;
    Ok(copies)
}

fn validate_post_shape(
    sublayer: &[u16],
    residual: &[u16],
    post: &[f32],
    comb: &[f32],
    output: &[u16],
) -> Result<usize, HcMixError> {
    let width = sublayer.len();
    validate_width(width)?;
    let copies = validate_copies(post.len())?;
    let expected_residual = copies
        .checked_mul(width)
        .ok_or(HcMixError::ShapeOverflow { field: "residual" })?;
    let expected_comb = copies
        .checked_mul(copies)
        .ok_or(HcMixError::ShapeOverflow { field: "comb" })?;
    check_length("residual", expected_residual, residual.len())?;
    check_length("comb", expected_comb, comb.len())?;
    check_length("output", expected_residual, output.len())?;
    Ok(copies)
}

fn validate_copies(copies: usize) -> Result<usize, HcMixError> {
    if copies == 0 {
        return Err(HcMixError::EmptyCopies);
    }
    if copies > MAX_HC_COPIES {
        return Err(HcMixError::CopyCountTooLarge {
            copies,
            max_copies: MAX_HC_COPIES,
        });
    }
    Ok(copies)
}

fn validate_width(width: usize) -> Result<(), HcMixError> {
    if width == 0 {
        return Err(HcMixError::EmptyWidth);
    }
    if width > MAX_HC_MIX_WIDTH {
        return Err(HcMixError::WidthTooLarge {
            width,
            max_width: MAX_HC_MIX_WIDTH,
        });
    }
    Ok(())
}

fn check_length(field: &'static str, expected: usize, actual: usize) -> Result<(), HcMixError> {
    if actual == expected {
        Ok(())
    } else {
        Err(HcMixError::Length {
            field,
            expected,
            actual,
        })
    }
}

fn validate_residual(residual: &[u16], copies: usize, width: usize) -> Result<(), HcMixError> {
    for (index, &bits) in residual.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(HcMixError::NonFiniteResidual {
                copy: index / width,
                feature: index % width,
            });
        }
    }
    debug_assert_eq!(residual.len(), copies * width);
    Ok(())
}

fn round_finite_bf16(
    value: f32,
    stage: &'static str,
    copy: usize,
    feature: usize,
) -> Result<u16, HcMixError> {
    let bits = f32_to_bf16_rne(value);
    if bf16_to_f32(bits).is_finite() {
        Ok(bits)
    } else {
        Err(HcMixError::ValueOverflow {
            stage,
            copy,
            feature,
        })
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("an FP32 high half always fits BF16 storage")
}

#[cfg(test)]
mod tests {
    use super::{HcMixError, hc_post_bf16_reference, hc_pre_bf16_reference};

    #[test]
    fn pre_collapses_copy_major_bf16_residuals() {
        let mut output = [0_u16; 2];
        hc_pre_bf16_reference(
            &[0x4000, 0xbf80, 0x40c0, 0x4040], // [[2, -1], [6, 3]]
            &[0.25, 0.5],
            2,
            &mut output,
        )
        .expect("finite collapse");
        assert_eq!(output, [0x4060, 0x3fa0]); // [3.5, 1.25]
    }

    #[test]
    fn post_comb_uses_input_copy_then_output_copy_axes() {
        let mut output = [0_u16; 2];
        hc_post_bf16_reference(
            &[0x4080],         // sublayer [4]
            &[0x4000, 0x40c0], // residual copies [2, 6]
            &[0.5, 1.5],
            &[0.25, 0.75, 0.5, 0.125], // comb[input_copy, output_copy]
            &mut output,
        )
        .expect("finite expansion");
        assert_eq!(output, [0x40b0, 0x4104]); // [5.5, 8.25]
    }

    #[test]
    fn post_reduces_the_residual_branch_before_adding_the_sublayer_branch() {
        let mut output = [0_u16; 2];
        hc_post_bf16_reference(
            &[0x3f80],         // sublayer [1]
            &[0x4b80, 0xcb80], // residual copies [2^24, -2^24]
            &[1.0, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &mut output,
        )
        .expect("finite ordered expansion");
        // The residual sum cancels to zero before +1; adding +1 first loses
        // it to FP32 precision and incorrectly produces zero.
        assert_eq!(output, [0x3f80, 0x0000]);
    }

    #[test]
    fn post_rounds_signed_nonexact_fp32_results_to_bf16() {
        let mut output = [0_u16; 2];
        hc_post_bf16_reference(&[0x3f80], &[0, 0], &[1.1, -1.1], &[0.0; 4], &mut output)
            .expect("finite BF16 round");
        assert_eq!(output, [0x3f8d, 0xbf8d]);
    }

    #[test]
    fn rejects_exact_shape_failures_without_writing_output() {
        let mut output = [0xdead_u16; 2];
        assert_eq!(
            hc_pre_bf16_reference(&[0x3f80], &[1.0, 1.0], 1, &mut output),
            Err(HcMixError::Length {
                field: "residual",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(output, [0xdead, 0xdead]);
        assert!(matches!(
            hc_post_bf16_reference(&[0x3f80], &[0x3f80], &[1.0], &[], &mut [0]),
            Err(HcMixError::Length { field: "comb", .. })
        ));
    }

    #[test]
    fn late_bf16_conversion_overflow_leaves_every_output_slot_unchanged() {
        let mut output = [0xdead_u16; 2];
        assert!(matches!(
            hc_post_bf16_reference(&[0x7f7f], &[0, 0], &[1.0, 1.003], &[0.0; 4], &mut output,),
            Err(HcMixError::ValueOverflow {
                stage: "post_bf16",
                copy: 1,
                feature: 0,
            })
        ));
        assert_eq!(output, [0xdead, 0xdead]);
    }
}
