//! Bounded scalar `DeepSeek` block, Engram, and final-head components.
//!
//! This module accepts runtime operands directly. It does not load a
//! checkpoint or a captured fixture, and it contains no expected-output
//! comparisons.
//!
//! ```
//! use deepseek::reduced::FinalHead;
//!
//! // One token with two HC copies and two hidden elements, in BF16 bits.
//! let norm = [0x3f80, 0x3f80];
//! let weights = [1.0, 0.0, 0.0, 1.0];
//! let head = FinalHead::new(&norm, &weights, 2, 2, 1.0)?;
//! let result = head.forward(&[0x3f80, 0, 0, 0], &[1.0, 0.0])?;
//! assert_eq!(result.logits(), &[0.816_406_25, 0.0]);
//! # Ok::<(), deepseek::reduced::FinalHeadError>(())
//! ```

mod ratio_two;
pub use ratio_two::{
    RatioTwoCompressedOwner, RatioTwoOwnerCall, RatioTwoOwnerDiagnostic, RatioTwoOwnerError,
    RatioTwoOwnerLayout, RatioTwoOwnerWeights,
};

mod layer_one;
pub use layer_one::{
    LayerOneCall, LayerOneConfig, LayerOneSession, LayerOneSessionError, LayerOneStepOutput,
    PreviousLayerThreeKeys,
};

mod layer_three;
pub use layer_three::{
    LayerThreeCall, LayerThreeConfig, LayerThreeSession, LayerThreeSessionError,
    LayerThreeStepOutput,
};

mod artifact;
mod candidates;
pub use artifact::{
    ArtifactError, MAX_REDUCED_ARTIFACT_BYTES, ReducedArtifact, ReducedGeneration,
    ReducedGenerationStop,
};
mod request;
pub use request::{
    BlockDefinition, EngramDefinition, HeadPositions, LayerFourDefinition, LayerKind,
    LayerOneDefinition, LayerStepOutput, LayerThreeDefinition, RequestError, RequestModel,
    RequestSession, RequestStepOutput, ReusedAttentionDefinition, ScheduleError,
    ScheduledAttentionOutput, ScheduledLayer, StartupDefinition, StepSources,
};
mod layer_four;
pub use candidates::{CandidateProjection, CandidateProjector, CandidateProjectorError};
pub use layer_four::{
    LayerFourCall, LayerFourConfig, LayerFourSession, LayerFourSessionError, LayerFourStepOutput,
    LayerThreePublication,
};

mod input;
pub use input::{AttentionInput, AttentionInputError, AttentionInputOutput};
mod startup;
pub use startup::{EmbeddingRowSource, StartupSession, StartupSessionError, StartupStepOutput};

mod engram;
pub use engram::{
    EngramSession, EngramSessionConfig, EngramSessionError, EngramSessionWeights, EngramStepOutput,
};

mod block;
pub use block::{BlockTailDiagnostic, BlockTailError, BlockTailReference};
pub mod checkpoint_model;

use thiserror::Error;

use crate::{
    hc::mixing::{HcMixError, MAX_HC_COPIES, MAX_HC_MIX_WIDTH, hc_pre_bf16_reference},
    norm::{MAX_RMS_NORM_WIDTH, RmsNormError, rms_norm_bf16_reference},
    precision::{Fp32LinearError, MAX_FP32_LINEAR_ELEMENTS, bf16_to_f32, fp32_linear_reference},
};

#[cfg(feature = "metal")]
use mlx_rs::{Array, StreamOrDevice};

/// Output-projection implementation selected by a model-local final head.
///
/// Scalar FP32 projection is the source-authoritative default. The optional
/// Metal variant replaces only the final vocabulary projection; HC collapse
/// and RMS normalization retain their established BF16 staging.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinalHeadExecution {
    /// Use the bounded scalar FP32 projection reference.
    #[default]
    Scalar,
    /// Use the bounded MLX Metal FP32 vocabulary projection.
    #[cfg(feature = "metal")]
    MetalFp32,
}

/// Largest BF16 output head: covers the real V4.1 129280 x 5120 head (662M
/// weights, 1.3 GB). An allocation guard on the borrowed weight, not a work cap.
pub const MAX_BF16_HEAD_ELEMENTS: usize = 1 << 30;

/// Borrowed output-head weights `[vocabulary, hidden_width]`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum HeadWeights<'a> {
    /// FP32 storage, bounded by the scalar FP32 linear limits.
    F32(&'a [f32]),
    /// BF16 storage, widened exactly to FP32 per weight inside the
    /// projection, so logits equal the FP32 path over the widened weights.
    Bf16(&'a [u16]),
}

/// Borrowed immutable operands for one final normalization and head.
#[derive(Clone, Copy, Debug)]
pub struct FinalHead<'a> {
    norm_weight: &'a [u16],
    head_weight: HeadWeights<'a>,
    vocabulary: usize,
    copies: usize,
    epsilon: f32,
    width: usize,
    execution: FinalHeadExecution,
}

impl<'a> FinalHead<'a> {
    /// Returns the fixed HC copy count and hidden width for request composition.
    #[must_use]
    pub(crate) const fn geometry(self) -> (usize, usize) {
        (self.copies, self.width)
    }

    /// Validates the static final-normalization and FP32 output-head operands.
    ///
    /// `norm_weight` is BF16 storage `[hidden_width]`; `head_weight` is FP32
    /// storage `[vocabulary, hidden_width]`; and `copies` is the number of
    /// copy-major residual rows to collapse for one token.
    pub fn new(
        norm_weight: &'a [u16],
        head_weight: &'a [f32],
        vocabulary: usize,
        copies: usize,
        epsilon: f32,
    ) -> Result<Self, FinalHeadError> {
        Self::with_weights(
            norm_weight,
            HeadWeights::F32(head_weight),
            vocabulary,
            copies,
            epsilon,
        )
    }

    /// Like [`Self::new`] for either head storage. FP32 weights keep the
    /// scalar FP32 linear limits; BF16 weights are bounded by
    /// [`MAX_BF16_HEAD_ELEMENTS`].
    pub fn with_weights(
        norm_weight: &'a [u16],
        head_weight: HeadWeights<'a>,
        vocabulary: usize,
        copies: usize,
        epsilon: f32,
    ) -> Result<Self, FinalHeadError> {
        let width = norm_weight.len();
        if width == 0 {
            return Err(FinalHeadError::EmptyWidth);
        }
        if width > MAX_RMS_NORM_WIDTH || width > MAX_HC_MIX_WIDTH {
            return Err(FinalHeadError::WidthTooLarge {
                width,
                maximum: MAX_RMS_NORM_WIDTH.min(MAX_HC_MIX_WIDTH),
            });
        }
        if copies == 0 {
            return Err(FinalHeadError::EmptyCopies);
        }
        if copies > MAX_HC_COPIES {
            return Err(FinalHeadError::CopyCountTooLarge {
                copies,
                maximum: MAX_HC_COPIES,
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(FinalHeadError::InvalidEpsilon);
        }
        if vocabulary == 0 {
            return Err(FinalHeadError::EmptyVocabulary);
        }
        let maximum = match head_weight {
            HeadWeights::F32(_) => MAX_FP32_LINEAR_ELEMENTS,
            HeadWeights::Bf16(_) => MAX_BF16_HEAD_ELEMENTS,
        };
        if vocabulary > maximum {
            return Err(FinalHeadError::VocabularyTooLarge {
                vocabulary,
                maximum,
            });
        }

        let expected_head_weight = checked_product(vocabulary, width, "head_weight")?;
        // One token performs one multiply-accumulate per weight, so the
        // weight element cap also bounds a projection's work.
        if expected_head_weight > maximum {
            return Err(FinalHeadError::ElementLimit {
                field: "head_weight",
                elements: expected_head_weight,
                maximum,
            });
        }
        let actual = match head_weight {
            HeadWeights::F32(weights) => weights.len(),
            HeadWeights::Bf16(weights) => weights.len(),
        };
        if actual != expected_head_weight {
            return Err(FinalHeadError::Length {
                field: "head_weight",
                actual,
                expected: expected_head_weight,
            });
        }
        for (element, &bits) in norm_weight.iter().enumerate() {
            if !bf16_to_f32(bits).is_finite() {
                return Err(FinalHeadError::NonFiniteNormWeight { element });
            }
        }
        let non_finite = match head_weight {
            HeadWeights::F32(weights) => weights.iter().position(|weight| !weight.is_finite()),
            HeadWeights::Bf16(weights) => weights
                .iter()
                .position(|&bits| !bf16_to_f32(bits).is_finite()),
        };
        if let Some(element) = non_finite {
            return Err(FinalHeadError::NonFiniteHeadWeight { element });
        }

        Ok(Self {
            norm_weight,
            head_weight,
            vocabulary,
            copies,
            epsilon,
            width,
            execution: FinalHeadExecution::Scalar,
        })
    }

    /// Selects the immutable final vocabulary-projection implementation.
    #[must_use]
    pub const fn with_execution(mut self, execution: FinalHeadExecution) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the final vocabulary-projection implementation.
    #[must_use]
    pub const fn execution(self) -> FinalHeadExecution {
        self.execution
    }

    /// Collapses, normalizes, and projects one copy-major token residual.
    ///
    /// `residual_bf16` must be `[copies, hidden_width]` and `incoming_pre`
    /// must be `[copies]`. The returned intermediate rows and logits are owned
    /// by the result, so a failed call leaves all caller-owned operands intact.
    pub fn forward(
        &self,
        residual_bf16: &[u16],
        incoming_pre: &[f32],
    ) -> Result<FinalHeadOutput, FinalHeadError> {
        let expected_residual = checked_product(self.copies, self.width, "residual")?;
        if residual_bf16.len() != expected_residual {
            return Err(FinalHeadError::Length {
                field: "residual_bf16",
                actual: residual_bf16.len(),
                expected: expected_residual,
            });
        }
        if incoming_pre.len() != self.copies {
            return Err(FinalHeadError::Length {
                field: "incoming_pre",
                actual: incoming_pre.len(),
                expected: self.copies,
            });
        }

        let mut collapsed_bf16 = allocate_u16("collapsed_bf16", self.width)?;
        hc_pre_bf16_reference(residual_bf16, incoming_pre, self.width, &mut collapsed_bf16)?;

        let mut normalized_bf16 = allocate_u16("normalized_bf16", self.width)?;
        rms_norm_bf16_reference(
            &collapsed_bf16,
            self.norm_weight,
            self.epsilon,
            &mut normalized_bf16,
        )?;

        let mut normalized_f32 = allocate_f32("normalized_f32", self.width)?;
        normalized_f32.extend(normalized_bf16.iter().copied().map(bf16_to_f32));
        let logits = match (self.execution, self.head_weight) {
            (FinalHeadExecution::Scalar, HeadWeights::F32(weights)) => {
                let mut logits = allocate_f32("logits", self.vocabulary)?;
                logits.resize(self.vocabulary, 0.0);
                fp32_linear_reference(
                    &normalized_f32,
                    weights,
                    1,
                    self.width,
                    self.vocabulary,
                    &mut logits,
                )?;
                logits
            }
            (FinalHeadExecution::Scalar, HeadWeights::Bf16(weights)) => {
                project_logits_bf16(&normalized_f32, weights, self.vocabulary)?
            }
            #[cfg(feature = "metal")]
            (FinalHeadExecution::MetalFp32, HeadWeights::F32(weights)) => {
                project_logits_metal(&normalized_f32, weights, self.vocabulary, self.width)?
            }
            #[cfg(feature = "metal")]
            (FinalHeadExecution::MetalFp32, HeadWeights::Bf16(_)) => {
                return Err(FinalHeadError::UnsupportedExecution);
            }
        };

        Ok(FinalHeadOutput {
            collapsed_bf16,
            normalized_bf16,
            logits,
        })
    }
}

/// Runs one validated FP32 vocabulary projection through MLX Metal.
///
/// The enclosing [`FinalHead`] has already checked the operand lengths, finite
/// static weights, and bounded vocabulary-by-width product. This helper repeats
/// its dynamic input and signed-dimension checks before constructing an MLX
/// graph, and returns host-owned logits because the request tail remains CPU
/// staged.
#[cfg(feature = "metal")]
fn project_logits_metal(
    input: &[f32],
    weights: &[f32],
    vocabulary: usize,
    width: usize,
) -> Result<Vec<f32>, FinalHeadError> {
    if input.len() != width {
        return Err(FinalHeadError::Length {
            field: "normalized_f32",
            actual: input.len(),
            expected: width,
        });
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(FinalHeadError::NonFiniteProjectionInput);
    }
    let expected_weights = checked_product(vocabulary, width, "head_weight")?;
    if weights.len() != expected_weights {
        return Err(FinalHeadError::Length {
            field: "head_weight",
            actual: weights.len(),
            expected: expected_weights,
        });
    }
    let rows = i32::try_from(vocabulary).map_err(|_| FinalHeadError::MetalDimension {
        field: "vocabulary",
    })?;
    let columns =
        i32::try_from(width).map_err(|_| FinalHeadError::MetalDimension { field: "width" })?;
    let stream = StreamOrDevice::gpu();
    let weights = Array::from_slice(weights, &[rows, columns]);
    let input = Array::from_slice(input, &[columns, 1]);
    let logits = weights
        .matmul_device(&input, &stream)
        .map_err(|error| FinalHeadError::Metal {
            message: error.to_string(),
        })?;
    logits.eval().map_err(|error| FinalHeadError::Metal {
        message: error.to_string(),
    })?;
    let values = logits.as_slice::<f32>();
    if values.len() != vocabulary {
        return Err(FinalHeadError::MetalOutputLength {
            actual: values.len(),
            expected: vocabulary,
        });
    }
    if let Some((element, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(FinalHeadError::NonFiniteMetalOutput { element });
    }
    Ok(values.to_vec())
}

/// Owned observations from one final normalization and output-head call.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalHeadOutput {
    collapsed_bf16: Vec<u16>,
    normalized_bf16: Vec<u16>,
    logits: Vec<f32>,
}

impl FinalHeadOutput {
    /// Returns the BF16 Hyper-Connections collapsed row.
    #[must_use]
    pub fn collapsed_bf16(&self) -> &[u16] {
        &self.collapsed_bf16
    }

    /// Returns the BF16 `RMSNorm` output row passed to the head projection.
    #[must_use]
    pub fn normalized_bf16(&self) -> &[u16] {
        &self.normalized_bf16
    }

    /// Returns the FP32 vocabulary logits.
    #[must_use]
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// An invalid final-head construction or execution request.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinalHeadError {
    /// The learned normalization row determines the hidden width and is empty.
    #[error("final head requires a nonempty hidden width")]
    EmptyWidth,
    /// The hidden width exceeds a scalar component bound.
    #[error("final head width {width} exceeds maximum {maximum}")]
    WidthTooLarge {
        /// Requested hidden width.
        width: usize,
        /// Maximum scalar hidden width.
        maximum: usize,
    },
    /// At least one Hyper-Connections residual copy is required.
    #[error("final head requires at least one residual copy")]
    EmptyCopies,
    /// The residual copy count exceeds the scalar HC pre-mix bound.
    #[error("final head copies {copies} exceeds maximum {maximum}")]
    CopyCountTooLarge {
        /// Requested copy count.
        copies: usize,
        /// Maximum scalar copy count.
        maximum: usize,
    },
    /// The `RMSNorm` epsilon must be finite and positive.
    #[error("final head epsilon must be finite and positive")]
    InvalidEpsilon,
    /// The output vocabulary needs at least one logit.
    #[error("final head requires a nonempty vocabulary")]
    EmptyVocabulary,
    /// The output vocabulary exceeds a bounded FP32 output row.
    #[error("final head vocabulary {vocabulary} exceeds maximum {maximum}")]
    VocabularyTooLarge {
        /// Requested vocabulary size.
        vocabulary: usize,
        /// Maximum scalar output elements.
        maximum: usize,
    },
    /// A derived static or runtime shape overflowed `usize`.
    #[error("final head {field} shape overflowed")]
    ShapeOverflow {
        /// Derived field that overflowed.
        field: &'static str,
    },
    /// A bounded static buffer would exceed the scalar linear limit.
    #[error("final head {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        /// Buffer role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
        /// Maximum supported element count.
        maximum: usize,
    },
    /// A caller buffer has an unexpected exact length.
    #[error("final head {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Actual element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// A learned BF16 normalization weight is NaN or infinity.
    #[error("final head normalization weight at element {element} is non-finite")]
    NonFiniteNormWeight {
        /// Weight element index.
        element: usize,
    },
    /// A learned FP32 output weight is NaN or infinity.
    #[error("final head output weight at element {element} is non-finite")]
    NonFiniteHeadWeight {
        /// Weight element index.
        element: usize,
    },
    /// The normalized FP32 row passed to the device projection is not finite.
    #[error("final head normalized projection input is non-finite")]
    NonFiniteProjectionInput,
    /// A bounded final-head dimension cannot be represented by the device API.
    #[error("final head device dimension {field} exceeds the backend limit")]
    MetalDimension {
        /// Dimension that could not be represented.
        field: &'static str,
    },
    /// The device backend rejected final-head projection execution.
    #[error("final head Metal projection failed: {message}")]
    Metal {
        /// Backend error rendered without operand contents.
        message: String,
    },
    /// The device projection returned an unexpected number of logits.
    #[error("final head Metal output length is {actual}, expected {expected}")]
    MetalOutputLength {
        /// Actual output element count.
        actual: usize,
        /// Expected vocabulary size.
        expected: usize,
    },
    /// The device projection produced a NaN or infinity.
    #[error("final head Metal output at element {element} is non-finite")]
    NonFiniteMetalOutput {
        /// Output element index.
        element: usize,
    },
    /// The selected execution does not support this head storage.
    #[error("final head execution does not support BF16 head weights")]
    UnsupportedExecution,
    /// A temporary output row could not be reserved.
    #[error("could not allocate {elements} final-head elements for {field}")]
    AllocationFailed {
        /// Temporary row role.
        field: &'static str,
        /// Required element count.
        elements: usize,
    },
    /// Hyper-Connections pre-mix rejected a runtime input.
    #[error(transparent)]
    HcMix(#[from] HcMixError),
    /// `RMSNorm` rejected a runtime input.
    #[error(transparent)]
    RmsNorm(#[from] RmsNormError),
    /// The FP32 output projection rejected a runtime input.
    #[error(transparent)]
    Linear(#[from] Fp32LinearError),
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, FinalHeadError> {
    left.checked_mul(right)
        .ok_or(FinalHeadError::ShapeOverflow { field })
}

/// One-row FP32 projection over BF16 weights in [`fp32_linear_reference`]'s
/// order and checks. Widening BF16 to FP32 is exact, so the logits equal that
/// reference over the widened weights bit for bit.
fn project_logits_bf16(
    input: &[f32],
    weights: &[u16],
    vocabulary: usize,
) -> Result<Vec<f32>, FinalHeadError> {
    if let Some(element) = input.iter().position(|value| !value.is_finite()) {
        return Err(Fp32LinearError::NonFiniteActivation { element }.into());
    }
    let mut logits = allocate_f32("logits", vocabulary)?;
    for (column, row) in weights.chunks_exact(input.len()).enumerate() {
        let mut sum = 0.0_f32;
        for (&activation, &bits) in input.iter().zip(row) {
            let product = activation * bf16_to_f32(bits);
            let stage = if product.is_finite() {
                sum += product;
                if sum.is_finite() {
                    continue;
                }
                "sum"
            } else {
                "product"
            };
            return Err(Fp32LinearError::ValueOverflow {
                stage,
                row: 0,
                output: column,
            }
            .into());
        }
        logits.push(sum);
    }
    Ok(logits)
}

fn allocate_u16(field: &'static str, elements: usize) -> Result<Vec<u16>, FinalHeadError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| FinalHeadError::AllocationFailed { field, elements })?;
    values.resize(elements, 0);
    Ok(values)
}

fn allocate_f32(field: &'static str, elements: usize) -> Result<Vec<f32>, FinalHeadError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| FinalHeadError::AllocationFailed { field, elements })?;
    Ok(values)
}
