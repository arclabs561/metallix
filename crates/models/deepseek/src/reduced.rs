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
use mlx_rs::{Array, Dtype, StreamOrDevice, ops::concatenate_device};

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
        let (mut output, normalized_f32) = self.normalize(residual_bf16, incoming_pre)?;
        output.logits = match (self.execution, self.head_weight) {
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
        Ok(output)
    }

    /// Like [`Self::forward`], but projects through `head`, a device-resident
    /// copy of this head's BF16 weights, regardless of the selected execution.
    ///
    /// `head` must have been built from the same `[vocabulary, hidden_width]`
    /// BF16 weights this head borrows; only its shape can be checked here.
    /// HC collapse and `RMSNorm` keep their scalar BF16 staging, and the
    /// projection is FP32 over the exactly widened weights, so logits differ
    /// from the scalar BF16 path only by FP32 summation order.
    #[cfg(feature = "metal")]
    pub fn forward_metal(
        &self,
        residual_bf16: &[u16],
        incoming_pre: &[f32],
        head: &MetalBf16Head,
    ) -> Result<FinalHeadOutput, FinalHeadError> {
        if !matches!(self.head_weight, HeadWeights::Bf16(_)) {
            return Err(FinalHeadError::UnsupportedExecution);
        }
        if head.vocabulary != self.vocabulary || head.width != self.width {
            return Err(FinalHeadError::Length {
                field: "metal_head",
                actual: checked_product(head.vocabulary, head.width, "metal_head")?,
                expected: checked_product(self.vocabulary, self.width, "head_weight")?,
            });
        }
        let (mut output, normalized_f32) = self.normalize(residual_bf16, incoming_pre)?;
        output.logits = head.project(&normalized_f32)?;
        Ok(output)
    }

    /// Validates one token's operands, then runs HC collapse and `RMSNorm`.
    /// Returns the BF16 rows in an output with no logits yet, and the
    /// normalized row widened to FP32 for the projection.
    fn normalize(
        &self,
        residual_bf16: &[u16],
        incoming_pre: &[f32],
    ) -> Result<(FinalHeadOutput, Vec<f32>), FinalHeadError> {
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
        let output = FinalHeadOutput {
            collapsed_bf16,
            normalized_bf16,
            logits: Vec::new(),
        };
        Ok((output, normalized_f32))
    }
}

/// Head rows uploaded and widened together: 8192 rows of a 5120-wide head
/// are an 84 MB BF16 staging copy and a 168 MB FP32 block.
#[cfg(feature = "metal")]
const METAL_HEAD_CHUNK_ROWS: usize = 8192;

/// A BF16 output head `[vocabulary, hidden_width]` widened exactly to FP32
/// and uploaded once to MLX.
///
/// Build it once per model and pass it to [`FinalHead::forward_metal`] for
/// every step; each call then uploads only the normalized row. The device
/// copy adds `4 * vocabulary * hidden_width` bytes of unified memory beside
/// the caller's host weights. Keeping FP32 rather than BF16 on the device
/// gives FP32 logits without a per-call widened copy, and a BF16 matmul
/// would round every logit to BF16.
///
/// The rows are held as separate row-block arrays: concatenating them on the
/// device would hold the blocks and the joined copy at once, doubling the
/// upload peak. Each projection concatenates only the per-block logits.
#[cfg(feature = "metal")]
pub struct MetalBf16Head {
    blocks: Vec<Array>,
    vocabulary: usize,
    width: usize,
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for MetalBf16Head {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalBf16Head")
            .field("vocabulary", &self.vocabulary)
            .field("width", &self.width)
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "metal")]
impl MetalBf16Head {
    /// Validates `weights` like [`FinalHead::with_weights`] does for
    /// [`HeadWeights::Bf16`], then uploads them to the GPU and widens them
    /// there in blocks of rows.
    pub fn new(weights: &[u16], vocabulary: usize, width: usize) -> Result<Self, FinalHeadError> {
        if width == 0 {
            return Err(FinalHeadError::EmptyWidth);
        }
        if vocabulary == 0 {
            return Err(FinalHeadError::EmptyVocabulary);
        }
        let expected = checked_product(vocabulary, width, "head_weight")?;
        if expected > MAX_BF16_HEAD_ELEMENTS {
            return Err(FinalHeadError::ElementLimit {
                field: "head_weight",
                elements: expected,
                maximum: MAX_BF16_HEAD_ELEMENTS,
            });
        }
        if weights.len() != expected {
            return Err(FinalHeadError::Length {
                field: "head_weight",
                actual: weights.len(),
                expected,
            });
        }
        if let Some(element) = first_non_finite_bf16(weights) {
            return Err(FinalHeadError::NonFiniteHeadWeight { element });
        }
        i32::try_from(vocabulary).map_err(|_| FinalHeadError::MetalDimension {
            field: "vocabulary",
        })?;
        let columns =
            i32::try_from(width).map_err(|_| FinalHeadError::MetalDimension { field: "width" })?;
        let _device = crate::device_lock();
        let stream = StreamOrDevice::gpu();
        let mut blocks = Vec::new();
        for block in weights.chunks(METAL_HEAD_CHUNK_ROWS * width) {
            let rows =
                i32::try_from(block.len() / width).map_err(|_| FinalHeadError::MetalDimension {
                    field: "vocabulary",
                })?;
            // Upload BF16 and widen on the GPU (exact, like `bf16_to_f32`):
            // half the bytes copied, and no host FP32 buffer. Evaluating per
            // block releases each BF16 staging copy before the next upload.
            let widened = Array::from_slice(block, &[rows, columns])
                .view_dtype_device(Dtype::Bfloat16, &stream)
                .and_then(|bf16| bf16.as_dtype_device(Dtype::Float32, &stream))
                .map_err(metal_error)?;
            widened.eval().map_err(metal_error)?;
            blocks.push(widened);
        }
        Ok(Self {
            blocks,
            vocabulary,
            width,
        })
    }

    /// FP32 logits for one normalized FP32 row over the resident weights.
    fn project(&self, input: &[f32]) -> Result<Vec<f32>, FinalHeadError> {
        if input.len() != self.width {
            return Err(FinalHeadError::Length {
                field: "normalized_f32",
                actual: input.len(),
                expected: self.width,
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(FinalHeadError::NonFiniteProjectionInput);
        }
        let columns = i32::try_from(self.width)
            .map_err(|_| FinalHeadError::MetalDimension { field: "width" })?;
        let _device = crate::device_lock();
        let stream = StreamOrDevice::gpu();
        let input = Array::from_slice(input, &[columns, 1]);
        let parts = self
            .blocks
            .iter()
            .map(|block| block.matmul_device(&input, &stream))
            .collect::<Result<Vec<_>, _>>()
            .map_err(metal_error)?;
        let logits = concatenate_device(&parts, &stream).map_err(metal_error)?;
        host_logits(&logits, self.vocabulary)
    }
}

/// Index of the first NaN or infinite BF16 value, scanning disjoint parts of
/// `weights` on scoped threads: a 662M-weight head takes over half a second
/// on one core.
#[cfg(feature = "metal")]
fn first_non_finite_bf16(weights: &[u16]) -> Option<usize> {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let part = weights.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let scans: Vec<_> = weights
            .chunks(part)
            .enumerate()
            .map(|(index, bits)| {
                scope.spawn(move || {
                    // All-ones exponent is infinity or NaN.
                    bits.iter()
                        .position(|&value| value & 0x7f80 == 0x7f80)
                        .map(|element| index * part + element)
                })
            })
            .collect();
        // Parts are in order, so the first hit is the lowest index.
        scans
            .into_iter()
            .find_map(|scan| scan.join().expect("finite scan thread panicked"))
    })
}

#[cfg(feature = "metal")]
#[allow(clippy::needless_pass_by_value)]
fn metal_error(error: mlx_rs::error::Exception) -> FinalHeadError {
    FinalHeadError::Metal {
        message: error.to_string(),
    }
}

/// Evaluates device `logits` and returns them as finite host logits.
#[cfg(feature = "metal")]
fn host_logits(logits: &Array, vocabulary: usize) -> Result<Vec<f32>, FinalHeadError> {
    logits.eval().map_err(metal_error)?;
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
    let _device = crate::device_lock();
    let stream = StreamOrDevice::gpu();
    let weights = Array::from_slice(weights, &[rows, columns]);
    let input = Array::from_slice(input, &[columns, 1]);
    let logits = weights
        .matmul_device(&input, &stream)
        .map_err(metal_error)?;
    host_logits(&logits, vocabulary)
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
    #[error("final head execution does not support this head weight storage")]
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

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::{FinalHead, FinalHeadError, HeadWeights, MetalBf16Head, bf16_to_f32};

    /// Deterministic finite BF16 bits with exponents `low..low + span`.
    fn bf16_values(count: usize, seed: u32, low: u32, span: u32) -> Vec<u16> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let sign = u16::from(state >> 31 == 1) << 15;
                let exponent = u16::try_from(low + (state >> 16) % span).unwrap() << 7;
                sign | exponent | u16::try_from(state & 0x7f).unwrap()
            })
            .collect()
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0
    }

    #[test]
    fn resident_bf16_head_matches_the_scalar_projection() {
        // One full and one partial resident row block.
        let (vocabulary, width, copies) = (super::METAL_HEAD_CHUNK_ROWS + 4099, 1024, 4);
        let norm = bf16_values(width, 3, 126, 3);
        // Weights about 2^-15..2^-6, the scale of a trained output head.
        let weights = bf16_values(vocabulary * width, 11, 112, 10);
        let head =
            FinalHead::with_weights(&norm, HeadWeights::Bf16(&weights), vocabulary, copies, 1e-6)
                .unwrap();
        let resident = MetalBf16Head::new(&weights, vocabulary, width).unwrap();
        assert_eq!(resident.blocks.len(), 2);
        // Two steps through one upload: the resident copy is reused, not stale.
        for seed in [13, 17] {
            let residual = bf16_values(copies * width, seed, 120, 12);
            let pre = [0.4, 0.3, 0.2, 0.1];
            let scalar = head.forward(&residual, &pre).unwrap();
            let metal = head.forward_metal(&residual, &pre, &resident).unwrap();
            assert_eq!(metal.collapsed_bf16(), scalar.collapsed_bf16());
            assert_eq!(metal.normalized_bf16(), scalar.normalized_bf16());
            // Order-independent FP32 bound: width * eps * sum |x w| is about
            // 6.1e-5 of the largest absolute row sum at width 1024.
            let scale = weights
                .chunks_exact(width)
                .map(|row| {
                    row.iter()
                        .zip(scalar.normalized_bf16())
                        .map(|(&w, &x)| (bf16_to_f32(w) * bf16_to_f32(x)).abs())
                        .sum::<f32>()
                })
                .fold(0.0_f32, f32::max);
            let error = metal
                .logits()
                .iter()
                .zip(scalar.logits())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "seed {seed}: max abs {error:e}, {:e} of scale {scale}",
                error / scale
            );
            assert!(error <= 1e-4 * scale, "{error} exceeds 1e-4 * {scale}");
            assert_eq!(argmax(metal.logits()), argmax(scalar.logits()));
        }
    }

    #[test]
    fn resident_head_rejects_mismatched_operands() {
        let norm = [0x3f80; 4];
        let weights = [0x3f80; 8];
        let head = FinalHead::with_weights(&norm, HeadWeights::Bf16(&weights), 2, 1, 1.0).unwrap();
        let residual = [0x3f80; 4];
        let other = MetalBf16Head::new(&weights, 4, 2).unwrap();
        assert!(matches!(
            head.forward_metal(&residual, &[1.0], &other),
            Err(FinalHeadError::Length {
                field: "metal_head",
                ..
            })
        ));
        let resident = MetalBf16Head::new(&weights, 2, 4).unwrap();
        assert!(
            head.forward_metal(&residual[..3], &[1.0], &resident)
                .is_err()
        );
        let f32_weights = [1.0; 8];
        let f32_head = FinalHead::new(&norm, &f32_weights, 2, 1, 1.0).unwrap();
        assert_eq!(
            f32_head.forward_metal(&residual, &[1.0], &resident),
            Err(FinalHeadError::UnsupportedExecution)
        );
        assert!(matches!(
            MetalBf16Head::new(&[0x7f80; 8], 2, 4),
            Err(FinalHeadError::NonFiniteHeadWeight { element: 0 })
        ));
        let mut late_nan = [0x3f80; 8];
        late_nan[5] = 0x7fc0;
        late_nan[7] = 0xff80;
        assert!(matches!(
            MetalBf16Head::new(&late_nan, 2, 4),
            Err(FinalHeadError::NonFiniteHeadWeight { element: 5 })
        ));
        assert!(MetalBf16Head::new(&weights, 3, 4).is_err());
        assert_eq!(
            head.forward_metal(&residual, &[1.0], &resident).unwrap(),
            head.forward(&residual, &[1.0]).unwrap()
        );
    }
}
