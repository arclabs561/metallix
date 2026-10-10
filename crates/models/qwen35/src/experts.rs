//! Where a `qwen3_5_moe` checkpoint in mlx-lm's layout stores each routed
//! expert, for reading experts on demand ([`checkpoint::expert`]).
//!
//! mlx-lm stores a layer's experts as three stacked projections,
//! `switch_mlp.{gate,up,down}_proj`, each as MLX affine `weight` codes plus
//! `scales` and `biases` with the experts on the outermost axis. One expert
//! is therefore nine contiguous byte ranges, one per stored tensor.

use checkpoint::{
    CheckpointError, TensorEncoding, TensorInfo, TensorSource,
    expert::{ExpertKey, ExpertLocator, ExpertRange},
};

/// The routed projections in the order the forward pass applies them.
pub const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
/// The stored tensors of one affine projection, in [`ExpertLocator`] order.
pub const PARTS: [&str; 3] = ["weight", "scales", "biases"];

/// The stored name of one part of a layer's routed projection.
#[must_use]
pub fn stored_name(layer: usize, projection: &str, part: &str) -> String {
    format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.{part}")
}

/// One stored part: its name and table entry.
#[derive(Clone, Debug)]
struct Part {
    tensor: String,
    info: TensorInfo,
}

/// The nine stored tensors behind every routed expert of every layer.
#[derive(Clone, Debug)]
pub struct Qwen35ExpertLocator {
    /// Per layer, [`PROJECTIONS`] x [`PARTS`] in order.
    layers: Vec<[Part; 9]>,
    experts: usize,
}

impl Qwen35ExpertLocator {
    /// Finds and checks every routed projection of `layers` layers with
    /// `experts` experts each: codes stored as MLX affine, `scales` and
    /// `biases` beside them, all with `experts` entries on the outermost
    /// axis that split into equal byte ranges.
    ///
    /// # Errors
    /// A missing part, codes that are not affine, or a part whose outer
    /// dimension is not `experts`.
    pub fn mlx_affine(
        source: &impl TensorSource,
        layers: usize,
        experts: usize,
    ) -> Result<Self, CheckpointError> {
        let mut found = Vec::with_capacity(layers);
        for layer in 0..layers {
            let mut parts = Vec::with_capacity(9);
            for projection in PROJECTIONS {
                for part in PARTS {
                    let tensor = stored_name(layer, projection, part);
                    let info = source
                        .tensor(&tensor)
                        .ok_or_else(|| CheckpointError::MissingTensor(tensor.clone()))?
                        .clone();
                    let affine = matches!(info.encoding(), TensorEncoding::Affine { .. });
                    let last = u64::try_from(experts.saturating_sub(1)).ok();
                    let splits = info.shape().first() == u64::try_from(experts).ok().as_ref()
                        && last.and_then(|last| info.slice_dim0(last)).is_some();
                    if (part == "weight") != affine || !splits {
                        return Err(CheckpointError::Range(tensor));
                    }
                    parts.push(Part { tensor, info });
                }
            }
            let parts: [Part; 9] = parts
                .try_into()
                .map_err(|_| CheckpointError::Range(format!("layer {layer} experts")))?;
            found.push(parts);
        }
        Ok(Self {
            layers: found,
            experts,
        })
    }

    /// Routed experts per layer.
    #[must_use]
    pub const fn experts(&self) -> usize {
        self.experts
    }

    /// Layers located.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The table entry of one stored part of `layer` (`index` in
    /// [`PROJECTIONS`] x [`PARTS`] order).
    #[must_use]
    pub fn part(&self, layer: usize, index: usize) -> Option<&TensorInfo> {
        self.layers.get(layer)?.get(index).map(|part| &part.info)
    }

    /// Bytes of one expert (every expert of a layer has the same size).
    #[must_use]
    pub fn expert_bytes(&self, layer: usize) -> u64 {
        self.layers.get(layer).map_or(0, |parts| {
            parts
                .iter()
                .map(|part| {
                    let range = part.info.range();
                    (range.end - range.start) / part.info.shape()[0]
                })
                .sum()
        })
    }
}

impl ExpertLocator for Qwen35ExpertLocator {
    fn ranges(&self, key: ExpertKey) -> Result<Vec<ExpertRange>, CheckpointError> {
        let missing = || CheckpointError::MissingTensor(format!("expert {key:?}"));
        let parts = self
            .layers
            .get(usize::from(key.layer))
            .ok_or_else(missing)?;
        parts
            .iter()
            .map(|part| {
                let range = part
                    .info
                    .slice_dim0(u64::from(key.expert))
                    .ok_or_else(missing)?;
                Ok(ExpertRange {
                    tensor: part.tensor.clone(),
                    range,
                })
            })
            .collect()
    }
}
