//! Six-range packed V4.1 experts through the shared checkpoint pager.
//!
//! This adapter changes byte residency only. It lends the existing scalar or
//! device-dispatched `Fp4ExpertWeights` to the caller, preserving route order
//! and arithmetic. The cache holds host packed bytes, not resident Metal arrays.
//! The source must already belong to the admitted checkpoint; checkpoint-wide
//! admission and receipt verification remain the source owner's responsibility.

use std::{
    num::NonZeroU16,
    sync::{Arc, Mutex},
};

use ::checkpoint::{
    CheckpointError, DType, TensorEncoding, TensorSource,
    expert::{ExpertCache, ExpertKey, ExpertLocator, ExpertPager, ExpertRange},
};
use thiserror::Error;

use crate::moe::{Fp4ExpertWeights, MoEConfig, MoEError, RoutedExpertSource};

/// Metadata or resource admission failure for the selected-expert pager.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41ExpertPagingError {
    /// The model's expert widths are invalid.
    #[error(transparent)]
    Geometry(#[from] MoEError),
    /// One selected tensor is missing or has incompatible storage geometry.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// The retained packed-byte cache cannot hold even one expert.
    #[error("one packed expert needs {required} bytes, above cache budget {budget}")]
    Budget {
        /// Packed code and scale bytes per expert.
        required: u64,
        /// Caller-provided cache budget.
        budget: u64,
    },
}

/// Resolves six full tensor ranges in W1 codes/scales, W2 codes/scales,
/// W3 codes/scales order. Each selected expert is validated before any read.
pub struct V41ExpertLocator<S> {
    source: Arc<S>,
    layers: NonZeroU16,
    experts: NonZeroU16,
    hidden: usize,
    intermediate: usize,
    expert_bytes: u64,
}

impl<S: TensorSource> V41ExpertLocator<S> {
    /// Declares the admitted model geometry without reading tensor payloads.
    /// Tensor names, exact I8 packed shapes and E8M0 scale shapes are checked
    /// when [`ExpertLocator::ranges`] selects an expert. No draft layers are
    /// inferred: `layers` is the explicit permitted range.
    ///
    /// # Errors
    /// Returns a geometry error for invalid widths or byte-count overflow.
    pub fn new(
        source: Arc<S>,
        layers: NonZeroU16,
        experts: NonZeroU16,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, V41ExpertPagingError> {
        MoEConfig::new(hidden, intermediate, 0.0, 1, 1.0, true, 1.0)?;
        let elements = hidden
            .checked_mul(intermediate)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(MoEError::ShapeOverflow {
                field: "packed expert",
            })?;
        let expert_bytes =
            (elements / 2 + elements / 32)
                .checked_mul(3)
                .ok_or(MoEError::ShapeOverflow {
                    field: "packed expert",
                })?;
        Ok(Self {
            source,
            layers,
            experts,
            hidden,
            intermediate,
            expert_bytes,
        })
    }

    /// Stored codes and E8M0 scales for one expert; excludes allocator overhead.
    #[must_use]
    pub const fn expert_bytes(&self) -> u64 {
        self.expert_bytes
    }
}

impl<S: TensorSource> ExpertLocator for V41ExpertLocator<S> {
    fn ranges(&self, key: ExpertKey) -> Result<Vec<ExpertRange>, CheckpointError> {
        if key.layer >= self.layers.get() || key.expert >= self.experts.get() {
            return Err(CheckpointError::Range(format!("expert {key:?}")));
        }
        let mut ranges = Vec::with_capacity(6);
        for (projection, rows, columns) in [
            ("w1", self.intermediate, self.hidden),
            ("w2", self.hidden, self.intermediate),
            ("w3", self.intermediate, self.hidden),
        ] {
            for (part, dtype, divisor) in [("weight", DType::I8, 2), ("scale", DType::F8E8M0, 32)] {
                let tensor = format!(
                    "layers.{}.ffn.experts.{}.{projection}.{part}",
                    key.layer, key.expert
                );
                let info = self
                    .source
                    .tensor(&tensor)
                    .ok_or_else(|| CheckpointError::MissingTensor(tensor.clone()))?;
                let expected = [rows as u64, (columns / divisor) as u64];
                let range = info.range();
                if info.encoding() != TensorEncoding::Plain(dtype)
                    || info.shape() != expected
                    || range.end.checked_sub(range.start) != expected[0].checked_mul(expected[1])
                {
                    return Err(CheckpointError::Range(tensor));
                }
                ranges.push(ExpertRange { tensor, range });
            }
        }
        Ok(ranges)
    }
}

struct PackedExpert {
    parts: [Vec<u8>; 6],
}

impl PackedExpert {
    fn weights(
        &self,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Fp4ExpertWeights<'_>, MoEError> {
        let [w1, s1, w2, s2, w3, s3] = &self.parts;
        Fp4ExpertWeights::new(hidden, intermediate, w1, s1, w2, s2, w3, s3)
    }
}

/// Packed host experts retained under a global byte budget across layers.
///
/// Misses read six ranges through the shared pager's eight reader threads.
/// Each cache fill completes before calling the numerical callback, exactly
/// once. Read errors never call it. The mutex is released before arithmetic.
/// A live callback retains its own expert reference across eviction; its bytes,
/// other in-flight callbacks, staging and output are outside the cache budget.
/// This is not a process-footprint cap or a device-residency implementation.
pub struct V41PagedExperts<S> {
    locator: V41ExpertLocator<S>,
    pager: ExpertPager<S>,
    cache: Mutex<ExpertCache<Arc<PackedExpert>>>,
}

impl<S: TensorSource + Send + Sync + 'static> V41PagedExperts<S> {
    /// Creates an empty cache and an eight-reader pager for the locator's source.
    ///
    /// # Errors
    /// Refuses a budget smaller than one expert before spawning reader threads.
    pub fn new(locator: V41ExpertLocator<S>, budget: u64) -> Result<Self, V41ExpertPagingError> {
        if budget < locator.expert_bytes {
            return Err(V41ExpertPagingError::Budget {
                required: locator.expert_bytes,
                budget,
            });
        }
        let pager = ExpertPager::new(Arc::clone(&locator.source), 8);
        Ok(Self {
            locator,
            pager,
            cache: Mutex::new(ExpertCache::new(budget)),
        })
    }

    /// Lends a layer-scoped provider suitable for `StepSources::experts`.
    ///
    /// # Errors
    /// Returns an error when the layer is outside the admitted range.
    pub fn layer(&self, layer: u16) -> Result<V41PagedExpertLayer<'_, S>, V41ExpertPagingError> {
        if layer >= self.locator.layers.get() {
            return Err(CheckpointError::Range(format!("expert layer {layer}")).into());
        }
        Ok(V41PagedExpertLayer { owner: self, layer })
    }

    fn load(&self, key: ExpertKey) -> Result<Arc<PackedExpert>, MoEError> {
        let unavailable = |reason: String| MoEError::ExpertUnavailable {
            index: usize::from(key.expert),
            reason,
        };
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| unavailable("expert cache lock poisoned".into()))?;
        if let Some(expert) = cache.get(key) {
            return Ok(Arc::clone(expert));
        }
        let ranges = self
            .locator
            .ranges(key)
            .map_err(|error| unavailable(error.to_string()))?;
        let mut fetched = self
            .pager
            .fetch(vec![(key, ranges)])
            .map_err(|error| unavailable(error.to_string()))?;
        let expert = fetched
            .pop()
            .ok_or_else(|| unavailable("pager returned no expert".into()))?;
        let parts = expert
            .parts
            .try_into()
            .map_err(|_| unavailable("pager returned wrong part count".into()))?;
        let expert = Arc::new(PackedExpert { parts });
        expert.weights(self.locator.hidden, self.locator.intermediate)?;
        cache
            .insert(key, Arc::clone(&expert), self.locator.expert_bytes, &[])
            .map_err(|error| unavailable(error.to_string()))?;
        Ok(expert)
    }
}

/// One layer borrowing the shared packed-expert cache.
pub struct V41PagedExpertLayer<'a, S> {
    owner: &'a V41PagedExperts<S>,
    layer: u16,
}

impl<S: TensorSource + Send + Sync + 'static> RoutedExpertSource for V41PagedExpertLayer<'_, S> {
    fn with_expert(
        &self,
        index: usize,
        run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
    ) -> Result<Vec<u16>, MoEError> {
        let expert = u16::try_from(index).map_err(|_| MoEError::ExpertUnavailable {
            index,
            reason: "expert index exceeds u16".into(),
        })?;
        let expert = self.owner.load(ExpertKey {
            layer: self.layer,
            expert,
        })?;
        run(expert.weights(self.owner.locator.hidden, self.owner.locator.intermediate)?)
    }
}

#[cfg(test)]
#[path = "paged_experts_tests.rs"]
mod tests;
