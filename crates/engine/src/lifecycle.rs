use thiserror::Error;

/// Observable lifecycle phase for one loaded model revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelPhase {
    /// No model resources are retained.
    Unloaded,
    /// Checkpoint metadata or weights are loading.
    Loading,
    /// The loaded model is compiling or warming its execution plans.
    Warming,
    /// The model may receive admitted inference requests.
    Ready,
    /// Loading or warming failed; this phase never exposes failure details.
    Failed,
}

impl ModelPhase {
    /// Returns whether admitted inference is safe in this phase.
    #[must_use]
    pub const fn accepts_requests(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// A transition-checked model lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelLifecycle {
    phase: ModelPhase,
}

impl Default for ModelLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelLifecycle {
    /// Creates an unloaded lifecycle.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: ModelPhase::Unloaded,
        }
    }

    /// Returns the current observable phase.
    #[must_use]
    pub const fn phase(self) -> ModelPhase {
        self.phase
    }

    /// Starts a fresh load after unloading or a failed attempt.
    pub fn start_loading(&mut self) -> Result<(), LifecycleError> {
        self.transition(
            ModelPhase::Loading,
            &[ModelPhase::Unloaded, ModelPhase::Failed],
        )
    }

    /// Records that the model is loaded and its plans are warming.
    pub fn start_warming(&mut self) -> Result<(), LifecycleError> {
        self.transition(ModelPhase::Warming, &[ModelPhase::Loading])
    }

    /// Records that all required warmup work completed.
    pub fn mark_ready(&mut self) -> Result<(), LifecycleError> {
        self.transition(ModelPhase::Ready, &[ModelPhase::Warming])
    }

    /// Records a failed load or warmup without retaining an error payload.
    pub fn fail(&mut self) -> Result<(), LifecycleError> {
        self.transition(
            ModelPhase::Failed,
            &[ModelPhase::Loading, ModelPhase::Warming],
        )
    }

    /// Releases model resources from any phase.
    pub fn unload(&mut self) {
        self.phase = ModelPhase::Unloaded;
    }

    fn transition(
        &mut self,
        next: ModelPhase,
        allowed: &[ModelPhase],
    ) -> Result<(), LifecycleError> {
        if !allowed.contains(&self.phase) {
            return Err(LifecycleError::InvalidTransition {
                from: self.phase,
                to: next,
            });
        }
        self.phase = next;
        Ok(())
    }
}

/// An attempted lifecycle transition did not preserve a safe readiness state.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum LifecycleError {
    /// The requested transition is not valid from the current phase.
    #[error("model lifecycle cannot transition from {from:?} to {to:?}")]
    InvalidTransition {
        /// The current model phase.
        from: ModelPhase,
        /// The requested model phase.
        to: ModelPhase,
    },
}

#[cfg(test)]
mod tests {
    use super::{LifecycleError, ModelLifecycle, ModelPhase};

    #[test]
    fn model_only_accepts_requests_when_ready() {
        let mut lifecycle = ModelLifecycle::new();
        assert!(!lifecycle.phase().accepts_requests());
        lifecycle.start_loading().expect("unloaded model can load");
        lifecycle.start_warming().expect("loaded model can warm");
        lifecycle
            .mark_ready()
            .expect("warmed model can become ready");
        assert!(lifecycle.phase().accepts_requests());
    }

    #[test]
    fn rejects_readiness_before_loading_and_warming() {
        let mut lifecycle = ModelLifecycle::new();
        assert_eq!(
            lifecycle.mark_ready(),
            Err(LifecycleError::InvalidTransition {
                from: ModelPhase::Unloaded,
                to: ModelPhase::Ready,
            })
        );
        lifecycle.start_loading().expect("unloaded model can load");
        assert_eq!(
            lifecycle.mark_ready(),
            Err(LifecycleError::InvalidTransition {
                from: ModelPhase::Loading,
                to: ModelPhase::Ready,
            })
        );
    }

    #[test]
    fn failed_model_can_retry_or_unload() {
        let mut lifecycle = ModelLifecycle::new();
        lifecycle.start_loading().expect("unloaded model can load");
        lifecycle.fail().expect("loading model can fail");
        lifecycle.start_loading().expect("failed model can retry");
        lifecycle.unload();
        assert_eq!(lifecycle.phase(), ModelPhase::Unloaded);
    }
}
