//! Reusable candidate-only executor and opt-in per-append host timing.

use std::path::Path;

use serde::Serialize;

use super::{
    Qwen3LayerKv, Qwen3MetalLoadError, STREAM_EXECUTOR_MAX_TOKENS, StreamForwardPlan,
    cached_kv_bytes, run_cached_append, validate_candidate_logits,
};

/// One opt-in host wall-clock profile for a successful streamed append.
///
/// The measurements cover synchronous payload reads and explicit MLX evaluation
/// and readback boundaries. MLX still evaluates lazily between those points, so
/// these are not GPU-only kernel timings or allocator measurements.
#[derive(Debug, Serialize)]
pub struct Qwen3StreamProfile {
    pub(super) schema_version: u32,
    pub(super) input_tokens: usize,
    pub(super) cached_tokens_before: usize,
    pub(super) cached_tokens_after: usize,
    pub(super) logical_kv_bytes: u64,
    pub(super) embedding_load_rebuild_ms: f64,
    pub(super) layer_load_conversion_ms: f64,
    pub(super) layer_execute_readback_ms: f64,
    pub(super) final_norm_ms: f64,
    pub(super) tiled_projection_load_ms: f64,
    pub(super) tiled_projection_execute_ms: f64,
    pub(super) total_ms: f64,
    pub(super) unattributed_ms: f64,
    pub(super) scope: &'static str,
}

impl Qwen3StreamProfile {
    pub(super) fn new(input_tokens: usize, cached_tokens_before: usize) -> Self {
        Self {
            schema_version: 1,
            input_tokens,
            cached_tokens_before,
            cached_tokens_after: 0,
            logical_kv_bytes: 0,
            embedding_load_rebuild_ms: 0.0,
            layer_load_conversion_ms: 0.0,
            layer_execute_readback_ms: 0.0,
            final_norm_ms: 0.0,
            tiled_projection_load_ms: 0.0,
            tiled_projection_execute_ms: 0.0,
            total_ms: 0.0,
            unattributed_ms: 0.0,
            scope: "opt-in host wall-clock phase profile for one candidate-only streamed append; payload reads, MLX evaluation, and host readback are interleaved by the lazy graph, so values are not isolated GPU-kernel, allocator, or serving measurements",
        }
    }

    pub(super) fn finish(
        &mut self,
        cached_tokens_after: usize,
        logical_kv_bytes: u64,
        total_ms: f64,
    ) {
        self.cached_tokens_after = cached_tokens_after;
        self.logical_kv_bytes = logical_kv_bytes;
        self.total_ms = total_ms;
        let attributed = self.embedding_load_rebuild_ms
            + self.layer_load_conversion_ms
            + self.layer_execute_readback_ms
            + self.final_norm_ms
            + self.tiled_projection_load_ms
            + self.tiled_projection_execute_ms;
        self.unattributed_ms = (total_ms - attributed).max(0.0);
    }
}

/// Reusable, candidate-only Qwen3 executor with layer-at-a-time weight reads.
///
/// This is deliberately adapter-local: it retains only detached Qwen3 K/V,
/// the initial prompt, and the most recently produced logits. It does not
/// construct a resident checkpoint oracle or retain a per-step history.
/// Weight and K/V budgets are logical planning contracts; they do not cover
/// MLX allocator retention, activations, or operator scratch.
pub struct Qwen3StreamExecutor {
    plan: StreamForwardPlan,
    prompt_ids: Vec<i32>,
    max_total_tokens: usize,
    max_kv_bytes: u64,
    cache: Vec<Option<Qwen3LayerKv>>,
    cached_tokens: usize,
    retained_kv_bytes: u64,
    current_logits: Vec<f32>,
    state: StreamExecutorState,
    profiling_enabled: bool,
    last_profile: Option<Qwen3StreamProfile>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamExecutorState {
    Ready,
    Prefilled,
    Poisoned,
}

impl Qwen3StreamExecutor {
    /// Plans a bounded streamed sequence before reading any tensor payload.
    ///
    /// `max_total_tokens` includes the supplied prompt and every later decode
    /// token. It is therefore a conservative, fixed K/V reservation contract.
    pub fn new(
        model: &Path,
        prompt_ids: &[i32],
        max_total_tokens: usize,
        max_weight_bytes: u64,
        max_kv_bytes: u64,
        tile_rows: usize,
    ) -> Result<Self, Qwen3MetalLoadError> {
        // `StreamForwardPlan::new` reads only configuration and safetensors
        // metadata. It validates prompt IDs against the inspected vocabulary;
        // tensor payloads are first read by prefill/decode below.
        let plan = StreamForwardPlan::new(model, prompt_ids, max_weight_bytes, tile_rows)?;
        validate_executor_context(
            prompt_ids.len(),
            max_total_tokens,
            STREAM_EXECUTOR_MAX_TOKENS.min(plan.config.maximum_cached_tokens()),
        )?;
        let planned_final_kv_bytes = plan.config.cached_kv_bytes(max_total_tokens)?;
        if planned_final_kv_bytes > max_kv_bytes {
            return Err(Qwen3MetalLoadError::CachedStateBudget {
                required: planned_final_kv_bytes,
                maximum: max_kv_bytes,
            });
        }
        let cache = (0..plan.config.hidden_layers()).map(|_| None).collect();
        Ok(Self {
            plan,
            prompt_ids: prompt_ids.to_vec(),
            max_total_tokens,
            max_kv_bytes,
            cache,
            cached_tokens: 0,
            retained_kv_bytes: 0,
            current_logits: Vec::new(),
            state: StreamExecutorState::Ready,
            profiling_enabled: false,
            last_profile: None,
        })
    }

    /// Enables or disables opt-in profile retention for future successful appends.
    pub fn enable_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }

    /// Takes the last successful append profile, if profiling was enabled.
    pub fn take_last_profile(&mut self) -> Option<Qwen3StreamProfile> {
        self.last_profile.take()
    }

    /// Fills the cache from the constructor prompt and returns the final logits.
    pub fn prefill_last_logits(&mut self) -> Result<Vec<f32>, Qwen3MetalLoadError> {
        self.last_profile = None;
        match self.state {
            StreamExecutorState::Ready => self.append_prefill(),
            StreamExecutorState::Prefilled => Err(Qwen3MetalLoadError::StreamPrefillAlreadyDone),
            StreamExecutorState::Poisoned => Err(Qwen3MetalLoadError::StreamPoisoned),
        }
    }

    /// Appends one token and returns its final logits.
    pub fn decode_last_logits(&mut self, token: i32) -> Result<Vec<f32>, Qwen3MetalLoadError> {
        self.last_profile = None;
        match self.state {
            StreamExecutorState::Ready => {
                return Err(Qwen3MetalLoadError::StreamDecodeWithoutPrefill);
            }
            StreamExecutorState::Poisoned => return Err(Qwen3MetalLoadError::StreamPoisoned),
            StreamExecutorState::Prefilled => {}
        }
        let next_tokens =
            self.cached_tokens
                .checked_add(1)
                .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                    "cached token count",
                ))?;
        if next_tokens > self.max_total_tokens {
            return Err(Qwen3MetalLoadError::StreamContextLimit {
                requested: next_tokens,
                maximum: self.max_total_tokens,
            });
        }
        validate_stream_token(token, self.plan.inspection.contract().vocab_size())?;
        self.append(&[token])
    }

    /// Number of tokens represented by every live cache layer.
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Logical byte total of the retained detached K/V arrays.
    #[must_use]
    pub const fn kv_bytes(&self) -> u64 {
        self.retained_kv_bytes
    }

    /// Peak planned weight and explicit staging bytes for one streamed stage.
    #[must_use]
    pub const fn planned_weight_bytes(&self) -> u64 {
        self.plan.memory.peak
    }

    fn append_prefill(&mut self) -> Result<Vec<f32>, Qwen3MetalLoadError> {
        let prompt_ids = self.prompt_ids.clone();
        self.append(&prompt_ids)
    }

    fn append(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3MetalLoadError> {
        self.last_profile = None;
        let mut profile =
            profile_for_append(self.profiling_enabled, input_ids.len(), self.cached_tokens);
        let result = run_cached_append(
            &self.plan,
            input_ids,
            self.cached_tokens,
            &mut self.cache,
            profile.as_mut(),
        )
        .and_then(|logits| {
            validate_candidate_logits(&logits)?;
            let next_tokens = self.cached_tokens.checked_add(input_ids.len()).ok_or(
                Qwen3MetalLoadError::DimensionOutOfRange("cached token count"),
            )?;
            let retained = cached_kv_bytes(&self.cache)?;
            let expected = self.plan.config.cached_kv_bytes(next_tokens)?;
            if retained != expected {
                return Err(crate::forward::Qwen3ForwardError::CacheInconsistent.into());
            }
            if retained > self.max_kv_bytes {
                return Err(Qwen3MetalLoadError::CachedStateBudget {
                    required: retained,
                    maximum: self.max_kv_bytes,
                });
            }
            Ok((logits, next_tokens, retained))
        });
        match result {
            Ok((logits, next_tokens, retained_kv_bytes)) => {
                self.cached_tokens = next_tokens;
                self.retained_kv_bytes = retained_kv_bytes;
                self.current_logits = logits;
                self.state = StreamExecutorState::Prefilled;
                self.last_profile = profile;
                Ok(self.current_logits.clone())
            }
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }

    fn poison(&mut self) {
        self.cache.iter_mut().for_each(|entry| *entry = None);
        self.cached_tokens = 0;
        self.retained_kv_bytes = 0;
        self.current_logits.clear();
        self.last_profile = None;
        self.state = StreamExecutorState::Poisoned;
    }
}

fn validate_executor_context(
    prompt_tokens: usize,
    max_total_tokens: usize,
    configured_maximum: usize,
) -> Result<(), Qwen3MetalLoadError> {
    if max_total_tokens < prompt_tokens {
        return Err(Qwen3MetalLoadError::StreamMaximumBelowPrompt {
            maximum: max_total_tokens,
            prompt_tokens,
        });
    }
    if max_total_tokens > configured_maximum {
        return Err(Qwen3MetalLoadError::StreamContextLimit {
            requested: max_total_tokens,
            maximum: configured_maximum,
        });
    }
    Ok(())
}

fn validate_stream_token(token: i32, vocab_size: u32) -> Result<(), Qwen3MetalLoadError> {
    if u32::try_from(token).map_or(true, |id| id >= vocab_size) {
        return Err(Qwen3MetalLoadError::InvalidTokenId { token, vocab_size });
    }
    Ok(())
}

fn profile_for_append(
    enabled: bool,
    input_tokens: usize,
    cached_tokens_before: usize,
) -> Option<Qwen3StreamProfile> {
    enabled.then(|| Qwen3StreamProfile::new(input_tokens, cached_tokens_before))
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use crate::GPU_TEST_LOCK;

    use super::{
        Qwen3StreamExecutor, Qwen3StreamProfile, profile_for_append, validate_executor_context,
        validate_stream_token,
    };

    #[test]
    fn disabled_profiling_allocates_no_profile() {
        assert!(profile_for_append(false, 1, 0).is_none());
        assert!(profile_for_append(true, 1, 0).is_some());
    }

    #[test]
    fn profile_accounting_serializes_scalar_phase_totals() {
        let mut profile = Qwen3StreamProfile::new(2, 3);
        profile.embedding_load_rebuild_ms = 1.0;
        profile.layer_load_conversion_ms = 2.0;
        profile.layer_execute_readback_ms = 3.0;
        profile.final_norm_ms = 4.0;
        profile.tiled_projection_load_ms = 5.0;
        profile.tiled_projection_execute_ms = 6.0;
        profile.finish(5, 40, 25.0);
        let value = serde_json::to_value(profile).expect("profile JSON");
        assert_eq!(value["input_tokens"], 2);
        assert_eq!(value["cached_tokens_before"], 3);
        assert_eq!(value["cached_tokens_after"], 5);
        assert_eq!(value["logical_kv_bytes"], 40);
        assert_eq!(value["unattributed_ms"], 4.0);
        assert!(value.get("logits").is_none());
        assert!(value.get("model").is_none());
    }

    #[test]
    fn executor_preflight_rejects_context_and_invalid_tokens_without_checkpoint() {
        assert!(validate_executor_context(3, 32, 32).is_ok());
        assert!(matches!(
            validate_executor_context(3, 2, 32),
            Err(super::Qwen3MetalLoadError::StreamMaximumBelowPrompt { .. })
        ));
        assert!(matches!(
            validate_executor_context(3, 33, 32),
            Err(super::Qwen3MetalLoadError::StreamContextLimit { .. })
        ));
        assert!(validate_stream_token(7, 8).is_ok());
        assert!(matches!(
            validate_stream_token(8, 8),
            Err(super::Qwen3MetalLoadError::InvalidTokenId { .. })
        ));
    }

    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
    fn repeated_candidate_lifetimes_emit_one_profile_per_successful_append() {
        let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
            eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
            return;
        };
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        for (request, prompt_ids, maximum_total_tokens, decode_ids) in [
            (0, &[9_707, 11][..], 4_usize, &[13, 13][..]),
            (1, &[9_707, 11, 1_879][..], 6_usize, &[13, 13, 13][..]),
        ] {
            let mut executor = Qwen3StreamExecutor::new(
                &model,
                prompt_ids,
                maximum_total_tokens,
                u64::MAX,
                u64::MAX,
                1_024,
            )
            .expect("candidate-only executor plan");
            executor.enable_profiling(true);
            assert!(
                executor
                    .prefill_last_logits()
                    .expect("prefill logits")
                    .len()
                    > 1
            );
            let prefill_profile = executor.last_profile.as_ref().expect("prefill profile");
            assert_eq!(prefill_profile.input_tokens, prompt_ids.len());
            assert_eq!(prefill_profile.cached_tokens_before, 0);
            assert_eq!(prefill_profile.cached_tokens_after, prompt_ids.len());
            assert_eq!(prefill_profile.logical_kv_bytes, executor.kv_bytes());
            assert_eq!(
                executor.kv_bytes(),
                executor
                    .plan
                    .config
                    .cached_kv_bytes(prompt_ids.len())
                    .expect("prefill KV")
            );

            let before_invalid = (executor.cached_tokens(), executor.kv_bytes());
            assert!(executor.decode_last_logits(i32::MAX).is_err());
            assert_eq!(
                (executor.cached_tokens(), executor.kv_bytes()),
                before_invalid
            );
            assert!(
                executor.last_profile.is_none(),
                "rejected decode clears stale profile"
            );
            assert!(matches!(
                executor.prefill_last_logits(),
                Err(super::Qwen3MetalLoadError::StreamPrefillAlreadyDone)
            ));

            for &token in decode_ids {
                assert!(
                    executor
                        .decode_last_logits(token)
                        .expect("decode logits")
                        .len()
                        > 1
                );
                let profile = executor.take_last_profile().expect("decode profile");
                assert_eq!(profile.input_tokens, 1);
                assert_eq!(profile.cached_tokens_after, executor.cached_tokens());
                assert_eq!(profile.logical_kv_bytes, executor.kv_bytes());
                assert_eq!(
                    executor.kv_bytes(),
                    executor
                        .plan
                        .config
                        .cached_kv_bytes(executor.cached_tokens())
                        .expect("decode KV")
                );
                assert!(
                    executor.take_last_profile().is_none(),
                    "profiles are single-consumer"
                );
            }
            let before_limit = (executor.cached_tokens(), executor.kv_bytes());
            assert!(executor.decode_last_logits(13).is_err());
            assert_eq!(
                (executor.cached_tokens(), executor.kv_bytes()),
                before_limit
            );
            assert!(executor.take_last_profile().is_none());
            eprintln!(
                "stream executor lifecycle request={request} total_tokens={} logical_kv_bytes={} planned_weight_bytes={}",
                executor.cached_tokens(),
                executor.kv_bytes(),
                executor.planned_weight_bytes(),
            );
        }
    }
}
