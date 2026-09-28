//! Test-only composition of Qwen cache forks with finite SMC accounting.
//!
//! This is a checkpoint qualification, not a particle-serving implementation,
//! exact conditioning result, or performance measurement.

use std::{env, path::PathBuf, sync::Mutex};

use engine::{sampling::sample_categorical, smc::ParticleSet};
use qwen::metal::Qwen3MlxWeights;

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

const TEMPERATURE: f64 = 0.7;
const FIRST_UNIFORMS: [f64; 3] = [0.05, 0.50, 0.95];
const SECOND_UNIFORMS: [f64; 3] = [0.20, 0.50, 0.80];
const FIRST_POTENTIALS: [f64; 3] = [1.0, 2.0, 0.5];
const SECOND_POTENTIALS: [f64; 3] = [0.5, 1.5, 2.0];

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 5e-5,
        "expected {expected:.8}, got {actual:.8}"
    );
}

fn assert_logits_match(actual: Vec<f32>, expected: Vec<f32>) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(f64::from(actual), f64::from(expected));
    }
}

/// Independent FP64 log-softmax calculation for a selected Qwen token.
fn selected_logprob(logits: &[f32], token: u32, temperature: f64) -> f64 {
    selected_logprob_and_positive_support(logits, token, temperature).0
}

/// Returns a selected log probability plus the non-underflowed FP64 support.
fn selected_logprob_and_positive_support(
    logits: &[f32],
    token: u32,
    temperature: f64,
) -> (f64, usize) {
    assert!(temperature.is_finite() && temperature > 0.0);
    let token = usize::try_from(token).expect("token ID fits usize");
    let scaled = logits
        .iter()
        .map(|&logit| f64::from(logit) / temperature)
        .collect::<Vec<_>>();
    let maximum = scaled
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("checkpoint vocabulary is nonempty");
    assert!(maximum.is_finite(), "checkpoint logits are finite");
    let normalizer = scaled
        .iter()
        .map(|logit| (logit - maximum).exp())
        .sum::<f64>();
    assert!(normalizer.is_finite() && normalizer > 0.0);
    let positive_support = scaled
        .iter()
        .filter(|&&logit| (logit - maximum).exp() > 0.0)
        .count();
    (scaled[token] - maximum - normalizer.ln(), positive_support)
}

fn independent_systematic_ancestry(weights: &[f64], offset: f64) -> Vec<usize> {
    assert!((0.0..1.0).contains(&offset));
    let total = weights.iter().sum::<f64>();
    assert!(total.is_finite() && total > 0.0);
    let count = f64::from(u32::try_from(weights.len()).expect("small population"));
    let probabilities = weights
        .iter()
        .map(|weight| weight / total)
        .collect::<Vec<_>>();
    let final_positive = probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .expect("positive reference support");
    (0..weights.len())
        .map(|index| {
            let target =
                (f64::from(u32::try_from(index).expect("small population")) + offset) / count;
            let mut cumulative = 0.0;
            for (ancestor, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if target < cumulative {
                    return ancestor;
                }
            }
            final_positive
        })
        .collect()
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keep the finite trace, resampling boundary, and cache ownership assertions together"
)]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_qwen_smc_resampling_composes_weights_and_replays_cache_ancestry() {
    let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    // Match the existing resident-chat and cache-fork qualification precision.
    weights.prepare_float32().expect("resident float32 weights");

    let prompt = [9_707_i32, 11];
    let mut root = weights.executor();
    let root_logits = root.prefill_last_logits(&prompt).expect("root prefill");
    let legal = vec![true; root_logits.len()];
    let mut particles = ParticleSet::new(vec![0_usize, 1, 2]).expect("three particles");
    let mut parents = Vec::with_capacity(3);

    for (index, uniform) in FIRST_UNIFORMS.into_iter().enumerate() {
        let sampled = sample_categorical(&root_logits, &legal, TEMPERATURE, uniform)
            .expect("full-vocabulary proposal has support");
        let first_token = i32::try_from(sampled.token_id).expect("Qwen token fits i32");
        let proposal = selected_logprob(&root_logits, sampled.token_id, TEMPERATURE);
        // This raw-model score is diagnostic only. The explicit test target
        // below is deployed temperature-conditioned q times a synthetic
        // potential, so no p/q correction is being claimed here.
        let raw_model = selected_logprob(&root_logits, sampled.token_id, 1.0);
        assert!(proposal.is_finite() && raw_model.is_finite());
        assert_close(proposal, sampled.sampling_logprob);
        particles.particles_mut()[index]
            .add_log_importance_ratio(proposal + FIRST_POTENTIALS[index].ln(), proposal)
            .expect("explicit finite first-stage target/proposal ratio");

        let mut parent = root.fork_prefilled().expect("prefilled root fork");
        let next_logits = parent
            .decode_last_logits(first_token)
            .expect("first model transition");
        parents.push((parent, first_token, next_logits));
    }

    // The target is the deployed model proposal multiplied by a deliberately
    // declared test potential [1, 2, 1/2], not a claim about an application
    // distribution. Its mean is 7/6. Offset 1/2 gives ancestor map [0, 1, 1].
    let first_log_mean = particles.systematic_resample(0.5).expect("resample");
    assert_close(first_log_mean.exp(), 7.0 / 6.0);
    assert_eq!(
        particles
            .particles()
            .iter()
            .map(|particle| *particle.state())
            .collect::<Vec<_>>(),
        [0, 1, 1]
    );
    assert!(
        particles
            .particles()
            .iter()
            .all(|particle| particle.log_weight() == 0.0)
    );

    let root_probe = root
        .fork_prefilled()
        .expect("root remains a valid immutable snapshot")
        .decode_last_logits(parents[0].1)
        .expect("root continuation");
    assert_logits_match(
        root_probe,
        weights
            .executor()
            .prefill_last_logits(&[prompt[0], prompt[1], parents[0].1])
            .expect("independent root replay"),
    );
    assert_eq!(root.cached_tokens(), prompt.len());

    let ancestry = particles
        .particles()
        .iter()
        .map(|particle| *particle.state())
        .collect::<Vec<_>>();
    for (child_index, state) in ancestry.into_iter().enumerate() {
        let (parent, first_token, parent_logits) = &parents[state];
        let sampled = sample_categorical(
            parent_logits,
            &legal,
            TEMPERATURE,
            SECOND_UNIFORMS[child_index],
        )
        .expect("full-vocabulary proposal has support");
        let second_token = i32::try_from(sampled.token_id).expect("Qwen token fits i32");
        let proposal = selected_logprob(parent_logits, sampled.token_id, TEMPERATURE);
        assert_close(proposal, sampled.sampling_logprob);
        particles.particles_mut()[child_index]
            .add_log_importance_ratio(proposal + SECOND_POTENTIALS[child_index].ln(), proposal)
            .expect("explicit finite second-stage target/proposal ratio");

        let next_logits = parent
            .fork_prefilled()
            .expect("resampled parent fork")
            .decode_last_logits(second_token)
            .expect("second model transition");
        assert_logits_match(
            next_logits,
            weights
                .executor()
                .prefill_last_logits(&[prompt[0], prompt[1], *first_token, second_token])
                .expect("independent child prefix replay"),
        );
        assert_eq!(parent.cached_tokens(), prompt.len() + 1);
    }

    // Children above appended divergent tokens after forks. Probe each
    // original parent only now: matching a fresh prefix after all child work
    // detects a shared mutable KV buffer that a token-count assertion misses.
    for (parent, first_token, parent_logits) in &parents {
        let probe = sample_categorical(parent_logits, &legal, TEMPERATURE, 0.37)
            .expect("full-vocabulary parent probe")
            .token_id;
        let probe = i32::try_from(probe).expect("Qwen token fits i32");
        let parent_probe = parent
            .fork_prefilled()
            .expect("original parent remains forkable")
            .decode_last_logits(probe)
            .expect("original parent probe transition");
        assert_logits_match(
            parent_probe,
            weights
                .executor()
                .prefill_last_logits(&[prompt[0], prompt[1], *first_token, probe])
                .expect("independent original-parent replay"),
        );
        assert_eq!(parent.cached_tokens(), prompt.len() + 1);
    }

    // After resampling, the second-stage weights are [1/2, 3/2, 2], with
    // mean 4/3. The product 14/9 is this finite trace's normalizer estimate;
    // it is neither exact conditioning nor a calibrated probability.
    let second_log_mean = particles.log_mean_weight().expect("second-stage mean");
    assert_close(second_log_mean.exp(), 4.0 / 3.0);
    assert_close((first_log_mean + second_log_mean).exp(), 14.0 / 9.0);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keep the p/q reference, resampling oracle, and cache replay in one checkpoint trace"
)]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_qwen_temperature_proposal_corrects_raw_model_weights_and_replays_ancestry() {
    let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");

    let prompt = [9_707_i32, 11];
    let mut root = weights.executor();
    let root_logits = root.prefill_last_logits(&prompt).expect("root prefill");
    let legal = vec![true; root_logits.len()];
    let mut particles = ParticleSet::new(vec![0_usize, 1, 2]).expect("three particles");
    let mut importance_weights = Vec::with_capacity(3);
    let mut parents = Vec::with_capacity(3);

    for (index, uniform) in FIRST_UNIFORMS.into_iter().enumerate() {
        let sampled = sample_categorical(&root_logits, &legal, TEMPERATURE, uniform)
            .expect("full-vocabulary temperature proposal");
        let (log_q, q_positive_support) =
            selected_logprob_and_positive_support(&root_logits, sampled.token_id, TEMPERATURE);
        // The production sampler intentionally permits FP64 tail underflow.
        // This target is raw model p, so an underflowed q would invalidate its
        // importance ratio; fail the qualification instead of claiming full
        // proposal support.
        assert_eq!(
            q_positive_support,
            root_logits.len(),
            "temperature proposal underflowed to {q_positive_support}/{} supported rows",
            root_logits.len()
        );
        let log_p = selected_logprob(&root_logits, sampled.token_id, 1.0);
        assert!(log_p.is_finite() && log_q.is_finite());
        assert_close(log_q, sampled.sampling_logprob);
        particles.particles_mut()[index]
            .add_log_importance_ratio(log_p, log_q)
            .expect("raw-p / temperature-q ratio with verified support");
        let importance_weight = (log_p - log_q).exp();
        assert!(importance_weight.is_finite() && importance_weight > 0.0);
        importance_weights.push(importance_weight);

        let first_token = i32::try_from(sampled.token_id).expect("Qwen token fits i32");
        let mut parent = root.fork_prefilled().expect("prefilled root fork");
        let next_logits = parent
            .decode_last_logits(first_token)
            .expect("first model transition");
        parents.push((parent, first_token, next_logits));
    }

    let reference_total = importance_weights.iter().sum::<f64>();
    let reference_log_mean = reference_total.ln() - 3.0_f64.ln();
    let reference_normalized = importance_weights
        .iter()
        .map(|weight| weight / reference_total)
        .collect::<Vec<_>>();
    let reference_ess = reference_normalized
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>()
        .recip();
    let (normalized, log_sum) = particles.normalized_weights().expect("finite population");
    assert_close(log_sum.exp(), reference_total);
    for (actual, expected) in normalized.iter().zip(&reference_normalized) {
        assert_close(*actual, *expected);
    }
    assert_close(
        particles.effective_sample_size().expect("finite ESS"),
        reference_ess,
    );

    // A distinct CDF implementation supplies the expected deterministic
    // ancestry. This finite N=3 normalizer is an estimate for this trace, not
    // exact conditioning under the raw model distribution.
    let expected_ancestry = independent_systematic_ancestry(&importance_weights, 0.37);
    let stage_log_mean = particles.systematic_resample(0.37).expect("resample");
    assert_close(stage_log_mean, reference_log_mean);
    assert_eq!(
        particles
            .particles()
            .iter()
            .map(|particle| *particle.state())
            .collect::<Vec<_>>(),
        expected_ancestry
    );
    assert!(
        particles
            .particles()
            .iter()
            .all(|particle| particle.log_weight() == 0.0)
    );

    let ancestry = particles
        .particles()
        .iter()
        .map(|particle| *particle.state())
        .collect::<Vec<_>>();
    for (child_index, parent_index) in ancestry.into_iter().enumerate() {
        let (parent, first_token, parent_logits) = &parents[parent_index];
        let probe = sample_categorical(
            parent_logits,
            &legal,
            TEMPERATURE,
            SECOND_UNIFORMS[child_index],
        )
        .expect("full-vocabulary branch probe");
        let probe = i32::try_from(probe.token_id).expect("Qwen token fits i32");
        let next_logits = parent
            .fork_prefilled()
            .expect("resampled parent fork")
            .decode_last_logits(probe)
            .expect("branch probe transition");
        assert_logits_match(
            next_logits,
            weights
                .executor()
                .prefill_last_logits(&[prompt[0], prompt[1], *first_token, probe])
                .expect("independent resampled-prefix replay"),
        );
    }

    for (parent, first_token, parent_logits) in &parents {
        let probe = sample_categorical(parent_logits, &legal, TEMPERATURE, 0.37)
            .expect("original parent probe")
            .token_id;
        let probe = i32::try_from(probe).expect("Qwen token fits i32");
        let parent_probe = parent
            .fork_prefilled()
            .expect("original parent remains forkable")
            .decode_last_logits(probe)
            .expect("original parent probe transition");
        assert_logits_match(
            parent_probe,
            weights
                .executor()
                .prefill_last_logits(&[prompt[0], prompt[1], *first_token, probe])
                .expect("independent original-parent replay"),
        );
        assert_eq!(parent.cached_tokens(), prompt.len() + 1);
    }
    assert_eq!(root.cached_tokens(), prompt.len());
}
