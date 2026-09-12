//! Stateless categorical sampling over an explicit legal-token mask.

use thiserror::Error;

/// A token selected from a temperature-transformed legal distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CategoricalSample {
    /// Index of the selected model-vocabulary row.
    pub token_id: u32,
    /// Natural-log probability under the temperature-transformed legal distribution.
    pub sampling_logprob: f64,
}

/// Invalid categorical-sampling input.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SamplingError {
    /// There are no model-vocabulary rows.
    #[error("logits must not be empty")]
    EmptyLogits,
    /// The legal mask does not describe every model-vocabulary row.
    #[error("logits and legal-token mask must have equal lengths")]
    MaskLengthMismatch,
    /// No token is legal to sample.
    #[error("at least one legal token is required")]
    EmptySupport,
    /// Temperature is zero, negative, non-finite, or would overflow transformed logits.
    #[error(
        "temperature must be finite, greater than zero, and preserve finite transformed logits"
    )]
    InvalidTemperature,
    /// The caller-supplied categorical variate is not in the half-open unit interval.
    #[error("uniform variate must be finite and in [0, 1)")]
    InvalidUniform,
    /// A model-vocabulary logit is non-finite.
    #[error("logits must all be finite")]
    NonFiniteLogit,
    /// The vocabulary is larger than the token-ID representation.
    #[error("model vocabulary exceeds the supported token-ID range")]
    VocabularyTooLarge,
}

/// Samples a legal model-vocabulary row from temperature-transformed logits.
///
/// `legal_mask` must have exactly one entry per `logits` row. `uniform` is supplied
/// by the caller so this primitive neither chooses an RNG nor owns a seed. The
/// returned probability is conditioned on legal rows and the temperature transform;
/// raw-model and grammar-conditioned metrics remain the caller's responsibility.
///
/// # Errors
///
/// Returns [`SamplingError`] for malformed input, a non-finite logit, or a
/// temperature whose scaling would make a transformed logit non-finite.
pub fn sample_categorical(
    logits: &[f32],
    legal_mask: &[bool],
    temperature: f64,
    uniform: f64,
) -> Result<CategoricalSample, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if logits.len() != legal_mask.len() {
        return Err(SamplingError::MaskLengthMismatch);
    }
    if u32::try_from(logits.len()).is_err() {
        return Err(SamplingError::VocabularyTooLarge);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
        return Err(SamplingError::InvalidUniform);
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(SamplingError::NonFiniteLogit);
    }

    let mut legal_maximum = f64::NEG_INFINITY;
    let mut has_legal_token = false;
    for (&logit, &legal) in logits.iter().zip(legal_mask) {
        if !legal {
            continue;
        }
        legal_maximum = legal_maximum.max(f64::from(logit));
        has_legal_token = true;
    }
    if !has_legal_token {
        return Err(SamplingError::EmptySupport);
    }

    let mut shifted_sum = 0.0;
    for (&logit, &legal) in logits.iter().zip(legal_mask) {
        if legal {
            shifted_sum += shifted_logit(logit, legal_maximum, temperature)?.exp();
        }
    }
    let log_normalizer = shifted_sum.ln();

    let target = uniform * shifted_sum;
    let mut cumulative_weight = 0.0;
    let mut final_legal_token = None;
    for (index, (&logit, &legal)) in logits.iter().zip(legal_mask).enumerate() {
        if !legal {
            continue;
        }
        let shifted = shifted_logit(logit, legal_maximum, temperature)?;
        let weight = shifted.exp();
        cumulative_weight += weight;
        let token_id = u32::try_from(index).map_err(|_| SamplingError::VocabularyTooLarge)?;
        if weight > 0.0 {
            final_legal_token = Some((token_id, shifted));
        }
        if target < cumulative_weight {
            return Ok(CategoricalSample {
                token_id,
                sampling_logprob: shifted - log_normalizer,
            });
        }
    }

    // Cumulative floating-point rounding can leave a variate immediately below one
    // beyond the final accumulated mass. The final positive-mass legal row is the
    // deterministic fallback, never an illegal, padded, or underflowed row.
    let (token_id, shifted) = final_legal_token.ok_or(SamplingError::EmptySupport)?;
    Ok(CategoricalSample {
        token_id,
        sampling_logprob: shifted - log_normalizer,
    })
}

fn shifted_logit(logit: f32, legal_maximum: f64, temperature: f64) -> Result<f64, SamplingError> {
    let shifted = (f64::from(logit) - legal_maximum) / temperature;
    if shifted.is_finite() {
        Ok(shifted)
    } else {
        Err(SamplingError::InvalidTemperature)
    }
}

#[cfg(test)]
mod tests {
    use super::{SamplingError, sample_categorical};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn samples_a_hand_computed_distribution() {
        let logits = [0.0, std::f32::consts::LN_2];
        let legal = [true, true];
        let ratio = f64::from(logits[1]).exp();

        let first = sample_categorical(&logits, &legal, 1.0, 0.2).expect("valid sample");
        assert_eq!(first.token_id, 0);
        assert_close(first.sampling_logprob, (1.0 / (1.0 + ratio)).ln());

        let second = sample_categorical(&logits, &legal, 1.0, 0.4).expect("valid sample");
        assert_eq!(second.token_id, 1);
        assert_close(second.sampling_logprob, (ratio / (1.0 + ratio)).ln());
    }

    #[test]
    fn uniform_boundaries_have_a_stable_token_order() {
        let logits = [0.0, 0.0];
        let legal = [true, true];

        assert_eq!(
            sample_categorical(&logits, &legal, 1.0, 0.0)
                .expect("valid sample")
                .token_id,
            0
        );
        assert_eq!(
            sample_categorical(&logits, &legal, 1.0, 0.5)
                .expect("valid sample")
                .token_id,
            1
        );
        assert_eq!(
            sample_categorical(&logits, &legal, 1.0, f64::from_bits(0x3fef_ffff_ffff_ffff))
                .expect("valid sample")
                .token_id,
            1
        );
    }

    #[test]
    fn illegal_and_padded_rows_are_excluded() {
        let logits = [0.0, 100.0, 2.0, 1_000.0];
        let legal = [true, false, true, false];

        let sample = sample_categorical(&logits, &legal, 1.0, 0.2).expect("valid sample");
        assert_eq!(sample.token_id, 2);
        assert_close(sample.sampling_logprob, -(1.0 + (-2.0_f64).exp()).ln());
    }

    #[test]
    fn temperature_changes_the_sampling_distribution() {
        let logits = [0.0, std::f32::consts::LN_2];
        let legal = [true, true];

        let sample = sample_categorical(&logits, &legal, 2.0, 0.5).expect("valid sample");
        let root_two = (f64::from(logits[1]) / 2.0).exp();
        assert_eq!(sample.token_id, 1);
        assert_close(sample.sampling_logprob, (root_two / (1.0 + root_two)).ln());
    }

    #[test]
    fn finite_extreme_logits_remain_numerically_stable() {
        let sample = sample_categorical(&[-f32::MAX, f32::MAX], &[true, true], 1.0, 0.5)
            .expect("finite logits should sample");
        assert_eq!(sample.token_id, 1);
        assert_close(sample.sampling_logprob, 0.0);
    }

    #[test]
    fn repeated_calls_with_the_same_variate_are_deterministic() {
        let logits = [-1.0, 0.0, 1.0];
        let legal = [true, true, true];
        let first = sample_categorical(&logits, &legal, 0.7, 0.8).expect("valid sample");
        let second = sample_categorical(&logits, &legal, 0.7, 0.8).expect("valid sample");
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            sample_categorical(&[], &[], 1.0, 0.0),
            Err(SamplingError::EmptyLogits)
        );
        assert_eq!(
            sample_categorical(&[0.0], &[], 1.0, 0.0),
            Err(SamplingError::MaskLengthMismatch)
        );
        assert_eq!(
            sample_categorical(&[0.0], &[false], 1.0, 0.0),
            Err(SamplingError::EmptySupport)
        );
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                sample_categorical(&[0.0], &[true], temperature, 0.0),
                Err(SamplingError::InvalidTemperature)
            );
        }
        for uniform in [-0.0_f64 - 1.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                sample_categorical(&[0.0], &[true], 1.0, uniform),
                Err(SamplingError::InvalidUniform)
            );
        }
        for logit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                sample_categorical(&[logit], &[true], 1.0, 0.0),
                Err(SamplingError::NonFiniteLogit)
            );
            assert_eq!(
                sample_categorical(&[0.0, logit], &[true, false], 1.0, 0.0),
                Err(SamplingError::NonFiniteLogit)
            );
        }
        assert_eq!(
            sample_categorical(&[0.0, f32::MAX], &[true, true], f64::MIN_POSITIVE, 0.0,),
            Err(SamplingError::InvalidTemperature)
        );
    }
}
