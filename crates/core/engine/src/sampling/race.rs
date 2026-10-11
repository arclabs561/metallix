//! The `race-v2` seeded sampler: Gumbel-max with counter-based noise.
//!
//! The token at generated position `t` is the legal row maximizing
//! `f64(logit) / temperature + g(seed, choice, t, row)`, lowest row on ties.
//! The noise `g` is a pure function of its key, so the token depends only on
//! the seed, the position, the logits and the legal set: not on batch
//! composition, speculation, or the order in which rows are inspected. Any
//! legal set fixed before the noise (top-k, top-p, a grammar mask) is exact:
//! the maximum over a subset is distributed as the categorical restricted to
//! it (Kool et al., arXiv 1903.06059v2, eq. 3).
//!
//! Noise: 64 bits per row from Philox4x32-10 (Salmon et al., SC'11) with key
//! `(seed mod 2^32, seed >> 32)` and counter `(row >> 1, t, choice, "race")`;
//! an even row takes output words 0 and 1, an odd row words 2 and 3, high word
//! first. The bits become an Exp(1) variate without cancellation in either
//! tail and then Gumbel noise, in f64 with the pure-Rust `libm` so the bits do
//! not depend on the platform's math library.

use super::{CategoricalSample, SamplingError, check_row, legal_shifted_sum, shifted_logit};

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
/// Counter word 3, "race" in ASCII: separates this stream from other uses of
/// the same seed.
const RACE_DOMAIN: u32 = 0x7261_6365;

/// The per-request part of a noise key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RaceKey {
    /// The request seed.
    pub seed: u64,
    /// The choice index when one request asks for several completions.
    pub choice: u32,
}

/// Philox4x32 with 10 rounds.
#[must_use]
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;
    for round in 0..10 {
        if round > 0 {
            k0 = k0.wrapping_add(PHILOX_W0);
            k1 = k1.wrapping_add(PHILOX_W1);
        }
        let (hi0, lo0) = mul_hi_lo(PHILOX_M0, c0);
        let (hi1, lo1) = mul_hi_lo(PHILOX_M1, c2);
        (c0, c1, c2, c3) = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0);
    }
    [c0, c1, c2, c3]
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "Philox splits a 64-bit product into its two 32-bit halves"
)]
fn mul_hi_lo(left: u32, right: u32) -> (u32, u32) {
    let product = u64::from(left) * u64::from(right);
    ((product >> 32) as u32, product as u32)
}

/// The 64 noise bits for `row` at generated position `position`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the key is the seed's low and high 32-bit halves"
)]
pub fn race_bits(key: RaceKey, position: u32, row: u32) -> u64 {
    let words = philox4x32_10(
        [row >> 1, position, key.choice, RACE_DOMAIN],
        [key.seed as u32, (key.seed >> 32) as u32],
    );
    let (high, low) = if row & 1 == 0 {
        (words[0], words[1])
    } else {
        (words[2], words[3])
    };
    (u64::from(high) << 32) | u64::from(low)
}

/// Gumbel noise from 64 bits: `-ln E` with `E = -ln(1 - w)` and
/// `w = (bits + 1/2) 2^-64`. Each half of the range uses the form that keeps
/// full relative precision, so `E` is never 0 or infinite and the noise lies
/// in about [-3.81, 45.06].
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "u64 to f64 rounds to nearest, which the contract specifies"
)]
pub fn gumbel_from_bits(bits: u64) -> f64 {
    const TWO_TO_MINUS_64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
    let exponential = if bits < 1 << 63 {
        -libm::log1p(-((bits as f64 + 0.5) * TWO_TO_MINUS_64))
    } else {
        -libm::log(((!bits) as f64 + 0.5) * TWO_TO_MINUS_64)
    };
    -libm::log(exponential)
}

/// The perturbed score the sampler maximizes for one row.
#[must_use]
pub fn race_score(logit: f32, temperature: f64, key: RaceKey, position: u32, row: u32) -> f64 {
    f64::from(logit) / temperature + gumbel_from_bits(race_bits(key, position, row))
}

/// Samples the legal row with the largest perturbed score.
///
/// `sampling_logprob` is the row's log probability under the temperature
/// transform conditioned on the legal rows, as for
/// [`super::sample_categorical`]; it does not affect which row is chosen.
///
/// # Errors
///
/// Returns [`SamplingError`] for malformed input, a non-finite logit, or a
/// temperature whose scaling would make a transformed logit non-finite.
pub fn sample_race(
    logits: &[f32],
    legal_mask: &[bool],
    temperature: f64,
    key: RaceKey,
    position: u32,
) -> Result<CategoricalSample, SamplingError> {
    check_row(logits, legal_mask, temperature)?;
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(SamplingError::NonFiniteLogit);
    }
    let (legal_maximum, shifted_sum) = legal_shifted_sum(logits, legal_mask, temperature)?;

    let mut best: Option<(u32, f64)> = None;
    for (row, (&logit, &legal)) in (0_u32..).zip(logits.iter().zip(legal_mask)) {
        if !legal {
            continue;
        }
        let score = race_score(logit, temperature, key, position, row);
        if !score.is_finite() {
            return Err(SamplingError::InvalidTemperature);
        }
        if best.is_none_or(|(_, top)| score > top) {
            best = Some((row, score));
        }
    }
    let (token_id, _) = best.ok_or(SamplingError::EmptySupport)?;
    let index = usize::try_from(token_id).map_err(|_| SamplingError::VocabularyTooLarge)?;
    Ok(CategoricalSample {
        token_id,
        sampling_logprob: shifted_logit(logits[index], legal_maximum, temperature)?
            - shifted_sum.ln(),
    })
}

/// Rows that may hold the race winner, each with its reference score, plus
/// a `floor` that no unlisted row's reference score exceeds.
///
/// A device that scores rows approximately can still hand over a short list:
/// it lists every row whose approximate score is near its maximum and sets the
/// floor from its error bound. The host recomputes the listed scores exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct RaceCandidates {
    /// Descending score, ascending row on ties.
    rows: Vec<(u32, f64)>,
    floor: f64,
}

impl RaceCandidates {
    /// `rows` pairs each listed row with [`race_score`] for it; `floor` bounds
    /// the score of every row not listed.
    #[must_use]
    pub fn new(mut rows: Vec<(u32, f64)>, floor: f64) -> Self {
        rows.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        Self { rows, floor }
    }

    /// The `count` best rows of `logits`, scored exactly, with the floor at
    /// the best unlisted score (negative infinity when every row is listed).
    #[must_use]
    pub fn top(
        logits: &[f32],
        temperature: f64,
        key: RaceKey,
        position: u32,
        count: usize,
    ) -> Self {
        let mut all = Self::new(
            (0_u32..)
                .zip(logits)
                .map(|(row, &logit)| (row, race_score(logit, temperature, key, position, row)))
                .collect(),
            f64::NEG_INFINITY,
        );
        if let Some(&(_, floor)) = all.rows.get(count) {
            all.floor = floor;
        }
        all.rows.truncate(count);
        all
    }

    /// The race winner among rows satisfying `legal`, or `None` when the list
    /// cannot settle it: the first legal listed row must score strictly above
    /// the floor, or an unlisted legal row could beat or tie it.
    #[must_use]
    pub fn first_legal(&self, legal: impl Fn(u32) -> bool) -> Option<u32> {
        self.rows
            .iter()
            .find(|&&(row, _)| legal(row))
            .filter(|&&(_, score)| score > self.floor)
            .map(|&(row, _)| row)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RaceCandidates, RaceKey, gumbel_from_bits, philox4x32_10, race_bits, race_score,
        sample_race,
    };
    use crate::sampling::SamplingError;

    #[test]
    fn philox_matches_the_random123_known_answers() {
        // Random123 tests/kat_vectors, "philox4x32 10" lines.
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn noise_matches_an_independent_reference() {
        // Values from a separate numpy implementation of the same contract.
        let key = RaceKey {
            seed: 12_345,
            choice: 0,
        };
        let expected = [
            (0, 0x84d2_817b_7fce_2df9, 0x3fd4_017c_2844_cf3b),
            (1, 0xdeb1_9838_b73b_512b, 0xbfe6_ce33_dc24_c79f),
            (2, 0x81fc_f40f_96bd_b61e, 0x3fd6_070c_fbc9_39f3),
            (151_935, 0xb1f3_f8c4_8b33_1b82, 0xbfc6_094c_7c17_6167),
        ];
        for (row, bits, noise) in expected {
            assert_eq!(race_bits(key, 7, row), bits, "row {row}");
            // The reference used the platform libm, so allow a few ulps.
            let noise = f64::from_bits(noise);
            assert!((gumbel_from_bits(bits) - noise).abs() < 1e-15, "row {row}");
        }
    }

    #[test]
    fn noise_is_finite_and_bounded_at_the_extreme_bits() {
        for bits in [0, 1, (1 << 63) - 1, 1 << 63, u64::MAX - 1, u64::MAX] {
            let noise = gumbel_from_bits(bits);
            assert!((-3.81..45.06).contains(&noise), "{bits:#x}: {noise}");
        }
    }

    #[test]
    fn race_v2_golden_tokens_pin_the_contract() {
        // Any change to the key, counter, noise mapping, score or tie rule
        // changes these tokens; such a change needs a new sampler id.
        let logits = [0.0, 0.4, -0.3, 0.9, 0.1, -1.2, 0.6, 0.2];
        let legal = [true, true, true, true, true, false, true, true];
        let key = RaceKey { seed: 7, choice: 0 };
        let tokens: Vec<u32> = (0..24)
            .map(|position| {
                sample_race(&logits, &legal, 0.7, key, position)
                    .expect("valid draw")
                    .token_id
            })
            .collect();
        assert_eq!(
            tokens,
            [
                7, 7, 3, 6, 3, 7, 6, 3, 2, 3, 3, 3, 6, 7, 7, 0, 3, 4, 3, 6, 3, 3, 7, 6
            ]
        );
    }

    #[test]
    fn draws_follow_the_legal_temperature_law() {
        // Pearson chi-square over 30000 positions. Legal cells: 5, so 4
        // degrees of freedom (mean 4, sd 2.8); the bound is ~12 sd, while a
        // sampler that ignores the temperature lands in the thousands.
        let logits = [0.3_f32, -0.6, 1.1, 0.0, 0.7, 2.0];
        let legal = [true, true, true, true, true, false];
        let temperature = 0.6;
        let key = RaceKey {
            seed: 11,
            choice: 0,
        };
        let draws = 30_000_u32;
        let mut counts = [0_u32; 6];
        for position in 0..draws {
            let sample = sample_race(&logits, &legal, temperature, key, position).expect("draw");
            counts[usize::try_from(sample.token_id).expect("row")] += 1;
        }
        assert_eq!(counts[5], 0, "an illegal row was drawn");
        let weights: Vec<f64> = logits[..5]
            .iter()
            .map(|&logit| (f64::from(logit) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let chi_square: f64 = weights
            .iter()
            .zip(counts)
            .map(|(weight, observed)| {
                let expected = weight / total * f64::from(draws);
                (f64::from(observed) - expected).powi(2) / expected
            })
            .sum();
        assert!(
            chi_square < 40.0,
            "chi-square {chi_square}, counts {counts:?}"
        );
    }

    #[test]
    fn logprob_is_the_legal_conditional_and_rows_are_independent_of_order() {
        let logits = [0.0, 100.0, 2.0, 1.0];
        let legal = [true, false, true, true];
        let key = RaceKey { seed: 3, choice: 1 };
        for position in 0..32 {
            let sample = sample_race(&logits, &legal, 1.0, key, position).expect("draw");
            assert_ne!(sample.token_id, 1);
            let index = usize::try_from(sample.token_id).expect("row");
            let normalizer = (1.0_f64 + 2.0_f64.exp() + 1.0_f64.exp()).ln();
            let expected = f64::from(logits[index]) - normalizer;
            assert!((sample.sampling_logprob - expected).abs() < 1e-12);
            // The winner is the maximum of per-row scores, which each depend
            // only on their own key; recomputing them in any order agrees.
            let best = [3_u32, 2, 0]
                .into_iter()
                .max_by(|&left, &right| {
                    let score =
                        |row: u32| race_score(logits[row as usize], 1.0, key, position, row);
                    score(left).total_cmp(&score(right)).then(right.cmp(&left))
                })
                .expect("legal rows");
            assert_eq!(best, sample.token_id);
        }
    }

    #[test]
    fn exact_candidate_lists_agree_with_the_whole_row() {
        let logits = [0.0, 0.4, -0.3, 0.9, 0.1, -1.2, 0.6, 0.2];
        let key = RaceKey { seed: 5, choice: 0 };
        for position in 0..64_u32 {
            let legal: Vec<bool> = (0..8_u32).map(|row| (row + position) % 3 != 0).collect();
            let expected = sample_race(&logits, &legal, 0.7, key, position)
                .expect("draw")
                .token_id;
            for count in 1..=8 {
                let candidates = RaceCandidates::top(&logits, 0.7, key, position, count);
                let settled = candidates.first_legal(|row| legal[row as usize]);
                if count == 8 {
                    assert_eq!(settled, Some(expected));
                } else if let Some(row) = settled {
                    assert_eq!(row, expected, "position {position}, count {count}");
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        let key = RaceKey { seed: 0, choice: 0 };
        assert_eq!(
            sample_race(&[], &[], 1.0, key, 0),
            Err(SamplingError::EmptyLogits)
        );
        assert_eq!(
            sample_race(&[0.0], &[], 1.0, key, 0),
            Err(SamplingError::MaskLengthMismatch)
        );
        assert_eq!(
            sample_race(&[0.0], &[false], 1.0, key, 0),
            Err(SamplingError::EmptySupport)
        );
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                sample_race(&[0.0], &[true], temperature, key, 0),
                Err(SamplingError::InvalidTemperature)
            );
        }
        assert_eq!(
            sample_race(&[0.0, f32::NAN], &[true, false], 1.0, key, 0),
            Err(SamplingError::NonFiniteLogit)
        );
    }
}
