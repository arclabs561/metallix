//! Token picks computed on the GPU from logit rows, so the host reads back
//! token IDs, and optionally a few top candidates, instead of whole
//! vocabulary rows.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use mlx_rs::{Array, StreamOrDevice, ops, ops::indexing::IndexOp};

use super::Qwen3ForwardError;

/// How each row's token is chosen on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub enum Qwen3Selection {
    /// The row's argmax, lowest ID on ties.
    Greedy,
    /// The `top_k` most likely tokens, then the smallest most-likely prefix
    /// whose temperature-transformed mass reaches `top_p` of theirs, then a
    /// categorical draw over that set in token-ID order at `uniforms[row]`.
    ///
    /// This is computed in f32, so near a cumulative-mass boundary it can
    /// pick a different token than an f64 host sampler; callers that owe an
    /// exact sampling contract re-derive the token from the candidates.
    TopKNucleus {
        /// Positive, finite.
        temperature: f64,
        /// In (0, 1].
        top_p: f64,
        /// At least one and fewer than the requested candidates.
        top_k: usize,
        /// One variate in [0, 1) per row.
        uniforms: Vec<f64>,
    },
}

/// A GPU pick and how many of each row's largest logits to read back with it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3PickRule {
    /// How the token is chosen.
    pub selection: Qwen3Selection,
    /// Largest logits per row to return, with their IDs; zero returns none.
    pub candidates: usize,
}

impl Qwen3PickRule {
    /// The argmax alone: one token ID per row crosses to the host.
    pub const GREEDY: Self = Self {
        selection: Qwen3Selection::Greedy,
        candidates: 0,
    };
}

/// One row's largest logits, read back next to its pick.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3RowCandidates {
    /// `(token ID, logit)` for the row's largest logits, in descending logit
    /// order and ascending ID on ties. These are the exact f32 logits.
    pub top: Vec<(i32, f32)>,
    /// The row's largest logit.
    pub max_logit: f32,
    /// `sum(exp(logit - max_logit))` over the whole row, summed in f32 on
    /// the GPU; the only value here that differs from an f64 host pass.
    pub shifted_exp_sum: f32,
}

impl Qwen3RowCandidates {
    /// Natural-log probability of a logit under the row's softmax.
    #[must_use]
    pub fn logprob(&self, logit: f32) -> f64 {
        f64::from(logit) - f64::from(self.max_logit) - f64::from(self.shifted_exp_sum).ln()
    }

    /// Whether the first `head` entries are the row's `head` largest logits
    /// with ties broken toward lower IDs, as a full-row host selection
    /// would choose them. That holds unless the entry at `head - 1` ties the
    /// smallest candidate, when an excluded lower-ID token could tie it too.
    #[must_use]
    pub fn settles_head(&self, head: usize) -> bool {
        match (
            head.checked_sub(1).and_then(|last| self.top.get(last)),
            self.top.last(),
        ) {
            (Some(&(_, edge)), Some(&(_, smallest))) => head < self.top.len() && edge > smallest,
            _ => false,
        }
    }
}

/// Token picks for one or more logit rows that may still be computing on
/// the GPU. Decode makes one row; a speculative verify can make several.
pub struct Qwen3TokenPicks {
    /// `[rows]` uint32 token IDs.
    pub(super) tokens: Array,
    finite: Array,
    /// The executor weights these picks may be appended to.
    pub(super) binding: usize,
    candidates: Option<Candidates>,
    /// The `[rows, vocab]` logits, kept so a caller can still read whole rows.
    logits: Array,
}

struct Candidates {
    /// `[rows, count]` uint32 IDs and f32 logits, in no particular order.
    ids: Array,
    logits: Array,
    /// `[rows]` f32.
    max_logit: Array,
    shifted_exp_sum: Array,
}

impl Qwen3PickRule {
    /// Validates this rule against `[rows, vocab]` logits and returns the
    /// row count and the number of candidates to read per row.
    fn checked_candidates(&self, shape: &[i32]) -> Result<(i32, i32), Qwen3ForwardError> {
        let &[rows, vocab] = shape else {
            return Err(Qwen3ForwardError::InvalidPickRule(
                "logits must be [rows, vocab]",
            ));
        };
        if let Qwen3Selection::TopKNucleus {
            temperature,
            top_p,
            top_k,
            uniforms,
        } = &self.selection
        {
            if !(temperature.is_finite() && *temperature > 0.0) {
                return Err(Qwen3ForwardError::InvalidPickRule("temperature"));
            }
            if !(*top_p > 0.0 && *top_p <= 1.0) {
                return Err(Qwen3ForwardError::InvalidPickRule("top_p"));
            }
            if *top_k == 0 || *top_k >= self.candidates {
                return Err(Qwen3ForwardError::InvalidPickRule(
                    "top_k must be positive and below the candidate count",
                ));
            }
            if uniforms.len() != usize::try_from(rows).unwrap_or(usize::MAX)
                || uniforms.iter().any(|u| !(0.0..1.0).contains(u))
            {
                return Err(Qwen3ForwardError::InvalidPickRule(
                    "one uniform in [0, 1) per row",
                ));
            }
        }
        let wanted =
            i32::try_from(self.candidates).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        if wanted >= vocab {
            return Err(Qwen3ForwardError::InvalidPickRule(
                "candidates must be fewer than the vocabulary",
            ));
        }
        Ok((rows, wanted))
    }
}

impl Candidates {
    /// Queues the `wanted` largest entries of each f32 `[rows, vocab]` row,
    /// the row maximum and the sum of shifted exponentials.
    fn start(wide: &Array, rows: i32, wanted: i32) -> Result<Self, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        // The first `wanted` positions of an ascending partition of the
        // negated row are the row's `wanted` largest logits.
        let ids =
            ops::argpartition_axis_device(wide.negative_device(&stream)?, wanted - 1, -1, &stream)?
                .index((.., 0..wanted))
                // The slice is a strided view; readback needs dense rows.
                .contiguous()?;
        let logits = ops::indexing::take_along_axis_device(wide, &ids, -1, &stream)?;
        let max_logit = wide.max_axis_device(-1, true, &stream)?;
        let shifted_exp_sum = ops::exp_device(wide.subtract_device(&max_logit, &stream)?, &stream)?
            .sum_axis_device(-1, false, &stream)?;
        Ok(Self {
            ids,
            logits,
            max_logit: max_logit.reshape_device(&[rows], &stream)?,
            shifted_exp_sum,
        })
    }
}

impl Qwen3TokenPicks {
    /// Queues the per-row argmax of `[rows, vocab]` logits and a check that
    /// every logit is finite.
    pub(crate) fn start(rows: &Array, binding: usize) -> Result<Self, Qwen3ForwardError> {
        Self::start_with(rows, binding, &Qwen3PickRule::GREEDY)
    }

    /// Queues `rule`'s pick over `[rows, vocab]` logits, its candidates, and
    /// a check that every logit is finite.
    pub(crate) fn start_with(
        rows: &Array,
        binding: usize,
        rule: &Qwen3PickRule,
    ) -> Result<Self, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let (row_count, wanted) = rule.checked_candidates(rows.shape())?;
        let finite = rows.is_finite_device(&stream)?.all_device(false, &stream)?;
        let wide = rows.as_type_device::<f32>(&stream)?;
        let candidates = if wanted > 0 {
            Some(Candidates::start(&wide, row_count, wanted)?)
        } else {
            None
        };
        let tokens = match &rule.selection {
            Qwen3Selection::Greedy => ops::indexing::argmax_axis_device(rows, -1, false, &stream)?,
            Qwen3Selection::TopKNucleus {
                temperature,
                top_p,
                top_k,
                uniforms,
            } => {
                let candidates = candidates
                    .as_ref()
                    .ok_or(Qwen3ForwardError::InvalidPickRule(
                        "sampling needs candidates",
                    ))?;
                sample_top_k_nucleus(
                    &candidates.ids,
                    &candidates.logits,
                    *temperature,
                    *top_p,
                    *top_k,
                    uniforms,
                )?
            }
        };
        let mut queued = vec![&tokens, &finite];
        if let Some(candidates) = &candidates {
            queued.extend([
                &candidates.ids,
                &candidates.logits,
                &candidates.max_logit,
                &candidates.shifted_exp_sum,
            ]);
        }
        mlx_rs::transforms::async_eval(queued)?;
        Ok(Self {
            tokens,
            finite,
            binding,
            candidates,
            logits: rows.clone(),
        })
    }

    /// Waits for the picks and reads back one token ID per row. Ties go to
    /// the lowest ID, as in a host argmax that keeps the first maximum.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3ForwardError::NonFiniteLogits`] when any logit was NaN
    /// or infinite, which the host argmax also refuses.
    pub fn wait(&self) -> Result<Vec<i32>, Qwen3ForwardError> {
        mlx_rs::transforms::eval([&self.tokens, &self.finite])?;
        if !self.finite.item::<bool>() {
            return Err(Qwen3ForwardError::NonFiniteLogits);
        }
        // MLX returns argmax and partition indices as uint32.
        self.tokens
            .as_slice::<u32>()
            .iter()
            .map(|&token| to_token(token))
            .collect()
    }

    /// [`Self::wait`] for a single-row pick, such as one decode step.
    pub fn wait_one(&self) -> Result<i32, Qwen3ForwardError> {
        match self.wait()?.as_slice() {
            [token] => Ok(*token),
            _ => Err(Qwen3ForwardError::CacheInconsistent),
        }
    }

    /// [`Self::wait`] plus each row's candidates, when the rule asked for
    /// any.
    pub fn wait_with_candidates(
        &self,
    ) -> Result<(Vec<i32>, Option<Vec<Qwen3RowCandidates>>), Qwen3ForwardError> {
        let tokens = self.wait()?;
        let Some(candidates) = &self.candidates else {
            return Ok((tokens, None));
        };
        mlx_rs::transforms::eval([
            &candidates.ids,
            &candidates.logits,
            &candidates.max_logit,
            &candidates.shifted_exp_sum,
        ])?;
        let width = candidates.ids.shape()[1];
        let width = usize::try_from(width).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let rows = candidates
            .ids
            .as_slice::<u32>()
            .chunks_exact(width)
            .zip(candidates.logits.as_slice::<f32>().chunks_exact(width))
            .zip(candidates.max_logit.as_slice::<f32>())
            .zip(candidates.shifted_exp_sum.as_slice::<f32>())
            .map(|(((ids, logits), &max_logit), &shifted_exp_sum)| {
                let mut top = ids
                    .iter()
                    .zip(logits)
                    .map(|(&id, &logit)| Ok((to_token(id)?, logit)))
                    .collect::<Result<Vec<_>, Qwen3ForwardError>>()?;
                top.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
                Ok(Qwen3RowCandidates {
                    top,
                    max_logit,
                    shifted_exp_sum,
                })
            })
            .collect::<Result<Vec<_>, Qwen3ForwardError>>()?;
        Ok((tokens, Some(rows)))
    }

    /// Reads back the whole `[rows, vocab]` logits as f32 rows, for a caller
    /// whose candidates could not settle its answer.
    pub fn full_rows(&self) -> Result<Vec<Vec<f32>>, Qwen3ForwardError> {
        let wide = self.logits.as_type_device::<f32>(StreamOrDevice::gpu())?;
        wide.eval()?;
        let vocab =
            usize::try_from(wide.shape()[1]).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        Ok(wide
            .as_slice::<f32>()
            .chunks_exact(vocab)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

fn to_token(id: u32) -> Result<i32, Qwen3ForwardError> {
    i32::try_from(id).map_err(|_| Qwen3ForwardError::ShapeOverflow)
}

/// f32 mirror of the host top-k, nucleus and categorical steps over the
/// unsorted `[rows, candidates]` IDs and logits.
fn sample_top_k_nucleus(
    ids: &Array,
    logits: &Array,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    uniforms: &[f64],
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let top_k = i32::try_from(top_k).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the GPU mirror runs in f32"
    )]
    let (temperature, top_p) = (temperature as f32, top_p as f32);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the GPU mirror runs in f32"
    )]
    let uniforms: Vec<f32> = uniforms.iter().map(|&u| u as f32).collect();
    let rows = i32::try_from(uniforms.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;

    // Most likely first; the host breaks exact ties by ID, which the
    // caller's exact re-derivation covers.
    let descending = ops::argsort_axis_device(logits.negative_device(&stream)?, -1, &stream)?
        .index((.., 0..top_k));
    let head_logits = ops::indexing::take_along_axis_device(logits, &descending, -1, &stream)?;
    let head_ids = ops::indexing::take_along_axis_device(ids, &descending, -1, &stream)?;
    let weights = ops::exp_device(
        head_logits
            .subtract_device(head_logits.index((.., 0..1)), &stream)?
            .divide_device(Array::from_f32(temperature), &stream)?,
        &stream,
    )?;
    // A token is kept when the mass before it is still short of the target.
    let kept = weights.sum_axis_device(-1, true, &stream)?;
    let before = ops::cumsum_device(&weights, -1, false, true, &stream)?
        .subtract_device(&weights, &stream)?;
    let legal = before.lt_device(
        kept.multiply_device(Array::from_f32(top_p), &stream)?,
        &stream,
    )?;
    let legal_weights = ops::which_device(&legal, &weights, Array::from_f32(0.0), &stream)?;

    // Draw in token-ID order, as the host sampler walks the vocabulary.
    let by_id = ops::argsort_axis_device(&head_ids, -1, &stream)?;
    let ordered_ids = ops::indexing::take_along_axis_device(&head_ids, &by_id, -1, &stream)?;
    let ordered = ops::indexing::take_along_axis_device(&legal_weights, &by_id, -1, &stream)?;
    let cumulative = ops::cumsum_device(&ordered, -1, false, true, &stream)?;
    let target = Array::from_slice(&uniforms, &[rows, 1])
        .multiply_device(ordered.sum_axis_device(-1, true, &stream)?, &stream)?;
    let position = cumulative
        .le_device(&target, &stream)?
        .sum_axis_device(-1, true, &stream)?;
    // Rounding can leave the target at or past the last cumulative mass;
    // then the last positive-weight token is drawn, as on the host.
    let columns = Array::arange_device::<i32, i32>(0, top_k, None, &stream)?;
    let last_positive = ops::which_device(
        ordered.gt_device(Array::from_f32(0.0), &stream)?,
        &columns,
        Array::from_int(-1),
        &stream,
    )?
    .max_axis_device(-1, true, &stream)?;
    let position = ops::minimum_device(
        position.as_type_device::<i32>(&stream)?,
        &last_positive,
        &stream,
    )?;
    Ok(
        ops::indexing::take_along_axis_device(&ordered_ids, &position, -1, &stream)?
            .reshape_device(&[rows], &stream)?,
    )
}

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    use super::{Qwen3PickRule, Qwen3RowCandidates, Qwen3Selection, Qwen3TokenPicks};
    use crate::GPU_TEST_LOCK;

    /// A deterministic, uneven logit row with repeated values.
    fn row(vocab: usize, salt: usize) -> Vec<f32> {
        (0..vocab)
            .map(|index| {
                let bucket = u16::try_from((index * 7_919 + salt * 104_729) % 4_001)
                    .expect("bucket fits u16");
                f32::from(bucket) / 400.0 - 5.0
            })
            .collect()
    }

    fn rows_array(rows: &[Vec<f32>]) -> Array {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Array::from_slice(
            &flat,
            &[
                i32::try_from(rows.len()).expect("rows"),
                i32::try_from(rows[0].len()).expect("vocab"),
            ],
        )
    }

    fn host_top(logits: &[f32], count: usize) -> Vec<(i32, f32)> {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&left, &right| {
            logits[right]
                .total_cmp(&logits[left])
                .then(left.cmp(&right))
        });
        order
            .into_iter()
            .take(count)
            .map(|index| (i32::try_from(index).expect("id"), logits[index]))
            .collect()
    }

    #[test]
    fn candidates_match_a_host_pass_over_the_full_row() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let logits = vec![row(5_000, 1), row(5_000, 2)];
        let rule = Qwen3PickRule {
            selection: Qwen3Selection::Greedy,
            candidates: 29,
        };
        let picks = Qwen3TokenPicks::start_with(&rows_array(&logits), 0, &rule).expect("queued");
        let (tokens, candidates) = picks.wait_with_candidates().expect("finite rows");
        let candidates = candidates.expect("candidates were requested");
        for ((row, token), candidates) in logits.iter().zip(tokens).zip(candidates) {
            let expected = host_top(row, 29);
            assert_eq!(token, expected[0].0);
            // Tied values may surface in either order from the partition;
            // the readback restores ascending IDs, so whole lists agree
            // whenever the 29th value is not tied past the boundary.
            if candidates.settles_head(28) {
                assert_eq!(&candidates.top[..28], &expected[..28]);
            }
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row
                .iter()
                .map(|&logit| (f64::from(logit) - f64::from(maximum)).exp())
                .sum();
            assert_eq!(candidates.max_logit, maximum);
            for &(id, logit) in &candidates.top {
                let host = f64::from(logit) - f64::from(maximum) - sum.ln();
                let index = usize::try_from(id).expect("id");
                assert_eq!(logit, row[index]);
                assert!((candidates.logprob(logit) - host).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn settled_heads_exclude_ties_across_the_boundary() {
        let candidates = |values: &[f32]| Qwen3RowCandidates {
            top: values
                .iter()
                .enumerate()
                .map(|(index, &value)| (i32::try_from(index).expect("id"), value))
                .collect(),
            max_logit: values[0],
            shifted_exp_sum: 1.0,
        };
        assert!(candidates(&[3.0, 2.0, 1.0]).settles_head(2));
        // The 2nd value ties the smallest candidate: an excluded token with a
        // lower ID could hold the same value.
        assert!(!candidates(&[3.0, 1.0, 1.0]).settles_head(2));
        // A head as wide as the candidates is never settled.
        assert!(!candidates(&[3.0, 2.0, 1.0]).settles_head(3));
        assert!(!candidates(&[3.0, 2.0, 1.0]).settles_head(0));
    }

    /// Definition-level f64 oracle for the top-k, nucleus and categorical
    /// steps, independent of the GPU mirror's operation order.
    fn oracle(
        logits: &[f32],
        temperature: f64,
        top_p: f64,
        top_k: usize,
        uniform: f64,
    ) -> (i32, f64) {
        let head = host_top(logits, top_k);
        let maximum = f64::from(head[0].1);
        let weight = |logit: f32| ((f64::from(logit) - maximum) / temperature).exp();
        let kept: f64 = head.iter().map(|&(_, logit)| weight(logit)).sum();
        let mut legal = Vec::new();
        let mut covered = 0.0;
        for &(id, logit) in &head {
            legal.push((id, weight(logit)));
            covered += weight(logit);
            if covered >= top_p * kept {
                break;
            }
        }
        legal.sort_by_key(|&(id, _)| id);
        let total: f64 = legal.iter().map(|&(_, weight)| weight).sum();
        let target = uniform * total;
        let mut cumulative = 0.0;
        let mut margin = f64::INFINITY;
        for &(id, weight) in &legal {
            cumulative += weight;
            margin = margin.min((cumulative - target).abs() / total);
            if target < cumulative {
                return (id, margin);
            }
        }
        (legal.last().expect("nonempty").0, margin)
    }

    #[test]
    fn gpu_sampler_matches_an_f64_oracle_away_from_cumulative_boundaries() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut checked = 0;
        for salt in 0..6 {
            let logits = row(3_000, salt);
            for (temperature, top_p, top_k) in [(0.6, 0.95, 20), (1.0, 0.5, 8), (1.3, 1.0, 40)] {
                for step in 0..40_u32 {
                    let uniform = f64::from(step) / 40.0 + 0.003;
                    let (expected, margin) = oracle(&logits, temperature, top_p, top_k, uniform);
                    let rule = Qwen3PickRule {
                        selection: Qwen3Selection::TopKNucleus {
                            temperature,
                            top_p,
                            top_k,
                            uniforms: vec![uniform],
                        },
                        candidates: top_k + 1,
                    };
                    let picks = Qwen3TokenPicks::start_with(
                        &rows_array(std::slice::from_ref(&logits)),
                        0,
                        &rule,
                    )
                    .expect("queued");
                    let token = picks.wait_one().expect("finite");
                    // f32 cumulative sums may land on the other side of a
                    // boundary within their rounding; elsewhere they agree.
                    if margin > 1e-4 {
                        assert_eq!(token, expected, "salt {salt} uniform {uniform}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked > 600,
            "only {checked} draws were away from a boundary"
        );
    }

    #[test]
    fn pick_rules_are_validated() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let logits = rows_array(&[row(100, 0)]);
        let sampled = |top_k: usize, candidates: usize, uniforms: Vec<f64>| Qwen3PickRule {
            selection: Qwen3Selection::TopKNucleus {
                temperature: 1.0,
                top_p: 1.0,
                top_k,
                uniforms,
            },
            candidates,
        };
        for rule in [
            sampled(5, 5, vec![0.5]),
            sampled(0, 5, vec![0.5]),
            sampled(5, 6, vec![1.0]),
            sampled(5, 6, vec![0.1, 0.2]),
            Qwen3PickRule {
                selection: Qwen3Selection::Greedy,
                candidates: 100,
            },
        ] {
            assert!(Qwen3TokenPicks::start_with(&logits, 0, &rule).is_err());
        }
    }
}
