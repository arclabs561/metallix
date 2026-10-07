//! Chunked evaluation of the gated delta rule for multi-token calls.
//!
//! The token recurrence (`S = exp(g) S`, `S = S + k (beta (v - k^T S))^T`,
//! `o = q^T S`) is evaluated `C` tokens at a time with the WY representation
//! of [Gated Delta Networks](https://arxiv.org/html/2412.06464v1) (§3.2, App. A). Per head and chunk, with rows indexed
//! by token, `G` the within-chunk cumulative sum of the log decay `g`, and
//! `Gamma_ij = exp(G_i - G_j)` for `i >= j` (zero above the diagonal):
//!
//! ```text
//! A    = strict_lower(beta_i (K K^T)_ij Gamma_ij)
//! U    = (I + A)^-1 (beta V)
//! W    = (I + A)^-1 (beta exp(G) K)
//! Vnew = U - W S
//! O    = exp(G) (Q S) + ((Q K^T) . Gamma) Vnew
//! S'   = exp(G_last) S + (exp(G_last - G) K)^T Vnew
//! ```
//!
//! The paper's printed output equation omits `Gamma` from the intra-chunk
//! term; the expansion in its App. A, and the Transformers and
//! flash-linear-attention chunk kernels, include it.
//!
//! Everything except the last three lines is independent of `S`, so it runs
//! for all chunks in one graph; only `Vnew`, `O` and `S'` follow chunk order.
//! The unit lower-triangular inverse is computed by forward substitution on
//! diagonal blocks and block merges, not by a product series in powers of
//! `A`: with correlated keys the entries of `A^m` grow combinatorially and the
//! series loses all precision (it produced NaN in a scratch test with
//! near-identical keys), while substitution stays within f32 rounding.

use mlx_rs::{
    Array, StreamOrDevice,
    ops::{
        self,
        indexing::{Ellipsis, IndexOp},
    },
};

use super::Qwen35Error;

/// Tokens per chunk, as in the source's chunked prefill.
pub(super) const CHUNK_TOKENS: i32 = 64;

/// Width of the diagonal blocks inverted by forward substitution.
const BLOCK_TOKENS: i32 = 16;

/// The gated delta rule over `seq` tokens in chunks. Shapes as in
/// `delta_rule_by_token`.
pub(super) fn chunked(
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    mut state: Array,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let seq = query.shape()[0];
    let chunks = (seq + CHUNK_TOKENS - 1) / CHUNK_TOKENS;
    let padded = chunks * CHUNK_TOKENS;
    // [seq, H, ...] -> [H, chunks, C, ...]. Padding tokens have zero key,
    // value and beta and zero log decay, so they leave the state unchanged.
    let split = |x: &Array| -> Result<Array, Qwen35Error> {
        let mut shape = x.shape().to_vec();
        let x = if padded > seq {
            shape[0] = padded - seq;
            ops::concatenate_axis_device(&[x, &Array::zeros_device::<f32>(&shape, &gpu)?], 0, &gpu)?
        } else {
            x.clone()
        };
        let shape = [&[chunks, CHUNK_TOKENS], &shape[1..]].concat();
        let x = x.reshape_device(&shape, &gpu)?;
        Ok(if x.ndim() == 4 {
            x.transpose_axes_device(&[2, 0, 1, 3], &gpu)?
        } else {
            x.transpose_axes_device(&[2, 0, 1], &gpu)?
        })
    };
    let (query, key, value) = (split(query)?, split(key)?, split(value)?);
    let beta = split(beta)?.expand_dims_device(-1, &gpu)?;
    // G, [H, chunks, C]
    let cumulative = ops::cumsum_device(split(log_decay)?, -1, None, None, &gpu)?;

    // Gamma_ij = exp(G_i - G_j) on and below the diagonal, 0 above it. The
    // mask comes before exp because G_i - G_j > 0 above the diagonal.
    let on_or_below = Array::tri_device::<bool>(CHUNK_TOKENS, None, Some(0), &gpu)?;
    let gaps = cumulative
        .expand_dims_device(-1, &gpu)?
        .subtract_device(cumulative.expand_dims_device(-2, &gpu)?, &gpu)?;
    let gamma = ops::r#where_device(
        &on_or_below,
        &gaps,
        Array::from_f32(f32::NEG_INFINITY),
        &gpu,
    )?
    .exp_device(&gpu)?;
    let key_rows = key.transpose_axes_device(&[0, 1, 3, 2], &gpu)?;
    let strictly_below = Array::tri_device::<f32>(CHUNK_TOKENS, None, Some(-1), &gpu)?;
    let coupling = key
        .matmul_device(&key_rows, &gpu)?
        .multiply_device(&gamma, &gpu)?
        .multiply_device(&beta, &gpu)?
        .multiply_device(&strictly_below, &gpu)?;
    let inverse = unit_lower_inverse(&coupling)?;
    let gain = cumulative.exp_device(&gpu)?.expand_dims_device(-1, &gpu)?;
    let corrected = inverse.matmul_device(value.multiply_device(&beta, &gpu)?, &gpu)?;
    let decayed_keys = inverse.matmul_device(
        key.multiply_device(beta.multiply_device(&gain, &gpu)?, &gpu)?,
        &gpu,
    )?;
    let scores = query
        .matmul_device(&key_rows, &gpu)?
        .multiply_device(&gamma, &gpu)?;
    let last = cumulative.index((.., .., CHUNK_TOKENS - 1..CHUNK_TOKENS));
    // exp(G_last - G_i) K_i, transposed to [H, chunks, Dk, C]
    let carried_keys = last
        .subtract_device(&cumulative, &gpu)?
        .exp_device(&gpu)?
        .expand_dims_device(-1, &gpu)?
        .multiply_device(&key, &gpu)?
        .transpose_axes_device(&[0, 1, 3, 2], &gpu)?;
    // exp(G_last), [H, chunks, 1, 1]
    let chunk_decay = last.exp_device(&gpu)?.expand_dims_device(-1, &gpu)?;

    let mut outputs =
        Vec::with_capacity(usize::try_from(chunks).map_err(|_| Qwen35Error::ShapeOverflow)?);
    for chunk in 0..chunks {
        let at = |x: &Array| x.index((.., chunk));
        let fresh =
            at(&corrected).subtract_device(at(&decayed_keys).matmul_device(&state, &gpu)?, &gpu)?;
        outputs.push(
            at(&gain)
                .multiply_device(at(&query).matmul_device(&state, &gpu)?, &gpu)?
                .add_device(at(&scores).matmul_device(&fresh, &gpu)?, &gpu)?,
        );
        state = at(&chunk_decay)
            .multiply_device(&state, &gpu)?
            .add_device(at(&carried_keys).matmul_device(&fresh, &gpu)?, &gpu)?;
    }
    // [H, chunks, C, Dv] -> [seq, H, Dv]
    let heads = value.shape()[0];
    let value_dim = value.shape()[3];
    let output = ops::stack_axis_device(&outputs, 1, &gpu)?
        .reshape_device(&[heads, padded, value_dim], &gpu)?
        .index((.., 0..seq, ..))
        .transpose_axes_device(&[1, 0, 2], &gpu)?;
    Ok((output, state))
}

/// `(I + A)^-1` for strictly lower-triangular `A` of shape `[..., C, C]`.
///
/// Each `BLOCK_TOKENS`-wide diagonal block is inverted by forward
/// substitution (row `i` of the inverse is `e_i - A_i T`, where only the rows
/// above `i` of `T` are filled yet); pairs of inverted blocks then merge as
/// `[[T11, 0], [-T22 A21 T11, T22]]` until one block covers the chunk.
fn unit_lower_inverse(coupling: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let width = coupling.shape()[coupling.ndim() - 1];
    let diagonal = (0..width / BLOCK_TOKENS)
        .map(|block| {
            let rows = block * BLOCK_TOKENS..(block + 1) * BLOCK_TOKENS;
            coupling.index((Ellipsis, rows.clone(), rows))
        })
        .collect::<Vec<_>>();
    // [..., blocks, B, B]
    let diagonal = ops::stack_axis_device(&diagonal, -3, &gpu)?;
    let identity = Array::eye_device::<f32>(BLOCK_TOKENS, None, None, &gpu)?;
    let mut inverse = ops::zeros_like_device(&diagonal, &gpu)?;
    for row in 0..BLOCK_TOKENS {
        let filled = identity.index(row..row + 1).subtract_device(
            diagonal
                .index((Ellipsis, row..row + 1, ..))
                .matmul_device(&inverse, &gpu)?,
            &gpu,
        )?;
        inverse = inverse.add_device(
            identity
                .index((.., row..row + 1))
                .multiply_device(&filled, &gpu)?,
            &gpu,
        )?;
    }
    let mut blocks = (0..width / BLOCK_TOKENS)
        .map(|block| inverse.index((Ellipsis, block, .., ..)))
        .collect::<Vec<_>>();
    let mut size = BLOCK_TOKENS;
    while blocks.len() > 1 {
        blocks = blocks
            .chunks(2)
            .zip((0..).step_by(2))
            .map(|(pair, first)| -> Result<Array, Qwen35Error> {
                let start = first * size;
                let below = coupling.index((
                    Ellipsis,
                    start + size..start + 2 * size,
                    start..start + size,
                ));
                let corner = pair[1]
                    .matmul_device(below.matmul_device(&pair[0], &gpu)?, &gpu)?
                    .negative_device(&gpu)?;
                let top = ops::concatenate_axis_device(
                    &[&pair[0], &ops::zeros_like_device(&pair[0], &gpu)?],
                    -1,
                    &gpu,
                )?;
                let bottom = ops::concatenate_axis_device(&[&corner, &pair[1]], -1, &gpu)?;
                Ok(ops::concatenate_axis_device(&[top, bottom], -2, &gpu)?)
            })
            .collect::<Result<_, _>>()?;
        size *= 2;
    }
    Ok(blocks.swap_remove(0))
}

#[cfg(test)]
mod tests {
    use mlx_rs::{
        Array, StreamOrDevice,
        ops::{self, indexing::IndexOp},
    };

    use super::super::delta_rule_by_token;
    use super::chunked;

    const HEADS: i32 = 4;
    const KEY_DIM: i32 = 128;
    const VALUE_DIM: i32 = 128;

    /// Relative bound declared from the reformulation's rounding, not from
    /// results: each output passes through at most three chained f32
    /// reductions of length <= C + Dk = 192, so a chunk adds at most about
    /// 3 * 192 * 2^-24 ~= 3.4e-5 relative error; the recurrence is a
    /// contraction, so 32 chunks accumulate it at most as a random walk,
    /// sqrt(32) * 3.4e-5 ~= 1.9e-4 in the worst case, with typical rounding
    /// far below that. Scratch runs measured 2e-6 to 3e-5.
    const MAX_RELATIVE: f32 = 1e-4;

    /// Deterministic values in [-1, 1) (xorshift64), so failures reproduce.
    fn values(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "24 high bits fit an f32 mantissa exactly"
                )]
                let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
                2.0 * unit - 1.0
            })
            .collect()
    }

    fn array(seed: u64, shape: &[i32]) -> Array {
        let count = shape.iter().product::<i32>();
        Array::from_slice(
            &values(seed, usize::try_from(count).expect("positive size")),
            shape,
        )
    }

    fn l2_rows(x: &Array) -> Array {
        let gpu = StreamOrDevice::gpu();
        let norm = x
            .square_device(&gpu)
            .and_then(|s| s.sum_axis_device(-1, true, &gpu))
            .and_then(|s| s.sqrt_device(&gpu))
            .expect("norm");
        x.divide_device(norm, &gpu).expect("normalize")
    }

    /// Inputs with the checkpoint's ranges: unit-norm keys, queries scaled by
    /// `Dk^-1/2`, beta in (0, 1), log decay below zero. `shared` in [0, 1]
    /// mixes one key per head into every token's key; near 1 the keys are
    /// almost parallel, the case that breaks a power-series inverse.
    struct Inputs {
        query: Array,
        key: Array,
        value: Array,
        log_decay: Array,
        beta: Array,
        state: Array,
    }

    fn inputs(seq: i32, shared: f32, seed: u64) -> Inputs {
        let gpu = StreamOrDevice::gpu();
        let common = array(seed, &[1, HEADS, KEY_DIM]);
        let key = array(seed + 1, &[seq, HEADS, KEY_DIM])
            .multiply_device(Array::from_f32(1.0 - shared), &gpu)
            .and_then(|k| {
                k.add_device(common.multiply_device(Array::from_f32(shared), &gpu)?, &gpu)
            })
            .expect("keys");
        #[allow(clippy::cast_precision_loss, reason = "128 is exact in f32")]
        let scale = (KEY_DIM as f32).powf(-0.5);
        let query = l2_rows(&array(seed + 2, &[seq, HEADS, KEY_DIM]))
            .multiply_device(Array::from_f32(scale), &gpu)
            .expect("queries");
        let unit = |x: Array| {
            x.add_device(Array::from_f32(1.0), &gpu)
                .and_then(|x| x.multiply_device(Array::from_f32(0.5), &gpu))
                .expect("unit interval")
        };
        // beta in (0, 1); decay exp(g) in about (0.37, 1): g = -u, u in [0, 1)
        let beta = unit(array(seed + 4, &[seq, HEADS]));
        let log_decay = unit(array(seed + 5, &[seq, HEADS]))
            .negative_device(&gpu)
            .expect("log decay");
        Inputs {
            query,
            key: l2_rows(&key),
            value: array(seed + 3, &[seq, HEADS, VALUE_DIM]),
            log_decay,
            beta,
            state: array(seed + 6, &[HEADS, KEY_DIM, VALUE_DIM])
                .multiply_device(Array::from_f32(0.1), &gpu)
                .expect("state"),
        }
    }

    fn relative(actual: &Array, expected: &Array) -> f32 {
        let gpu = StreamOrDevice::gpu();
        let worst = actual
            .subtract_device(expected, &gpu)
            .and_then(|d| d.abs_device(&gpu))
            .and_then(|d| d.max_device(None, &gpu))
            .expect("difference");
        let scale = expected
            .abs_device(&gpu)
            .and_then(|e| e.max_device(None, &gpu))
            .expect("scale");
        worst.item::<f32>() / scale.item::<f32>()
    }

    // Bound only the test oracle's lazy graph. Production calls the unchanged
    // token recurrence for one decode token; prefill uses the chunked solver.
    fn bounded_token_oracle(x: &Inputs) -> (Array, Array) {
        const ORACLE_TOKENS: i32 = 64;
        let gpu = StreamOrDevice::gpu();
        let seq = x.query.shape()[0];
        let mut state = x.state.clone();
        let mut outputs = Vec::new();
        for start in (0..seq).step_by(64) {
            let end = (start + ORACLE_TOKENS).min(seq);
            let slice = |array: &Array| {
                array
                    .index(start..end)
                    .contiguous()
                    .expect("contiguous oracle input")
            };
            let (output, next_state) = delta_rule_by_token(
                &slice(&x.query),
                &slice(&x.key),
                &slice(&x.value),
                &slice(&x.log_decay),
                &slice(&x.beta),
                state,
            )
            .expect("token recurrence slice");
            mlx_rs::transforms::eval([&output, &next_state])
                .expect("joint oracle slice evaluation");
            outputs.push(output);
            state = next_state;
        }
        let output =
            ops::concatenate_axis_device(&outputs, 0, &gpu).expect("oracle outputs in token order");
        (output, state)
    }

    #[test]
    fn bounded_oracle_preserves_small_monolithic_results_bit_for_bit() {
        for seq in [1, 17, 64] {
            let x = inputs(seq, 0.3, 11);
            let (expected, expected_state) = delta_rule_by_token(
                &x.query,
                &x.key,
                &x.value,
                &x.log_decay,
                &x.beta,
                x.state.clone(),
            )
            .expect("small monolithic oracle");
            let (actual, actual_state) = bounded_token_oracle(&x);
            mlx_rs::transforms::eval([&expected, &expected_state, &actual, &actual_state])
                .expect("oracle control evaluation");
            for (label, actual, expected) in [
                ("outputs", &actual, &expected),
                ("final state", &actual_state, &expected_state),
            ] {
                assert_eq!(actual.shape(), expected.shape(), "{label}, seq={seq}");
                let bits = |array: &Array| {
                    let array = array.contiguous().expect("logical oracle element order");
                    array.eval().expect("oracle readback");
                    array
                        .as_slice::<f32>()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(actual), bits(expected), "{label}, seq={seq}");
            }
        }
    }

    fn check(seq: i32, shared: f32) {
        let x = inputs(seq, shared, 7);
        let (expected, expected_state) = bounded_token_oracle(&x);
        let (actual, actual_state) =
            chunked(&x.query, &x.key, &x.value, &x.log_decay, &x.beta, x.state).expect("chunked");
        mlx_rs::transforms::eval([&actual, &actual_state]).expect("chunked evaluation");
        assert_eq!(actual.shape(), expected.shape());
        let (output, state) = (
            relative(&actual, &expected),
            relative(&actual_state, &expected_state),
        );
        eprintln!("seq {seq}, shared key weight {shared}: output {output:.3e}, state {state:.3e}");
        assert!(
            output <= MAX_RELATIVE && state <= MAX_RELATIVE,
            "seq {seq}, shared {shared}: relative output {output:.3e}, state {state:.3e} exceed {MAX_RELATIVE:e}"
        );
    }

    // Scalar recurrence with k=q=1, beta=1/2, g=0 and S0=0:
    // v=[2,0] gives S=[1,1/2], so outputs are [1,1/2]. The remaining
    // tokens have beta=0 and preserve S=1/2. These exact dyadic values
    // test the update itself without taking the token implementation as oracle.
    fn check_exact_adjacent_pair(first: usize) {
        const TOKENS: usize = 64;
        let mut values = [0.0_f32; TOKENS];
        let mut beta = [0.0_f32; TOKENS];
        values[first] = 2.0;
        beta[first] = 0.5;
        beta[first + 1] = 0.5;
        let query = Array::from_slice(&[1.0_f32; TOKENS], &[64, 1, 1]);
        let key = Array::from_slice(&[1.0_f32; TOKENS], &[64, 1, 1]);
        let value = Array::from_slice(&values, &[64, 1, 1]);
        let log_decay = Array::from_slice(&[0.0_f32; TOKENS], &[64, 1]);
        let beta = Array::from_slice(&beta, &[64, 1]);
        let state = Array::from_slice(&[0.0_f32], &[1, 1, 1]);
        let (output, state) =
            chunked(&query, &key, &value, &log_decay, &beta, state).expect("exact chunk");
        output.eval().expect("output evaluation");
        state.eval().expect("state evaluation");
        let mut expected = [0.0_f32; TOKENS];
        expected[first] = 1.0;
        expected[first + 1..].fill(0.5);
        assert_eq!(output.shape(), &[64, 1, 1]);
        assert_eq!(output.as_slice::<f32>(), &expected);
        assert_eq!(state.as_slice::<f32>(), &[0.5]);
    }

    #[test]
    fn exact_adjacent_delta_updates_within_a_diagonal_block() {
        check_exact_adjacent_pair(0);
    }

    #[test]
    fn exact_adjacent_delta_updates_across_the_block_merge() {
        // The active edge 15 -> 16 is in A21, outside either diagonal solve.
        check_exact_adjacent_pair(15);
    }

    #[test]
    fn chunks_match_the_token_recurrence_over_a_2k_prompt() {
        check(2048, 0.0);
    }

    #[test]
    fn chunks_match_the_token_recurrence_with_nearly_parallel_keys() {
        check(512, 0.95);
    }

    #[test]
    fn a_partial_last_chunk_matches_the_token_recurrence() {
        // 1 token, one short of a chunk, one past it, and one past a block.
        for seq in [1, 63, 65, 81] {
            check(seq, 0.3);
        }
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        let gpu = StreamOrDevice::gpu();
        let strictly_below = Array::tri_device::<f32>(64, None, Some(-1), &gpu).expect("mask");
        let coupling = array(11, &[3, 64, 64])
            .multiply_device(Array::from_f32(0.2), &gpu)
            .and_then(|a| a.multiply_device(&strictly_below, &gpu))
            .expect("strictly lower");
        let identity = Array::eye_device::<f32>(64, None, None, &gpu).expect("identity");
        let inverse = super::unit_lower_inverse(&coupling).expect("inverse");
        let product = identity
            .add_device(&coupling, &gpu)
            .and_then(|m| m.matmul_device(&inverse, &gpu))
            .expect("product");
        let error = ops::subtract_device(&product, &identity, &gpu)
            .and_then(|d| d.abs_device(&gpu))
            .and_then(|d| d.max_device(None, &gpu))
            .expect("error")
            .item::<f32>();
        assert!(error < 1e-4, "(I + A) T differs from I by {error}");
    }
}
