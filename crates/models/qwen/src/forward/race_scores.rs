//! `race-v2` scores on the GPU: an f32 mirror of `engine::sampling::race`.
//!
//! Each thread scores one row entry: Philox4x32-10 noise keyed by the row's
//! seed, choice and generated position and by the token ID, turned into
//! Gumbel noise, plus `(logit - shift) / temperature` with the row maximum as
//! `shift`. The host reference in f64 remains the definition; these scores
//! only decide a token when one row leads all others by [`CERTIFY_WINDOW`].
//!
//! Error bound against the f64 reference, from the MSL 4.1 accuracy tables
//! (`precise::log` within 4 ulp) and f32 rounding of each step:
//! - the uniform `1 - w`, or `w` itself, is formed from the bits with a
//!   relative error of at most 2^-23;
//! - `E = -ln(1 - w)` uses a five-term series for `w < 2^-6` (truncation below
//!   2^-30 relative) and `-precise::log(1 - w)` otherwise, where `E >= 2^-6`
//!   turns the 2^-23 input error into at most 2^-17 relative;
//! - `-precise::log(E)` adds 4 ulp of a value below 46, about 1.6e-5, to that
//!   2^-17;
//! - `(logit - shift) * (1 / T) + noise` rounds each of three steps once
//!   (the multiply-add may fuse), so at most `3 * 2^-24 |score|` more.
//!
//! So `|f32 - f64| <= 4e-5 + 2^-22 |score|` ([`score_error_bound`]), well
//! under a quarter of the window for scores near a winner (|score| below
//! about 50). Subnormals flush to zero on the GPU; no step here makes one.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::cell::OnceCell;

use engine::sampling::race::RaceKey;
use mlx_kernels::{
    Grid, KernelError, KernelSpec, Launch, MetalKernel, Output, TemplateArg, ThreadGroup,
};
use mlx_rs::{Array, Dtype, StreamOrDevice};

use super::Qwen3ForwardError;

/// The f32 score lead below which the host settles a pick: no other row may
/// score within it of the winner. A quarter of it bounds the kernel's error
/// near a winner (see the module documentation).
pub const CERTIFY_WINDOW: f32 = 1.0 / 256.0;

/// The kernel's error bound at an f32 score `score`, in the reference's
/// units (see the module documentation).
#[must_use]
pub fn score_error_bound(score: f32) -> f64 {
    4e-5 + f64::from(score.abs()) / f64::from(1_u32 << 22)
}

const BODY: &str = r"
    uint column = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (column >= uint(V)) {
        return;
    }
    uint entry = row * uint(V) + column;
    uint token = TOKEN;
    uint c0 = token >> 1;
    uint c1 = params[row * 4 + 2];
    uint c2 = params[row * 4 + 3];
    uint c3 = 0x72616365u;
    uint k0 = params[row * 4];
    uint k1 = params[row * 4 + 1];
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        uint hi0 = mulhi(0xD2511F53u, c0);
        uint lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2);
        uint lo1 = 0xCD9E8D57u * c2;
        uint n0 = hi1 ^ c1 ^ k0;
        uint n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    uint high = (token & 1u) ? c2 : c0;
    uint low = (token & 1u) ? c3 : c1;
    const float two_to_minus_32 = 2.3283064365386963e-10f;
    // E = -ln(1 - w), w = (bits + 1/2) 2^-64. Small w: a series, since
    // 1 - w would round away the digits that set E. Otherwise 1 - w is
    // formed from the complemented bits, exact to f32 precision.
    float exponential;
    if (high < (1u << 26)) {
        float w = (float(high) + (float(low) + 0.5f) * two_to_minus_32) * two_to_minus_32;
        exponential = w * (1.0f + w * (0.5f + w * (1.0f / 3.0f + w * (0.25f + w * 0.2f))));
    } else {
        float rest = (float(~high) + (float(~low) + 0.5f) * two_to_minus_32) * two_to_minus_32;
        exponential = -metal::precise::log(rest);
    }
    float noise = -metal::precise::log(exponential);
    out[entry] = (logits[entry] - shift[row]) * inverse_temperature[0] + noise;
";

thread_local! {
    static FULL_ROW: OnceCell<Result<MetalKernel, KernelError>> = const { OnceCell::new() };
    static LISTED: OnceCell<Result<MetalKernel, KernelError>> = const { OnceCell::new() };
}

/// Per-row kernel parameters: seed halves, generated position and choice.
fn params(key: RaceKey, positions: &[u32]) -> Result<Array, Qwen3ForwardError> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the key is the seed's low and high 32-bit halves"
    )]
    let (low, high) = (key.seed as u32, (key.seed >> 32) as u32);
    let values: Vec<u32> = positions
        .iter()
        .flat_map(|&position| [low, high, position, key.choice])
        .collect();
    let rows = i32::try_from(positions.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    Ok(Array::from_slice(&values, &[rows, 4]))
}

/// Scores every entry of `[rows, width]` f32 `logits` whose token IDs are
/// the column indices (`ids` is `None`) or the matching uint32 `ids`
/// entries. `shift` is `[rows]` f32, subtracted from each row before scaling.
pub(super) fn race_scores(
    logits: &Array,
    ids: Option<&Array>,
    shift: &Array,
    temperature: f64,
    key: RaceKey,
    positions: &[u32],
) -> Result<Array, Qwen3ForwardError> {
    let &[rows, width] = logits.shape() else {
        return Err(Qwen3ForwardError::InvalidPickRule(
            "race scores need [rows, width] logits",
        ));
    };
    if usize::try_from(rows).ok() != Some(positions.len()) {
        return Err(Qwen3ForwardError::InvalidPickRule(
            "one generated position per row",
        ));
    }
    let rows_usize = usize::try_from(rows).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let width_usize = usize::try_from(width).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the GPU mirror runs in f32"
    )]
    let inverse_temperature = Array::from_slice(&[(1.0 / temperature) as f32], &[1]);
    let params = params(key, positions)?;
    let cell = if ids.is_some() { &LISTED } else { &FULL_ROW };
    cell.with(|cell| {
        let kernel = cell
            .get_or_init(|| match ids {
                Some(_) => MetalKernel::new(&KernelSpec::new(
                    "race_scores_listed",
                    &["logits", "ids", "params", "shift", "inverse_temperature"],
                    &["out"],
                    &BODY.replace("TOKEN", "ids[entry]"),
                )),
                None => MetalKernel::new(&KernelSpec::new(
                    "race_scores_full_row",
                    &["logits", "params", "shift", "inverse_temperature"],
                    &["out"],
                    &BODY.replace("TOKEN", "column"),
                )),
            })
            .as_ref()
            .map_err(|error| Qwen3ForwardError::Kernel(error.clone()))?;
        let mut inputs = vec![logits];
        inputs.extend(ids);
        inputs.extend([&params, shift, &inverse_temperature]);
        let count = rows_usize
            .checked_mul(width_usize)
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        let outputs = [Output::new(&[count], Dtype::Float32).map_err(Qwen3ForwardError::Kernel)?];
        let launch = Launch {
            outputs: &outputs,
            template: &[("V", TemplateArg::Int(width))],
            grid: Grid::new(width_usize, rows_usize, 1).map_err(Qwen3ForwardError::Kernel)?,
            threadgroup: ThreadGroup::new(width_usize.min(256), 1, 1)
                .map_err(Qwen3ForwardError::Kernel)?,
            init_value: None,
        };
        let scores = kernel
            .apply(&inputs, &launch)
            .map_err(Qwen3ForwardError::Kernel)?
            .pop()
            .ok_or(Qwen3ForwardError::InvalidPickRule(
                "the race kernel has one output",
            ))?;
        Ok(scores.reshape_device(&[rows, width], StreamOrDevice::gpu())?)
    })
}
