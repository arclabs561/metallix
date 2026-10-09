//! `FLUX.2`'s four-axis rotary position tables.
//!
//! Every token carries a `(t, h, w, l)` position. Each axis owns a contiguous
//! slice of the head dimension (32 of 128 lanes each for klein) with its own
//! frequencies, and pairs of adjacent lanes rotate together. `MLX`'s fused
//! rope takes one position per token, so the tables are built here once per
//! resolution and applied as `x * cos + rotate_pairs(x) * sin` (see
//! `Flux2PosEmbed` and `apply_rotary_emb` in diffusers).

use thiserror::Error;

/// Position ids for `text_len` text tokens: `(0, 0, 0, l)`.
///
/// # Panics
///
/// Panics if a generated position exceeds `u32::MAX` or vector capacity
/// exceeds the allocation limit.
#[must_use]
pub fn text_position_ids(text_len: usize) -> Vec<[u32; 4]> {
    (0..text_len)
        .map(|l| [0, 0, 0, u32::try_from(l).expect("text length fits u32")])
        .collect()
}

/// Position ids for an `height x width` latent grid, row-major:
/// `(0, h, w, 0)`.
///
/// # Panics
///
/// Panics if a generated coordinate exceeds `u32::MAX`, vector capacity
/// overflows, or `height * width` overflows with overflow checks enabled.
/// Callers must bound both dimensions and their product.
#[must_use]
pub fn image_position_ids(height: usize, width: usize) -> Vec<[u32; 4]> {
    let mut ids = Vec::with_capacity(height * width);
    for h in 0..height {
        for w in 0..width {
            ids.push([
                0,
                u32::try_from(h).expect("latent height fits u32"),
                u32::try_from(w).expect("latent width fits u32"),
                0,
            ]);
        }
    }
    ids
}

/// Invalid rotary dimensions, frequency base or table storage.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum RopeError {
    /// Axis widths must be even and their total must be positive.
    #[error("rotary axes must be even with a positive total width: {0:?}")]
    Axes([usize; 4]),
    /// The supplied frequency base is not finite and positive.
    #[error("rotary frequency base must be finite and positive: {0}")]
    Theta(f64),
    /// Axis sums or table storage exceed addressable memory.
    #[error("rotary table shape exceeds addressable memory")]
    ShapeOverflow,
    /// Storage for validated table dimensions could not be reserved.
    #[error("cannot reserve rotary table storage")]
    Allocation,
}

/// Cosine and sine tables, `[positions, head_dim]` row-major, float32.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTables {
    /// Cosines, two identical adjacent entries per rotary pair.
    cos: Vec<f32>,
    /// Sines in the same row-major layout as `cos`.
    sin: Vec<f32>,
    /// Number of token-position rows.
    positions: usize,
    /// Scalar lanes in each row.
    head_dim: usize,
}

impl RopeTables {
    /// Builds the tables for `ids` in the given order (text ids first, then
    /// image ids, as the transformer concatenates them).
    ///
    /// Frequencies and angles are computed in float64 and rounded to float32
    /// once, as the source does (`freqs_dtype=float64`, then `.float()`).
    /// Each axis width must be even and their total positive. Zero-width axes
    /// are allowed. `theta` must be finite and positive.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError::Axes`] or [`RopeError::Theta`] for invalid inputs,
    /// [`RopeError::ShapeOverflow`] for unrepresentable table dimensions, and
    /// [`RopeError::Allocation`] if storage reservation fails.
    pub fn new(ids: &[[u32; 4]], axes: [usize; 4], theta: f64) -> Result<Self, RopeError> {
        let head_dim = axes.iter().try_fold(0_usize, |total, &dim| {
            total.checked_add(dim).ok_or(RopeError::ShapeOverflow)
        })?;
        if head_dim == 0 || axes.iter().any(|dim| dim % 2 != 0) {
            return Err(RopeError::Axes(axes));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(RopeError::Theta(theta));
        }
        let table_len = ids
            .len()
            .checked_mul(head_dim)
            .ok_or(RopeError::ShapeOverflow)?;
        let bytes = table_len
            .checked_mul(size_of::<f32>())
            .ok_or(RopeError::ShapeOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(RopeError::ShapeOverflow);
        }
        let mut frequencies = Vec::new();
        frequencies
            .try_reserve_exact(head_dim / 2)
            .map_err(|_| RopeError::Allocation)?;
        for &dim in &axes {
            #[allow(clippy::cast_precision_loss)]
            let width = dim as f64;
            for j in (0..dim).step_by(2) {
                #[allow(clippy::cast_precision_loss)]
                let exponent = j as f64 / width;
                frequencies.push(1.0 / theta.powf(exponent));
            }
        }
        let mut axis_of = Vec::new();
        axis_of
            .try_reserve_exact(head_dim / 2)
            .map_err(|_| RopeError::Allocation)?;
        axis_of.extend(
            axes.iter()
                .enumerate()
                .flat_map(|(axis, &dim)| std::iter::repeat_n(axis, dim / 2)),
        );

        let mut cos = Vec::new();
        let mut sin = Vec::new();
        cos.try_reserve_exact(table_len)
            .map_err(|_| RopeError::Allocation)?;
        sin.try_reserve_exact(table_len)
            .map_err(|_| RopeError::Allocation)?;
        for position in ids {
            for (pair, &frequency) in frequencies.iter().enumerate() {
                let angle = f64::from(position[axis_of[pair]]) * frequency;
                #[allow(clippy::cast_possible_truncation)]
                let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                // repeat_interleave(2): both lanes of a pair share the angle.
                cos.extend([c, c]);
                sin.extend([s, s]);
            }
        }
        Ok(Self {
            cos,
            sin,
            positions: ids.len(),
            head_dim,
        })
    }

    /// Cosines in contiguous `[positions, head_dim]` row-major order.
    #[must_use]
    pub fn cos(&self) -> &[f32] {
        &self.cos
    }

    /// Sines in the same layout as [`Self::cos`].
    #[must_use]
    pub fn sin(&self) -> &[f32] {
        &self.sin
    }

    /// Number of token-position rows.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Number of scalar lanes per row.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_ids_are_row_major_over_the_latent_grid() {
        let ids = image_position_ids(2, 3);
        assert_eq!(ids[0], [0, 0, 0, 0]);
        assert_eq!(ids[2], [0, 0, 2, 0]);
        assert_eq!(ids[3], [0, 1, 0, 0]);
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "exact values are the point of this test")]
    fn each_axis_rotates_only_its_own_lanes() {
        // A token at h=5 and nothing else: only lanes 32..64 (axis 1) turn.
        let tables = RopeTables::new(&[[0, 5, 0, 0]], [32, 32, 32, 32], 2000.0).unwrap();
        for lane in 0..128 {
            let turned = tables.sin[lane] != 0.0;
            assert_eq!(turned, (32..64).contains(&lane), "lane {lane}");
        }
        // The pair's first frequency is 1, so lanes 32 and 33 share angle 5.
        // cos(5) = 0.28366218546322626446..., rounded to IEEE binary32.
        assert_eq!(tables.cos[32], f32::from_bits(0x3e91_3c2c));
        assert_eq!(tables.cos[33], tables.cos[32]);
    }
    #[test]
    fn malformed_axes_are_rejected_before_indexing_or_allocation() {
        assert_eq!(
            RopeTables::new(&[[0; 4]], [1, 0, 0, 0], 2000.0),
            Err(RopeError::Axes([1, 0, 0, 0]))
        );
        assert_eq!(
            RopeTables::new(&[[0; 4]], [0; 4], 2000.0),
            Err(RopeError::Axes([0; 4]))
        );
        assert_eq!(
            RopeTables::new(&[[0; 4]], [usize::MAX - 1, 2, 0, 0], 2000.0),
            Err(RopeError::ShapeOverflow)
        );
        assert_eq!(
            RopeTables::new(&[[0; 4]; 2], [usize::MAX - 1, 0, 0, 0], 2000.0),
            Err(RopeError::ShapeOverflow)
        );
    }

    #[test]
    fn invalid_frequency_bases_do_not_produce_nonfinite_tables() {
        for theta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                RopeTables::new(&[[0, 1, 2, 3]], [2; 4], theta),
                Err(RopeError::Theta(_))
            ));
        }
    }
}
