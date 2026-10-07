//! The output image size and its latent token grid.

use thiserror::Error;

/// Pixels per latent token: the VAE's 8x downsampling times the 2x2 patch.
pub const PIXELS_PER_TOKEN: u32 = 16;
const MIN_SIDE: u32 = 256;
const MAX_SIDE: u32 = 2048;

/// A validated output size: each side a multiple of 16 in `[256, 2048]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    width: u32,
    height: u32,
}

/// An output side violates the supported pixel bounds or token alignment.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ImageSizeError {
    /// A side is not divisible by the pixel width of a latent token.
    #[error("{side} {value} is not a multiple of {PIXELS_PER_TOKEN}")]
    NotAligned {
        /// The input dimension, `width` or `height`.
        side: &'static str,
        /// Supplied side length in pixels.
        value: u32,
    },
    /// A side falls outside the supported pixel range.
    #[error("{side} {value} is outside [{MIN_SIDE}, {MAX_SIDE}]")]
    OutOfRange {
        /// The input dimension, `width` or `height`.
        side: &'static str,
        /// Supplied side length in pixels.
        value: u32,
    },
}

impl ImageSize {
    /// Validates pixel dimensions without rounding or cropping.
    ///
    /// # Errors
    ///
    /// Returns [`ImageSizeError::OutOfRange`] unless each side is in
    /// `[256, 2048]`, or [`ImageSizeError::NotAligned`] unless divisible by 16.
    pub fn new(width: u32, height: u32) -> Result<Self, ImageSizeError> {
        for (side, value) in [("width", width), ("height", height)] {
            if !(MIN_SIDE..=MAX_SIDE).contains(&value) {
                return Err(ImageSizeError::OutOfRange { side, value });
            }
            if value % PIXELS_PER_TOKEN != 0 {
                return Err(ImageSizeError::NotAligned { side, value });
            }
        }
        Ok(Self { width, height })
    }

    /// Output width in pixels.
    #[must_use]
    pub fn width(self) -> u32 {
        self.width
    }

    /// Output height in pixels.
    #[must_use]
    pub fn height(self) -> u32 {
        self.height
    }

    /// The latent token grid, `(rows, columns)`.
    #[must_use]
    pub fn grid(self) -> (u32, u32) {
        (
            self.height / PIXELS_PER_TOKEN,
            self.width / PIXELS_PER_TOKEN,
        )
    }

    /// Image tokens the transformer sees.
    #[must_use]
    pub fn tokens(self) -> usize {
        let (rows, columns) = self.grid();
        rows as usize * columns as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sides_must_be_aligned_to_latent_tokens_and_in_range() {
        let size = ImageSize::new(1024, 768).unwrap();
        assert_eq!(size.grid(), (48, 64));
        assert_eq!(size.tokens(), 3072);
        assert_eq!(
            ImageSize::new(1000, 1024),
            Err(ImageSizeError::NotAligned {
                side: "width",
                value: 1000
            })
        );
        assert_eq!(
            ImageSize::new(1024, 4096),
            Err(ImageSizeError::OutOfRange {
                side: "height",
                value: 4096
            })
        );
    }
}
