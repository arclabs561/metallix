//! Image-space comparison for pipeline parity gates.
//!
//! Both metrics take interleaved 8-bit RGB of the same size, the format the
//! pipelines emit, so a gate compares what a user would receive.
//!
//! SSIM follows Wang et al. 2004 with the usual constants: an 11x11 Gaussian
//! window (sigma 1.5), `K1 = 0.01`, `K2 = 0.03`, `L = 255`, computed per
//! channel over valid windows only (no padding) and averaged.

/// Peak signal-to-noise ratio in dB over all channels; infinite when equal.
///
/// # Panics
///
/// If the images differ in length.
#[must_use]
pub fn psnr_rgb8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "images differ in size");
    let squared: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    if squared == 0.0 {
        return f64::INFINITY;
    }
    let mse = squared / a.iter().fold(0.0, |count, _| count + 1.0);
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Mean structural similarity of two `width x height` RGB8 images.
///
/// # Panics
///
/// If either image is not `width * height * 3` bytes, or a side is not
/// at least 11 pixels.
#[must_use]
pub fn ssim_rgb8(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    const RADIUS: usize = 5;
    assert_eq!(a.len(), width * height * 3, "first image size");
    assert_eq!(b.len(), a.len(), "images differ in size");
    assert!(
        width > 2 * RADIUS && height > 2 * RADIUS,
        "image smaller than the window"
    );
    let kernel = gaussian(RADIUS, 1.5);
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let mut total = 0.0;
    let mut count = 0.0;
    for channel in 0..3 {
        let x: Vec<f64> = a
            .iter()
            .skip(channel)
            .step_by(3)
            .map(|&v| f64::from(v))
            .collect();
        let y: Vec<f64> = b
            .iter()
            .skip(channel)
            .step_by(3)
            .map(|&v| f64::from(v))
            .collect();
        let xx: Vec<f64> = x.iter().map(|v| v * v).collect();
        let yy: Vec<f64> = y.iter().map(|v| v * v).collect();
        let xy: Vec<f64> = x.iter().zip(&y).map(|(p, q)| p * q).collect();
        let [mx, my, sxx, syy, sxy] =
            [&x, &y, &xx, &yy, &xy].map(|plane| blur(plane, width, height, &kernel));
        for i in 0..mx.len() {
            let (mu_x, mu_y) = (mx[i], my[i]);
            let var_x = sxx[i] - mu_x * mu_x;
            let var_y = syy[i] - mu_y * mu_y;
            let cov = sxy[i] - mu_x * mu_y;
            total += ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2))
                / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2));
            count += 1.0;
        }
    }
    total / count
}

fn gaussian(radius: usize, sigma: f64) -> Vec<f64> {
    let weights: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let d = f64::from(u32::try_from(i).expect("Gaussian window index"))
                - f64::from(u32::try_from(radius).expect("Gaussian window radius"));
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / sum).collect()
}

/// Separable Gaussian filter over the valid region:
/// `(width - 2r) x (height - 2r)` outputs.
fn blur(plane: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let span = kernel.len();
    let out_w = width - span + 1;
    let out_h = height - span + 1;
    let mut rows = vec![0.0; out_w * height];
    for r in 0..height {
        for c in 0..out_w {
            rows[r * out_w + c] = (0..span)
                .map(|k| kernel[k] * plane[r * width + c + k])
                .sum();
        }
    }
    let mut out = vec![0.0; out_w * out_h];
    for r in 0..out_h {
        for c in 0..out_w {
            out[r * out_w + c] = (0..span)
                .map(|k| kernel[k] * rows[(r + k) * out_w + c])
                .sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{psnr_rgb8, ssim_rgb8};

    fn gradient(width: usize, height: usize) -> Vec<u8> {
        let mut image = Vec::with_capacity(width * height * 3);
        for r in 0..height {
            for c in 0..width {
                let v = u8::try_from((r * 7 + c * 3) % 256).unwrap();
                image.extend([v, v.wrapping_add(40), 255 - v]);
            }
        }
        image
    }

    #[test]
    fn identical_images_score_perfectly() {
        let image = gradient(32, 24);
        assert_eq!(psnr_rgb8(&image, &image), f64::INFINITY);
        assert!((ssim_rgb8(&image, &image, 32, 24) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_uniform_offset_has_the_textbook_psnr() {
        // Every channel off by 1: MSE 1, PSNR = 20 log10(255) = 48.13 dB.
        let a = vec![100u8; 16 * 16 * 3];
        let b = vec![101u8; 16 * 16 * 3];
        assert!((psnr_rgb8(&a, &b) - 48.130_803_6).abs() < 1e-6);
    }

    #[test]
    fn ssim_drops_for_noise_but_barely_for_a_small_offset() {
        let image = gradient(48, 48);
        let shifted: Vec<u8> = image.iter().map(|v| v.saturating_add(2)).collect();
        let scrambled: Vec<u8> = image
            .iter()
            .enumerate()
            .map(|(i, v)| v.wrapping_add(u8::try_from((i * 7919) % 97).unwrap()))
            .collect();
        let near = ssim_rgb8(&image, &shifted, 48, 48);
        let far = ssim_rgb8(&image, &scrambled, 48, 48);
        assert!(near > 0.99, "offset SSIM {near}");
        assert!(far < 0.5, "noise SSIM {far}");
    }
}
