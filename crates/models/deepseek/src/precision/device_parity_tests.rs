//! The model's device-routed FP8 linears and FP4 experts against their scalar references.
//!
//! Declared tolerances, fixed before the runs they check:
//! - FP8 linear: the device sums each block's 32 products as a tree, the
//!   reference serially; both then accumulate blocks in the same order. With
//!   `S` the same linear over absolute values (sign bits cleared) and `G =
//!   reduction / 32` blocks, `|got - want| <= 2 * (32 + 3 * G) * eps * S`, a
//!   first-order bound on both summations' rounding error, so cancellation
//!   cannot hide behind a small `|want|`.
//! - FP4 expert, synthetic weights: BF16 output equal to the scalar expert's.
//! - FP4 expert, real weights: the FP32 order difference can flip a BF16
//!   rounding of the gate or up value and so an FP8 code of the down
//!   projection's input; the output must have cosine `>= 0.99999` and relative
//!   L2 error `<= 1e-3` against the scalar expert.

use std::sync::Arc;

use super::{
    ActivationGroup, DeviceCounts, DeviceLinears, Fp8ForwardError, Fp8MetalError, bf16_to_f32,
    fp8_linear_f32, fp8_linear_runtime_f32, metal_fp4_expert, metal_fp8_linear,
    quantize_bf16_activations_e4m3fn,
};
use crate::moe::{Fp4ExpertWeights, project_fp4_expert_scalar};

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    u32::try_from(*seed >> 33).expect("31-bit value")
}

fn bytes(seed: &mut u64, count: usize, map: impl Fn(u8) -> u8) -> Vec<u8> {
    (0..count)
        .map(|_| map(u8::try_from(lcg(seed) & 0xff).expect("byte")))
        .collect()
}

/// BF16 activations of roughly unit scale.
fn activations(rows: usize, width: usize) -> Vec<u16> {
    (0..rows * width)
        .map(|index| {
            let value = (f32::from(u16::try_from(index % 97).expect("small")) - 48.0) / 19.0;
            u16::try_from(value.to_bits() >> 16).expect("bf16")
        })
        .collect()
}

fn quantize(input: &[u16], rows: usize, width: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0; rows * width];
    let mut scales = vec![0; rows * width / 32];
    quantize_bf16_activations_e4m3fn(
        input,
        rows,
        width,
        ActivationGroup::Elements32,
        &mut codes,
        &mut scales,
    )
    .expect("activation quantization");
    (codes, scales)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn routed_fp8(
    (codes, scales): (&[u8], &[u8]),
    (weight_codes, weight_scales): (&[u8], &[u8]),
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> Vec<f32> {
    let mut routed = vec![0.0; rows * outputs];
    fp8_linear_f32(
        codes,
        scales,
        weight_codes,
        weight_scales,
        rows,
        reduction,
        outputs,
        ActivationGroup::Elements32,
        &mut routed,
    )
    .expect("routed linear");
    routed
}

/// Checks one FP8 linear: outside a device scope the routed call is the
/// scalar reference; inside one it is the device result, per call or over
/// resident weights alike, within the declared tolerance of the reference.
/// Returns the largest observed `|got - want| / (eps * S)`.
fn check_fp8(
    name: &str,
    weight_codes: &[u8],
    weight_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> f64 {
    let (codes, scales) = quantize(&activations(rows, reduction), rows, reduction);
    let scalar = |codes: &[u8], weights: &[u8]| {
        let mut expected = vec![0.0; rows * outputs];
        fp8_linear_runtime_f32(
            codes,
            &scales,
            weights,
            weight_scales,
            rows,
            reduction,
            outputs,
            ActivationGroup::Elements32,
            &mut expected,
        )
        .expect("scalar reference");
        expected
    };
    let expected = scalar(&codes, weight_codes);
    let magnitude = |bytes: &[u8]| bytes.iter().map(|&code| code & 0x7f).collect::<Vec<_>>();
    let abs_sum = scalar(&magnitude(&codes), &magnitude(weight_codes));
    let activations = (&codes[..], &scales[..]);
    let weights = (weight_codes, weight_scales);
    assert_eq!(
        bits(&routed_fp8(activations, weights, rows, reduction, outputs)),
        bits(&expected),
        "{name} rows {rows}: outside a scope the routed call is the scalar reference"
    );
    let device = metal_fp8_linear(
        &codes,
        &scales,
        weight_codes,
        weight_scales,
        rows,
        reduction,
        outputs,
    )
    .expect("device path");
    // The resident entry must be the very buffers the call passes.
    let (codes_arc, scales_arc): (Arc<[u8]>, Arc<[u8]>) =
        (weight_codes.into(), weight_scales.into());
    for (label, linears, resident) in [
        ("per call", Arc::new(DeviceLinears::default()), false),
        (
            "resident",
            Arc::new(DeviceLinears::new([(
                Arc::clone(&codes_arc),
                Arc::clone(&scales_arc),
            )])),
            true,
        ),
    ] {
        let _scope = linears.enter();
        // Twice: the resident path uploads on the first call and reuses after.
        for _ in 0..2 {
            let routed = routed_fp8(
                activations,
                (&codes_arc[..], &scales_arc[..]),
                rows,
                reduction,
                outputs,
            );
            assert_eq!(
                bits(&routed),
                bits(&device),
                "{name} rows {rows} {label}: routed call must take the device path"
            );
        }
        let counts = linears.counts();
        assert_eq!(
            (
                counts.resident_fp8,
                counts.uploaded_fp8,
                counts.scalar_fallbacks
            ),
            if resident { (2, 0, 0) } else { (0, 2, 0) },
            "{name} {label}: {counts:?}"
        );
        assert_eq!(counts.resident_bytes > 0, resident);
    }
    let groups = reduction / 32;
    let factor =
        2.0 * f64::from(u32::try_from(32 + 3 * groups).expect("small")) * f64::from(f32::EPSILON);
    let mut worst = 0.0_f64;
    for (index, ((&got, &want), &sum)) in device.iter().zip(&expected).zip(&abs_sum).enumerate() {
        let error = (f64::from(got) - f64::from(want)).abs();
        let bound = factor * f64::from(sum);
        assert!(
            error <= bound,
            "{name} rows {rows} output {index}: {got} vs {want}, bound {bound}"
        );
        if sum > 0.0 {
            worst = worst.max(error / (f64::from(f32::EPSILON) * f64::from(sum)));
        }
    }
    worst
}

#[test]
fn fp8_linear_routes_to_the_device_and_matches_the_scalar_reference() {
    let _gpu = crate::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (reduction, outputs) = (256, 96);
    let mut seed = 7_u64;
    let codes = bytes(&mut seed, outputs * reduction, |byte| {
        if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
    });
    let scales = bytes(&mut seed, outputs.div_ceil(32) * (reduction / 32), |byte| {
        120 + byte % 14
    });
    for rows in [1, 3] {
        check_fp8("synthetic", &codes, &scales, rows, reduction, outputs);
    }
}

#[test]
fn a_device_rejection_is_returned_not_rerun_on_the_scalar_path() {
    let _gpu = crate::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (codes, scales) = quantize(&activations(1, 32), 1, 32);
    // A NaN weight code is refused by the device upload and by the scalar reference.
    let weights = [0x7f_u8; 32];
    let run = || {
        let mut output = [0.0; 1];
        fp8_linear_f32(
            &codes,
            &scales,
            &weights,
            &[127],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
    };
    assert!(matches!(run(), Err(Fp8ForwardError::Scalar(_))));
    let linears = Arc::new(DeviceLinears::default());
    let routed = {
        let _scope = linears.enter();
        run()
    };
    assert_eq!(
        routed,
        Err(Fp8ForwardError::Device(Fp8MetalError::NonFinite {
            field: "weight codes",
            index: 0
        }))
    );
    assert_eq!(linears.counts(), DeviceCounts::default());
}

/// Cosine and relative L2 error of `got` against `want`, over BF16 values.
fn agreement(got: &[u16], want: &[u16]) -> (f64, f64) {
    let (mut dot, mut gg, mut ww, mut dd) = (0.0, 0.0, 0.0, 0.0);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(bf16_to_f32(g)), f64::from(bf16_to_f32(w)));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        dd += (g - w) * (g - w);
    }
    (dot / (gg.sqrt() * ww.sqrt()), (dd / ww).sqrt())
}

/// Checks one routed FP4 expert: scalar outside a scope, the device result
/// inside one, and that result within the declared tolerance of the scalar
/// expert (`exact` for synthetic weights). Returns (cosine, relative L2, BF16
/// elements that differ).
fn check_fp4_expert(
    name: &str,
    dim: usize,
    inter: usize,
    weights: [&[u8]; 6],
    (limit, route): (f32, Option<f32>),
    exact: bool,
) -> (f64, f64, usize) {
    let [w1, s1, w2, s2, w3, s3] = weights;
    let input = activations(1, dim);
    let expert = Fp4ExpertWeights::new(dim, inter, w1, s1, w2, s2, w3, s3).expect("scalar expert");
    let expected = project_fp4_expert_scalar(&input, &expert, limit, route).expect("scalar");
    assert_eq!(
        expert
            .forward_token(&input, limit, route)
            .expect("scalar routed"),
        expected,
        "{name}: outside a scope the routed expert is the scalar expert"
    );
    let (codes, scales) = quantize(&input, 1, dim);
    let device = metal_fp4_expert(
        (&codes, &scales),
        dim,
        inter,
        (w1, s1),
        (w2, s2),
        (w3, s3),
        limit,
        route,
    )
    .expect("device path")
    .expect("an encodable limit");
    let linears = Arc::new(DeviceLinears::default());
    {
        let _scope = linears.enter();
        assert_eq!(
            expert
                .forward_token(&input, limit, route)
                .expect("routed expert"),
            device,
            "{name}: routed expert must equal the device result"
        );
    }
    assert_eq!(
        linears.counts(),
        DeviceCounts {
            fp4_experts: 1,
            ..DeviceCounts::default()
        }
    );
    let differing = device.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let (cosine, relative) = agreement(&device, &expected);
    if exact {
        assert_eq!(device, expected, "{name}: BF16 expert output");
    } else {
        assert!(
            cosine >= 0.99999 && relative <= 1e-3,
            "{name}: cosine {cosine}, relative L2 {relative}, {differing} differ"
        );
    }
    (cosine, relative, differing)
}

#[test]
fn fp4_expert_routes_to_the_device_and_matches_the_scalar_expert() {
    let _gpu = crate::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (dim, inter) = (128, 96);
    let mut seed = 11_u64;
    let w1 = bytes(&mut seed, inter * dim / 2, |b| b);
    let w2 = bytes(&mut seed, dim * inter / 2, |b| b);
    let w3 = bytes(&mut seed, inter * dim / 2, |b| b);
    let s1 = bytes(&mut seed, inter * dim / 32, |b| 120 + b % 10);
    let s2 = bytes(&mut seed, dim * inter / 32, |b| 120 + b % 10);
    let s3 = bytes(&mut seed, inter * dim / 32, |b| 120 + b % 10);
    let weights = [&w1[..], &s1, &w2, &s2, &w3, &s3];
    check_fp4_expert("weighted", dim, inter, weights, (2.0, Some(0.75)), true);
    check_fp4_expert(
        "unweighted, no clamp",
        dim,
        inter,
        weights,
        (0.0, None),
        true,
    );
    // A non-integer limit has no device encoding: the scalar expert runs, counted.
    let (codes, scales) = quantize(&activations(1, dim), 1, dim);
    assert_eq!(
        metal_fp4_expert(
            (&codes, &scales),
            dim,
            inter,
            (&w1, &s1),
            (&w2, &s2),
            (&w3, &s3),
            2.5,
            None
        ),
        Ok(None)
    );
    let input = activations(1, dim);
    let expert = Fp4ExpertWeights::new(dim, inter, &w1, &s1, &w2, &s2, &w3, &s3).expect("expert");
    let linears = Arc::new(DeviceLinears::default());
    let routed = {
        let _scope = linears.enter();
        expert.forward_token(&input, 2.5, None).expect("fallback")
    };
    assert_eq!(
        routed,
        project_fp4_expert_scalar(&input, &expert, 2.5, None).expect("scalar")
    );
    assert_eq!(linears.counts().scalar_fallbacks, 1);
}

fn real(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.agents/receipts/route-trace/weights")
        .join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

#[test]
#[ignore = "needs real V4.1 tensors under .agents/receipts/route-trace/weights"]
fn real_shapes_match_the_scalar_references() {
    let _gpu = crate::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (name, outputs, reduction) in [
        ("attn.wq_a", 1_280, 5_120),
        ("attn.wq_b", 32_768, 1_280),
        ("attn.wkv", 512, 5_120),
        ("attn.wo_b", 5_120, 8_192),
        ("ffn.shared_experts.w1", 2_304, 5_120),
        ("ffn.shared_experts.w2", 5_120, 2_304),
        ("ffn.shared_experts.w3", 2_304, 5_120),
    ] {
        let codes = real(&format!("layers.1.{name}.weight"));
        let scales = real(&format!("layers.1.{name}.scale"));
        for rows in [1, 17] {
            let worst = check_fp8(name, &codes, &scales, rows, reduction, outputs);
            let bound = 2 * (32 + 3 * reduction / 32);
            eprintln!("{name} rows {rows}: max |diff| / (eps S) {worst:.2}, declared {bound}");
        }
    }
    for expert in [2, 4, 16, 20, 28, 35] {
        let tensor = |projection: &str, kind: &str| {
            real(&format!(
                "layers.0.ffn.experts.{expert}.{projection}.{kind}"
            ))
        };
        let parts = [
            tensor("w1", "weight"),
            tensor("w1", "scale"),
            tensor("w2", "weight"),
            tensor("w2", "scale"),
            tensor("w3", "weight"),
            tensor("w3", "scale"),
        ];
        let weights = [
            &parts[0][..],
            &parts[1],
            &parts[2],
            &parts[3],
            &parts[4],
            &parts[5],
        ];
        let (cosine, relative, differing) = check_fp4_expert(
            &format!("expert {expert}"),
            5_120,
            2_304,
            weights,
            (10.0, Some(0.25)),
            false,
        );
        eprintln!(
            "expert {expert}: cosine {cosine:.9}, relative L2 {relative:.2e}, {differing}/5120 BF16 differ"
        );
    }
}
