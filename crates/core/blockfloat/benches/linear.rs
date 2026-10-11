//! Scalar FP4 and BF16 linears at DeepSeek-V4.1 Flash projection shapes.
//!
//! `rows` of 1 (one decode token; routed experts always see one row per call)
//! and 17 (the 17-token teacher-forced prefill the end-to-end profile uses):
//! - `fp4_expert_w1`: routed-expert `w1` through `fp4_linear_runtime_f32_owned`
//!   (the `MoE` path), G32, 5120 -> 2304 (`w3` has the same shape).
//! - `fp4_expert_w2`: the same for `w2`, 2304 -> 5120.
//! - `fp4_expert_w1_public`: `w1` through `fp4_linear_runtime_f32`.
//! - `bf16_wo_a`: one of the 8 `wo_a` groups, `bf16_linear_reference`,
//!   4096 -> `o_lora_rank` 1024.
//!
//! Inputs are deterministic pseudo-random finite codes with scales that keep
//! every output finite. Input generation is outside the timing.

use std::hint::black_box;

use blockfloat::{
    ActivationGroup, bf16_linear_reference, fp4_linear_runtime_f32, fp4_linear_runtime_f32_owned,
};

/// Bytes from a fixed xorshift stream.
fn bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

struct Fp4 {
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u8>,
    rows: usize,
    reduction: usize,
    outputs: usize,
}

fn fp4(rows: usize, reduction: usize, outputs: usize) -> Fp4 {
    let blocks = reduction / 32;
    Fp4 {
        // Finite E4M3FN: never `0x7f`/`0xff`.
        activation_codes: bytes(rows * reduction, 0x9e37_79b9_7f4a_7c15)
            .into_iter()
            .map(|code| {
                if code & 0x7f == 0x7f {
                    code & 0xf0
                } else {
                    code
                }
            })
            .collect(),
        activation_scales: bytes(rows * blocks, 3)
            .into_iter()
            .map(|code| 120 + code % 8)
            .collect(),
        weight_codes: bytes(outputs * reduction / 2, 0x2545_f491_4f6c_dd1d),
        weight_scales: bytes(outputs * blocks, 5)
            .into_iter()
            .map(|code| 110 + code % 8)
            .collect(),
        rows,
        reduction,
        outputs,
    }
}

fn run_fp4_owned(bencher: divan::Bencher, problem: &Fp4) {
    bencher.bench_local(|| {
        black_box(
            fp4_linear_runtime_f32_owned(
                black_box(&problem.activation_codes),
                &problem.activation_scales,
                black_box(&problem.weight_codes),
                &problem.weight_scales,
                problem.rows,
                problem.reduction,
                problem.outputs,
                ActivationGroup::Elements32,
            )
            .expect("finite FP4 linear"),
        )
    });
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn fp4_expert_w1(bencher: divan::Bencher, rows: usize) {
    run_fp4_owned(bencher, &fp4(rows, 5120, 2304));
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn fp4_expert_w2(bencher: divan::Bencher, rows: usize) {
    run_fp4_owned(bencher, &fp4(rows, 2304, 5120));
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn fp4_expert_w1_public(bencher: divan::Bencher, rows: usize) {
    let problem = fp4(rows, 5120, 2304);
    let mut output = vec![0.0_f32; rows * problem.outputs];
    bencher.bench_local(|| {
        fp4_linear_runtime_f32(
            black_box(&problem.activation_codes),
            &problem.activation_scales,
            black_box(&problem.weight_codes),
            &problem.weight_scales,
            problem.rows,
            problem.reduction,
            problem.outputs,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("finite FP4 linear");
        black_box(&output);
    });
}

/// Finite BF16 values in roughly [-2, 2]: sign, exponent 126..128, mantissa.
fn bf16(count: usize, seed: u64) -> Vec<u16> {
    bytes(2 * count, seed)
        .chunks_exact(2)
        .map(|pair| {
            let sign = u16::from(pair[0] & 0x80) << 8;
            let exponent = (126 + u16::from(pair[0] % 3)) << 7;
            sign | exponent | u16::from(pair[1] & 0x7f)
        })
        .collect()
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn bf16_wo_a(bencher: divan::Bencher, rows: usize) {
    let (reduction, outputs) = (4096, 1024);
    let activations = bf16(rows * reduction, 7);
    let weights = bf16(outputs * reduction, 11);
    let mut output = vec![0_u16; rows * outputs];
    bencher.bench_local(|| {
        bf16_linear_reference(
            black_box(&activations),
            black_box(&weights),
            rows,
            reduction,
            outputs,
            &mut output,
        )
        .expect("finite BF16 linear");
        black_box(&output);
    });
}

fn main() {
    divan::main();
}
