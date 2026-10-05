//! Scalar `fp8_linear_runtime_f32` at DeepSeek-V4.1 Flash projection shapes.
//!
//! G32 activations, `rows` of 1 (one decode token) and 17 (the 17-token
//! teacher-forced prefill the end-to-end profile uses):
//! - `shared_expert_w1`: 5120 -> 2304 (`w3` has the same shape).
//! - `shared_expert_w2`: 2304 -> 5120.
//! - `wq_b`: `q_lora_rank` 1280 -> 64 heads x 512 = 32768.
//!
//! Codes are deterministic pseudo-random finite E4M3FN values; scales sit at
//! or below 1 so no output overflows. Input generation is outside the timing.

use std::hint::black_box;

use blockfloat::{ActivationGroup, fp8_linear_runtime_f32};

const GROUP: usize = 32;

struct Problem {
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u8>,
    rows: usize,
    reduction: usize,
    outputs: usize,
}

/// Finite E4M3FN codes (never `0x7f`/`0xff`) from a fixed xorshift stream.
fn codes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let code = state.to_le_bytes()[0];
            if code & 0x7f == 0x7f {
                code & 0xf0
            } else {
                code
            }
        })
        .collect()
}

fn problem(rows: usize, reduction: usize, outputs: usize) -> Problem {
    let groups = reduction / GROUP;
    Problem {
        activation_codes: codes(rows * reduction, 0x9e37_79b9_7f4a_7c15),
        activation_scales: codes(rows * groups, 3)
            .into_iter()
            .map(|code| 120 + code % 8)
            .collect(),
        weight_codes: codes(outputs * reduction, 0x2545_f491_4f6c_dd1d),
        weight_scales: codes(outputs.div_ceil(GROUP) * groups, 5)
            .into_iter()
            .map(|code| 110 + code % 8)
            .collect(),
        rows,
        reduction,
        outputs,
    }
}

fn run(bencher: divan::Bencher, problem: &Problem) {
    let mut output = vec![0.0_f32; problem.rows * problem.outputs];
    bencher.bench_local(|| {
        fp8_linear_runtime_f32(
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
        .expect("finite FP8 linear");
        black_box(&output);
    });
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn shared_expert_w1(bencher: divan::Bencher, rows: usize) {
    run(bencher, &problem(rows, 5120, 2304));
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn shared_expert_w2(bencher: divan::Bencher, rows: usize) {
    run(bencher, &problem(rows, 2304, 5120));
}

#[divan::bench(args = [1, 17], sample_count = 10)]
fn wq_b(bencher: divan::Bencher, rows: usize) {
    run(bencher, &problem(rows, 1280, 32768));
}

fn main() {
    divan::main();
}
