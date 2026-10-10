//! The few-row 4-bit kernel against MLX's `quantized_matmul`.
//!
//! Gate (declared before the kernel was built; see the module docs):
//! - (a) FP32 activations, scales and biases: |kernel - MLX| <= 1e-2;
//! - (b) BF16: the kernel's error against that FP32 result is at most twice
//!   MLX's own BF16 error plus 1e-2. A bit-exact BF16 bar cannot hold: any
//!   change of summation order moves the last bit of outputs whose BF16
//!   spacing is above 1e-2.
//!
//! One row is never routed, so it stays bit-identical to MLX.

use mlx_rs::{Array, Dtype, ops, random};

use super::super::small_m::{MAX_ROWS, applies, matmul, matmul_with_source, set_small_m_qmv};
use crate::GPU_TEST_LOCK;
use crate::quantization::{Qwen3AffineBits, Qwen3AffineGroupSize, Qwen3AffineQuantization};

/// (K, N): N not a multiple of 8 exercises the clamped last tile.
const SHAPES: [(i32, i32); 2] = [(256, 200), (1024, 512)];
const GROUPS: [i32; 3] = [32, 64, 128];

struct Weights {
    packed: Array,
    scales: Array,
    biases: Array,
}

fn weights(k: i32, n: i32, group: i32, seed: u64) -> Weights {
    let dense = random::normal::<f32>(&[n, k][..], None, None, &random::key(seed).expect("key"))
        .expect("normal")
        .as_dtype(Dtype::Bfloat16)
        .expect("bf16");
    let (packed, scales, biases) = ops::quantize(&dense, group, 4).expect("quantize");
    Weights {
        packed,
        scales,
        biases,
    }
}

fn mlx(x: &Array, w: &Weights, scales: &Array, biases: &Array, group: i32) -> Array {
    ops::quantized_matmul(x, &w.packed, scales, biases, true, group, 4).expect("quantized_matmul")
}

fn max_abs(left: &Array, right: &Array) -> f32 {
    let diff = left
        .as_dtype(Dtype::Float32)
        .expect("f32")
        .subtract(right.as_dtype(Dtype::Float32).expect("f32"))
        .expect("subtract")
        .abs()
        .expect("abs")
        .max(None)
        .expect("max");
    diff.eval().expect("eval");
    diff.item::<f32>()
}

type Kernel<'a> = &'a dyn Fn(&Array, &Weights, &Array, &Array, i32) -> Array;

/// Checks (a) and (b) for every shape, group size and row count; returns the
/// number of failing cases.
fn failures(kernel: Kernel<'_>) -> usize {
    let mut failed = 0;
    for (case, &(k, n)) in SHAPES.iter().enumerate() {
        for &group in &GROUPS {
            let w = weights(k, n, group, 7 + case as u64);
            let (s32, b32) = (
                w.scales.as_dtype(Dtype::Float32).expect("f32"),
                w.biases.as_dtype(Dtype::Float32).expect("f32"),
            );
            for rows in 2..=MAX_ROWS {
                let m = i32::try_from(rows).expect("rows");
                let x = random::normal::<f32>(
                    &[m, k][..],
                    None,
                    None,
                    &random::key(100 + u64::from(m.unsigned_abs())).expect("key"),
                )
                .expect("normal");
                let reference = mlx(&x, &w, &s32, &b32, group);
                let fp32 = max_abs(&kernel(&x, &w, &s32, &b32, group), &reference);
                let x16 = x.as_dtype(Dtype::Bfloat16).expect("bf16");
                let ours = max_abs(&kernel(&x16, &w, &w.scales, &w.biases, group), &reference);
                let theirs = max_abs(&mlx(&x16, &w, &w.scales, &w.biases, group), &reference);
                let bound = 2.0 * theirs + 1e-2;
                if fp32 > 1e-2 || ours > bound {
                    println!(
                        "small_m_qmv FAIL K={k} N={n} g={group} M={rows}: fp32 {fp32:e}, \
                         bf16 {ours:e} vs bound {bound:e}"
                    );
                    failed += 1;
                }
            }
        }
    }
    failed
}

#[test]
fn few_row_kernel_matches_mlx_in_fp32_and_within_its_bf16_error() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let failed = failures(&|x, w, s, b, g| matmul(x, &w.packed, s, b, g).expect("kernel"));
    assert_eq!(failed, 0, "{failed} cases outside the gate");
}

/// The kernel without the per-group bias term.
fn kernel_without_bias() -> String {
    let source = include_str!("../small_m/qmv.metal");
    let biased = "& 0xfu) + bias;";
    assert_eq!(source.matches(biased).count(), 2, "mutation site moved");
    source.replace(biased, "& 0xfu);")
}

#[test]
fn a_kernel_that_drops_the_group_bias_fails_the_gate() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let broken = kernel_without_bias();
    let failed = failures(&|x, w, s, b, g| {
        matmul_with_source(&broken, x, &w.packed, s, b, g).expect("kernel")
    });
    assert!(failed > 0, "the mutated kernel passed every case");
}

#[test]
fn only_two_to_sixteen_rows_of_four_bit_weights_are_routed() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let four = Qwen3AffineQuantization {
        bits: Qwen3AffineBits::Four,
        group_size: Qwen3AffineGroupSize::G64,
    };
    let six = Qwen3AffineQuantization {
        bits: Qwen3AffineBits::Six,
        ..four
    };
    let input = Array::zeros::<f32>(&[1, 4, 256]).expect("zeros");
    set_small_m_qmv(false);
    assert!(!applies(&input, 4, four), "routing is off by default");
    set_small_m_qmv(true);
    let routed: Vec<usize> = (1..=20)
        .filter(|&rows| applies(&input, rows, four))
        .collect();
    let six_bit = applies(&input, 4, six);
    let ragged_k = applies(&Array::zeros::<f32>(&[4, 100]).expect("zeros"), 4, four);
    set_small_m_qmv(false);
    assert_eq!(routed, (2..=16).collect::<Vec<_>>());
    assert!(!six_bit, "6-bit weights stay on MLX");
    assert!(!ragged_k, "K must be whole 32-value chunks");
}

/// Top-1 index (lowest on ties) and top-1 minus top-2 logit.
fn top_two(row: &[f32]) -> (usize, f32) {
    let mut first = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[first] {
            first = index;
        }
    }
    let second = row
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != first)
        .map(|(_, &value)| value)
        .fold(f32::NEG_INFINITY, f32::max);
    (first, row[first] - second)
}

type Executor<'a> = super::super::Qwen3ForwardExecutor<'a, std::collections::hash_map::RandomState>;

/// Verifies `draft[..m]` for m = 2..=16 with routing off and on; a routed
/// pick may differ from MLX's only where MLX's top-two gap is below 0.5.
/// Returns the number of differing positions.
fn near_tie_divergences(executor: &mut Executor<'_>, draft: &[i32]) -> usize {
    let mut divergences = 0;
    for m in 2..=MAX_ROWS {
        let mut rows = Vec::new();
        for routed in [false, true] {
            set_small_m_qmv(routed);
            rows.push(executor.extend_all_logits(&draft[..m]).expect("verify"));
            executor.truncate_cached_tokens(m).expect("rollback");
        }
        set_small_m_qmv(false);
        for position in 0..m {
            let (want, gap) = top_two(rows[0].row(position).expect("row"));
            let (got, _) = top_two(rows[1].row(position).expect("row"));
            if want != got {
                assert!(
                    gap < 0.5,
                    "m={m} position {position}: diverged with gap {gap}"
                );
                divergences += 1;
            }
        }
    }
    divergences
}

/// Seconds per greedy verify of `draft[..m]`, `[m - 1][arm]`, over `rounds`
/// rounds of every (m, arm) in shuffled order. Arms: 0 today, 1 today again
/// (the A/A noise floor), 2 routed.
fn verify_times(executor: &mut Executor<'_>, draft: &[i32], rounds: usize) -> Vec<[Vec<f64>; 3]> {
    let mut run = |m: usize, routed: bool| {
        set_small_m_qmv(routed);
        let started = std::time::Instant::now();
        executor
            .extend_greedy(&draft[..m])
            .expect("verify")
            .wait()
            .expect("picks");
        let elapsed = started.elapsed().as_secs_f64();
        executor.truncate_cached_tokens(m).expect("rollback");
        elapsed
    };
    let mut arms: Vec<(usize, usize)> = (1..=MAX_ROWS)
        .flat_map(|m| [(m, 0), (m, 1), (m, 2)])
        .collect();
    for &(m, arm) in &arms {
        run(m, arm == 2);
    }
    let mut times = vec![[Vec::new(), Vec::new(), Vec::new()]; MAX_ROWS];
    let mut state = 0x9e37_79b9_u64;
    for _ in 0..rounds {
        for i in (1..arms.len()).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let j = usize::try_from(state >> 33).expect("fits") % (i + 1);
            arms.swap(i, j);
        }
        for &(m, arm) in &arms {
            let elapsed = run(m, arm == 2);
            times[m - 1][arm].push(elapsed);
        }
    }
    set_small_m_qmv(false);
    times
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Opt-in: the verify(m) cost curve on a 4-bit Qwen3 checkpoint.
///
/// Set `METALLIX_QWEN_4BIT_MODEL` to an MLX 4-bit Qwen3 directory. After a
/// 512-token prefill, each round verifies m = 1..=16 drafted tokens with
/// routing on and off, in shuffled order, rolling each append back, and
/// prints the per-m median of the paired on/off ratio. Speed bars (card,
/// declared before the run): routed m must not be slower than today, and
/// verify(8) <= 0.75x today. First, the near-tie rule end to end. Run on an
/// idle machine; the test prints the load average.
#[test]
#[ignore = "requires METALLIX_QWEN_4BIT_MODEL and an idle Apple-Silicon GPU"]
fn verify_cost_curve_with_and_without_the_few_row_kernel() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let model = std::env::var_os("METALLIX_QWEN_4BIT_MODEL")
        .map(std::path::PathBuf::from)
        .expect("METALLIX_QWEN_4BIT_MODEL is required for this ignored test");
    let rounds: usize = std::env::var("METALLIX_VERIFY_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(15);
    let weights = crate::metal::Qwen3MlxWeights::load(&model).expect("checkpoint load");
    let mut executor = weights
        .resident_chat_executor(4096, 1 << 30)
        .expect("executor");
    let token = |index: usize| i32::try_from((index * 7919 + 101) % 151_000).expect("small");
    let prompt: Vec<i32> = (0..512).map(token).collect();
    let draft: Vec<i32> = (512..512 + MAX_ROWS).map(token).collect();
    executor.prefill_last_logits(&prompt).expect("prefill");

    let divergences = near_tie_divergences(&mut executor, &draft);
    println!("small_m_qmv verify near-tie divergences: {divergences}");
    let times = verify_times(&mut executor, &draft, rounds);
    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default();
    let architecture = mlx_kernels::gpu_architecture().unwrap_or_else(|| "unknown".into());
    println!("small_m_qmv verify curve: {architecture}, {rounds} rounds; {load}");
    // Cost relative to today's one-row verify, both arms: the drafter's
    // VerifyCost table takes t(q) / t(1) per GPU architecture.
    let one_row = median(&times[0][0]);
    for (index, [today, again, routed]) in times.iter().enumerate() {
        let ratios: Vec<f64> = routed.iter().zip(today).map(|(a, b)| a / b).collect();
        // A/A floor: minimum detectable effect from the today/today ratios.
        let logs: Vec<f64> = again.iter().zip(today).map(|(a, b)| (a / b).ln()).collect();
        #[allow(clippy::cast_precision_loss)]
        let n = logs.len() as f64;
        let mean = logs.iter().sum::<f64>() / n;
        let sd = (logs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let mde = (2.8 * sd / n.sqrt()).exp() - 1.0;
        println!(
            "small_m_qmv verify m={}: today {:.2} ms ({:.2}x one row), routed {:.2} ms \
             ({:.2}x one row), paired routed/today {:.3}, A/A MDE {:.1}%",
            index + 1,
            1e3 * median(today),
            median(today) / one_row,
            1e3 * median(routed),
            median(routed) / one_row,
            median(&ratios),
            100.0 * mde
        );
    }
}
