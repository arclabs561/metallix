//! Integration coverage for the pinned FP4 activation staging.
//!
//! The public scalar reference owns software encoding and scale diagnostics.
//! This suite keeps its fixture outputs and cross-operator composition checks;
//! it is not CUDA/TileLang instruction-level parity or packed FP4 storage.

use deepseek::attention::{SparseAttentionLayout, sparse_attention_reference};
use deepseek::precision::{
    Fp4ActivationError, Fp4ActivationMode as Fp4Mode, MAX_FP4_ACTIVATION_ELEMENTS as MAX_ELEMENTS,
    requantize_bf16_activations_e2m1 as fp4_activation_reference,
};
use deepseek::rotary::{RotaryDirection, RotaryFrequency, RotaryTailLayout, rotate_tail};
use std::num::NonZeroUsize;

// This integration binary has its own process; the library's cfg(test) guard
// is not linked into it. Match the unit-test guard for MLX global device state.
#[cfg(feature = "metal")]
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16)
        .expect("high FP32 half fits")
}

fn fixture_words(value: &serde_json::Value, field: &str) -> Vec<u16> {
    value[field]
        .as_array()
        .expect("fixture words")
        .iter()
        .map(|word| u16::try_from(word.as_u64().expect("word")).expect("u16 word"))
        .collect()
}

#[test]
fn independent_fixture_cases_match_software_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/fp4-activation-reference.json"
    ))
    .expect("checked-in FP4 fixture");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["kernel_sha256"],
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(fixture["receipt"]["device"], "cpu");
    let cases = fixture["cases"].as_array().expect("fixture cases");
    assert_eq!(cases.len(), 12);
    for mode in ["compressed_kv", "index"] {
        for category in [
            "zero",
            "ties",
            "nonpower_scale",
            "subnormal",
            "scale_boundary",
            "upper_finite_scale",
        ] {
            let name = format!("{mode}_{category}");
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case["name"] == name && case["mode"] == mode)
                    .count(),
                1,
                "required fixture case {name} must occur exactly once"
            );
        }
    }
    for case in cases {
        let mode = match case["mode"].as_str().expect("mode") {
            "compressed_kv" => Fp4Mode::CompressedKv16E4m3,
            "index" => Fp4Mode::Index32E8m0,
            other => panic!("unexpected fixture mode {other}"),
        };
        let input = fixture_words(case, "input_bf16");
        let expected = fixture_words(case, "output_bf16");
        let mut output = vec![0_u16; input.len()];
        fp4_activation_reference(&input, 1, input.len(), mode, &mut output)
            .expect("finite fixture case");
        assert_eq!(output, expected, "{}", case["name"]);
    }
}

#[test]
fn known_groups_preserve_zero_signs_across_scale_modes() {
    let mut compressed = [0_u16; 16];
    compressed[..4].copy_from_slice(&[0x40c0, 0xc0c0, 0x3f80, 0x8000]); // 6,-6,1,-0
    let mut output = [0xdead_u16; 16];
    fp4_activation_reference(&compressed, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output)
        .expect("finite E4M3 group");
    assert_eq!(&output[..4], &[0x40c0, 0xc0c0, 0x3f80, 0x8000]);

    let mut zeros = [0_u16; 32];
    for bits in zeros.iter_mut().skip(1).step_by(2) {
        *bits = 0x8000;
    }
    let mut index_output = [0xdead_u16; 32];
    fp4_activation_reference(&zeros, 1, 32, Fp4Mode::Index32E8m0, &mut index_output)
        .expect("zero E8M0 group");
    assert_eq!(index_output, zeros);
}

#[test]
fn multigroup_modes_keep_expected_outputs_and_late_failure_is_atomic() {
    let mut compressed = [0_u16; 32];
    compressed[..16].fill(0x40c0); // 6 -> E4M3 scale 1
    compressed[16..].fill(0x4140); // 12 -> E4M3 scale 2
    compressed[0] = 0x3f00; // 0.5 survives scale 1, but ties to zero at scale 2.
    compressed[16] = 0x3f00;
    let mut compressed_output = [0_u16; 32];
    fp4_activation_reference(
        &compressed,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut compressed_output,
    )
    .expect("two compressed groups");
    let mut compressed_expected = compressed;
    compressed_expected[16] = 0x0000;
    assert_eq!(compressed_output, compressed_expected);

    let mut index = [0_u16; 64];
    index[..32].fill(0x4110); // 9 -> power-of-two scale 2
    index[32..].fill(0x4190); // 18 -> power-of-two scale 4
    index[0] = 0x3fa0; // 1.25 -> 1 at scale 2
    index[32] = 0x3fa0; // 1.25 -> 2 at scale 4
    let mut index_output = [0_u16; 64];
    fp4_activation_reference(&index, 2, 32, Fp4Mode::Index32E8m0, &mut index_output)
        .expect("two index groups");
    let mut index_expected = [0x4100_u16; 64]; // 9 -> 8 at scale 2
    index_expected[32..].fill(0x4180); // 18 -> 16 at scale 4
    index_expected[0] = 0x3f80;
    index_expected[32] = 0x4000;
    assert_eq!(index_output, index_expected);

    let mut late = compressed;
    late[16..].fill(0x7f7f); // second group exceeds E4M3 finite scale range
    let mut output = [0xdead_u16; 32];
    assert!(matches!(
        fp4_activation_reference(&late, 2, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::E4m3ScaleOverflow)
    ));
    assert_eq!(output, [0xdead; 32]);
}

#[test]
fn failures_are_atomic_for_shape_nonfinite_and_scale_overflow() {
    let mut output = [0xdead_u16; 16];
    assert!(matches!(
        fp4_activation_reference(&[0; 15], 1, 15, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::PartialBlock { .. })
    ));
    assert_eq!(output, [0xdead; 16]);
    let mut nonfinite = [0_u16; 16];
    nonfinite[9] = 0x7f80;
    assert!(matches!(
        fp4_activation_reference(&nonfinite, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::NonFiniteInput { index: 9 })
    ));
    assert_eq!(output, [0xdead; 16]);
    let huge = [0x7f7f_u16; 16];
    assert!(matches!(
        fp4_activation_reference(&huge, 1, 16, Fp4Mode::CompressedKv16E4m3, &mut output),
        Err(Fp4ActivationError::E4m3ScaleOverflow)
    ));
    assert_eq!(output, [0xdead; 16]);

    let mut index_output = [0xdead_u16; 32];
    let index_huge = [0x7f7f_u16; 32];
    assert!(matches!(
        fp4_activation_reference(&index_huge, 1, 32, Fp4Mode::Index32E8m0, &mut index_output,),
        Err(Fp4ActivationError::ReconstructionOverflow { .. })
    ));
    assert_eq!(index_output, [0xdead; 32]);
}

#[test]
fn malformed_shapes_and_buffers_leave_output_untouched_without_large_allocations() {
    let input = [0_u16; 16];
    for (rows, width, input_len) in [
        (0, 16, 16),
        (1, 0, 16),
        (usize::MAX, 16, 16),
        (MAX_ELEMENTS, 16, 16),
        (1, 16, 15),
        (1, 32, 16),
    ] {
        let mut output = [0xdead_u16; 16];
        let error = fp4_activation_reference(
            &input[..input_len],
            rows,
            width,
            Fp4Mode::CompressedKv16E4m3,
            &mut output,
        )
        .expect_err("invalid shape or input buffer");
        match (rows, width, input_len) {
            (0, _, _) | (_, 0, _) => {
                assert!(matches!(error, Fp4ActivationError::EmptyShape));
            }
            (usize::MAX, _, _) => {
                assert!(matches!(error, Fp4ActivationError::ShapeOverflow));
            }
            (rows, 16, 16) if rows == MAX_ELEMENTS => {
                assert!(matches!(
                    error,
                    Fp4ActivationError::ElementLimit { elements }
                        if elements == MAX_ELEMENTS * 16
                ));
            }
            (1, 16, 15) => {
                assert!(matches!(
                    error,
                    Fp4ActivationError::Length {
                        field: "input",
                        actual: 15,
                        expected: 16,
                    }
                ));
            }
            (1, 32, 16) => {
                assert!(matches!(
                    error,
                    Fp4ActivationError::Length {
                        field: "input",
                        actual: 16,
                        expected: 32,
                    }
                ));
            }
            _ => panic!("unexpected malformed shape case"),
        }
        assert_eq!(output, [0xdead; 16]);
    }
    let mut short_output = [0xdead_u16; 15];
    let error = fp4_activation_reference(
        &input,
        1,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut short_output,
    )
    .expect_err("short output buffer");
    assert!(matches!(
        error,
        Fp4ActivationError::Length {
            field: "output",
            actual: 15,
            expected: 16
        }
    ));
    assert_eq!(short_output, [0xdead; 15]);
}

fn fixture_floats(fixture: &serde_json::Value, field: &str) -> Vec<f32> {
    serde_json::from_value(fixture[field].clone()).expect("FP32 fixture array")
}

// Fixed two-key, width-16 composition. This is not a runtime cache API.
fn rotate_compressed_tails(input: &[u16], frequencies: &[f32]) -> Vec<u16> {
    rotate_two_row_tails(input, 16, frequencies)
}

fn rotate_two_row_tails(input: &[u16], width: usize, frequencies: &[f32]) -> Vec<u16> {
    let one = NonZeroUsize::new(1).expect("one");
    let two = NonZeroUsize::new(2).expect("two");
    assert!(width >= 4);
    assert_eq!(input.len(), 2 * width);
    assert_eq!(frequencies.len(), 8);
    let mut tail: Vec<f32> = input
        .chunks_exact(width)
        .flat_map(|row| row[width - 4..].iter().copied().map(bf16_to_f32))
        .collect();
    let frequencies: Vec<_> = frequencies
        .chunks_exact(2)
        .map(|pair| RotaryFrequency::new(pair[0], pair[1]).expect("finite frequency"))
        .collect();
    rotate_tail(
        &mut tail,
        RotaryTailLayout::new(one, two, one, two).expect("tail layout"),
        &frequencies,
        RotaryDirection::Forward,
    )
    .expect("rotated supplied latent");
    let mut result = input.to_vec();
    for (row, rotated) in result.chunks_exact_mut(width).zip(tail.chunks_exact(4)) {
        for (output, &value) in row[width - 4..].iter_mut().zip(rotated) {
            *output = f32_to_bf16_rne(value);
        }
    }
    result
}

fn compressed_attention_output(kv: &[u16], fixture: &serde_json::Value) -> Vec<f32> {
    let nz = |value| NonZeroUsize::new(value).expect("fixed nonzero dimension");
    let keys: Vec<_> = kv.iter().copied().map(bf16_to_f32).collect();
    let indices: Vec<i32> =
        serde_json::from_value(fixture["indices_i32"].clone()).expect("indices");
    let scale: f32 = serde_json::from_value(fixture["softmax_scale"].clone()).expect("scale");
    sparse_attention_reference(
        &fixture_floats(fixture, "query_f32"),
        &keys,
        &fixture_floats(fixture, "sink_f32"),
        &indices,
        scale,
        SparseAttentionLayout::new(nz(1), nz(2), nz(1), nz(16), nz(2), nz(4))
            .expect("attention layout"),
    )
    .expect("compressed-only mathematical attention")
}

#[test]
fn rotated_fp4_compressed_keys_feed_sparse_attention_in_source_order() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-attention-reference.json"
    ))
    .expect("independent composition oracle");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["indices_i32"],
        serde_json::json!([1, 0, 1, -1, -1, -1, -1, -1])
    );
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["model_sha256"],
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture["source"]["kernel_sha256"],
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(fixture["receipt"]["device"], "cpu");
    let latent = fixture_words(&fixture, "latent_bf16");
    let frequencies = fixture_floats(&fixture, "frequencies_f32");
    let rotated = rotate_compressed_tails(&latent, &frequencies);
    assert_eq!(rotated, fixture_words(&fixture, "rotated_bf16"));
    let mut reconstructed = vec![0; 32];
    fp4_activation_reference(
        &rotated,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut reconstructed,
    )
    .expect("post-rotation quantization");
    assert_eq!(reconstructed, fixture_words(&fixture, "reconstructed_bf16"));
    let actual = compressed_attention_output(&reconstructed, &fixture);
    let expected = fixture_floats(&fixture, "output_f32");
    assert_eq!(expected.len(), 32);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "attention element {index}: {actual} != {expected}"
        );
    }
    assert!(actual[16..].iter().all(|value| value.to_bits() == 0));
    let mut early_quantized = vec![0; 32];
    fp4_activation_reference(
        &latent,
        2,
        16,
        Fp4Mode::CompressedKv16E4m3,
        &mut early_quantized,
    )
    .expect("wrong-order distinguisher");
    let wrong_order = rotate_compressed_tails(&early_quantized, &frequencies);
    assert_ne!(wrong_order, reconstructed);
    for wrong_keys in [&wrong_order, &rotated] {
        let wrong = compressed_attention_output(wrong_keys, &fixture);
        assert!(
            wrong
                .iter()
                .zip(&actual)
                .any(|(wrong, actual)| (wrong - actual).abs() > 1e-3),
            "oracle must distinguish reordered or omitted quantization"
        );
    }
}

/// Joins the actual GPU score core and CPU selection with attention. The index
/// operands are supplied post-projection/post-RoPE, not produced by a full indexer.
#[cfg(feature = "metal")]
#[test]
fn fp4_index_scores_select_the_compressed_vector_consumed_by_attention() {
    use deepseek::indexer::index_scores_f32;
    use deepseek::selection::select_indices;

    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let mut query = [0x3f80_u16; 64]; // two heads: +1, -1
    query[32..].fill(0xbf80);
    let mut keys = [0x4110_u16; 64]; // two keys: +9, -3
    keys[32..].fill(0xc040);
    let mut quantized_query = [0; 64];
    let mut quantized_keys = [0; 64];
    fp4_activation_reference(&query, 2, 32, Fp4Mode::Index32E8m0, &mut quantized_query)
        .expect("index query preparation");
    fp4_activation_reference(&keys, 2, 32, Fp4Mode::Index32E8m0, &mut quantized_keys)
        .expect("index key preparation");
    assert_eq!(&quantized_keys[..32], &[0x4100; 32]); // 9 -> 8 at scale 2
    assert_eq!(&quantized_keys[32..], &[0xc040; 32]); // -3 preserved
    let q: Vec<_> = quantized_query.iter().copied().map(bf16_to_f32).collect();
    let k: Vec<_> = quantized_keys.iter().copied().map(bf16_to_f32).collect();
    let scores = index_scores_f32(
        &q,
        &k,
        &[-4.0, 1.0],
        NonZeroUsize::new(32).expect("index width"),
    )
    .expect("Metal score reduction");
    // Dot rows [256,-96],[-256,96]; ReLU BEFORE signed head weighting.
    assert_eq!(
        scores
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [-1024.0_f32, 96.0].map(f32::to_bits)
    );
    let selected = select_indices(&scores, 2, 1, 0).expect("strict score cutoff");
    assert_eq!(selected, [1]);

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-attention-reference.json"
    ))
    .expect("supplied compressed latent fixture");
    let rotated = rotate_compressed_tails(
        &fixture_words(&fixture, "latent_bf16"),
        &fixture_floats(&fixture, "frequencies_f32"),
    );
    let mut cache = vec![0; 32];
    fp4_activation_reference(&rotated, 2, 16, Fp4Mode::CompressedKv16E4m3, &mut cache)
        .expect("compressed KV preparation");
    let cache: Vec<_> = cache.into_iter().map(bf16_to_f32).collect();
    let nz = |value| NonZeroUsize::new(value).expect("fixed nonzero dimension");
    let layout = SparseAttentionLayout::new(nz(1), nz(1), nz(1), nz(16), nz(2), nz(1))
        .expect("one selected compressed position");
    // Zero attention query and zero sink yield exactly half the selected KV:
    // one key contributes exp(0), the sink contributes exp(0) only to denominator.
    let output = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &selected, 0.25, layout)
        .expect("attention consumes calculated indices");
    let expected: Vec<_> = cache[16..]
        .iter()
        .map(|value| (value * 0.5).to_bits())
        .collect();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    let earlier = select_indices(&[scores[0], f32::NEG_INFINITY], 1, 1, 0)
        .expect("future compressed key is masked before selection");
    assert_eq!(earlier, [0]);
    let wrong = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &earlier, 0.25, layout)
        .expect("earlier causal prefix selects the other vector");
    assert!(
        output
            .iter()
            .zip(wrong)
            .any(|(actual, wrong)| (actual - wrong).abs() > 0.1)
    );
    let none =
        select_indices(&[f32::NEG_INFINITY; 2], 0, 1, 0).expect("no completed compressed group");
    assert_eq!(none, [-1]);
    let empty = sparse_attention_reference(&[0.0; 16], &cache, &[0.0], &none, 0.25, layout)
        .expect("unreachable keys do not contribute");
    assert!(empty.iter().all(|value| value.to_bits() == 0));
}

#[cfg(feature = "metal")]
fn projected_normalized_index_keys(latent: &[u16], frequencies: &[f32]) -> Vec<f32> {
    use deepseek::norm::rms_norm_bf16_reference;
    use deepseek::precision::bf16_linear_reference;

    // Every output selects latent column 12, which changes sign under the
    // fixture's first rotation. This makes premature latent mutation observable.
    let mut weight = [0_u16; 32 * 16];
    for row in weight.chunks_exact_mut(16) {
        row[12] = 0x3f80;
    }
    let mut projected = [0; 64];
    bf16_linear_reference(latent, &weight, 2, 16, 32, &mut projected).expect("index wk projection");
    let mut normalized = [0; 64];
    for (row, output) in projected
        .chunks_exact(32)
        .zip(normalized.chunks_exact_mut(32))
    {
        rms_norm_bf16_reference(row, &[0x4110; 32], 1e-6, output)
            .expect("index k_norm with learned scale 9");
    }
    let rotated = rotate_two_row_tails(&normalized, 32, frequencies);
    let mut quantized = [0; 64];
    fp4_activation_reference(&rotated, 2, 32, Fp4Mode::Index32E8m0, &mut quantized)
        .expect("index key FP4 preparation");
    quantized.into_iter().map(bf16_to_f32).collect()
}

#[cfg(feature = "metal")]
#[test]
fn projected_index_queries_rotate_before_fp4_and_scale_signed_head_weights() {
    use deepseek::indexer::index_scores_f32;
    use deepseek::precision::bf16_linear_reference;
    use deepseek::selection::select_indices;

    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let nz = |value| NonZeroUsize::new(value).expect("fixed dimension");
    // Qualify the source's BF16-configured wq_b branch, not FP8 checkpoint GEMM.
    // One supplied qr row [9, 2]; output heads select +qr[0] and -qr[0].
    let mut wq_b = [0_u16; 64 * 2];
    for (head, rows) in wq_b.chunks_exact_mut(32 * 2).enumerate() {
        for row in rows.chunks_exact_mut(2) {
            row[0] = if head == 0 { 0x3f80 } else { 0xbf80 };
        }
    }
    let mut projected = [0_u16; 64];
    bf16_linear_reference(&[0x4110, 0x4000], &wq_b, 1, 2, 64, &mut projected)
        .expect("BF16 query projection");
    assert_eq!(&projected[..32], &[0x4110; 32]);
    assert_eq!(&projected[32..], &[0xc110; 32]);

    let rotate = |input: &[u16; 64]| {
        let mut tail: Vec<_> = input
            .chunks_exact(32)
            .flat_map(|head| head[28..].iter().copied().map(bf16_to_f32))
            .collect();
        rotate_tail(
            &mut tail,
            RotaryTailLayout::new(nz(1), nz(1), nz(2), nz(2)).expect("one query, two heads"),
            &[RotaryFrequency::new(0.6, 0.8).expect("frequency"); 2],
            RotaryDirection::Forward,
        )
        .expect("query position frequency broadcasts over heads");
        let mut output = *input;
        for (head, tail) in output.chunks_exact_mut(32).zip(tail.chunks_exact(4)) {
            for (word, &value) in head[28..].iter_mut().zip(tail) {
                *word = f32_to_bf16_rne(value);
            }
        }
        output
    };
    let quantize = |input: &[u16; 64]| {
        let mut output = [0; 64];
        fp4_activation_reference(input, 2, 32, Fp4Mode::Index32E8m0, &mut output)
            .expect("independent group per query head");
        output
    };
    let query = quantize(&rotate(&projected));
    // (9+9i)*(0.6+0.8i) = -1.8+12.6i. BF16 narrowing then scale-4
    // E2M1 gives (-2,12); the nonrotary 9s become 8. Head sum is 244.
    assert_eq!(&query[..28], &[0x4100; 28]);
    assert_eq!(&query[28..32], &[0xc000, 0x4140, 0xc000, 0x4140]);
    for (&positive, &negative) in query[..32].iter().zip(&query[32..]) {
        assert_eq!(positive ^ 0x8000, negative);
    }

    // weights_proj(x): [2,1] @ [[-16,0],[0,8]]^T = [-32,8].
    let mut weights_bf16 = [0; 2];
    bf16_linear_reference(
        &[0x4000, 0x3f80],
        &[0xc180, 0, 0, 0x4100],
        1,
        2,
        2,
        &mut weights_bf16,
    )
    .expect("signed BF16 weights projection");
    assert_eq!(weights_bf16, [0xc200, 0x4100]);
    // Source scales by D^-0.5 * H^-0.5; D=32,H=2 gives 1/8.
    // Narrow back to BF16 after multiplication, as the source tensor does.
    let scale = 0.125_f32;
    let weights = weights_bf16.map(|word| bf16_to_f32(f32_to_bf16_rne(bf16_to_f32(word) * scale)));
    assert_eq!(weights.map(f32::to_bits), [-4.0_f32, 1.0].map(f32::to_bits));
    let mut keys = [1.0_f32; 64]; // supplied, exactly FP4-representable keys
    keys[32..].fill(-1.0);
    let score = |query: &[u16; 64], weights: &[f32]| {
        let query: Vec<_> = query.iter().copied().map(bf16_to_f32).collect();
        index_scores_f32(&query, &keys, weights, nz(32)).expect("Metal score core")
    };
    let scores = score(&query, &weights);
    assert_eq!(scores, [-976.0, 244.0]); // ReLU([244,-244],[-244,244]), signed sum
    assert_eq!(
        select_indices(&scores, 2, 1, 0).expect("strict cutoff"),
        [1]
    );
    assert_eq!(score(&quantize(&projected), &weights), [-1024.0, 256.0]);
    assert!(
        score(&rotate(&quantize(&projected)), &weights)
            .iter()
            .zip(&scores)
            .any(|(wrong, right)| (wrong - right).abs() > 1.0)
    );
    assert_eq!(
        score(&query, &weights_bf16.map(bf16_to_f32)),
        [-7808.0, 1952.0]
    );
}

#[cfg(feature = "metal")]
#[test]
fn index_projection_must_read_latents_before_attention_rotates_them() {
    use deepseek::indexer::index_scores_f32;
    use deepseek::selection::{SelectionError, select_indices};

    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-attention-reference.json"
    ))
    .expect("compressed latent fixture");
    let latent = fixture_words(&fixture, "latent_bf16");
    let frequencies = fixture_floats(&fixture, "frequencies_f32");
    let keys = projected_normalized_index_keys(&latent, &frequencies);
    // wk selects +1 and -4; k_norm rounds to +9 and -9. After key RoPE
    // and scale-4 FP4 reconstruction, row sums are exactly +248 and -248.
    assert_eq!(
        keys[..32].iter().sum::<f32>().to_bits(),
        248.0_f32.to_bits()
    );
    assert_eq!(
        keys[32..].iter().sum::<f32>().to_bits(),
        (-248.0_f32).to_bits()
    );
    let mut query = [1.0_f32; 64];
    query[32..].fill(-1.0); // supplied, exactly FP4-representable query heads
    let score = |keys: &[f32]| {
        index_scores_f32(
            &query,
            keys,
            &[-4.0, 1.0],
            NonZeroUsize::new(32).expect("index width"),
        )
        .expect("Metal index scores")
    };
    let scores = score(&keys);
    assert_eq!(
        scores
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [-992.0_f32, 248.0].map(f32::to_bits)
    );
    let selected = select_indices(&scores, 2, 1, 0).expect("strict cutoff");
    assert_eq!(selected, [1]);
    let mutated = rotate_compressed_tails(&latent, &frequencies);
    let mut cache = [0; 32];
    fp4_activation_reference(&mutated, 2, 16, Fp4Mode::CompressedKv16E4m3, &mut cache)
        .expect("attention cache reconstruction");
    let cache: Vec<_> = cache.into_iter().map(bf16_to_f32).collect();
    let nz = |value| NonZeroUsize::new(value).expect("fixed dimension");
    let output = sparse_attention_reference(
        &[0.0; 16],
        &cache,
        &[0.0],
        &selected,
        0.25,
        SparseAttentionLayout::new(nz(1), nz(1), nz(1), nz(16), nz(2), nz(1))
            .expect("attention layout"),
    )
    .expect("projected index keys drive attention selection");
    for (&actual, &value) in output.iter().zip(&cache[16..]) {
        assert_eq!(actual.to_bits(), (value * 0.5).to_bits());
    }
    let wrong_keys = projected_normalized_index_keys(&mutated, &frequencies);
    let wrong_scores = score(&wrong_keys);
    assert_eq!(
        wrong_scores
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [248.0_f32, 248.0].map(f32::to_bits)
    );
    assert_eq!(
        select_indices(&wrong_scores, 2, 1, 0),
        Err(SelectionError::AmbiguousCutoffTie)
    );
}
