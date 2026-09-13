//! Native complete `MoE` sublayer against encoded synthetic source-forward data.
//! Source-provided input means this is not a complete native block or model.

use std::collections::BTreeMap;

use deepseek::moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference};
use deepseek::{
    ffn::FfnSublayerReference,
    hc::{
        HcCoefficients,
        mixing::{hc_post_bf16_reference, hc_pre_bf16_reference},
        projection::project_hc_coefficients,
        split_hc_coefficients,
    },
    rms_norm_bf16_reference,
};
use serde::Deserialize;

#[path = "support/hc_chain_bounds.rs"]
mod hc_chain_bounds;
#[path = "support/hc_coefficient_bounds.rs"]
mod hc_coefficient_bounds;
#[path = "support/hc_projection_bounds.rs"]
mod hc_projection_bounds;
#[path = "support/rounding_interval.rs"]
mod rounding_interval;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
}

#[derive(Deserialize)]
struct BlockConfig {
    copies: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    norm_eps: f32,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    model_sha256: String,
    cpu_backend_sha256: String,
    complete_capture_sha256: String,
    storage_byteorder: String,
    kernel_source_sha256: String,
    loader_sha256: String,
    engram_sha256: String,
    manifest_canonical_sha256: String,
    runner_sha256: String,
}

#[derive(Deserialize)]
struct Model {
    dim: usize,
    moe_inter_dim: usize,
    n_routed_experts: usize,
    n_activated_experts: usize,
    n_shared_experts: usize,
    score_func: String,
    gate_temp: f32,
    norm_topk_prob: bool,
    route_scale: f32,
    swiglu_limit: f32,
    expert_dtype: String,
}

#[derive(Deserialize)]
struct Policy {
    output_bf16: String,
    route_weight_abs_error_max: f32,
    fixed_before_candidate_execution: bool,
    block_next_pre_abs_error_max: f32,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    input: Tensor,
    gate_weights: Tensor,
    gate_indices: Tensor,
    output: Tensor,
    block_input: Tensor,
    block_incoming_pre: Tensor,
    attention_input: Tensor,
    attention_output: Tensor,
    after_attention_residual: Tensor,
    attention_hc_mixes: Tensor,
    attention_coefficients: Coefficients,
    ffn_collapsed: Tensor,
    ffn_hc_mixes: Tensor,
    ffn_coefficients: Coefficients,
    block_output: Tensor,
    block_next_pre: Tensor,
}

#[derive(Deserialize)]
struct Coefficients {
    pre: Tensor,
    post: Tensor,
    comb: Tensor,
}

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    dtype: String,
    storage_hex: String,
}

impl Tensor {
    fn bytes(&self) -> Vec<u8> {
        assert_eq!(self.storage_hex.len() % 2, 0);
        self.storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 2);
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn fp32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "torch.float32");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 4);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn indices(&self) -> Vec<usize> {
        assert_eq!(self.dtype, "torch.int64");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 8);
        bytes
            .chunks_exact(8)
            .map(|b| usize::try_from(i64::from_le_bytes(b.try_into().unwrap())).unwrap())
            .collect()
    }
}

fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-moe-reference.json"
    ))
    .unwrap();
    assert_eq!(f.schema_version, 1);
    assert_source_provenance(&f);
    assert_encoded_parameter_schema(&f);
    assert_model_and_case_contract(&f);
    f
}

fn assert_source_provenance(f: &Fixture) {
    assert_eq!(
        f.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        f.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        f.source.cpu_backend_sha256,
        "b1f1f3cfdb93b674a5f96a114cf45bf5be9ad3a555ae95ac24add567f9f5232e"
    );
    assert_eq!(
        f.source.complete_capture_sha256,
        "1802d5fd528ce271c1dcf559bbcc3160c166ec8d18e5f4ca468680a2de7e93a5"
    );
    assert_eq!(
        f.source.runner_sha256,
        "a3ae0375fae76a3ad3e8802a4fb647a8a2756457dc1da7e5a32d847daf5eda2f"
    );
    assert_eq!(f.source.storage_byteorder, "little");
    assert_eq!(
        f.source.kernel_source_sha256,
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(
        f.source.loader_sha256,
        "359c4c961bdc8e200e2ccd13e7499974220a8d6942b5f6627316ab54210bef03"
    );
    assert_eq!(
        f.source.engram_sha256,
        "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
    );
    assert_eq!(
        f.source.manifest_canonical_sha256,
        "fd69a8fce4d5048f87db705603e05e3077c4f9bda402ec08be848aaa5cbdb92e"
    );
}

fn assert_encoded_parameter_schema(f: &Fixture) {
    assert_eq!(f.encoded_parameters.len(), 32);
    for expert in (0..4)
        .map(|id| format!("experts.{id}"))
        .chain(["shared_experts".to_owned()])
    {
        let shared = expert == "shared_experts";
        for projection in ["w1", "w2", "w3"] {
            let prefix = format!("layers.4.ffn.{expert}.{projection}");
            let weight = &f.encoded_parameters[&format!("{prefix}.weight")];
            let scale = &f.encoded_parameters[&format!("{prefix}.scale")];
            assert_eq!(
                weight.dtype,
                if shared {
                    "torch.float8_e4m3fn"
                } else {
                    "torch.float4_e2m1fn_x2"
                }
            );
            assert_eq!(weight.shape, [128, if shared { 128 } else { 64 }]);
            assert_eq!(scale.dtype, "torch.float8_e8m0fnu");
            assert_eq!(scale.shape, [if shared { 4 } else { 128 }, 4]);
        }
    }
}

fn assert_model_and_case_contract(f: &Fixture) {
    assert_eq!((f.model.dim, f.model.moe_inter_dim), (128, 128));
    assert_eq!(
        (
            f.model.n_routed_experts,
            f.model.n_activated_experts,
            f.model.n_shared_experts
        ),
        (4, 2, 1)
    );
    assert_eq!(f.model.expert_dtype, "fp4");
    assert_eq!(f.model.score_func, "sqrtsoftplus");
    assert_eq!(f.comparison_policy.output_bf16, "exact storage bits");
    assert_eq!(
        f.comparison_policy.route_weight_abs_error_max.to_bits(),
        2.0_f32.powi(-20).to_bits()
    );
    assert!(f.comparison_policy.fixed_before_candidate_execution);
    assert_eq!(f.cases.len(), 3);
    for (case, (start, positions)) in f.cases.iter().zip([(0, 5), (5, 1), (6, 1)]) {
        assert_eq!(case.start_pos, start);
        assert_eq!(case.input.shape, [1, positions, 128]);
        assert_eq!(case.output.shape, [1, positions, 128]);
        assert_eq!(case.gate_weights.shape, [positions, 2]);
        assert_eq!(case.gate_indices.shape, [positions, 2]);
        assert_eq!(case.block_input.shape, [1, positions, 2, 128]);
        assert_eq!(case.after_attention_residual.shape, case.block_input.shape);
        assert_eq!(case.attention_input.shape, [1, positions, 128]);
        assert_eq!(case.attention_output.shape, case.attention_input.shape);
        assert_eq!(case.ffn_collapsed.shape, case.attention_input.shape);
        for coefficients in [&case.attention_coefficients, &case.ffn_coefficients] {
            assert_eq!(coefficients.pre.shape, [1, positions, 2]);
            assert_eq!(coefficients.post.shape, [1, positions, 2]);
            assert_eq!(coefficients.comb.shape, [1, positions, 2, 2]);
        }
        assert_eq!(case.attention_hc_mixes.shape, [1, positions, 8]);
        assert_eq!(case.ffn_hc_mixes.shape, [1, positions, 8]);
    }
}

fn coefficient_bit_differences(
    actual: &HcCoefficients,
    expected_pre: &[f32],
    expected_post: &[f32],
    expected_comb: &[f32],
    context: &str,
) -> Vec<String> {
    assert_eq!(actual.copies(), 2, "{context} copies");
    let mut differences = Vec::new();
    for (field, actual, expected) in [
        ("pre", actual.pre(), expected_pre),
        ("post", actual.post(), expected_post),
        ("comb", actual.comb(), expected_comb),
    ] {
        assert_eq!(actual.len(), expected.len(), "{context} {field} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                differences.push(format!(
                    "{context} {field}[{index}]: {actual:?} ({:#010x}) != {expected:?} ({:#010x})",
                    actual.to_bits(),
                    expected.to_bits(),
                ));
            }
        }
    }
    differences
}

fn with_model<R>(f: &Fixture, omit_shared: bool, body: impl FnOnce(MoEReference<'_>) -> R) -> R {
    let mut encoded: BTreeMap<String, Vec<u8>> = f
        .encoded_parameters
        .iter()
        .map(|(k, v)| (k.clone(), v.bytes()))
        .collect();
    if omit_shared {
        encoded
            .get_mut("layers.4.ffn.shared_experts.w2.weight")
            .unwrap()
            .fill(0);
    }
    let expert_bytes = |prefix: &str| -> [&[u8]; 6] {
        [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ]
        .map(|suffix| encoded[&format!("layers.4.ffn.{prefix}.{suffix}")].as_slice())
    };
    let routed: Vec<_> = (0..4)
        .map(|id| {
            let [w1, s1, w2, s2, w3, s3] = expert_bytes(&format!("experts.{id}"));
            Fp4ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3).unwrap()
        })
        .collect();
    let [w1, s1, w2, s2, w3, s3] = expert_bytes("shared_experts");
    let shared = Fp8ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3).unwrap();
    let gate = f.encoded_parameters["layers.4.ffn.gate.weight"].bf16();
    let bias = f.encoded_parameters["layers.4.ffn.gate.bias"].fp32();
    let c = &f.model;
    let config = MoEConfig::new(
        c.dim,
        c.moe_inter_dim,
        c.swiglu_limit,
        c.n_activated_experts,
        c.gate_temp,
        c.norm_topk_prob,
        c.route_scale,
    )
    .unwrap();
    let model = MoEReference::new(config, &gate, &bias, &routed, shared).unwrap();
    body(model)
}

fn run(f: &Fixture, omit_shared: bool) -> Vec<Vec<u16>> {
    with_model(f, omit_shared, |model| {
        let mut outputs = Vec::new();
        for case in &f.cases {
            let input = case.input.bf16();
            let expected_ids = case.gate_indices.indices();
            let expected_weights = case.gate_weights.fp32();
            let mut output = Vec::new();
            for (position, row) in input.chunks_exact(128).enumerate() {
                let result = model.forward_token(row).unwrap();
                assert_eq!(result.routes().len(), 2);
                let ids = &expected_ids[position * 2..(position + 1) * 2];
                let weights = &expected_weights[position * 2..(position + 1) * 2];
                let mut sorted_ids = ids.to_vec();
                sorted_ids.sort_unstable();
                let native_ids: Vec<_> = result.routes().iter().map(|r| r.expert_index()).collect();
                assert_eq!(
                    native_ids, sorted_ids,
                    "complete selected expert set in ascending execution order"
                );
                for route in result.routes() {
                    let index = ids
                        .iter()
                        .position(|&id| id == route.expert_index())
                        .expect("native selected expert is selected by source");
                    assert!(
                        (route.weight() - weights[index]).abs()
                            <= f.comparison_policy.route_weight_abs_error_max,
                        "start {} position {position} expert {} route weight {} != {}",
                        case.start_pos,
                        route.expert_index(),
                        route.weight(),
                        weights[index]
                    );
                }
                output.extend_from_slice(result.output_bf16());
            }
            outputs.push(output);
        }
        outputs
    })
}

#[test]
fn native_moe_matches_all_source_prefill_and_decode_outputs() {
    let f = fixture();
    for (case, actual) in f.cases.iter().zip(run(&f, false)) {
        assert_eq!(
            actual,
            case.output.bf16(),
            "source MoE at start {}",
            case.start_pos
        );
    }
}

#[test]
fn source_oracle_rejects_omitting_shared_expert() {
    let f = fixture();
    for (case, actual) in f.cases.iter().zip(run(&f, true)) {
        assert_ne!(
            actual,
            case.output.bf16(),
            "shared contribution must matter at start {}",
            case.start_pos
        );
    }
}

#[test]
fn captured_attention_hc_post_reproduces_source_residual_exactly() {
    let f = fixture();
    for case in &f.cases {
        let positions = case.input.shape[1];
        let residual = case.block_input.bf16();
        let sublayer = case.attention_output.bf16();
        let expected = case.after_attention_residual.bf16();
        let post = case.attention_coefficients.post.fp32();
        let comb = case.attention_coefficients.comb.fp32();
        for position in 0..positions {
            let mut actual = vec![0; 256];
            hc_post_bf16_reference(
                &sublayer[position * 128..(position + 1) * 128],
                &residual[position * 256..(position + 1) * 256],
                &post[position * 2..(position + 1) * 2],
                &comb[position * 4..(position + 1) * 4],
                &mut actual,
            )
            .unwrap();
            assert_eq!(
                actual,
                expected[position * 256..(position + 1) * 256],
                "captured attention post start {} position {position}",
                case.start_pos
            );
        }
    }
}

#[test]
fn captured_attention_hc_pre_reproduces_source_ffn_collapse_exactly() {
    let f = fixture();
    for case in &f.cases {
        let positions = case.input.shape[1];
        let residual = case.after_attention_residual.bf16();
        let expected = case.ffn_collapsed.bf16();
        let pre = case.attention_coefficients.pre.fp32();
        for position in 0..positions {
            let mut actual = vec![0; 128];
            hc_pre_bf16_reference(
                &residual[position * 256..(position + 1) * 256],
                &pre[position * 2..(position + 1) * 2],
                128,
                &mut actual,
            )
            .unwrap();
            assert_eq!(
                actual,
                expected[position * 128..(position + 1) * 128],
                "captured attention pre start {} position {position}",
                case.start_pos
            );
        }
    }
}

#[test]
fn attention_hc_coefficients_isolate_split_before_projection() {
    let f = fixture();
    let c = &f.block_config;
    let scale: [f32; 3] = f.block_parameters["layers.4.hc_attn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = f.block_parameters["layers.4.hc_attn_base"].fp32();
    let projection = f.block_parameters["layers.4.hc_attn_fn"].fp32();
    let parameters = AttentionHcParameters {
        scale: &scale,
        base: &base,
        projection: &projection,
        config: c,
    };
    let mut differences = Vec::new();
    for case in &f.cases {
        for position in 0..case.input.shape[1] {
            check_attention_hc_position(case, position, &parameters, &mut differences);
        }
    }
    if !differences.is_empty() {
        eprintln!(
            "attention HC exact-bit diagnostic observed {} differences:\n{}",
            differences.len(),
            differences.join("\n")
        );
    }
}

struct AttentionHcParameters<'a> {
    scale: &'a [f32; 3],
    base: &'a [f32],
    projection: &'a [f32],
    config: &'a BlockConfig,
}

fn check_attention_hc_position(
    case: &Case,
    position: usize,
    parameters: &AttentionHcParameters<'_>,
    differences: &mut Vec<String>,
) {
    let residual = case.block_input.bf16();
    let mixes = case.attention_hc_mixes.fp32();
    let expected_pre = case.attention_coefficients.pre.fp32();
    let expected_post = case.attention_coefficients.post.fp32();
    let expected_comb = case.attention_coefficients.comb.fp32();
    let context = format!(
        "attention HC coefficients start {} position {position}",
        case.start_pos
    );
    let split = split_hc_coefficients(
        &mixes[position * 8..(position + 1) * 8],
        parameters.scale,
        parameters.base,
        parameters.config.copies,
        parameters.config.hc_sinkhorn_iters,
        parameters.config.hc_eps,
    )
    .unwrap();
    differences.extend(coefficient_bit_differences(
        &split,
        &expected_pre[position * 2..(position + 1) * 2],
        &expected_post[position * 2..(position + 1) * 2],
        &expected_comb[position * 4..(position + 1) * 4],
        &format!("{context} source mixes split"),
    ));
    let projected = project_hc_coefficients(
        &residual[position * 256..(position + 1) * 256],
        parameters.projection,
        parameters.scale,
        parameters.base,
        parameters.config.copies,
        parameters.config.norm_eps,
        parameters.config.hc_sinkhorn_iters,
        parameters.config.hc_eps,
    )
    .unwrap();
    differences.extend(coefficient_bit_differences(
        &projected,
        &expected_pre[position * 2..(position + 1) * 2],
        &expected_post[position * 2..(position + 1) * 2],
        &expected_comb[position * 4..(position + 1) * 4],
        &format!("{context} scalar projection"),
    ));
    if case.start_pos == 6 && position == 0 {
        diagnose_decode_hc_position(case, position, parameters, &split, &projected, differences);
    }
}

fn diagnose_decode_hc_position(
    case: &Case,
    position: usize,
    parameters: &AttentionHcParameters<'_>,
    split: &HcCoefficients,
    projected: &HcCoefficients,
    differences: &mut Vec<String>,
) {
    let residual = case.block_input.bf16();
    let attention = case.attention_output.bf16();
    let expected_residual = case.after_attention_residual.bf16();
    let expected_collapse = case.ffn_collapsed.bf16();
    let mixes = case.attention_hc_mixes.fp32();
    let expected_pre = case.attention_coefficients.pre.fp32();
    let expected_post = case.attention_coefficients.post.fp32();
    let expected_comb = case.attention_coefficients.comb.fp32();
    let residual = &residual[position * 256..(position + 1) * 256];
    let attention = &attention[position * 128..(position + 1) * 128];
    let expected_residual = &expected_residual[position * 256..(position + 1) * 256];
    let expected_collapse = &expected_collapse[position * 128..(position + 1) * 128];
    let captured_pre = &expected_pre[position * 2..(position + 1) * 2];
    let captured_post = &expected_post[position * 2..(position + 1) * 2];
    let captured_comb = &expected_comb[position * 4..(position + 1) * 4];
    let mut native_residual = vec![0; 256];
    hc_post_bf16_reference(
        attention,
        residual,
        projected.post(),
        projected.comb(),
        &mut native_residual,
    )
    .unwrap();
    append_bf16_differences(
        differences,
        "decode start 6 scalar projection attention residual",
        &native_residual,
        expected_residual,
    );

    let reciprocal_post: Vec<f32> = (0..parameters.config.copies)
        .map(|copy| {
            let affine = mixes[position * 8 + parameters.config.copies + copy]
                * parameters.scale[1]
                + parameters.base[parameters.config.copies + copy];
            2.0 / (1.0 + (-affine).exp())
        })
        .collect();
    eprintln!(
        "decode start 6 sigmoid-form probe: scalar post={:?}; reciprocal post={reciprocal_post:?}",
        projected.post(),
    );
    for (label, comb) in [
        ("reciprocal raw post plus captured comb", captured_comb),
        ("reciprocal raw post plus projected comb", projected.comb()),
    ] {
        report_residual_mismatches(
            label,
            attention,
            residual,
            expected_residual,
            &reciprocal_post,
            comb,
        );
    }
    let state = DecodeHcState {
        attention,
        residual,
        expected_residual,
        expected_collapse,
        captured_pre,
        captured_post,
        captured_comb,
        native_residual: &native_residual,
    };
    diagnose_decode_coefficients(&state, split, projected, differences);
}

struct DecodeHcState<'a> {
    attention: &'a [u16],
    residual: &'a [u16],
    expected_residual: &'a [u16],
    expected_collapse: &'a [u16],
    captured_pre: &'a [f32],
    captured_post: &'a [f32],
    captured_comb: &'a [f32],
    native_residual: &'a [u16],
}

fn diagnose_decode_coefficients(
    state: &DecodeHcState<'_>,
    split: &HcCoefficients,
    projected: &HcCoefficients,
    differences: &mut Vec<String>,
) {
    for (label, coefficients) in [
        ("source-mix scalar split", split),
        ("scalar projection", projected),
    ] {
        report_comb_differences(label, coefficients.comb(), state.captured_comb);
        report_residual_mismatches(
            &format!("{label} comb plus captured post"),
            state.attention,
            state.residual,
            state.expected_residual,
            state.captured_post,
            coefficients.comb(),
        );
        append_collapse_differences(
            differences,
            label,
            state.expected_residual,
            coefficients.pre(),
            state.expected_collapse,
            state.captured_pre,
        );
    }
    report_residual_mismatches(
        "source-mix scalar split post plus captured comb",
        state.attention,
        state.residual,
        state.expected_residual,
        split.post(),
        state.captured_comb,
    );
    for (label, pre) in [
        ("captured pre", state.captured_pre),
        ("scalar projection pre", projected.pre()),
    ] {
        append_collapse_differences(
            differences,
            &format!("native attention residual with {label}"),
            state.native_residual,
            pre,
            state.expected_collapse,
            state.captured_pre,
        );
    }
}

fn append_bf16_differences(
    differences: &mut Vec<String>,
    label: &str,
    actual: &[u16],
    expected: &[u16],
) {
    for (feature, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual != expected {
            differences.push(format!("{label}[{feature}]: {actual} != {expected}"));
        }
    }
}

fn report_residual_mismatches(
    label: &str,
    attention: &[u16],
    residual: &[u16],
    expected: &[u16],
    post: &[f32],
    comb: &[f32],
) {
    let mut candidate = vec![0; 256];
    hc_post_bf16_reference(attention, residual, post, comb, &mut candidate).unwrap();
    let mismatches: Vec<_> = candidate
        .iter()
        .zip(expected)
        .enumerate()
        .filter_map(|(feature, (&actual, &expected))| {
            (actual != expected).then_some(format!("{feature}: {actual} != {expected}"))
        })
        .collect();
    eprintln!(
        "decode start 6 {label} residual mismatches: {}{}",
        mismatches.len(),
        if mismatches.is_empty() {
            String::new()
        } else {
            format!(" ({})", mismatches.join(", "))
        }
    );
}

fn report_comb_differences(label: &str, actual: &[f32], expected: &[f32]) {
    let differences: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter_map(|(index, (&actual, &expected))| {
            (actual.to_bits() != expected.to_bits()).then_some(format!(
                "comb[{index}]={actual:?} ({:#010x}) != {expected:?} ({:#010x})",
                actual.to_bits(),
                expected.to_bits(),
            ))
        })
        .collect();
    eprintln!(
        "decode start 6 {label} comb differences: {}{}",
        differences.len(),
        if differences.is_empty() {
            String::new()
        } else {
            format!(" ({})", differences.join(", "))
        }
    );
}

fn append_collapse_differences(
    differences: &mut Vec<String>,
    label: &str,
    residual: &[u16],
    pre: &[f32],
    expected: &[u16],
    source_pre: &[f32],
) {
    let mut collapse = vec![0; 128];
    hc_pre_bf16_reference(residual, pre, 128, &mut collapse).unwrap();
    for (feature, (&actual, &expected)) in collapse.iter().zip(expected).enumerate() {
        if actual != expected {
            differences.push(format!(
                "decode start 6 {label} collapse[{feature}]: {actual} != {expected}; pre={pre:?} source_pre={source_pre:?}"
            ));
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum BlockControl {
    SourceAttention,
    WrongFfnPre,
    ZeroAttention,
}

struct BlockTailParameters {
    attn_projection: Vec<f32>,
    attn_scale: [f32; 3],
    attn_base: Vec<f32>,
    ffn_projection: Vec<f32>,
    ffn_scale: [f32; 3],
    ffn_base: Vec<f32>,
    attn_norm: Vec<u16>,
    ffn_norm: Vec<u16>,
}

fn block_tail(f: &Fixture, control: BlockControl, verify_contract: bool) -> Vec<Vec<u16>> {
    validate_block_tail_fixture(f);
    let parameters = block_tail_parameters(f);
    let config = &f.block_config;
    with_model(f, false, |model| {
        let ffn = FfnSublayerReference::new(
            model,
            &parameters.ffn_norm,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .unwrap();
        let context = BlockTailContext {
            fixture: f,
            parameters: &parameters,
            ffn: &ffn,
            control,
            verify_contract,
        };
        f.cases
            .iter()
            .map(|case| run_block_tail_case(&context, case))
            .collect()
    })
}

fn validate_block_tail_fixture(f: &Fixture) {
    let c = &f.block_config;
    assert_eq!(c.copies, 2);
    assert_eq!(c.hc_sinkhorn_iters, 20);
    assert_eq!(c.norm_eps.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(c.hc_eps.to_bits(), 1e-6_f32.to_bits());
    assert_eq!(f.block_parameters.len(), 8);
    assert_eq!(
        f.comparison_policy.block_next_pre_abs_error_max.to_bits(),
        2.0_f32.powi(-20).to_bits()
    );
}

fn block_tail_parameters(f: &Fixture) -> BlockTailParameters {
    let fp32 = |name: &str| f.block_parameters[&format!("layers.4.{name}")].fp32();
    BlockTailParameters {
        attn_projection: fp32("hc_attn_fn"),
        attn_scale: fp32("hc_attn_scale").try_into().unwrap(),
        attn_base: fp32("hc_attn_base"),
        ffn_projection: fp32("hc_ffn_fn"),
        ffn_scale: fp32("hc_ffn_scale").try_into().unwrap(),
        ffn_base: fp32("hc_ffn_base"),
        attn_norm: f.block_parameters["layers.4.attn_norm.weight"].bf16(),
        ffn_norm: f.block_parameters["layers.4.ffn_norm.weight"].bf16(),
    }
}

struct BlockTailContext<'a> {
    fixture: &'a Fixture,
    parameters: &'a BlockTailParameters,
    ffn: &'a FfnSublayerReference<'a>,
    control: BlockControl,
    verify_contract: bool,
}

struct BlockCaseData<'a> {
    block_input: &'a [u16],
    incoming: &'a [f32],
    attention: &'a [u16],
    expected_attention_input: &'a [u16],
}

fn run_block_tail_case(context: &BlockTailContext<'_>, case: &Case) -> Vec<u16> {
    let positions = case.input.shape[1];
    assert_block_case_shapes(case, positions);
    let block_input = case.block_input.bf16();
    let incoming = case.block_incoming_pre.fp32();
    let attention = case.attention_output.bf16();
    let expected_attention_input = case.attention_input.bf16();
    let data = BlockCaseData {
        block_input: &block_input,
        incoming: &incoming,
        attention: &attention,
        expected_attention_input: &expected_attention_input,
    };
    (0..positions)
        .flat_map(|position| run_block_tail_position(context, case, position, &data))
        .collect()
}

fn assert_block_case_shapes(case: &Case, positions: usize) {
    assert_eq!(case.block_input.shape, [1, positions, 2, 128]);
    assert_eq!(case.block_output.shape, case.block_input.shape);
    assert_eq!(case.block_incoming_pre.shape, [1, positions, 2]);
    assert_eq!(case.block_next_pre.shape, [1, positions, 2]);
    assert_eq!(case.attention_input.shape, [1, positions, 128]);
    assert_eq!(case.attention_output.shape, case.attention_input.shape);
    assert_eq!(case.ffn_collapsed.shape, case.attention_input.shape);
}

fn run_block_tail_position(
    context: &BlockTailContext<'_>,
    case: &Case,
    position: usize,
    data: &BlockCaseData<'_>,
) -> Vec<u16> {
    let parameters = context.parameters;
    let config = &context.fixture.block_config;
    let residual = &data.block_input[position * 256..(position + 1) * 256];
    let row = position * 128..(position + 1) * 128;
    let attn_coefficients = project_hc_coefficients(
        residual,
        &parameters.attn_projection,
        &parameters.attn_scale,
        &parameters.attn_base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    assert_attention_input(context, case, position, residual, data, &row);
    let attention_row = if context.control == BlockControl::ZeroAttention {
        &[0; 128][..]
    } else {
        &data.attention[row.clone()]
    };
    let mut after_attention = vec![0; 256];
    hc_post_bf16_reference(
        attention_row,
        residual,
        attn_coefficients.post(),
        attn_coefficients.comb(),
        &mut after_attention,
    )
    .unwrap();
    let own_coefficients = project_hc_coefficients(
        &after_attention,
        &parameters.ffn_projection,
        &parameters.ffn_scale,
        &parameters.ffn_base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    let pre = if context.control == BlockControl::WrongFfnPre {
        own_coefficients.pre()
    } else {
        attn_coefficients.pre()
    };
    let result = context.ffn.forward_token(&after_attention, pre).unwrap();
    if context.verify_contract {
        hc_chain_bounds::check_position(
            context.fixture,
            case,
            position,
            &attn_coefficients,
            &after_attention,
            &result,
        );
    }
    result.output_bf16().to_vec()
}

fn assert_attention_input(
    context: &BlockTailContext<'_>,
    case: &Case,
    position: usize,
    residual: &[u16],
    data: &BlockCaseData<'_>,
    row: &std::ops::Range<usize>,
) {
    // Attention output is source-supplied, but its incoming HC handoff remains
    // observable at this source attention-input boundary.
    let mut collapsed = vec![0; 128];
    hc_pre_bf16_reference(
        residual,
        &data.incoming[position * 2..(position + 1) * 2],
        128,
        &mut collapsed,
    )
    .unwrap();
    let mut normalized = vec![0; 128];
    rms_norm_bf16_reference(
        &collapsed,
        &context.parameters.attn_norm,
        context.fixture.block_config.norm_eps,
        &mut normalized,
    )
    .unwrap();
    assert_eq!(
        normalized,
        data.expected_attention_input[row.clone()],
        "attention input start {} position {position}",
        case.start_pos
    );
}

#[test]
fn native_block_tail_matches_source_numerical_contract() {
    let f = fixture();
    // Numerical and exact discrete assertions run at each joined boundary.
    let output = block_tail(&f, BlockControl::SourceAttention, true);
    assert_eq!(output.len(), f.cases.len());
}

#[test]
fn block_tail_rejects_wrong_hc_handoff_and_unused_attention() {
    let f = fixture();
    for control in [BlockControl::WrongFfnPre, BlockControl::ZeroAttention] {
        let changed = block_tail(&f, control, false);
        assert!(
            f.cases
                .iter()
                .zip(changed)
                .any(|(case, actual)| actual != case.block_output.bf16()),
            "counterexample must change the captured block output"
        );
    }
}

#[test]
#[should_panic(expected = "FFN collapse native")]
fn numerical_contract_rejects_wrong_hc_handoff() {
    block_tail(&fixture(), BlockControl::WrongFfnPre, true);
}

#[test]
#[should_panic(expected = "MoE checkpoint must agree exactly")]
fn numerical_contract_rejects_omitted_attention() {
    block_tail(&fixture(), BlockControl::ZeroAttention, true);
}
