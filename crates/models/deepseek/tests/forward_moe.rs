//! Native complete `MoE` sublayer against encoded synthetic source-forward data.
//! Source-provided input means this is not a complete native block or model.

use std::collections::BTreeMap;

use deepseek::moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
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
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    input: Tensor,
    gate_weights: Tensor,
    gate_indices: Tensor,
    output: Tensor,
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
        "b9ec31b115b0e12067bba7ddbee7526859027dde4e7b24062038a7b030e621e7"
    );
    assert_eq!(
        f.source.runner_sha256,
        "c473ce3a4c799d5bd9849c67c77dededccec817d54cc7b0c5431c27a0ac5a704"
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
    }
    f
}

fn run(f: &Fixture, omit_shared: bool) -> Vec<Vec<u16>> {
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
