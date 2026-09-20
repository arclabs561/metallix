//! Source-owned layer-three HC pre-mix and `RMSNorm` capture boundary.

use std::collections::BTreeMap;

use deepseek::{hc::mixing::hc_pre_bf16_reference, rms_norm_bf16_reference};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const WIDTH: usize = 128;
const COPIES: usize = 2;
const CAPTURE_SHA256: &str = "8379042b0d90f09b4c5b3d91f781a8f67171bc93603925e891f0e4254870a9a9";
const MODEL_SHA256: &str = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65";
const OBSERVER_SHA256: &str = "ec100119e51bf3cb4fbef2907e39bca07d389616af5e3b54682736d75f5cb00c";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u8,
    source: Source,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    scope: String,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    model_sha256: String,
    complete_capture_sha256: String,
    forward_observers_sha256: String,
    storage_byteorder: String,
}

#[derive(Deserialize)]
struct Model {
    input_dimension: usize,
    norm_epsilon: f32,
    expected_start_positions: Vec<usize>,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    block_input: BlockInput,
    attention_input: Tensor,
}

#[derive(Deserialize)]
struct BlockInput {
    residual: Tensor,
    incoming_pre: Tensor,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    finite: bool,
    numel: usize,
    shape: Vec<usize>,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn bytes(&self) -> Vec<u8> {
        let elements = self
            .shape
            .iter()
            .copied()
            .try_fold(1_usize, usize::checked_mul)
            .expect("source tensor shape product fits usize");
        assert_eq!(self.numel, elements, "source tensor numel");
        assert!(
            self.storage_hex.len().is_multiple_of(2),
            "source hex alignment"
        );
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("hex UTF-8"), 16)
                    .expect("source hex")
            })
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.storage_sha256);
        bytes
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        assert!(self.finite, "source BF16 is finite");
        self.bytes()
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
            .collect()
    }

    fn fp32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "torch.float32");
        assert!(self.finite, "source FP32 is finite");
        self.bytes()
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("FP32 word")))
            .collect()
    }
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-candidate-hc-reference.json"
    ))
    .expect("valid layer-three HC candidate fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(fixture.source.model_sha256, MODEL_SHA256);
    assert_eq!(fixture.source.complete_capture_sha256, CAPTURE_SHA256);
    assert_eq!(fixture.source.forward_observers_sha256, OBSERVER_SHA256);
    assert_eq!(fixture.source.storage_byteorder, "little");
    assert_eq!(fixture.model.input_dimension, WIDTH);
    assert_eq!(fixture.model.expected_start_positions, [0, 5, 6]);
    assert!(fixture.scope.contains("HC block operands"));
    assert_eq!(fixture.cases.len(), 3);
    fixture
}

fn attention_input(case: &Case, norm_weight: &[u16], epsilon: f32) -> Vec<u16> {
    let positions = case.attention_input.shape[1];
    assert_eq!(
        case.block_input.residual.shape,
        [1, positions, COPIES, WIDTH]
    );
    assert_eq!(case.block_input.incoming_pre.shape, [1, positions, COPIES]);
    assert_eq!(case.attention_input.shape, [1, positions, WIDTH]);
    let residual = case.block_input.residual.bf16();
    let incoming_pre = case.block_input.incoming_pre.fp32();
    let mut result = Vec::with_capacity(positions * WIDTH);
    for position in 0..positions {
        let mut collapsed = vec![0; WIDTH];
        hc_pre_bf16_reference(
            &residual[position * COPIES * WIDTH..(position + 1) * COPIES * WIDTH],
            &incoming_pre[position * COPIES..(position + 1) * COPIES],
            WIDTH,
            &mut collapsed,
        )
        .expect("captured HC pre-mix");
        let mut normalized = vec![0; WIDTH];
        rms_norm_bf16_reference(&collapsed, norm_weight, epsilon, &mut normalized)
            .expect("captured attention RMSNorm");
        result.extend(normalized);
    }
    result
}

/// Derives the layer-three candidate-producer input from source HC operands.
pub(super) fn derived_inputs() -> Vec<(usize, Vec<u16>)> {
    let fixture = fixture();
    let norm_weight = fixture.encoded_parameters["layers.3.attn_norm.weight"].bf16();
    assert_eq!(norm_weight.len(), WIDTH);
    fixture
        .cases
        .iter()
        .map(|case| {
            let derived = attention_input(case, &norm_weight, fixture.model.norm_epsilon);
            assert_eq!(
                derived,
                case.attention_input.bf16(),
                "HC input at start {}",
                case.start_pos
            );
            (case.start_pos, derived)
        })
        .collect()
}

/// Mutation controls keep the HC/RMSNorm source boundary sensitive.
#[allow(
    dead_code,
    reason = "the dedicated candidate-HC test uses this shared fixture control"
)]
pub(super) fn assert_mutations_rejected() {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let norm_weight = fixture.encoded_parameters["layers.3.attn_norm.weight"].bf16();
    let expected = case.attention_input.bf16();
    let mut residual = case.block_input.residual.bf16();
    residual.fill(0);
    let mut collapsed = vec![0; WIDTH];
    hc_pre_bf16_reference(
        &residual[..COPIES * WIDTH],
        &[1.0, 0.0],
        WIDTH,
        &mut collapsed,
    )
    .expect("mutated HC pre-mix remains bounded");
    let mut normalized = vec![0; WIDTH];
    rms_norm_bf16_reference(
        &collapsed,
        &norm_weight,
        fixture.model.norm_epsilon,
        &mut normalized,
    )
    .expect("mutated RMSNorm remains bounded");
    assert_ne!(normalized, expected, "zeroed HC residual must not pass");
    assert_ne!(
        attention_input(case, &vec![0; WIDTH], fixture.model.norm_epsilon),
        expected,
        "zeroed RMSNorm weight must not pass"
    );
}
