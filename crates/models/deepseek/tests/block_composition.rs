//! Reduced, test-only `Block.forward` sequencing gate.
//!
//! This composes the bounded scalar HC pre/post and `RMSNorm` primitives around
//! deterministic attention/FFN stubs. It proves coefficient handoff and
//! residual sequencing only; it is not attention, `MoE`, checkpoint, Metal, or
//! full-graph parity.

use deepseek::{
    RmsNormError,
    hc::mixing::{HcMixError, hc_post_bf16_reference, hc_pre_bf16_reference},
    rms_norm_bf16_reference,
};
use serde_json::Value;

const COPIES: usize = 2;
const WIDTH: usize = 2;
const NORM_WEIGHT: [u16; WIDTH] = [0x3f80; WIDTH];
const NORM_EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy)]
struct Mix<'a> {
    pre: &'a [f32],
    post: &'a [f32],
    comb: &'a [f32],
}

#[derive(Debug, thiserror::Error)]
enum ReducedBlockError {
    #[error(transparent)]
    Hc(#[from] HcMixError),
    #[error(transparent)]
    Norm(#[from] RmsNormError),
}

struct BlockTrace<'a> {
    attention_input: [u16; WIDTH],
    ffn_input: [u16; WIDTH],
    output: Vec<u16>,
    next_pre: &'a [f32],
}

/// Mirrors only the ordering in `Block.forward`:
///
/// 1. caller-owned previous-FFN `pre` collapses attention input;
/// 2. current attention `pre` collapses FFN input;
/// 3. current FFN `pre` becomes the next block's caller-owned input.
///
/// The two sublayer functions are intentionally deterministic BF16 stubs so
/// this gate has no claim about either real sublayer's implementation.
fn reduced_block<'a>(
    residual: &[u16],
    incoming_pre: &[f32],
    attention_mix: Mix<'a>,
    ffn_mix: Mix<'a>,
) -> Result<BlockTrace<'a>, ReducedBlockError> {
    let attention_input = collapse_and_norm(residual, incoming_pre)?;
    let attention_output = stub_attention(attention_input);
    let mut after_attention = vec![0_u16; residual.len()];
    hc_post_bf16_reference(
        &attention_output,
        residual,
        attention_mix.post,
        attention_mix.comb,
        &mut after_attention,
    )?;

    let ffn_input = collapse_and_norm(&after_attention, attention_mix.pre)?;
    let ffn_output = stub_ffn(ffn_input);
    let mut output = vec![0_u16; residual.len()];
    hc_post_bf16_reference(
        &ffn_output,
        &after_attention,
        ffn_mix.post,
        ffn_mix.comb,
        &mut output,
    )?;

    Ok(BlockTrace {
        attention_input,
        ffn_input,
        output,
        next_pre: ffn_mix.pre,
    })
}

fn collapse_and_norm(residual: &[u16], pre: &[f32]) -> Result<[u16; WIDTH], ReducedBlockError> {
    let mut collapsed = [0_u16; WIDTH];
    hc_pre_bf16_reference(residual, pre, WIDTH, &mut collapsed)?;
    let mut normalized = [0_u16; WIDTH];
    rms_norm_bf16_reference(&collapsed, &NORM_WEIGHT, NORM_EPSILON, &mut normalized)?;
    Ok(normalized)
}

/// Explicit deterministic stand-in for attention: swap normalized features.
fn stub_attention(input: [u16; WIDTH]) -> [u16; WIDTH] {
    [input[1], input[0]]
}

/// Explicit deterministic stand-in for the FFN: preserve normalized features.
fn stub_ffn(input: [u16; WIDTH]) -> [u16; WIDTH] {
    input
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/block-reference.json"
    ))
    .expect("checked-in source block fixture")
}

fn tensor<'a>(fixture: &'a Value, name: &str) -> &'a Value {
    fixture["tensors"]
        .as_array()
        .expect("fixture tensors")
        .iter()
        .find(|candidate| candidate["name"] == name)
        .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
}

fn bf16_values(tensor: &Value) -> Vec<u16> {
    tensor["values"]
        .as_array()
        .expect("BF16 tensor values")
        .iter()
        .map(|value| u16::try_from(value.as_u64().expect("BF16 word")).expect("BF16 word range"))
        .collect()
}

fn f32_values(tensor: &Value) -> Vec<f32> {
    tensor["values"]
        .as_array()
        .expect("FP32 tensor values")
        .iter()
        .map(|value| {
            let bits = u32::try_from(value.as_u64().expect("FP32 word")).expect("FP32 word range");
            f32::from_bits(bits)
        })
        .collect()
}

#[test]
fn source_fixture_identifies_the_qualified_revision_and_methods() {
    let fixture = fixture();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["receipt"]["device"], "cpu");
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["sha256"],
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture["source"]["symbols"],
        serde_json::json!([
            "Block.forward",
            "Block.hc_pre",
            "Block.hc_post",
            "RMSNorm.forward"
        ])
    );
}

#[test]
fn reduced_harness_matches_pinned_source_block_sequence() {
    let fixture = fixture();
    let block0_input = bf16_values(tensor(&fixture, "initial"));
    let previous_ffn_pre = f32_values(tensor(&fixture, "previous_ffn_pre"));
    let block0_attn_pre = f32_values(tensor(&fixture, "block0_attn_pre"));
    let block0_attn_post = f32_values(tensor(&fixture, "block0_attn_post"));
    let block0_attn_comb = f32_values(tensor(&fixture, "block0_attn_comb"));
    let block0_ffn_pre = f32_values(tensor(&fixture, "block0_ffn_pre"));
    let block0_ffn_post = f32_values(tensor(&fixture, "block0_ffn_post"));
    let block0_ffn_comb = f32_values(tensor(&fixture, "block0_ffn_comb"));
    let block0_attention = Mix {
        pre: &block0_attn_pre,
        post: &block0_attn_post,
        comb: &block0_attn_comb,
    };
    let block0_ffn = Mix {
        pre: &block0_ffn_pre,
        post: &block0_ffn_post,
        comb: &block0_ffn_comb,
    };
    let block1_attn_pre = f32_values(tensor(&fixture, "block1_attn_pre"));
    let block1_attn_post = f32_values(tensor(&fixture, "block1_attn_post"));
    let block1_attn_comb = f32_values(tensor(&fixture, "block1_attn_comb"));
    let block1_ffn_pre = f32_values(tensor(&fixture, "block1_ffn_pre"));
    let block1_ffn_post = f32_values(tensor(&fixture, "block1_ffn_post"));
    let block1_ffn_comb = f32_values(tensor(&fixture, "block1_ffn_comb"));
    let block1_attention = Mix {
        pre: &block1_attn_pre,
        post: &block1_attn_post,
        comb: &block1_attn_comb,
    };
    let block1_ffn = Mix {
        pre: &block1_ffn_pre,
        post: &block1_ffn_post,
        comb: &block1_ffn_comb,
    };

    let block0 = reduced_block(
        &block0_input,
        &previous_ffn_pre,
        block0_attention,
        block0_ffn,
    )
    .expect("bounded first reduced block");

    assert_eq!(
        block0.attention_input.as_slice(),
        bf16_values(tensor(&fixture, "block0_attn_norm_input"))
    );
    assert_eq!(
        block0.ffn_input.as_slice(),
        bf16_values(tensor(&fixture, "block0_ffn_norm_input"))
    );
    assert_eq!(
        block0.output,
        bf16_values(tensor(&fixture, "block0_output"))
    );
    assert_eq!(
        block0.next_pre,
        f32_values(tensor(&fixture, "block0_next_pre"))
    );

    let block1 = reduced_block(
        &block0.output,
        block0.next_pre,
        block1_attention,
        block1_ffn,
    )
    .expect("bounded second reduced block");
    assert_eq!(
        block1.attention_input.as_slice(),
        bf16_values(tensor(&fixture, "block1_attn_norm_input"))
    );
    assert_eq!(
        block1.ffn_input.as_slice(),
        bf16_values(tensor(&fixture, "block1_ffn_norm_input"))
    );
    assert_eq!(
        block1.output,
        bf16_values(tensor(&fixture, "block1_output"))
    );
    assert_eq!(
        block1.next_pre,
        f32_values(tensor(&fixture, "block1_next_pre"))
    );
}

#[test]
fn using_the_previous_mix_for_ffn_is_distinguished_from_current_attention_pre() {
    let residual = [0x4000, 0xbf80, 0x40c0, 0x4040];
    let incoming_pre = [0.25, 0.5];
    let attention = Mix {
        pre: &[0.75, 0.125],
        post: &[1.0, 0.5],
        comb: &[1.0, 0.0, 0.0, 1.0],
    };
    let ffn = Mix {
        pre: &[0.625, 0.375],
        post: &[0.75, 1.25],
        comb: &[0.0, 1.0, 1.0, 0.0],
    };
    let correct =
        reduced_block(&residual, &incoming_pre, attention, ffn).expect("correct reduced ordering");

    let mut after_attention = vec![0_u16; COPIES * WIDTH];
    hc_post_bf16_reference(
        &stub_attention(correct.attention_input),
        &residual,
        attention.post,
        attention.comb,
        &mut after_attention,
    )
    .expect("bounded attention expansion");
    let wrong_ffn_input = collapse_and_norm(&after_attention, &incoming_pre)
        .expect("deliberately stale pre still has valid shape");
    let mut wrong_output = vec![0_u16; COPIES * WIDTH];
    hc_post_bf16_reference(
        &stub_ffn(wrong_ffn_input),
        &after_attention,
        ffn.post,
        ffn.comb,
        &mut wrong_output,
    )
    .expect("deliberately stale but finite FFN expansion");

    assert_ne!(correct.ffn_input, wrong_ffn_input);
    assert_ne!(correct.ffn_input, correct.attention_input);
    assert_ne!(correct.output, wrong_output);
}

#[test]
fn reduced_harness_propagates_hc_pre_shape_errors() {
    let attention = Mix {
        pre: &[0.75, 0.125],
        post: &[1.0, 0.5],
        comb: &[1.0, 0.0, 0.0, 1.0],
    };
    let ffn = Mix {
        pre: &[0.625, 0.375],
        post: &[0.75, 1.25],
        comb: &[0.0, 1.0, 1.0, 0.0],
    };
    let Err(error) = reduced_block(&[0x3f80; COPIES * WIDTH], &[1.0], attention, ffn) else {
        panic!("one-copy pre cannot consume two-copy residual");
    };
    assert!(matches!(
        error,
        ReducedBlockError::Hc(HcMixError::Length {
            field: "residual",
            ..
        })
    ));
}
