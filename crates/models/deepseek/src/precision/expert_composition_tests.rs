//! Test-only scalar composition for the pinned routed `Expert` ordering.
//!
//! This checks BF16 boundaries around the existing FP8-activation/FP4-linear
//! references. It is neither a hardware kernel oracle nor a full `MoE` test.

use super::{
    ActivationGroup, fp4_linear_runtime_f32, fp8_linear_runtime_f32,
    quantize_bf16_activations_e4m3fn,
};

const WIDTH: usize = 32;
const PACKED_WIDTH: usize = WIDTH / 2;

#[derive(Clone, Copy)]
enum Variant {
    Pinned,
    RouteAfterW2,
    RouteAfterHiddenBf16Cast,
    WithoutLinearBf16Casts,
    SymmetricGateClamp,
    UpperOnlyUpClamp,
}

struct Trace {
    gate_after_clamp: f32,
    up_after_clamp: f32,
    hidden_before_w2: f32,
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("an FP32 high half always fits BF16 storage")
}

fn bf16_value(value: f32) -> f32 {
    bf16_from_bits(bf16_bits(value))
}

fn bf16_from_bits(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_row(values: &[f32; WIDTH]) -> [u16; WIDTH] {
    let mut row = [0_u16; WIDTH];
    for (destination, &value) in row.iter_mut().zip(values) {
        *destination = bf16_bits(value);
    }
    row
}

fn project(input: &[u16; WIDTH], weights: &[u8; WIDTH * PACKED_WIDTH]) -> [f32; WIDTH] {
    let mut activation_codes = [0_u8; WIDTH];
    let mut activation_scales = [0_u8; 1];
    quantize_bf16_activations_e4m3fn(
        input,
        1,
        WIDTH,
        ActivationGroup::Elements32,
        &mut activation_codes,
        &mut activation_scales,
    )
    .expect("finite complete scalar expert input");
    let mut output = [0.0_f32; WIDTH];
    fp4_linear_runtime_f32(
        &activation_codes,
        &activation_scales,
        weights,
        &[127; WIDTH],
        1,
        WIDTH,
        WIDTH,
        ActivationGroup::Elements32,
        &mut output,
    )
    .expect("finite complete scalar expert projection");
    output
}

fn routed_expert(
    input: &[u16; WIDTH],
    w1: &[u8; WIDTH * PACKED_WIDTH],
    w2: &[u8; WIDTH * PACKED_WIDTH],
    w3: &[u8; WIDTH * PACKED_WIDTH],
    swiglu_limit: f32,
    route_weight: f32,
    variant: Variant,
) -> ([f32; WIDTH], Trace) {
    let gate_raw = project(input, w1);
    let up_raw = project(input, w3);
    let cast_linears = !matches!(variant, Variant::WithoutLinearBf16Casts);
    let mut gate = gate_raw.map(|value| {
        if cast_linears {
            bf16_value(value)
        } else {
            value
        }
    });
    let mut up = up_raw.map(|value| {
        if cast_linears {
            bf16_value(value)
        } else {
            value
        }
    });
    if swiglu_limit > 0.0 {
        for value in &mut up {
            *value = if matches!(variant, Variant::UpperOnlyUpClamp) {
                value.min(swiglu_limit)
            } else {
                value.clamp(-swiglu_limit, swiglu_limit)
            };
        }
        for value in &mut gate {
            *value = if matches!(variant, Variant::SymmetricGateClamp) {
                value.clamp(-swiglu_limit, swiglu_limit)
            } else {
                value.min(swiglu_limit)
            };
        }
    }
    let mut hidden = [0.0_f32; WIDTH];
    for ((destination, &gate), &up) in hidden.iter_mut().zip(&gate).zip(&up) {
        *destination = (gate / (1.0 + (-gate).exp())) * up;
        if !matches!(
            variant,
            Variant::RouteAfterW2 | Variant::RouteAfterHiddenBf16Cast
        ) {
            *destination *= route_weight;
        }
    }
    let trace = Trace {
        gate_after_clamp: gate[0],
        up_after_clamp: up[0],
        hidden_before_w2: hidden[0],
    };
    let hidden = bf16_row(&hidden);
    let hidden = if matches!(variant, Variant::RouteAfterHiddenBf16Cast) {
        let routed = hidden.map(|bits| bf16_from_bits(bits) * route_weight);
        bf16_row(&routed)
    } else {
        hidden
    };
    let mut output = project(&hidden, w2).map(bf16_value);
    if matches!(variant, Variant::RouteAfterW2) {
        for value in &mut output {
            *value *= route_weight;
        }
    }
    (output, trace)
}

fn uniform_weight(code: u8) -> [u8; WIDTH * PACKED_WIDTH] {
    [code; WIDTH * PACKED_WIDTH]
}

fn assert_bits_eq(actual: &[f32; WIDTH], expected: f32) {
    assert!(
        actual
            .iter()
            .all(|value| value.to_bits() == expected.to_bits()),
        "actual scalar routed-expert output was {actual:?}, expected every lane {expected:?}"
    );
}

#[test]
fn routed_weight_precedes_bf16_hidden_cast_and_w2() {
    let input = [0x3f80_u16; WIDTH]; // BF16 1.0
    let w1 = uniform_weight(0x11); // every logical FP4 lane is +0.5
    let w2 = uniform_weight(0x11);
    let w3 = uniform_weight(0x11);

    let (pinned, trace) = routed_expert(&input, &w1, &w2, &w3, 4.0, 0.3, Variant::Pinned);
    // Independent scalar stages: each first projection is 16, both clamps make
    // it 4, and route-weighted SwiGLU rounds to BF16 before W2. Its FP8/FP4
    // scalar reconstruction is 4.5, so 32 lanes of +0.5 produce 72.
    assert_eq!(trace.gate_after_clamp.to_bits(), 4.0_f32.to_bits());
    assert_eq!(trace.up_after_clamp.to_bits(), 4.0_f32.to_bits());
    assert_bits_eq(&pinned, 72.0);

    let (route_after_w2, _) = routed_expert(&input, &w1, &w2, &w3, 4.0, 0.3, Variant::RouteAfterW2);
    assert_ne!(route_after_w2[0].to_bits(), pinned[0].to_bits());
}

#[test]
fn routed_weight_precedes_the_hidden_bf16_boundary() {
    let input = [0x3f80_u16; WIDTH];
    let weights = uniform_weight(0x11);
    let (pinned, _) = routed_expert(
        &input,
        &weights,
        &weights,
        &weights,
        4.0,
        0.1995,
        Variant::Pinned,
    );
    // `bf16(0.1995 * silu(4) * 4)` is 3.140625. Moving route weighting
    // after the first BF16 rounding gives 3.125. The next FP8 boundary maps
    // those values to 3.25 and 3.0, respectively.
    assert_bits_eq(&pinned, 52.0);

    let (route_after_hidden_cast, _) = routed_expert(
        &input,
        &weights,
        &weights,
        &weights,
        4.0,
        0.1995,
        Variant::RouteAfterHiddenBf16Cast,
    );
    assert_bits_eq(&route_after_hidden_cast, 48.0);
    assert_ne!(route_after_hidden_cast[0].to_bits(), pinned[0].to_bits());
}

#[test]
fn bf16_casts_after_w1_and_w3_change_the_quantized_w2_input() {
    let mut input = [0x3ff0_u16; WIDTH]; // BF16 1.875
    input[WIDTH - 1] = 0x3fe0; // BF16 1.75
    let weights = uniform_weight(0x11);

    let (pinned, trace) = routed_expert(
        &input,
        &weights,
        &weights,
        &weights,
        0.0,
        0.609,
        Variant::Pinned,
    );
    // The independent uncast dot is 29.9375. The pinned BF16 boundary turns it
    // into 30, which changes the subsequent BF16/FP8 boundary before W2.
    assert_eq!(trace.gate_after_clamp.to_bits(), 30.0_f32.to_bits());
    assert_eq!(trace.up_after_clamp.to_bits(), 30.0_f32.to_bits());
    assert_bits_eq(&pinned, 9_216.0);

    let (without_casts, without_casts_trace) = routed_expert(
        &input,
        &weights,
        &weights,
        &weights,
        0.0,
        0.609,
        Variant::WithoutLinearBf16Casts,
    );
    assert_eq!(
        without_casts_trace.gate_after_clamp.to_bits(),
        29.9375_f32.to_bits()
    );
    assert_bits_eq(&without_casts, 8_192.0);
    assert_ne!(without_casts[0].to_bits(), pinned[0].to_bits());
}

#[test]
fn gate_clamp_is_upper_only_while_up_clamp_is_symmetric() {
    let input = [0x3f80_u16; WIDTH];
    let negative_gate = uniform_weight(0x99); // every logical FP4 lane is -0.5
    let positive_up_and_down = uniform_weight(0x11);

    let (pinned, trace) = routed_expert(
        &input,
        &negative_gate,
        &positive_up_and_down,
        &positive_up_and_down,
        4.0,
        1.0,
        Variant::Pinned,
    );
    assert_eq!(trace.gate_after_clamp.to_bits(), (-16.0_f32).to_bits());
    assert_eq!(trace.up_after_clamp.to_bits(), 4.0_f32.to_bits());
    assert!(trace.hidden_before_w2.is_sign_negative());

    let (symmetric_gate, symmetric_trace) = routed_expert(
        &input,
        &negative_gate,
        &positive_up_and_down,
        &positive_up_and_down,
        4.0,
        1.0,
        Variant::SymmetricGateClamp,
    );
    assert_eq!(
        symmetric_trace.gate_after_clamp.to_bits(),
        (-4.0_f32).to_bits()
    );
    assert_ne!(symmetric_gate[0].to_bits(), pinned[0].to_bits());
}

#[test]
fn up_clamp_has_a_lower_bound_while_gate_clamp_does_not() {
    let input = [0x3f80_u16; WIDTH];
    let positive_gate_and_down = uniform_weight(0x11);
    let negative_up = uniform_weight(0x99);
    let (pinned, trace) = routed_expert(
        &input,
        &positive_gate_and_down,
        &positive_gate_and_down,
        &negative_up,
        4.0,
        1.0,
        Variant::Pinned,
    );
    assert_eq!(trace.gate_after_clamp.to_bits(), 4.0_f32.to_bits());
    assert_eq!(trace.up_after_clamp.to_bits(), (-4.0_f32).to_bits());

    let (upper_only_up, upper_only_trace) = routed_expert(
        &input,
        &positive_gate_and_down,
        &positive_gate_and_down,
        &negative_up,
        4.0,
        1.0,
        Variant::UpperOnlyUpClamp,
    );
    assert_eq!(
        upper_only_trace.up_after_clamp.to_bits(),
        (-16.0_f32).to_bits()
    );
    assert_ne!(upper_only_up[0].to_bits(), pinned[0].to_bits());
}

// Shared experts use FP8 weights and two-dimensional weight-scale blocks,
// unlike the routed FP4 experts above. This fixture has one 32x32 scale block.
fn shared_projection(input: &[u16; WIDTH]) -> [f32; WIDTH] {
    let mut codes = [0_u8; WIDTH];
    let mut scales = [0_u8; 1];
    quantize_bf16_activations_e4m3fn(
        input,
        1,
        WIDTH,
        ActivationGroup::Elements32,
        &mut codes,
        &mut scales,
    )
    .expect("finite shared-expert activation");
    let mut output = [0.0; WIDTH];
    fp8_linear_runtime_f32(
        &codes,
        &scales,
        &[0x30; WIDTH * WIDTH], // E4M3FN +0.5
        &[127],                 // one unit scale for the whole weight tile
        1,
        WIDTH,
        WIDTH,
        ActivationGroup::Elements32,
        &mut output,
    )
    .expect("finite shared-expert projection");
    output.map(bf16_value)
}

#[test]
fn fp4_routed_and_fp8_shared_outputs_join_before_final_bf16_cast() {
    let input = [0x3f80_u16; WIDTH];
    let weights = uniform_weight(0x11);
    let (routed, _) = routed_expert(
        &input,
        &weights,
        &weights,
        &weights,
        4.0,
        0.3,
        Variant::Pinned,
    );
    let gate = shared_projection(&input);
    let up = shared_projection(&input);
    assert_bits_eq(&gate, 16.0);
    assert_bits_eq(&up, 16.0);
    let mut hidden = [0.0; WIDTH];
    for ((value, gate), up) in hidden.iter_mut().zip(gate).zip(up) {
        let gate = gate.min(4.0);
        let up = up.clamp(-4.0, 4.0);
        *value = (gate / (1.0 + (-gate).exp())) * up;
    }
    let shared = shared_projection(&bf16_row(&hidden));
    // Unweighted hidden rounds/requantizes to 16; 32 products with 0.5
    // yield 256. The routed branch independently contributes 72.
    assert_bits_eq(&shared, 256.0);
    assert_bits_eq(&routed, 72.0);
    let combined = std::array::from_fn(|i| bf16_value(routed[i] + shared[i]));
    assert_bits_eq(&combined, 328.0);
    // Shared contribution is added once and is not multiplied by route weight.
    let incorrectly_routed_shared = bf16_value(routed[0] + 0.3 * shared[0]);
    assert_ne!(combined[0].to_bits(), incorrectly_routed_shared.to_bits());
}

#[test]
fn two_flash_selected_fp4_experts_and_one_fp8_shared_expert_compose() {
    // This supplies logits directly. It qualifies the text-route/expert join,
    // not the gate projection or a complete `MoE` block.
    let routes =
        crate::flash_sqrt_softplus_routes(&[0.0, 1.0, 3.0], &[0.0, 0.0, -10.0], 2, 1.0, true, 1.0)
            .expect("finite distinct selected expert scores");
    assert_eq!(
        routes
            .iter()
            .map(|route| route.expert_index())
            .collect::<Vec<_>>(),
        [0, 1]
    );

    // Independent scalar route arithmetic keeps the correction bias out of
    // each gathered route weight, then normalizes the two selected scores.
    let score1 = (1.0_f32.exp().ln_1p()).sqrt();
    let score0 = (0.0_f32.exp().ln_1p()).sqrt();
    let denominator = score0 + score1 + 1.0e-20_f32;
    assert_eq!(
        routes[0].weight().to_bits(),
        (score0 / denominator).to_bits()
    );
    assert_eq!(
        routes[1].weight().to_bits(),
        (score1 / denominator).to_bits()
    );

    let input = [0x3f80_u16; WIDTH];
    let expert0_weights = uniform_weight(0x11); // FP4 +0.5
    let expert1_weights = uniform_weight(0x22); // FP4 +1.0
    let (expert0, _) = routed_expert(
        &input,
        &expert0_weights,
        &expert0_weights,
        &expert0_weights,
        4.0,
        routes[0].weight(),
        Variant::Pinned,
    );
    let (expert1, _) = routed_expert(
        &input,
        &expert1_weights,
        &expert1_weights,
        &expert1_weights,
        4.0,
        routes[1].weight(),
        Variant::Pinned,
    );
    let shared_gate = shared_projection(&input);
    let shared_up = shared_projection(&input);
    let mut shared_hidden = [0.0_f32; WIDTH];
    for ((destination, &gate), &up) in shared_hidden.iter_mut().zip(&shared_gate).zip(&shared_up) {
        *destination = (gate.min(4.0) / (1.0 + (-gate.min(4.0)).exp())) * up.clamp(-4.0, 4.0);
    }
    let shared = shared_projection(&bf16_row(&shared_hidden));

    // The selected weights are about 0.421 and 0.579. After SwiGLU,
    // BF16 and FP8 rounding, W2 sees 6.5 and 9 respectively. Thus the
    // distinct down projections give 32*0.5*6.5=104 and 32*1*9=288.
    assert_bits_eq(&expert0, 104.0);
    assert_bits_eq(&expert1, 288.0);
    assert_bits_eq(&shared, 256.0);
    let combined =
        std::array::from_fn(|index| bf16_value(expert0[index] + expert1[index] + shared[index]));
    assert_bits_eq(&combined, 648.0);
    assert_ne!(
        combined[0].to_bits(),
        bf16_value(expert0[0] + shared[0]).to_bits()
    );
    assert_ne!(
        combined[0].to_bits(),
        bf16_value(expert1[0] + shared[0]).to_bits()
    );
    assert_ne!(
        combined[0].to_bits(),
        bf16_value(expert0[0] + expert1[0]).to_bits()
    );
}
