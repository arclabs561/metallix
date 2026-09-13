//! Fixture-local HC propagation under an explicit target arithmetic model.
//!
//! This compares two observed executions, not every execution reachable from
//! an interval. Their `MoE` outputs and routing IDs must agree exactly before
//! that observed output can narrow the next HC input. No `MoE` Lipschitz or
//! universal backend transcendental guarantee is implied.

use deepseek::{
    ffn::FfnDiagnostic,
    hc::{HcCoefficients, projection::project_hc_diagnostics},
};

use super::hc_coefficient_bounds::coefficient_envelopes;
use super::hc_projection_bounds::normalized_projection_interval_envelopes;
use super::rounding_interval::{bf16_rounded_enclosure, bf16_rounding_cell_intersects};
use super::{Case, Coefficients, Fixture, Tensor};

type Span = [f64; 2];
const U: f64 = 1.0 / 16_777_216.0;

fn point(value: f64) -> Span {
    assert!(value.is_finite());
    [value, value]
}
fn add(a: Span, b: Span) -> Span {
    [(a[0] + b[0]).next_down(), (a[1] + b[1]).next_up()]
}
fn mul(a: Span, b: Span) -> Span {
    let products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
    [
        products
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .next_down(),
        products
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .next_up(),
    ]
}
fn rounded(a: Span) -> Span {
    // Relative RNE error plus half the smallest subnormal covers cancellation.
    let error = (a[0].abs().max(a[1].abs()) * U + 2.0_f64.powi(-150)).next_up();
    let result = [(a[0] - error).next_down(), (a[1] + error).next_up()];
    assert!(
        result
            .iter()
            .all(|x| x.is_finite() && x.abs() <= f64::from(f32::MAX))
    );
    result
}
fn decoded(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
fn points(bits: &[u16]) -> Vec<Span> {
    bits.iter().map(|&b| point(decoded(b))).collect()
}
fn row_bits(tensor: &Tensor, pos: usize, width: usize) -> Vec<u16> {
    tensor.bf16()[pos * width..(pos + 1) * width].to_vec()
}
fn row_floats(tensor: &Tensor, pos: usize, width: usize) -> Vec<f32> {
    tensor.fp32()[pos * width..(pos + 1) * width].to_vec()
}
fn contains(span: Span, value: f32) -> bool {
    value.is_finite() && span[0] <= f64::from(value) && f64::from(value) <= span[1]
}

fn cast_checked(label: &str, spans: &[Span], native: &[u16], source: &[u16]) -> Vec<Span> {
    assert_eq!(spans.len(), native.len());
    assert_eq!(spans.len(), source.len());
    spans
        .iter()
        .zip(native)
        .zip(source)
        .enumerate()
        .map(|(index, ((&span, &native), &source))| {
            for (who, bits) in [("native", native), ("source", source)] {
                assert!(
                    bf16_rounding_cell_intersects(bits, span[0], span[1]).unwrap(),
                    "{label} {who} element {index}: bits={bits} outside {span:?}"
                );
            }
            bf16_rounded_enclosure(span[0], span[1]).unwrap()
        })
        .collect()
}

fn post_mix(sublayer: &[u16], residual: &[Span], post: &[Span], comb: &[Span]) -> Vec<Span> {
    assert_eq!(post.len(), 2);
    assert_eq!(comb.len(), 4);
    assert_eq!(residual.len(), 2 * sublayer.len());
    let width = sublayer.len();
    (0..2 * width)
        .map(|index| {
            let copy = index / width;
            let feature = index % width;
            let left = rounded(mul(comb[copy], residual[feature]));
            let right = rounded(mul(comb[2 + copy], residual[width + feature]));
            let mixed = rounded(add(left, right));
            rounded(add(
                mixed,
                rounded(mul(post[copy], point(decoded(sublayer[feature])))),
            ))
        })
        .collect()
}
fn pre_mix(residual: &[Span], pre: &[Span]) -> Vec<Span> {
    assert_eq!(pre.len(), 2);
    let width = residual.len() / 2;
    (0..width)
        .map(|i| {
            rounded(add(
                rounded(mul(pre[0], residual[i])),
                rounded(mul(pre[1], residual[width + i])),
            ))
        })
        .collect()
}
fn norm(input: &[Span], weight: &[u16], epsilon: f32) -> Vec<Span> {
    assert_eq!(input.len(), weight.len());
    assert!(!input.is_empty());
    let width = f64::from(u32::try_from(input.len()).expect("bounded reference width"));
    let mut squares = point(0.0);
    for &x in input {
        let square = if x[0] <= 0.0 && x[1] >= 0.0 {
            [0.0, (x[0] * x[0]).max(x[1] * x[1]).next_up()]
        } else {
            mul(x, x)
        };
        squares = add(squares, square);
    }
    let nu = 2.0 * width * U;
    assert!(nu < 1.0);
    let gamma = (nu / (1.0 - nu)).next_up();
    // Every square/add contributes at most half a minimum subnormal in the
    // gradual-underflow target model, amplified by the remaining operations.
    // This absolute term is necessary even when the exact square is tiny.
    let underflow = ((2.0 * width * 2.0_f64.powi(-150)) / (1.0 - nu)).next_up();
    let error = (gamma * squares[1].abs() + underflow).next_up();
    squares = [
        (squares[0] - error).next_down().max(0.0),
        (squares[1] + error).next_up(),
    ];
    assert!(
        squares[1] <= f64::from(f32::MAX),
        "square reduction may overflow"
    );
    let mean = rounded(mul(squares, point(1.0 / width)));
    let variance = rounded(add(mean, point(f64::from(epsilon))));
    assert!(variance[0] > 0.0);
    let inverse = [
        ((1.0 / variance[1].sqrt()).next_down() * (1.0 - U) / (1.0 + U)).next_down(),
        ((1.0 / variance[0].sqrt()).next_up() * (1.0 + U) / (1.0 - U)).next_up(),
    ];
    input
        .iter()
        .zip(weight)
        .map(|(&x, &w)| rounded(mul(rounded(mul(x, inverse)), point(decoded(w)))))
        .collect()
}

#[test]
fn norm_envelope_covers_subnormal_square_rounding() {
    let input = [0x1a81, 0x1a83];
    let weight = [0x3f80; 2];
    let epsilon = f32::MIN_POSITIVE;
    let mut actual = [0; 2];
    deepseek::rms_norm_bf16_reference(&input, &weight, epsilon, &mut actual).unwrap();
    let bound = norm(&points(&input), &weight, epsilon);
    for (&bits, span) in actual.iter().zip(bound) {
        assert!(bf16_rounding_cell_intersects(bits, span[0], span[1]).unwrap());
    }
}

#[allow(clippy::too_many_arguments)]
fn coefficients(
    fixture: &Fixture,
    prefix: &str,
    residual: &[Span],
    native_residual: &[u16],
    native_coefficients: &HcCoefficients,
    source_mixes: &[f32],
    source: &Coefficients,
    pos: usize,
) -> (Vec<Span>, Vec<Span>, Vec<Span>) {
    let c = &fixture.block_config;
    assert_eq!(c.copies, 2, "bounded reduced-graph contract");
    let projection = fixture.block_parameters[&format!("layers.4.hc_{prefix}_fn")].fp32();
    let scale: [f32; 3] = fixture.block_parameters[&format!("layers.4.hc_{prefix}_scale")]
        .fp32()
        .try_into()
        .unwrap();
    let base = fixture.block_parameters[&format!("layers.4.hc_{prefix}_base")].fp32();
    let observed = project_hc_diagnostics(
        native_residual,
        &projection,
        &scale,
        &base,
        2,
        c.norm_eps,
        c.hc_sinkhorn_iters,
        c.hc_eps,
    )
    .unwrap();
    assert_eq!(
        observed.coefficients(),
        native_coefficients,
        "diagnostics must observe actual arithmetic"
    );
    let envelopes =
        normalized_projection_interval_envelopes(residual, &projection, c.norm_eps).unwrap();
    assert_eq!(source_mixes.len(), envelopes.len());
    for (i, ((&native, &source), span)) in observed
        .mixes()
        .iter()
        .zip(source_mixes)
        .zip(&envelopes)
        .enumerate()
    {
        assert!(
            span.contains(native) && span.contains(source),
            "{prefix} mix {i}: native={native} source={source} outside {span:?}"
        );
    }
    let spans: Vec<Span> = envelopes.iter().map(|s| [s.lo, s.hi]).collect();
    let bound =
        coefficient_envelopes(&spans, &scale, &base, c.hc_sinkhorn_iters, c.hc_eps).unwrap();
    for (role, spans, native, source) in [
        (
            "pre",
            &bound.pre,
            native_coefficients.pre(),
            row_floats(&source.pre, pos, 2),
        ),
        (
            "post",
            &bound.post,
            native_coefficients.post(),
            row_floats(&source.post, pos, 2),
        ),
        (
            "comb",
            &bound.comb,
            native_coefficients.comb(),
            row_floats(&source.comb, pos, 4),
        ),
    ] {
        assert_eq!(spans.len(), native.len());
        for (i, ((&span, &native), source)) in spans.iter().zip(native).zip(source).enumerate() {
            assert!(
                contains(span, native) && contains(span, source),
                "{prefix} {role}[{i}]: native={native} source={source} outside {span:?}"
            );
        }
    }
    (bound.pre, bound.post, bound.comb)
}

pub(super) fn check_position(
    fixture: &Fixture,
    case: &Case,
    pos: usize,
    attention_coefficients: &HcCoefficients,
    after_attention: &[u16],
    result: &FfnDiagnostic,
) {
    let block_input = row_bits(&case.block_input, pos, 256);
    let residual = points(&block_input);
    let (attn_pre, attn_post, attn_comb) = coefficients(
        fixture,
        "attn",
        &residual,
        &block_input,
        attention_coefficients,
        &row_floats(&case.attention_hc_mixes, pos, 8),
        &case.attention_coefficients,
        pos,
    );
    let attention_output = row_bits(&case.attention_output, pos, 128);
    let mixed = post_mix(&attention_output, &residual, &attn_post, &attn_comb);
    let residual = cast_checked(
        "attention residual",
        &mixed,
        after_attention,
        &row_bits(&case.after_attention_residual, pos, 256),
    );
    let (next_pre, ffn_post, ffn_comb) = coefficients(
        fixture,
        "ffn",
        &residual,
        after_attention,
        result.coefficients(),
        &row_floats(&case.ffn_hc_mixes, pos, 8),
        &case.ffn_coefficients,
        pos,
    );
    let collapse = cast_checked(
        "FFN collapse",
        &pre_mix(&residual, &attn_pre),
        result.collapsed_bf16(),
        &row_bits(&case.ffn_collapsed, pos, 128),
    );
    let norm_weight = fixture.block_parameters["layers.4.ffn_norm.weight"].bf16();
    cast_checked(
        "FFN norm",
        &norm(&collapse, &norm_weight, fixture.block_config.norm_eps),
        result.normalized_bf16(),
        &row_bits(&case.input, pos, 128),
    );
    let source_moe = row_bits(&case.output, pos, 128);
    assert_eq!(
        result.moe().output_bf16(),
        source_moe,
        "MoE checkpoint must agree exactly before narrowing propagated uncertainty"
    );
    let mut ids = case.gate_indices.indices()
        [pos * fixture.model.n_activated_experts..(pos + 1) * fixture.model.n_activated_experts]
        .to_vec();
    ids.sort_unstable();
    assert_eq!(
        result
            .moe()
            .routes()
            .iter()
            .map(|r| r.expert_index())
            .collect::<Vec<_>>(),
        ids,
        "observed route IDs remain exact"
    );
    let mixed = post_mix(&source_moe, &residual, &ffn_post, &ffn_comb);
    cast_checked(
        "block output",
        &mixed,
        result.output_bf16(),
        &row_bits(&case.block_output, pos, 256),
    );
    for (&span, source) in next_pre
        .iter()
        .zip(row_floats(&case.block_next_pre, pos, 2))
    {
        assert!(
            contains(span, source),
            "returned pre outside propagated envelope"
        );
    }
    // Zeroing this nonzero output must not satisfy the arithmetic envelope.
    assert!(
        mixed
            .iter()
            .any(|s| !bf16_rounding_cell_intersects(0, s[0], s[1]).unwrap()),
        "output envelope must reject an omitted nonzero block"
    );
}
