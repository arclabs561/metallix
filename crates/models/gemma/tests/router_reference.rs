//! Router-only qualification against actual pinned Transformers module outputs.
use gemma::routing::{Gemma4MoeGeometry, Gemma4RouterError};
#[cfg(feature = "metal")]
use serde::Deserialize;

#[cfg(feature = "metal")]
#[derive(Deserialize)]
struct Reference {
    hidden: usize,
    experts: usize,
    epsilon: f32,
    cases: Vec<Case>,
}
#[cfg(feature = "metal")]
#[derive(Deserialize)]
struct Case {
    top_k: usize,
    input: Vec<Vec<f32>>,
    scale: Vec<f32>,
    projection: Vec<Vec<f32>>,
    expert_scale: Vec<f32>,
    indices: Vec<Vec<usize>>,
    weights: Vec<Vec<f32>>,
}
#[cfg(feature = "metal")]
fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../../fixtures/gemma-4-tiny/router-reference.json"
    ))
    .unwrap()
}

#[cfg(feature = "metal")]
mod metal {
    use mlx_rs::Array;

    /// MLX's native test operations share process-global device initialization.
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub fn lock() -> std::sync::MutexGuard<'static, ()> {
        GPU.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn array(values: &[f32], shape: &[usize]) -> Array {
        let shape: Vec<i32> = shape.iter().map(|&d| i32::try_from(d).unwrap()).collect();
        Array::from_slice(values, &shape)
    }

    /// Routes `rows` `[n, H]` through gemma's router and returns each row's
    /// `(expert, weight)` pairs, largest weight first.
    pub fn route(
        rows: &[Vec<f32>],
        scale: &[f32],
        projection: &[f32],
        expert_scale: &[f32],
        top_k: usize,
        epsilon: f32,
    ) -> Vec<Vec<(usize, f32)>> {
        let hidden = scale.len();
        let experts = expert_scale.len();
        let input: Vec<f32> = rows.iter().flatten().copied().collect();
        let routing = gemma::routing::route(
            &array(&input, &[rows.len(), hidden]),
            &array(scale, &[hidden]),
            &array(projection, &[experts, hidden]),
            &array(expert_scale, &[experts]),
            top_k,
            epsilon,
        )
        .unwrap();
        // The top-k slice of the sort is strided; read row-contiguous copies.
        let chosen = routing.chosen.contiguous().unwrap();
        let weights = routing.weights.contiguous().unwrap();
        chosen.eval().unwrap();
        weights.eval().unwrap();
        chosen
            .as_slice::<u32>()
            .chunks(top_k)
            .zip(weights.as_slice::<f32>().chunks(top_k))
            .map(|(chosen, weights)| {
                chosen
                    .iter()
                    .map(|&e| usize::try_from(e).unwrap())
                    .zip(weights.iter().copied())
                    .collect()
            })
            .collect()
    }
}

#[cfg(feature = "metal")]
#[test]
fn router_matches_pinned_cpu_module() {
    let _gpu = metal::lock();
    let reference = reference();
    assert_eq!(reference.cases.len(), 12);
    for case in &reference.cases {
        let projection: Vec<f32> = case.projection.iter().flatten().copied().collect();
        assert_eq!(case.input.len(), 9);
        let routes = metal::route(
            &case.input,
            &case.scale,
            &projection,
            &case.expert_scale,
            case.top_k,
            reference.epsilon,
        );
        for ((actual, indices), weights) in routes.iter().zip(&case.indices).zip(&case.weights) {
            assert_eq!(actual.len(), case.top_k);
            for ((&(expert, weight), index), source) in actual.iter().zip(indices).zip(weights) {
                assert_eq!(expert, *index);
                assert!(weight.is_finite() && source.is_finite());
                assert!(
                    (weight - source).abs() <= 1e-6,
                    "native={weight} source={source}"
                );
            }
        }
    }
}

#[cfg(feature = "metal")]
#[test]
fn nonunit_scales_are_observable_in_source_cases() {
    let _gpu = metal::lock();
    let reference = reference();
    let mut dropping_router_scale_changes = false;
    let mut dropping_expert_scale_changes = false;
    for case in reference.cases.iter().filter(|c| c.top_k == 3) {
        let projection: Vec<_> = case.projection.iter().flatten().copied().collect();
        let ones_hidden = vec![1.0; reference.hidden];
        let ones_experts = vec![1.0; reference.experts];
        for (scale, expert_scale, changed) in [
            (
                &ones_hidden,
                &case.expert_scale,
                &mut dropping_router_scale_changes,
            ),
            (
                &case.scale,
                &ones_experts,
                &mut dropping_expert_scale_changes,
            ),
        ] {
            let routes = metal::route(
                &case.input,
                scale,
                &projection,
                expert_scale,
                case.top_k,
                reference.epsilon,
            );
            for ((selected, indices), weights) in
                routes.iter().zip(&case.indices).zip(&case.weights)
            {
                *changed |= selected
                    .iter()
                    .zip(indices)
                    .zip(weights)
                    .any(|((&(e, w), i), source)| e != *i || (w - source).abs() > 1e-5);
            }
        }
    }
    assert!(dropping_router_scale_changes);
    assert!(dropping_expert_scale_changes);
}

#[cfg(feature = "metal")]
#[test]
#[allow(clippy::float_cmp, reason = "the weights are exact by construction")]
fn tie_policy_and_post_normalization_expert_scale() {
    let _gpu = metal::lock();
    // Zero logits tie all four experts: the lower indices win, and each
    // renormalized 0.5 is then scaled.
    let selected = metal::route(
        &[vec![1.0, -1.0]],
        &[1.0, 1.0],
        &[0.0; 8],
        &[2.0, 3.0, 4.0, 5.0],
        2,
        1e-6,
    );
    assert_eq!(selected, vec![vec![(0, 1.0), (1, 1.5)]]);
}

#[cfg(feature = "metal")]
#[test]
fn more_experts_than_exist_are_refused() {
    let _gpu = metal::lock();
    let result = gemma::routing::route(
        &metal::array(&[1.0, -1.0], &[1, 2]),
        &metal::array(&[1.0, 1.0], &[2]),
        &metal::array(&[0.0; 8], &[4, 2]),
        &metal::array(&[1.0; 4], &[4]),
        5,
        1e-6,
    );
    assert!(matches!(
        result,
        Err(layers::moe::MoeError::TopK { top_k: 5, .. })
    ));
}

#[test]
fn invalid_geometry_is_rejected() {
    assert_eq!(
        Gemma4MoeGeometry::new(0, 4, 2, 3),
        Err(Gemma4RouterError::ZeroDimension)
    );
    for top_k in [0, 5] {
        assert_eq!(
            Gemma4MoeGeometry::new(2, 4, top_k, 3),
            Err(Gemma4RouterError::InvalidTopK)
        );
    }
    assert_eq!(
        Gemma4MoeGeometry::new(usize::MAX, 4, 2, 3),
        Err(Gemma4RouterError::ShapeOverflow)
    );
}
