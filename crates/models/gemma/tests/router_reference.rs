//! Router-only qualification against actual pinned Transformers module outputs.
use gemma::routing::{Gemma4MoeGeometry, Gemma4Router, Gemma4RouterError};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    hidden: usize,
    experts: usize,
    intermediate: usize,
    epsilon: f32,
    cases: Vec<Case>,
}
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
fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../../fixtures/gemma-4-tiny/router-reference.json"
    ))
    .unwrap()
}

#[test]
fn router_matches_pinned_cpu_module() {
    let reference = reference();
    assert_eq!(reference.cases.len(), 12);
    for case in &reference.cases {
        let geometry = Gemma4MoeGeometry::new(
            reference.hidden,
            reference.experts,
            case.top_k,
            reference.intermediate,
        )
        .unwrap();
        let projection: Vec<f32> = case.projection.iter().flatten().copied().collect();
        let router = Gemma4Router::new(
            geometry,
            &case.scale,
            &projection,
            &case.expert_scale,
            reference.epsilon,
        )
        .unwrap();
        assert_eq!(case.input.len(), 9);
        for ((input, indices), weights) in case.input.iter().zip(&case.indices).zip(&case.weights) {
            let actual = router.route(input).unwrap();
            assert_eq!(actual.len(), case.top_k);
            for ((route, index), weight) in actual.iter().zip(indices).zip(weights) {
                assert_eq!(route.expert(), *index);
                assert!(route.weight().is_finite() && weight.is_finite());
                assert!(
                    (route.weight() - weight).abs() <= 1e-6,
                    "native={} source={weight}",
                    route.weight()
                );
            }
        }
    }
}

#[test]
fn nonunit_scales_are_observable_in_source_cases() {
    let reference = reference();
    let mut dropping_router_scale_changes = false;
    let mut dropping_expert_scale_changes = false;
    for case in reference.cases.iter().filter(|c| c.top_k == 3) {
        let geometry = Gemma4MoeGeometry::new(
            reference.hidden,
            reference.experts,
            case.top_k,
            reference.intermediate,
        )
        .unwrap();
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
            let router = Gemma4Router::new(
                geometry,
                scale,
                &projection,
                expert_scale,
                reference.epsilon,
            )
            .unwrap();
            for ((input, indices), weights) in
                case.input.iter().zip(&case.indices).zip(&case.weights)
            {
                let selected = router.route(input).unwrap();
                *changed |= selected
                    .iter()
                    .zip(indices)
                    .zip(weights)
                    .any(|((r, i), w)| r.expert() != *i || (r.weight() - w).abs() > 1e-5);
            }
        }
    }
    assert!(dropping_router_scale_changes);
    assert!(dropping_expert_scale_changes);
}

#[test]
fn tie_policy_and_post_normalization_expert_scale() {
    let geometry = Gemma4MoeGeometry::new(2, 4, 2, 3).unwrap();
    let router = Gemma4Router::new(
        geometry,
        &[1.0, 1.0],
        &[0.0; 8],
        &[2.0, 3.0, 4.0, 5.0],
        1e-6,
    )
    .unwrap();
    let selected = router.route(&[1.0, -1.0]).unwrap();
    assert_eq!(
        selected.iter().map(|r| r.expert()).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        selected.iter().map(|r| r.weight()).collect::<Vec<_>>(),
        vec![1.0, 1.5]
    );
}

#[test]
fn invalid_geometry_and_nonfinite_values_are_rejected() {
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
    let g = Gemma4MoeGeometry::new(2, 4, 2, 3).unwrap();
    assert!(matches!(
        Gemma4Router::new(g, &[1.0; 2], &[0.0; 7], &[1.0; 4], 1e-6),
        Err(Gemma4RouterError::ShapeMismatch)
    ));
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Gemma4Router::new(g, &[1.0; 2], &[0.0; 8], &[1.0; 4], eps),
            Err(Gemma4RouterError::InvalidEpsilon)
        ));
    }
    let router = Gemma4Router::new(g, &[1.0; 2], &[0.0; 8], &[1.0; 4], 1e-6).unwrap();
    assert!(matches!(
        router.route(&[f32::NAN, 1.0]),
        Err(Gemma4RouterError::NonFinite)
    ));
    assert!(matches!(
        router.route(&[f32::MAX, 1.0]),
        Err(Gemma4RouterError::NonFinite)
    ));
}

#[test]
fn router_contract_does_not_enable_moe_adapter_admission() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/gemma-4-tiny/moe-reference.json"
    ))
    .unwrap();
    assert!(matches!(
        gemma::Gemma4TextConfig::parse(&fixture["config"].to_string()),
        Err(gemma::Gemma4ConfigError::Unsupported(
            "mixture-of-experts block"
        ))
    ));
}
