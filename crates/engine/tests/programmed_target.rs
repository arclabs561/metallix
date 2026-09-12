//! Offline target/proposal accounting for one finite, two-step program.
//!
//! This is deliberately a test-only oracle, not an interpreter or an SMC API.

// Keep the paper's p/q and A/B notation. Exact comparisons below check only
// hard-zero support and negative infinity, not rounded finite calculations.
#![allow(clippy::similar_names, clippy::float_cmp)]

use engine::sampling::sample_categorical;

const BASE_X: [f64; 2] = [0.6, 0.4];
const BASE_Y: [[f64; 2]; 2] = [[0.7, 0.3], [0.2, 0.8]];
const OBSERVATION: [[f64; 2]; 2] = [[0.5, 0.25], [0.75, 0.9]];
const PROPOSAL_X: [f64; 2] = [0.5, 0.5];
const PROPOSAL_Y: [[f64; 2]; 2] = [[0.5, 0.5], [0.25, 0.75]];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum X {
    A,
    B,
}

impl X {
    const ALL: [Self; 2] = [Self::A, Self::B];

    const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Y {
    Zero,
    One,
}

impl Y {
    const ALL: [Self; 2] = [Self::Zero, Self::One];

    const fn index(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Receipt {
    log_p_x: f64,
    log_p_y_given_x: f64,
    log_p: f64,
    condition_holds: bool,
    log_observation: f64,
    log_q_x: f64,
    log_q_y_given_x: f64,
    log_q: f64,
    log_weight: f64,
}

impl Receipt {
    fn target_mass(self) -> f64 {
        if self.condition_holds {
            self.log_p.exp() * self.log_observation.exp()
        } else {
            0.0
        }
    }
}

fn receipt(x: X, y: Y) -> Receipt {
    let x_index = x.index();
    let y_index = y.index();
    let log_p_x = BASE_X[x_index].ln();
    let log_p_y_given_x = BASE_Y[x_index][y_index].ln();
    let log_p = log_p_x + log_p_y_given_x;
    let condition_holds = !(x == X::B && y == Y::Zero);
    let log_observation = OBSERVATION[x_index][y_index].ln();
    let log_q_x = PROPOSAL_X[x_index].ln();
    let log_q_y_given_x = PROPOSAL_Y[x_index][y_index].ln();
    let log_q = log_q_x + log_q_y_given_x;
    let log_weight = if condition_holds {
        log_p + log_observation - log_q
    } else {
        f64::NEG_INFINITY
    };

    Receipt {
        log_p_x,
        log_p_y_given_x,
        log_p,
        condition_holds,
        log_observation,
        log_q_x,
        log_q_y_given_x,
        log_q,
        log_weight,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected:.16}, got {actual:.16}"
    );
}

#[test]
fn hard_condition_and_soft_observation_have_distinct_target_weights() {
    // Hand-enumerated p(x, y) * condition(x, y) * r(x, y), ordered A0/A1/B0/B1.
    // Base leaves are 0.42, 0.18, 0.08, 0.32; their observations are
    // 0.5, 0.25, 0.75, 0.9. B0 is excluded only by the hard condition.
    let expected_target_masses = [0.21, 0.045, 0.0, 0.288];
    let expected_normalizer = 0.543;
    let leaves = [
        receipt(X::A, Y::Zero),
        receipt(X::A, Y::One),
        receipt(X::B, Y::Zero),
        receipt(X::B, Y::One),
    ];

    assert_close(BASE_X.into_iter().sum(), 1.0);
    for probabilities in BASE_Y {
        assert_close(probabilities.into_iter().sum(), 1.0);
    }
    for likelihoods in OBSERVATION {
        assert!(likelihoods.into_iter().all(|likelihood| likelihood > 0.0));
    }

    for (&expected, leaf) in expected_target_masses.iter().zip(leaves) {
        assert_close(leaf.log_p, leaf.log_p_x + leaf.log_p_y_given_x);
        assert_close(leaf.log_q, leaf.log_q_x + leaf.log_q_y_given_x);
        assert!(leaf.log_observation.is_finite());
        assert!(leaf.log_observation.exp() > 0.0);
        if expected == 0.0 {
            assert!(!leaf.condition_holds);
            assert_eq!(leaf.log_weight, f64::NEG_INFINITY);
            assert_eq!(leaf.target_mass(), 0.0);
            // The observation remains positive: hard and soft control differ.
            assert!(leaf.log_p.exp() * leaf.log_observation.exp() > 0.0);
        } else {
            assert!(leaf.condition_holds);
            assert_close(leaf.log_weight.exp(), expected / leaf.log_q.exp());
            assert_close(leaf.target_mass(), expected);
        }
    }

    let target_masses = leaves.map(Receipt::target_mass);
    let enumerated_normalizer: f64 = target_masses.into_iter().sum();
    assert_close(enumerated_normalizer, expected_normalizer);
    assert_close(target_masses[0] / enumerated_normalizer, 70.0 / 181.0);
    assert_close(target_masses[1] / enumerated_normalizer, 15.0 / 181.0);
    assert_close(target_masses[3] / enumerated_normalizer, 96.0 / 181.0);
}

#[test]
fn full_support_proposal_weights_recover_the_unnormalized_target() {
    assert_close(PROPOSAL_X.into_iter().sum(), 1.0);
    for probabilities in PROPOSAL_Y {
        assert_close(probabilities.into_iter().sum(), 1.0);
        assert!(
            probabilities
                .into_iter()
                .all(|probability| probability > 0.0)
        );
    }

    let leaves = X::ALL
        .into_iter()
        .flat_map(|x| Y::ALL.into_iter().map(move |y| receipt(x, y)))
        .collect::<Vec<_>>();

    let importance_estimate: f64 = leaves
        .iter()
        .map(|leaf| leaf.log_q.exp() * leaf.log_weight.exp())
        .sum();
    assert_close(importance_estimate, 0.543);

    let omitted_proposal_correction: f64 = leaves
        .iter()
        .map(|leaf| leaf.log_q.exp() * leaf.target_mass())
        .sum();
    assert_close(omitted_proposal_correction, 0.171_75);
    assert!((omitted_proposal_correction - importance_estimate).abs() > 0.1);

    let normalized_target_b_one = receipt(X::B, Y::One).target_mass() / importance_estimate;
    let normalized_without_correction = receipt(X::B, Y::One).log_q.exp()
        * receipt(X::B, Y::One).target_mass()
        / omitted_proposal_correction;
    assert_close(normalized_target_b_one, 96.0 / 181.0);
    assert!(
        (normalized_without_correction - normalized_target_b_one).abs() > 0.09,
        "omitting log p + log r - log q must alter the normalized leaf distribution"
    );

    let forbidden = receipt(X::B, Y::Zero);
    assert_close(forbidden.log_q.exp(), 0.125);
    assert_eq!(forbidden.log_weight, f64::NEG_INFINITY);
    assert_eq!(forbidden.log_q.exp() * forbidden.log_weight.exp(), 0.0);
}

#[test]
fn categorical_primitive_matches_each_staged_proposal_log_probability() {
    // Equal zero logits produce exact q(x)=(0.5, 0.5). For B's second draw,
    // [0, 1] at T=1/ln(3) gives an exact analytic ratio of three without
    // pretending an f32 representation of ln(3) is exact f64 input data.
    let x_logits = [0.0, 0.0];
    let y_a_logits = [0.0, 0.0];
    let y_b_logits = [0.0, 1.0];
    let legal = [true, true];
    let x_draws = [(0.25, 0_u32), (0.75, 1_u32)];
    let y_a_draws = [(0.25, 0_u32), (0.75, 1_u32)];
    let y_b_draws = [(0.125, 0_u32), (0.875, 1_u32)];

    for (x_draw, expected_x) in x_draws {
        let x_sample =
            sample_categorical(&x_logits, &legal, 1.0, x_draw).expect("valid proposal draw");
        assert_eq!(x_sample.token_id, expected_x);
        let (y_logits, temperature, y_draws) = if expected_x == 0 {
            (&y_a_logits, 1.0, &y_a_draws[..])
        } else {
            (&y_b_logits, 1.0 / 3.0_f64.ln(), &y_b_draws[..])
        };
        for &(y_draw, expected_y) in y_draws {
            let y_sample = sample_categorical(y_logits, &legal, temperature, y_draw)
                .expect("valid proposal draw");
            assert_eq!(y_sample.token_id, expected_y);

            let x = X::ALL[usize::try_from(expected_x).expect("two token IDs")];
            let y = Y::ALL[usize::try_from(expected_y).expect("two token IDs")];
            let leaf = receipt(x, y);
            assert_close(x_sample.sampling_logprob, leaf.log_q_x);
            assert_close(y_sample.sampling_logprob, leaf.log_q_y_given_x);
            assert_close(
                x_sample.sampling_logprob + y_sample.sampling_logprob,
                leaf.log_q,
            );
        }
    }
}
