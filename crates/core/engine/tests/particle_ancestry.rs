//! Finite particle ancestry accounting around an absorbing EOS state.
//!
//! This complements `eos_particle_target`: it is a test-only, enumerable
//! resampling receipt, not an SMC runtime, cache-fork API, or claim about
//! finite-particle convergence.

#![allow(clippy::float_cmp)]

use engine::sampling::sample_categorical;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Token {
    A,
    B,
    Eos,
}

#[derive(Clone, Debug)]
struct Particle {
    prefix: Vec<Token>,
    terminated: bool,
    log_weight: f64,
    parent: Option<usize>,
}

impl Particle {
    fn advance_to_eos(&mut self, live_increment: f64) {
        if self.terminated {
            return;
        }
        self.prefix.push(Token::Eos);
        self.terminated = true;
        self.log_weight += live_increment.ln();
    }
}

struct NormalizedWeights {
    log_sum: f64,
    probabilities: Vec<f64>,
    ess: f64,
}

fn normalize(log_weights: &[f64]) -> NormalizedWeights {
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        maximum.is_finite(),
        "this finite oracle requires live support"
    );
    let shifted_sum = log_weights
        .iter()
        .map(|weight| (*weight - maximum).exp())
        .sum::<f64>();
    let log_sum = maximum + shifted_sum.ln();
    let probabilities = log_weights
        .iter()
        .map(|weight| (*weight - log_sum).exp())
        .collect::<Vec<_>>();
    let ess = probabilities
        .iter()
        .map(|probability| probability * probability)
        .sum::<f64>()
        .recip();
    NormalizedWeights {
        log_sum,
        probabilities,
        ess,
    }
}

fn independent_cdf(probabilities: &[f64], uniform: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if uniform < cumulative {
            return index;
        }
    }
    probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .expect("finite oracle has positive support")
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "production categorical logits are FP32; this oracle's accounting remains FP64"
)]
fn production_categorical(log_weights: &[f64], uniform: f64) -> usize {
    let logits = log_weights
        .iter()
        .map(|weight| *weight as f32)
        .collect::<Vec<_>>();
    let legal = vec![true; logits.len()];
    usize::try_from(
        sample_categorical(&logits, &legal, 1.0, uniform)
            .expect("finite particle weights must sample")
            .token_id,
    )
    .expect("small particle population")
}

fn resample(parents: &[Particle], ancestors: &[usize]) -> Vec<Particle> {
    ancestors
        .iter()
        .map(|&ancestor| Particle {
            prefix: parents[ancestor].prefix.clone(),
            terminated: parents[ancestor].terminated,
            // The preceding normalizer is accounted separately; the next
            // segment starts from equal post-resampling weights.
            log_weight: 0.0,
            parent: Some(ancestor),
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected:.16}, got {actual:.16}"
    );
}

#[test]
fn eos_ancestry_keeps_duplicates_discards_and_weighted_terminal_draws_explicit() {
    // One already-finished EOS path and two live paths. Their first-stage
    // unnormalized weights are [1/2, 2, 3/2], total 4.
    let parents = [
        Particle {
            prefix: vec![Token::A],
            terminated: false,
            log_weight: 0.5_f64.ln(),
            parent: None,
        },
        Particle {
            prefix: vec![Token::B],
            terminated: false,
            log_weight: 2.0_f64.ln(),
            parent: None,
        },
        Particle {
            prefix: vec![Token::Eos],
            terminated: true,
            log_weight: 1.5_f64.ln(),
            parent: None,
        },
    ];
    let first_weights = parents
        .iter()
        .map(|particle| particle.log_weight)
        .collect::<Vec<_>>();
    let first = normalize(&first_weights);
    assert_close(first.log_sum.exp(), 4.0);
    assert_eq!(first.probabilities.len(), 3);
    assert_close(first.probabilities[0], 1.0 / 8.0);
    assert_close(first.probabilities[1], 1.0 / 2.0);
    assert_close(first.probabilities[2], 3.0 / 8.0);
    assert_close(first.ess, 32.0 / 13.0);

    // Parent 1 is duplicated, parent 0 is discarded, and the already-ended
    // EOS parent remains in the population through resampling.
    let uniforms = [0.20, 0.60, 0.90];
    let expected = uniforms.map(|uniform| independent_cdf(&first.probabilities, uniform));
    assert_eq!(expected, [1, 1, 2]);
    let ancestors = uniforms.map(|uniform| production_categorical(&first_weights, uniform));
    assert_eq!(ancestors, expected);
    let mut children = resample(&parents, &ancestors);
    assert_eq!(
        children
            .iter()
            .map(|child| child.parent)
            .collect::<Vec<_>>(),
        [Some(1), Some(1), Some(2)]
    );
    assert!(children.iter().all(|child| child.log_weight == 0.0));
    assert_eq!(children[2].prefix, vec![Token::Eos]);
    assert!(children[2].terminated);

    // Only live copies append EOS and acquire this stage's 1/4 potential.
    // The retained EOS child is absorbing: it neither appends another EOS nor
    // receives a fictitious terminal potential.
    for child in &mut children {
        child.advance_to_eos(0.25);
    }
    assert_eq!(children[0].prefix, vec![Token::B, Token::Eos]);
    assert_eq!(children[1].prefix, vec![Token::B, Token::Eos]);
    assert_eq!(children[2].prefix, vec![Token::Eos]);
    assert!(children.iter().all(|child| child.terminated));

    let terminal_weights = children
        .iter()
        .map(|particle| particle.log_weight)
        .collect::<Vec<_>>();
    let terminal = normalize(&terminal_weights);
    assert_close(terminal.log_sum.exp(), 1.5);
    assert_close(terminal.probabilities[0], 1.0 / 6.0);
    assert_close(terminal.probabilities[1], 1.0 / 6.0);
    assert_close(terminal.probabilities[2], 2.0 / 3.0);
    assert_close(terminal.ess, 2.0);

    // This must be a weighted categorical terminal draw, not optimization or
    // equal-particle selection. The production primitive is exercised against
    // an independent FP64 CDF at fixed entropy values.
    assert_eq!(independent_cdf(&terminal.probabilities, 0.10), 0);
    assert_eq!(production_categorical(&terminal_weights, 0.10), 0);
    let map = 2;
    assert_ne!(production_categorical(&terminal_weights, 0.10), map);
    assert_eq!(independent_cdf(&terminal.probabilities, 0.40), 2);
    assert_eq!(production_categorical(&terminal_weights, 0.40), 2);
    let unweighted = if 0.40 < 1.0 / 3.0 {
        0
    } else if 0.40 < 2.0 / 3.0 {
        1
    } else {
        2
    };
    assert_eq!(unweighted, 1);
    assert_ne!(production_categorical(&terminal_weights, 0.40), unweighted);
}
