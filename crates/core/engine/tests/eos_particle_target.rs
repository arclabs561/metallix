//! Offline EOS-tree accounting for a finite, resampled particle trace.
//!
//! This is intentionally an executable oracle, not an SMC runtime, cache API,
//! model adapter, or programmable-inference surface.

#![allow(clippy::float_cmp, clippy::similar_names)]

use engine::sampling::sample_categorical;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FirstToken {
    Eos,
    A,
    B,
    Dead,
}

impl FirstToken {
    const ALL: [Self; 4] = [Self::Eos, Self::A, Self::B, Self::Dead];

    const fn index(self) -> usize {
        match self {
            Self::Eos => 0,
            Self::A => 1,
            Self::B => 2,
            Self::Dead => 3,
        }
    }

    const fn terminal_after_first(self) -> bool {
        matches!(self, Self::Eos | Self::Dead)
    }
}

// The named proposal has full support. `Dead` is removed by a hard condition,
// so it is a valid proposal draw with a zero target weight.
const BASE_FIRST: [f64; 4] = [0.20, 0.50, 0.25, 0.05];
const PROPOSAL_FIRST: [f64; 4] = [0.40, 0.25, 0.25, 0.10];
const FIRST_POTENTIAL: [f64; 4] = [0.50, 0.80, 0.24, 0.0];
const EOS_POTENTIAL_AFTER: [f64; 4] = [1.0, 0.50, 1.0, 1.0];

#[derive(Clone, Debug)]
struct Particle {
    prefix: Vec<FirstToken>,
    terminated: bool,
    log_weight: f64,
}

impl Particle {
    fn after_first(token: FirstToken) -> Self {
        let index = token.index();
        let target_increment = BASE_FIRST[index] * FIRST_POTENTIAL[index];
        let log_weight = if target_increment == 0.0 {
            f64::NEG_INFINITY
        } else {
            (target_increment / PROPOSAL_FIRST[index]).ln()
        };
        Self {
            prefix: vec![token],
            terminated: token.terminal_after_first(),
            log_weight,
        }
    }

    fn absorb_or_emit_eos(&mut self) {
        if self.terminated {
            return;
        }
        let first = self.prefix[0];
        let increment = EOS_POTENTIAL_AFTER[first.index()].ln();
        self.log_weight += increment;
        self.prefix.push(FirstToken::Eos);
        self.terminated = true;
    }
}

struct NormalizedWeights {
    log_sum: f64,
    probabilities: Vec<f64>,
    ess: f64,
}

fn normalize(log_weights: &[f64]) -> Option<NormalizedWeights> {
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return None;
    }
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
        .map(|weight| weight * weight)
        .sum::<f64>()
        .recip();
    Some(NormalizedWeights {
        log_sum,
        probabilities,
        ess,
    })
}

fn resample(particles: &[Particle], ancestors: &[usize]) -> Vec<Particle> {
    ancestors
        .iter()
        .map(|&ancestor| Particle {
            prefix: particles[ancestor].prefix.clone(),
            terminated: particles[ancestor].terminated,
            // Resampling records the prior normalizer separately and restarts
            // the next incremental-weight segment at equal particle weights.
            log_weight: 0.0,
        })
        .collect()
}

fn oracle_weighted_select(probabilities: &[f64], uniform: f64) -> usize {
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
        .expect("nonempty weighted selection")
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the production sampler accepts FP32 logits; the independent oracle stays FP64"
)]
fn sampled_weighted_select(log_weights: &[f64], uniform: f64) -> usize {
    let logits = log_weights
        .iter()
        .map(|weight| *weight as f32)
        .collect::<Vec<_>>();
    let legal = vec![true; logits.len()];
    usize::try_from(
        sample_categorical(&logits, &legal, 1.0, uniform)
            .expect("finite live particle weights")
            .token_id,
    )
    .expect("small particle population")
}

fn sampled_multinomial_resample(log_weights: &[f64], uniforms: &[f64]) -> Vec<usize> {
    uniforms
        .iter()
        .map(|&uniform| sampled_weighted_select(log_weights, uniform))
        .collect()
}

fn exact_path_masses() -> [f64; 4] {
    FirstToken::ALL.map(|token| {
        let index = token.index();
        BASE_FIRST[index] * FIRST_POTENTIAL[index] * EOS_POTENTIAL_AFTER[index]
    })
}

fn proposal_draw(uniform: f64) -> FirstToken {
    let mut cumulative = 0.0;
    for token in FirstToken::ALL {
        cumulative += PROPOSAL_FIRST[token.index()];
        if uniform < cumulative {
            return token;
        }
    }
    FirstToken::Dead
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected:.16}, got {actual:.16}"
    );
}

#[test]
fn exact_eos_tree_enumeration_names_the_target_before_particles_exist() {
    // Paths are EOS, A→EOS, B→EOS, and a hard-rejected Dead branch.
    let masses = exact_path_masses();
    for (actual, expected) in masses.into_iter().zip([0.10, 0.20, 0.06, 0.0]) {
        assert_close(actual, expected);
    }
    let normalizer = masses.into_iter().sum::<f64>();
    assert_close(normalizer, 0.36);
    assert_close(masses[0] / normalizer, 5.0 / 18.0);
    assert_close(masses[1] / normalizer, 5.0 / 9.0);
    assert_close(masses[2] / normalizer, 1.0 / 6.0);
    assert_eq!(masses[3], 0.0);

    assert_eq!(proposal_draw(0.10), FirstToken::Eos);
    assert_eq!(proposal_draw(0.50), FirstToken::A);
    assert_eq!(proposal_draw(0.80), FirstToken::B);
    assert_eq!(proposal_draw(0.95), FirstToken::Dead);
}

#[test]
fn finite_resampled_trace_records_weights_ancestry_absorption_and_weighted_selection() {
    // A fixed proposal-entropy schedule gives A, A, EOS. This is one finite
    // trace, not a draw from the exact enumerated target above.
    let first_tokens = [
        proposal_draw(0.50),
        proposal_draw(0.50),
        proposal_draw(0.10),
    ];
    assert_eq!(
        first_tokens,
        [FirstToken::A, FirstToken::A, FirstToken::Eos]
    );
    let parents = first_tokens.map(Particle::after_first);
    let first_weights = parents
        .iter()
        .map(|particle| particle.log_weight)
        .collect::<Vec<_>>();
    assert_close(first_weights[0].exp(), 1.6);
    assert_close(first_weights[1].exp(), 1.6);
    assert_close(first_weights[2].exp(), 0.25);
    let first = normalize(&first_weights).expect("live first-stage population");
    assert_close(first.log_sum.exp(), 3.45);
    assert_close(first.probabilities[0], 32.0 / 69.0);
    assert_close(first.probabilities[1], 32.0 / 69.0);
    assert_close(first.probabilities[2], 5.0 / 69.0);
    assert_close(first.ess, 4_761.0 / 2_073.0);

    // Parent 1 is duplicated, parent 0 is discarded, and the already-ended
    // EOS parent remains present as an absorbed particle.
    let resample_uniforms = [0.50, 0.60, 0.95];
    let oracle_ancestors =
        resample_uniforms.map(|uniform| oracle_weighted_select(&first.probabilities, uniform));
    assert_eq!(oracle_ancestors, [1, 1, 2]);
    let ancestors = sampled_multinomial_resample(&first_weights, &resample_uniforms);
    assert_eq!(ancestors, vec![1, 1, 2]);
    let mut children = resample(&parents, &ancestors);
    assert_eq!(children[0].prefix, vec![FirstToken::A]);
    assert_eq!(children[1].prefix, vec![FirstToken::A]);
    assert_eq!(children[2].prefix, vec![FirstToken::Eos]);
    assert!(!children[0].terminated);
    assert!(!children[1].terminated);
    assert!(children[2].terminated);
    assert!(children.iter().all(|particle| particle.log_weight == 0.0));

    for particle in &mut children {
        particle.absorb_or_emit_eos();
    }
    assert_eq!(children[0].prefix, vec![FirstToken::A, FirstToken::Eos]);
    assert_eq!(children[1].prefix, vec![FirstToken::A, FirstToken::Eos]);
    assert_eq!(children[2].prefix, vec![FirstToken::Eos]);
    assert!(children.iter().all(|particle| particle.terminated));

    let final_weights = children
        .iter()
        .map(|particle| particle.log_weight)
        .collect::<Vec<_>>();
    let final_stage = normalize(&final_weights).expect("live terminal population");
    assert_close(final_stage.probabilities[0], 0.25);
    assert_close(final_stage.probabilities[1], 0.25);
    assert_close(final_stage.probabilities[2], 0.50);
    assert_close(final_stage.ess, 8.0 / 3.0);
    assert_eq!(oracle_weighted_select(&final_stage.probabilities, 0.10), 0);
    assert_eq!(oracle_weighted_select(&final_stage.probabilities, 0.55), 2);
    assert_eq!(sampled_weighted_select(&final_weights, 0.10), 0);
    assert_eq!(sampled_weighted_select(&final_weights, 0.55), 2);

    // SMC's standard product-of-stage-average estimator is finite-N evidence,
    // not the exact target normalizer. This trace also loses the B→EOS path.
    let particle_normalizer_estimate = (first.log_sum
        - f64::from(u32::try_from(parents.len()).expect("small particle count")).ln())
    .exp()
        * (final_stage.log_sum
            - f64::from(u32::try_from(children.len()).expect("small particle count")).ln())
        .exp();
    assert_close(particle_normalizer_estimate, 23.0 / 30.0);
    assert!((particle_normalizer_estimate - 0.36).abs() > 0.4);
    let particle_path_mass = [0.50, 0.50, 0.0, 0.0];
    assert_ne!(
        particle_path_mass,
        exact_path_masses().map(|mass| mass / 0.36)
    );
}

#[test]
fn hard_condition_can_extinguish_a_finite_particle_population() {
    let particles = [
        Particle::after_first(FirstToken::Dead),
        Particle::after_first(FirstToken::Dead),
        Particle::after_first(FirstToken::Dead),
    ];
    let weights = particles
        .iter()
        .map(|particle| particle.log_weight)
        .collect::<Vec<_>>();
    assert!(weights.iter().all(|weight| *weight == f64::NEG_INFINITY));
    assert!(normalize(&weights).is_none());
    // No normalized weights means there is no valid ancestry distribution or
    // terminal weighted selection. A runtime needs an explicit extinction
    // outcome; this oracle intentionally supplies no fallback token.
}
