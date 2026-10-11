//! A finite EOS-tree oracle against the public SMC population primitive.
//!
//! This fixes a single resampling schedule; it is not a model runtime, a
//! convergence test, or an evidence estimate for arbitrary particle counts.

#![allow(clippy::float_cmp)]

use engine::smc::ParticleSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Prefix {
    A,
    B,
    Eos,
    Dead,
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected:.16}, got {actual:.16}"
    );
}

#[test]
fn finite_oracle_keeps_eos_death_and_stage_normalizers_separate() {
    // The first-stage unnormalized masses are [1/2, 2, 3/2, 0]. The final
    // zero is a hard condition, not an EOS state: it has no target mass and
    // must never be selected or revived.
    let mut particles = ParticleSet::new(vec![Prefix::A, Prefix::B, Prefix::Eos, Prefix::Dead])
        .expect("nonempty population");
    {
        let particles = particles.particles_mut();
        particles[0]
            .add_log_weight(0.5_f64.ln())
            .expect("finite increment");
        particles[1]
            .add_log_weight(2.0_f64.ln())
            .expect("finite increment");
        particles[2]
            .add_log_weight(1.5_f64.ln())
            .expect("finite increment");
        particles[2].absorb().expect("live EOS particle");
        particles[3].reject().expect("live rejected particle");
    }

    let (first_weights, first_log_sum) = particles.normalized_weights().expect("support");
    assert_eq!(first_weights.len(), 4);
    assert_close(first_weights[0], 1.0 / 8.0);
    assert_close(first_weights[1], 1.0 / 2.0);
    assert_close(first_weights[2], 3.0 / 8.0);
    assert_eq!(first_weights[3], 0.0);
    assert_close(first_log_sum.exp(), 4.0);
    // The total mass and mean stage weight differ by the population size.
    assert_close(particles.log_mean_weight().expect("support").exp(), 1.0);

    // Systematic targets at 0.125, 0.375, 0.625, 0.875 select B, B, EOS, EOS.
    // A is discarded, B is duplicated, EOS is copied as terminal, and Dead
    // has zero mass. The receipt is the first stage's mean, not its total.
    let first_stage_log_mean = particles.systematic_resample(0.5).expect("resample");
    assert_close(first_stage_log_mean.exp(), 1.0);
    assert_eq!(
        particles
            .particles()
            .iter()
            .map(|particle| *particle.state())
            .collect::<Vec<_>>(),
        [Prefix::B, Prefix::B, Prefix::Eos, Prefix::Eos]
    );
    assert!(particles.particles()[2].is_absorbed());
    assert!(particles.particles()[3].is_absorbed());
    assert!(
        particles
            .particles()
            .iter()
            .all(|particle| particle.log_weight() == 0.0)
    );

    // Only live B children receive the second-stage potential and terminate.
    // Absorbed EOS children reject the same update, preventing a duplicated
    // EOS transition or a fictitious terminal potential.
    {
        let particles = particles.particles_mut();
        for particle in &mut particles[..2] {
            particle
                .add_log_weight(0.25_f64.ln())
                .expect("live B child");
            particle.absorb().expect("live B child");
        }
        assert!(particles[2].add_log_weight(0.0).is_err());
        assert!(particles[3].add_log_weight(0.0).is_err());
    }

    let (terminal_weights, terminal_log_sum) = particles.normalized_weights().expect("support");
    assert_close(terminal_weights[0], 1.0 / 10.0);
    assert_close(terminal_weights[1], 1.0 / 10.0);
    assert_close(terminal_weights[2], 2.0 / 5.0);
    assert_close(terminal_weights[3], 2.0 / 5.0);
    assert_close(terminal_log_sum.exp(), 2.5);
    assert_close(
        particles.effective_sample_size().expect("support"),
        100.0 / 34.0,
    );

    // The product of stage means is the finite-particle normalizer estimate.
    // It differs from the final population's total mass and from any exact
    // target normalizer a model-specific driver might define.
    let second_stage_log_mean = particles.log_mean_weight().expect("support");
    assert_close((first_stage_log_mean + second_stage_log_mean).exp(), 0.625);
    assert_close(second_stage_log_mean.exp(), 0.625);
}

#[test]
fn importance_ratio_oracle_distinguishes_zero_target_from_zero_proposal_support() {
    // This is an independent one-particle support oracle. A zero target under
    // a proposal that drew the event is an ordinary hard rejection. A zero
    // proposal cannot supply a valid importance ratio, even if the target is
    // also zero, and it must leave the particle unchanged for the caller to
    // handle as a failed driver step.
    let mut target_zero = ParticleSet::new(vec![Prefix::A]).expect("population");
    target_zero.particles_mut()[0]
        .add_log_importance_ratio(f64::NEG_INFINITY, 0.0)
        .expect("proposal drew zero-target event");
    assert!(target_zero.particles()[0].is_impossible());
    assert_eq!(
        target_zero.normalized_weights(),
        Err(engine::smc::SmcError::NoFiniteWeight)
    );

    let mut finite_ratio = ParticleSet::new(vec![Prefix::A]).expect("population");
    finite_ratio.particles_mut()[0]
        .add_log_importance_ratio(3.0_f64.ln(), 1.5_f64.ln())
        .expect("finite target/proposal ratio");
    assert_close(finite_ratio.particles()[0].log_weight().exp(), 2.0);

    for (log_target, log_proposal) in [
        (0.0, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        let mut no_proposal_support = ParticleSet::new(vec![Prefix::B]).expect("population");
        let before = no_proposal_support.clone();
        assert_eq!(
            no_proposal_support.particles_mut()[0]
                .add_log_importance_ratio(log_target, log_proposal),
            Err(engine::smc::SmcError::ProposalHasNoSupport)
        );
        assert_eq!(no_proposal_support, before);
    }

    let mut nonfinite = ParticleSet::new(vec![Prefix::Eos]).expect("population");
    let before = nonfinite.clone();
    assert_eq!(
        nonfinite.particles_mut()[0].add_log_importance_ratio(f64::NAN, 0.0),
        Err(engine::smc::SmcError::NonFiniteImportanceLogprob)
    );
    assert_eq!(nonfinite, before);

    let mut overflow = ParticleSet::new(vec![Prefix::Eos]).expect("population");
    let before = overflow.clone();
    assert_eq!(
        overflow.particles_mut()[0].add_log_importance_ratio(f64::MAX, -f64::MAX),
        Err(engine::smc::SmcError::WeightOverflow)
    );
    assert_eq!(overflow, before);
}

#[test]
fn two_resampling_rounds_compose_prior_stage_means_once_each() {
    // Each `systematic_resample` returns the outgoing stage mean and resets
    // all child weights. This trace makes a second resample observable: using
    // the current post-reset weights, or carrying either old normalizer into a
    // child, changes the final finite-particle estimate.
    let mut particles =
        ParticleSet::new(vec![Prefix::A, Prefix::B, Prefix::Eos]).expect("nonempty population");
    {
        let particles = particles.particles_mut();
        particles[0]
            .add_log_weight(0.5_f64.ln())
            .expect("first stage A");
        particles[1]
            .add_log_weight(1.5_f64.ln())
            .expect("first stage B");
        particles[2].absorb().expect("first stage EOS");
    }
    // Offset 0.51 keeps every systematic target away from a binary CDF
    // boundary: the ancestry is B, B, EOS on every supported platform.
    let first = particles.systematic_resample(0.51).expect("first resample");
    assert_close(first.exp(), 1.0);
    assert_eq!(
        particles
            .particles()
            .iter()
            .map(|particle| *particle.state())
            .collect::<Vec<_>>(),
        [Prefix::B, Prefix::B, Prefix::Eos]
    );

    // The first B descendant has mass 3, the second has mass 1/2, and the
    // absorbed EOS descendant carries neutral potential 1. The stage mean is
    // (3 + 1/2 + 1) / 3 = 3/2. Systematic ancestry duplicates the first B and
    // discards the second B while retaining EOS.
    {
        let particles = particles.particles_mut();
        particles[0]
            .add_log_weight(3.0_f64.ln())
            .expect("second stage first B");
        particles[1]
            .add_log_weight(0.5_f64.ln())
            .expect("second stage second B");
    }
    let second = particles
        .systematic_resample(0.51)
        .expect("second resample");
    assert_close(second.exp(), 1.5);
    assert_eq!(
        particles
            .particles()
            .iter()
            .map(|particle| *particle.state())
            .collect::<Vec<_>>(),
        [Prefix::B, Prefix::B, Prefix::Eos]
    );
    assert!(particles.particles()[2].is_absorbed());
    assert!(
        particles
            .particles()
            .iter()
            .all(|particle| particle.log_weight() == 0.0)
    );

    // The final stage is not resampled, so its mean is read directly. EOS
    // stays neutral; both live B descendants receive a 1/4 potential.
    for particle in &mut particles.particles_mut()[..2] {
        particle
            .add_log_weight(0.25_f64.ln())
            .expect("third stage B");
    }
    let third = particles.log_mean_weight().expect("third-stage mean");
    assert_close(third.exp(), 0.5);
    assert_close((first + second + third).exp(), 0.75);
}
