//! Bounded sequential Monte Carlo state independent of model graphs.

use thiserror::Error;

/// One particle and its request-local state.
#[derive(Clone, Debug, PartialEq)]
pub struct Particle<T> {
    state: T,
    log_weight: f64,
    status: ParticleStatus,
}

/// Whether a particle can still take a transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParticleStatus {
    Live,
    Absorbed,
    Impossible,
}

impl<T> Particle<T> {
    /// Creates a live particle with a unit weight.
    #[must_use]
    pub fn new(state: T) -> Self {
        Self {
            state,
            log_weight: 0.0,
            status: ParticleStatus::Live,
        }
    }

    /// Returns the model-owned state.
    #[must_use]
    pub const fn state(&self) -> &T {
        &self.state
    }

    /// Returns the log weight.
    #[must_use]
    pub const fn log_weight(&self) -> f64 {
        self.log_weight
    }

    /// Returns whether this particle has reached an absorbing terminal state.
    #[must_use]
    pub const fn is_absorbed(&self) -> bool {
        matches!(self.status, ParticleStatus::Absorbed)
    }

    /// Returns whether this particle has zero target mass and cannot transition.
    #[must_use]
    pub const fn is_impossible(&self) -> bool {
        matches!(self.status, ParticleStatus::Impossible)
    }

    /// Adds a finite incremental log weight to a live particle.
    ///
    /// # Errors
    ///
    /// The particle is unchanged on error.
    ///
    /// * [`SmcError::NonFiniteWeight`] for a NaN or infinite `increment`.
    /// * [`SmcError::AbsorbedParticle`] or [`SmcError::ImpossibleParticle`]
    ///   when the particle is not live.
    /// * [`SmcError::WeightOverflow`] when the sum is not finite.
    pub fn add_log_weight(&mut self, increment: f64) -> Result<(), SmcError> {
        if !increment.is_finite() {
            return Err(SmcError::NonFiniteWeight);
        }
        self.require_live()?;
        let updated = self.log_weight + increment;
        if !updated.is_finite() {
            return Err(SmcError::WeightOverflow);
        }
        self.log_weight = updated;
        Ok(())
    }

    /// Applies a target/proposal log-importance ratio to a live particle.
    ///
    /// A zero target mass under a positive-mass proposal makes this particle
    /// impossible. A zero proposal mass is never a valid importance update:
    /// it either cannot explain a positive-target particle or describes an
    /// event that the proposal could not have drawn. Both cases fail without
    /// changing the particle.
    ///
    /// # Errors
    ///
    /// The particle is unchanged on error.
    ///
    /// * [`SmcError::AbsorbedParticle`] or [`SmcError::ImpossibleParticle`]
    ///   when the particle is not live.
    /// * [`SmcError::NonFiniteImportanceLogprob`] for a NaN or positive
    ///   infinite log mass.
    /// * [`SmcError::ProposalHasNoSupport`] when `log_proposal` is negative
    ///   infinity.
    /// * [`SmcError::WeightOverflow`] when the ratio or the new weight is not
    ///   finite.
    pub fn add_log_importance_ratio(
        &mut self,
        log_target: f64,
        log_proposal: f64,
    ) -> Result<(), SmcError> {
        self.require_live()?;
        let target_is_zero = log_target == f64::NEG_INFINITY;
        let proposal_is_zero = log_proposal == f64::NEG_INFINITY;
        if (!log_target.is_finite() && !target_is_zero)
            || (!log_proposal.is_finite() && !proposal_is_zero)
        {
            return Err(SmcError::NonFiniteImportanceLogprob);
        }
        if proposal_is_zero {
            return Err(SmcError::ProposalHasNoSupport);
        }
        if target_is_zero {
            return self.reject();
        }
        let increment = log_target - log_proposal;
        if !increment.is_finite() {
            return Err(SmcError::WeightOverflow);
        }
        self.add_log_weight(increment)
    }

    /// Assigns zero target mass to a live particle.
    ///
    /// This is distinct from [`Self::absorb`]: an impossible particle remains
    /// observable for ancestry accounting, but has no normalized mass and
    /// cannot be revived by a later increment.
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::AbsorbedParticle`] or [`SmcError::ImpossibleParticle`]
    /// when the particle is not live.
    pub fn reject(&mut self) -> Result<(), SmcError> {
        self.require_live()?;
        self.log_weight = f64::NEG_INFINITY;
        self.status = ParticleStatus::Impossible;
        Ok(())
    }

    /// Marks a live particle terminal; subsequent transitions fail closed.
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::AbsorbedParticle`] or [`SmcError::ImpossibleParticle`]
    /// when the particle is not live.
    pub fn absorb(&mut self) -> Result<(), SmcError> {
        self.require_live()?;
        self.status = ParticleStatus::Absorbed;
        Ok(())
    }

    fn require_live(&self) -> Result<(), SmcError> {
        match self.status {
            ParticleStatus::Live => Ok(()),
            ParticleStatus::Absorbed => Err(SmcError::AbsorbedParticle),
            ParticleStatus::Impossible => Err(SmcError::ImpossibleParticle),
        }
    }
}

/// A finite particle population with deterministic, caller-supplied entropy.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSet<T> {
    particles: Vec<Particle<T>>,
}

impl<T: Clone> ParticleSet<T> {
    /// Creates a population from nonempty initial states.
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::EmptyPopulation`] when `states` is empty.
    pub fn new(states: Vec<T>) -> Result<Self, SmcError> {
        if states.is_empty() {
            return Err(SmcError::EmptyPopulation);
        }
        Ok(Self {
            particles: states.into_iter().map(Particle::new).collect(),
        })
    }

    /// Returns the population size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Returns whether the population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Returns all particles.
    #[must_use]
    pub fn particles(&self) -> &[Particle<T>] {
        &self.particles
    }

    /// Returns mutable particles for driver-owned bookkeeping transitions.
    ///
    /// Callers must preserve the particle's target/proposal ledger. Weight and
    /// terminal changes remain checked by [`Particle`] methods.
    #[must_use]
    pub fn particles_mut(&mut self) -> &mut [Particle<T>] {
        &mut self.particles
    }

    /// Returns normalized weights and the log sum of current unnormalized weights.
    ///
    /// The returned log sum describes this population only. It is not a
    /// sequence-level evidence estimate; a caller that resamples must retain
    /// each stage's [`Self::log_mean_weight`] separately.
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::NoFiniteWeight`] when no particle has a finite
    /// weight, so the population has no mass to normalize.
    pub fn normalized_weights(&self) -> Result<(Vec<f64>, f64), SmcError> {
        let maximum = self
            .particles
            .iter()
            .map(Particle::log_weight)
            .filter(|weight| weight.is_finite())
            .max_by(f64::total_cmp)
            .ok_or(SmcError::NoFiniteWeight)?;
        let scaled = self
            .particles
            .iter()
            .map(|particle| (particle.log_weight - maximum).exp())
            .collect::<Vec<_>>();
        let total = scaled.iter().sum::<f64>();
        if !total.is_finite() || total == 0.0 {
            return Err(SmcError::NoFiniteWeight);
        }
        let log_normalizer = maximum + total.ln();
        Ok((
            scaled.into_iter().map(|weight| weight / total).collect(),
            log_normalizer,
        ))
    }

    /// Returns the log mean of the current unnormalized particle weights.
    ///
    /// This is the per-stage normalizer that an SMC driver may accumulate.
    /// It remains distinct from the log sum returned by
    /// [`Self::normalized_weights`].
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::NoFiniteWeight`] when no particle has a finite
    /// weight, and [`SmcError::PopulationTooLarge`] for more than `u32::MAX`
    /// particles.
    pub fn log_mean_weight(&self) -> Result<f64, SmcError> {
        let (_, log_weight_sum) = self.normalized_weights()?;
        let count = f64::from(u32::try_from(self.len()).map_err(|_| SmcError::PopulationTooLarge)?);
        Ok(log_weight_sum - count.ln())
    }

    /// Computes effective sample size from normalized log weights.
    ///
    /// # Errors
    ///
    /// Returns [`SmcError::NoFiniteWeight`] when no particle has a finite
    /// weight, so the population has no mass to normalize.
    pub fn effective_sample_size(&self) -> Result<f64, SmcError> {
        let (weights, _) = self.normalized_weights()?;
        Ok(weights
            .iter()
            .map(|weight| weight * weight)
            .sum::<f64>()
            .recip())
    }

    /// Systematically resamples to the existing population size.
    ///
    /// `offset` is a deterministic uniform variate in `[0, 1)`. Resampling
    /// preserves terminal status and resets every child to a unit stage weight.
    ///
    /// The returned value is the previous population's log mean weight. An SMC
    /// driver must retain it before advancing the fresh stage; it is not stored
    /// in this state-only primitive.
    ///
    /// # Errors
    ///
    /// The population is unchanged on error.
    ///
    /// * [`SmcError::InvalidOffset`] unless `offset` is in `[0, 1)`.
    /// * [`SmcError::NoFiniteWeight`] when no particle has a finite weight.
    /// * [`SmcError::PopulationTooLarge`] for more than `u32::MAX` particles.
    pub fn systematic_resample(&mut self, offset: f64) -> Result<f64, SmcError> {
        if !(0.0..1.0).contains(&offset) || !offset.is_finite() {
            return Err(SmcError::InvalidOffset);
        }
        let (weights, log_weight_sum) = self.normalized_weights()?;
        let count = self.len();
        let count_u32 = u32::try_from(count).map_err(|_| SmcError::PopulationTooLarge)?;
        let population_width = f64::from(count_u32);
        // `normalized_weights` has live support, but floating-point division
        // need not make the positive entries sum to exactly one. Keep the CDF
        // cursor at or before the last supported parent: a rounded target of
        // one (or a rounded CDF below one) must select that parent, never a
        // trailing zero-mass particle.
        let last_supported = weights
            .iter()
            .rposition(|weight| *weight > 0.0)
            .ok_or(SmcError::NoFiniteWeight)?;
        let mut cumulative = 0.0;
        let mut ancestor = 0;
        let parents = self.particles.clone();
        let mut children = Vec::with_capacity(count);
        for index in 0..count {
            let index_u32 = u32::try_from(index).map_err(|_| SmcError::PopulationTooLarge)?;
            let target = (f64::from(index_u32) + offset) / population_width;
            while target >= cumulative + weights[ancestor] && ancestor < last_supported {
                cumulative += weights[ancestor];
                ancestor += 1;
            }
            children.push(Particle {
                state: parents[ancestor].state.clone(),
                log_weight: 0.0,
                status: parents[ancestor].status,
            });
        }
        self.particles = children;
        Ok(log_weight_sum - population_width.ln())
    }
}

/// SMC state or input was invalid.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SmcError {
    /// No particles were supplied.
    #[error("particle population must not be empty")]
    EmptyPopulation,
    /// Every particle had an unusable weight.
    #[error("particle population has no finite weight")]
    NoFiniteWeight,
    /// A weight increment was NaN or infinite.
    #[error("particle weight must be finite")]
    NonFiniteWeight,
    /// Adding a finite increment overflowed the finite log-weight representation.
    #[error("particle weight update overflowed")]
    WeightOverflow,
    /// A target or proposal log mass was NaN or positive infinity.
    #[error("target and proposal log masses must be finite or negative infinity")]
    NonFiniteImportanceLogprob,
    /// The proposal assigns zero mass to an importance update.
    #[error("proposal has zero support for an importance update")]
    ProposalHasNoSupport,
    /// A terminal particle was asked to take another transition.
    #[error("absorbed particle cannot transition")]
    AbsorbedParticle,
    /// A zero-mass particle was asked to take another transition.
    #[error("impossible particle cannot transition")]
    ImpossibleParticle,
    /// Systematic resampling requires an offset in `[0, 1)`.
    #[error("resampling offset must be finite and in [0, 1)")]
    InvalidOffset,
    /// The population cannot be represented by the deterministic sampler.
    #[error("particle population is too large")]
    PopulationTooLarge,
}

#[cfg(test)]
mod tests {
    use super::{ParticleSet, SmcError};

    #[test]
    fn weights_ess_and_absorption_are_explicit() {
        let mut set = ParticleSet::new(vec!["a", "b", "c"]).expect("population");
        set.particles[0].add_log_weight(0.0).expect("finite");
        set.particles[1]
            .add_log_weight(2.0_f64.ln())
            .expect("finite");
        set.particles[2].absorb().expect("live");
        let (weights, log_weight_sum) = set.normalized_weights().expect("weights");
        assert!((log_weight_sum.exp() - 4.0).abs() < 1e-12);
        assert!((weights[1] - 0.5).abs() < 1e-12);
        assert!((set.effective_sample_size().expect("ess") - 8.0 / 3.0).abs() < 1e-12);
        assert!(set.particles()[2].is_absorbed());
    }

    #[test]
    fn systematic_resampling_is_deterministic_and_preserves_absorption() {
        let mut set = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        set.particles[2]
            .add_log_weight(2.0_f64.ln())
            .expect("finite");
        set.particles[2].absorb().expect("live");
        let log_mean = set.systematic_resample(0.2).expect("resample");
        assert!((log_mean.exp() - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(
            set.particles()
                .iter()
                .map(|p| *p.state())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(set.particles()[2].is_absorbed());
        assert!(
            set.particles()
                .iter()
                .all(|particle| particle.log_weight() == 0.0)
        );
    }

    #[test]
    fn invalid_population_inputs_fail_closed() {
        assert_eq!(
            ParticleSet::<u8>::new(Vec::new()),
            Err(SmcError::EmptyPopulation)
        );
        let mut set = ParticleSet::new(vec![1_u8]).expect("population");
        assert_eq!(set.systematic_resample(1.0), Err(SmcError::InvalidOffset));
        assert_eq!(
            set.particles[0].add_log_weight(f64::NAN),
            Err(SmcError::NonFiniteWeight)
        );
    }

    #[test]
    fn terminal_impossible_and_overflowed_particles_fail_closed() {
        let mut set = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        set.particles[0].absorb().expect("live");
        assert_eq!(
            set.particles[0].add_log_weight(0.0),
            Err(SmcError::AbsorbedParticle)
        );

        set.particles[1].reject().expect("live");
        assert!(set.particles[1].is_impossible());
        assert_eq!(
            set.particles[1].log_weight().to_bits(),
            f64::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            set.particles[1].add_log_weight(0.0),
            Err(SmcError::ImpossibleParticle)
        );
        assert_eq!(set.particles[1].absorb(), Err(SmcError::ImpossibleParticle));

        set.particles[2]
            .add_log_weight(f64::MAX)
            .expect("representable");
        assert_eq!(
            set.particles[2].add_log_weight(f64::MAX),
            Err(SmcError::WeightOverflow)
        );
        assert_eq!(set.particles[2].log_weight().to_bits(), f64::MAX.to_bits());
    }

    #[test]
    fn rejected_particles_have_zero_normalized_mass() {
        let mut set = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        set.particles[2].reject().expect("live");
        let (weights, log_weight_sum) = set.normalized_weights().expect("live support");
        assert_eq!(weights, [0.5, 0.5, 0.0]);
        assert!((log_weight_sum.exp() - 2.0).abs() < 1e-12);
        assert!((set.log_mean_weight().expect("live support").exp() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn extinct_population_has_no_ancestry_distribution() {
        let mut set = ParticleSet::new(vec![0_u8, 1]).expect("population");
        for particle in &mut set.particles {
            particle.reject().expect("live");
        }
        let before = set.clone();
        assert_eq!(set.normalized_weights(), Err(SmcError::NoFiniteWeight));
        assert_eq!(set.effective_sample_size(), Err(SmcError::NoFiniteWeight));
        assert_eq!(set.systematic_resample(0.5), Err(SmcError::NoFiniteWeight));
        assert_eq!(set, before);
    }

    #[test]
    #[allow(
        clippy::manual_midpoint,
        reason = "reproduce the resampling target rounding expression exactly"
    )]
    fn systematic_resampling_never_revives_zero_mass_cdf_endpoints() {
        let final_offset = f64::from_bits(1.0_f64.to_bits() - 1);
        // The mathematical target remains below one, but this ordinary
        // floating-point formulation rounds its final systematic point up.
        assert_eq!(((1.0 + final_offset) / 2.0).to_bits(), 1.0_f64.to_bits());

        // A trailing rejected parent must remain unselectable at that rounded
        // endpoint. The two supported parents have normalized weights 1/2.
        let mut trailing = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        trailing.particles[2].reject().expect("live");
        trailing
            .systematic_resample(final_offset)
            .expect("live support");
        assert_eq!(
            trailing
                .particles()
                .iter()
                .map(|particle| *particle.state())
                .collect::<Vec<_>>(),
            [0, 1, 1]
        );
        assert!(
            trailing
                .particles()
                .iter()
                .all(|particle| !particle.is_impossible())
        );

        // A leading zero must be skipped even when offset zero produces the
        // left CDF endpoint exactly.
        let mut leading = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        leading.particles[0].reject().expect("live");
        leading.systematic_resample(0.0).expect("live support");
        assert_eq!(
            leading
                .particles()
                .iter()
                .map(|particle| *particle.state())
                .collect::<Vec<_>>(),
            [1, 1, 2]
        );

        // A single supported parent absorbs every systematic point, including
        // the rounded endpoint, with both zero-mass neighbours excluded.
        let mut concentrated = ParticleSet::new(vec![0_u8, 1, 2]).expect("population");
        concentrated.particles[0].reject().expect("live");
        concentrated.particles[2].reject().expect("live");
        concentrated
            .systematic_resample(final_offset)
            .expect("live support");
        assert!(
            concentrated
                .particles()
                .iter()
                .all(|particle| *particle.state() == 1 && !particle.is_impossible())
        );
    }
}
