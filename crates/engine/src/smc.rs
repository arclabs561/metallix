//! Bounded sequential Monte Carlo state independent of model graphs.

use thiserror::Error;

/// One particle and its request-local state.
#[derive(Clone, Debug, PartialEq)]
pub struct Particle<T> {
    state: T,
    log_weight: f64,
    absorbed: bool,
}

impl<T> Particle<T> {
    /// Creates a live particle with a unit weight.
    #[must_use]
    pub fn new(state: T) -> Self {
        Self {
            state,
            log_weight: 0.0,
            absorbed: false,
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
        self.absorbed
    }

    /// Adds a finite incremental log weight.
    pub fn add_log_weight(&mut self, increment: f64) -> Result<(), SmcError> {
        if !increment.is_finite() {
            return Err(SmcError::NonFiniteWeight);
        }
        self.log_weight += increment;
        Ok(())
    }

    /// Marks the particle terminal; subsequent transitions must leave it unchanged.
    pub const fn absorb(&mut self) {
        self.absorbed = true;
    }
}

/// A finite particle population with deterministic, caller-supplied entropy.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSet<T> {
    particles: Vec<Particle<T>>,
}

impl<T: Clone> ParticleSet<T> {
    /// Creates a population from nonempty initial states.
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

    /// Returns normalized weights in population order and the log normalizer.
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

    /// Computes effective sample size from normalized log weights.
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
    /// resets child weights to equal mass and preserves absorbed states.
    pub fn systematic_resample(&mut self, offset: f64) -> Result<(), SmcError> {
        if !(0.0..1.0).contains(&offset) || !offset.is_finite() {
            return Err(SmcError::InvalidOffset);
        }
        let (weights, _) = self.normalized_weights()?;
        let count = self.len();
        let count_u32 = u32::try_from(count).map_err(|_| SmcError::PopulationTooLarge)?;
        let population_width = f64::from(count_u32);
        let mut cumulative = 0.0;
        let mut ancestor = 0;
        let parents = self.particles.clone();
        let mut children = Vec::with_capacity(count);
        for index in 0..count {
            let index_u32 = u32::try_from(index).map_err(|_| SmcError::PopulationTooLarge)?;
            let target = (f64::from(index_u32) + offset) / population_width;
            while target >= cumulative + weights[ancestor] && ancestor + 1 < count {
                cumulative += weights[ancestor];
                ancestor += 1;
            }
            children.push(Particle {
                state: parents[ancestor].state.clone(),
                log_weight: -population_width.ln(),
                absorbed: parents[ancestor].absorbed,
            });
        }
        self.particles = children;
        Ok(())
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
        set.particles[2].absorb();
        let (weights, normalizer) = set.normalized_weights().expect("weights");
        assert!((normalizer.exp() - 4.0).abs() < 1e-12);
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
        set.particles[2].absorb();
        set.systematic_resample(0.2).expect("resample");
        assert_eq!(
            set.particles()
                .iter()
                .map(|p| *p.state())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(set.particles()[2].is_absorbed());
        assert!(set.particles().iter().all(|p| p.log_weight().is_finite()));
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
}
