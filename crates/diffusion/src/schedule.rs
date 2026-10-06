//! Flow-matching Euler schedule, as diffusers' `FlowMatchEulerDiscreteScheduler`
//! computes it for the `FLUX.2` and Qwen-Image pipelines.
//!
//! The arithmetic follows the source's dtypes, not just its formulas: the
//! pipeline builds its starting sigmas with a float64 `linspace`, the
//! scheduler casts them to float32 and does every later shift in float32
//! (`NumPy` 2 keeps a float32 array float32 when combined with a Python float),
//! and only the dynamic-shift `mu` and its exponential stay in float64 until
//! they meet the array. Matching that order is what lets the sigmas agree to
//! the last bit rather than to a tolerance.
//!
//! Source: `schedulers/scheduling_flow_match_euler_discrete.py` and
//! `pipelines/flux2/pipeline_flux2_klein.py` at diffusers v0.41.0.

use serde::Deserialize;
use thiserror::Error;

const SCHEDULER_CLASS: &str = "FlowMatchEulerDiscreteScheduler";

/// How a dynamic `mu` bends the sigma curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeShift {
    /// `e^mu / (e^mu + (1/t - 1))`.
    Exponential,
    /// `mu / (mu + (1/t - 1))`.
    Linear,
}

/// The subset of a `scheduler_config.json` that this schedule executes.
///
/// Options that change the sampler itself (Karras, exponential or beta
/// sigmas, inverted sigmas, stochastic sampling) are refused at parse time
/// rather than ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchConfig {
    num_train_timesteps: u32,
    shift: f32,
    use_dynamic_shifting: bool,
    base_shift: f64,
    max_shift: f64,
    base_image_seq_len: u32,
    max_image_seq_len: u32,
    shift_terminal: Option<f32>,
    time_shift: TimeShift,
}

#[derive(Debug, Error, PartialEq)]
pub enum ScheduleError {
    #[error("scheduler config is not valid JSON for {SCHEDULER_CLASS}: {0}")]
    Json(String),
    #[error("scheduler class {0:?} is not {SCHEDULER_CLASS}")]
    Class(String),
    #[error("scheduler option {0} is set; this schedule does not implement it")]
    Unsupported(&'static str),
    #[error("time_shift_type {0:?} is neither \"exponential\" nor \"linear\"")]
    TimeShiftType(String),
    #[error("dynamic shifting is enabled, so the pipeline must supply mu")]
    MissingMu,
    #[error("dynamic shifting is disabled, so mu must not be supplied")]
    UnexpectedMu,
    #[error("a schedule needs at least one step")]
    NoSteps,
    #[error("num_train_timesteps must be positive")]
    NoTrainingTimesteps,
    #[error("shift {0} must be finite and positive in float32")]
    InvalidShift(f32),
    #[error("mu {0} must be finite and yield a finite positive float32 shift scale")]
    InvalidMu(f64),
    #[error("shift_terminal {0} must lie in (0, 1)")]
    ShiftTerminal(f64),
    #[error("sigma at step {index} is outside the finite positive descending schedule: {value}")]
    InvalidSigma { index: usize, value: f32 },
    #[error("max_image_seq_len must exceed base_image_seq_len")]
    SeqLenRange,
    /// diffusers divides by `1 - last_sigma` to stretch to `shift_terminal`;
    /// when the last sigma is 1 (a one-step schedule) it returns NaN sigmas.
    #[error("terminal stretching requires a finite positive float32 denominator")]
    DegenerateTerminal,
}

// The flags mirror the JSON's own booleans; each is checked once and dropped.
#[allow(clippy::struct_excessive_bools)]
#[derive(Deserialize)]
struct RawConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    num_train_timesteps: u32,
    #[serde(default = "one")]
    shift: f32,
    #[serde(default)]
    use_dynamic_shifting: bool,
    #[serde(default = "half")]
    base_shift: f64,
    #[serde(default = "default_max_shift")]
    max_shift: f64,
    #[serde(default = "default_base_len")]
    base_image_seq_len: u32,
    #[serde(default = "default_max_len")]
    max_image_seq_len: u32,
    #[serde(default)]
    shift_terminal: Option<f64>,
    #[serde(default = "default_time_shift")]
    time_shift_type: String,
    #[serde(default)]
    invert_sigmas: bool,
    #[serde(default)]
    use_karras_sigmas: bool,
    #[serde(default)]
    use_exponential_sigmas: bool,
    #[serde(default)]
    use_beta_sigmas: bool,
    #[serde(default)]
    stochastic_sampling: bool,
}

// Defaults are the scheduler constructor's keyword defaults.
fn one() -> f32 {
    1.0
}
fn half() -> f64 {
    0.5
}
fn default_max_shift() -> f64 {
    1.15
}
fn default_base_len() -> u32 {
    256
}
fn default_max_len() -> u32 {
    4096
}
fn default_time_shift() -> String {
    "exponential".to_owned()
}

impl FlowMatchConfig {
    /// Parses a diffusers `scheduler/scheduler_config.json`.
    pub fn from_json(text: &str) -> Result<Self, ScheduleError> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|error| ScheduleError::Json(error.to_string()))?;
        if raw.class_name != SCHEDULER_CLASS {
            return Err(ScheduleError::Class(raw.class_name));
        }
        for (set, name) in [
            (raw.invert_sigmas, "invert_sigmas"),
            (raw.use_karras_sigmas, "use_karras_sigmas"),
            (raw.use_exponential_sigmas, "use_exponential_sigmas"),
            (raw.use_beta_sigmas, "use_beta_sigmas"),
            (raw.stochastic_sampling, "stochastic_sampling"),
        ] {
            if set {
                return Err(ScheduleError::Unsupported(name));
            }
        }
        let time_shift = match raw.time_shift_type.as_str() {
            "exponential" => TimeShift::Exponential,
            "linear" => TimeShift::Linear,
            _ => return Err(ScheduleError::TimeShiftType(raw.time_shift_type)),
        };
        if raw.num_train_timesteps == 0 {
            return Err(ScheduleError::NoTrainingTimesteps);
        }
        if !raw.shift.is_finite() || raw.shift <= 0.0 {
            return Err(ScheduleError::InvalidShift(raw.shift));
        }
        // Test zero before narrowing: a nonzero terminal that underflows must
        // not silently turn off stretching. diffusers treats exact zero as unset.
        let shift_terminal = raw
            .shift_terminal
            .filter(|&value| value != 0.0)
            .map(|value| {
                #[allow(clippy::cast_possible_truncation)]
                let narrowed = value as f32;
                if value > 0.0 && value < 1.0 && narrowed > 0.0 && narrowed < 1.0 {
                    Ok(narrowed)
                } else {
                    Err(ScheduleError::ShiftTerminal(value))
                }
            })
            .transpose()?;
        if raw.max_image_seq_len <= raw.base_image_seq_len {
            return Err(ScheduleError::SeqLenRange);
        }
        Ok(Self {
            num_train_timesteps: raw.num_train_timesteps,
            shift: raw.shift,
            use_dynamic_shifting: raw.use_dynamic_shifting,
            base_shift: raw.base_shift,
            max_shift: raw.max_shift,
            base_image_seq_len: raw.base_image_seq_len,
            max_image_seq_len: raw.max_image_seq_len,
            shift_terminal,
            time_shift,
        })
    }

    /// Whether [`Sigmas::new`] needs a resolution-dependent `mu`.
    #[must_use]
    pub fn uses_dynamic_shifting(&self) -> bool {
        self.use_dynamic_shifting
    }
}

/// `FLUX.2`'s step- and resolution-dependent `mu`
/// (`compute_empirical_mu` in the `FLUX.2` pipelines).
///
/// Two lines fitted at 10 and 200 steps are interpolated in the step count;
/// beyond 4300 image tokens the 200-step line is used regardless of steps.
#[must_use]
#[allow(clippy::cast_precision_loss)] // sequence lengths and step counts are far below 2^52
pub fn flux2_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.738_095_24e-05, 1.898_333_33);
    let (a2, b2) = (0.000_169_27, 0.456_666_66);
    let len = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * len + b2;
    }
    let m_200 = a2 * len + b2;
    let m_10 = a1 * len + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

/// The linear-in-sequence-length `mu` (`calculate_shift` in the Qwen-Image
/// and `FLUX.1` pipelines), between the config's base and max shifts.
#[must_use]
pub fn linear_mu(config: &FlowMatchConfig, image_seq_len: usize) -> f64 {
    let base_len = f64::from(config.base_image_seq_len);
    let max_len = f64::from(config.max_image_seq_len);
    let m = (config.max_shift - config.base_shift) / (max_len - base_len);
    let b = config.base_shift - m * base_len;
    #[allow(clippy::cast_precision_loss)]
    let len = image_seq_len as f64;
    len * m + b
}

/// One denoising step: the model is called at `timestep` and the latent then
/// moves from `sigma` to `sigma_next`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowMatchStep {
    pub index: usize,
    pub sigma: f32,
    pub sigma_next: f32,
    /// `sigma * num_train_timesteps`, in float32 like the source.
    pub timestep: f32,
}

impl FlowMatchStep {
    /// The Euler update's step size: `x_next = x + dt * velocity`.
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.sigma_next - self.sigma
    }
}

/// A complete schedule: one sigma per step plus the terminal zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Sigmas {
    values: Vec<f32>,
    num_train_timesteps: f32,
}

impl Sigmas {
    /// Builds the schedule the `FLUX.2` and Qwen-Image pipelines pass to
    /// `set_timesteps`: starting sigmas `linspace(1, 1/steps, steps)`, then the
    /// configured shift and terminal stretch, then a terminal zero.
    ///
    /// `mu` is required exactly when the config enables dynamic shifting; use
    /// [`flux2_empirical_mu`] or [`linear_mu`] as the pipeline does.
    pub fn new(
        config: &FlowMatchConfig,
        steps: usize,
        mu: Option<f64>,
    ) -> Result<Self, ScheduleError> {
        if steps == 0 {
            return Err(ScheduleError::NoSteps);
        }
        #[allow(clippy::cast_possible_truncation)] // the source casts float64 to float32 here
        let mut values: Vec<f32> = linspace_to_inverse(steps)
            .into_iter()
            .map(|value| value as f32)
            .collect();

        match (config.use_dynamic_shifting, mu) {
            (true, None) => return Err(ScheduleError::MissingMu),
            (false, Some(_)) => return Err(ScheduleError::UnexpectedMu),
            (true, Some(mu)) => {
                if !mu.is_finite() {
                    return Err(ScheduleError::InvalidMu(mu));
                }
                // The scalar is formed in float64 and meets the float32 array as a
                // weak scalar, so it is rounded to float32 once.
                #[allow(clippy::cast_possible_truncation)]
                let scale = match config.time_shift {
                    TimeShift::Exponential => mu.exp() as f32,
                    TimeShift::Linear => mu as f32,
                };
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(ScheduleError::InvalidMu(mu));
                }
                for value in &mut values {
                    *value = scale / (scale + (1.0 / *value - 1.0));
                }
            }
            (false, None) => {
                let shift = config.shift;
                for value in &mut values {
                    *value = shift * *value / (1.0 + (shift - 1.0) * *value);
                }
            }
        }

        if let Some(terminal) = config.shift_terminal {
            let last = 1.0 - values[values.len() - 1];
            if !last.is_finite() || last <= 0.0 {
                return Err(ScheduleError::DegenerateTerminal);
            }
            let scale = last / (1.0 - terminal);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(ScheduleError::DegenerateTerminal);
            }
            for value in &mut values {
                *value = 1.0 - ((1.0 - *value) / scale);
            }
        }

        // Even finite positive parameters can overflow, underflow or cancel in
        // the reference's float32 arithmetic. Refuse the result rather than
        // clamp it and cease to match the reference.
        let mut previous = 1.0;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 || value > previous {
                return Err(ScheduleError::InvalidSigma { index, value });
            }
            previous = value;
        }
        values.push(0.0);
        #[allow(clippy::cast_precision_loss)]
        let num_train_timesteps = config.num_train_timesteps as f32;
        Ok(Self {
            values,
            num_train_timesteps,
        })
    }

    /// Number of model evaluations (without CFG).
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.values.len() - 1
    }

    /// Every sigma including the terminal zero, as the scheduler stores them.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The steps in order.
    pub fn steps(&self) -> impl ExactSizeIterator<Item = FlowMatchStep> + '_ {
        self.values
            .windows(2)
            .enumerate()
            .map(|(index, pair)| FlowMatchStep {
                index,
                sigma: pair[0],
                sigma_next: pair[1],
                timestep: pair[0] * self.num_train_timesteps,
            })
    }
}

/// `numpy.linspace(1.0, 1.0 / steps, steps)` in float64: `start + i * step`,
/// with the endpoint written exactly, and `[1.0]` for one step.
#[allow(clippy::cast_precision_loss)]
fn linspace_to_inverse(steps: usize) -> Vec<f64> {
    let start = 1.0_f64;
    let stop = 1.0 / steps as f64;
    if steps == 1 {
        return vec![start];
    }
    let increment = (stop - start) / (steps - 1) as f64;
    let mut values: Vec<f64> = (0..steps).map(|i| i as f64 * increment + start).collect();
    values[steps - 1] = stop;
    values
}

#[cfg(test)]
mod tests;
