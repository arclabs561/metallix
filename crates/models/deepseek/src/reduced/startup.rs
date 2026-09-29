//! Request-local token startup through the window-only first block.
use thiserror::Error;

use super::{
    AttentionInput, AttentionInputError, BlockTailDiagnostic, BlockTailError, BlockTailReference,
};
use crate::{
    RotaryFrequency, StartupError, StartupLayout, StartupOutput,
    attention::layer::{
        LayerAttentionDiagnostic, LayerAttentionError, LayerAttentionLayout, LayerAttentionState,
        LayerAttentionWeights,
    },
    startup_bf16_reference,
};

const MAX_ELEMENTS: usize = 1 << 20;

/// Borrowed immutable operands and owned first-block window state.
///
/// A failure after an admitted call poisons this session. `reset` is required
/// before reuse because attention may have committed before a block-tail error.
#[derive(Debug)]
pub struct StartupSession<'a> {
    table: &'a [u16],
    layout: StartupLayout,
    input: AttentionInput<'a>,
    weights: LayerAttentionWeights<'a>,
    tail: BlockTailReference<'a>,
    attention: LayerAttentionState,
    width: usize,
    copies: usize,
    next_start: usize,
    poisoned: bool,
}

impl<'a> StartupSession<'a> {
    /// Constructs a bounded batch-one window-only startup block.
    pub fn new(
        table: &'a [u16],
        norm: &'a [u16],
        epsilon: f32,
        attention_layout: LayerAttentionLayout,
        weights: LayerAttentionWeights<'a>,
        tail: BlockTailReference<'a>,
    ) -> Result<Self, StartupSessionError> {
        let (copies, width) = tail.geometry();
        if norm.len() != width
            || !table.len().is_multiple_of(width)
            || !attention_layout.is_batch_one_window_only(width)
        {
            return Err(StartupSessionError::Geometry);
        }
        let layout = StartupLayout::new(table.len() / width, width, copies)?;
        let input = AttentionInput::new(norm, copies, epsilon)?;
        Ok(Self {
            table,
            layout,
            input,
            weights,
            tail,
            attention: LayerAttentionState::new(attention_layout),
            width,
            copies,
            next_start: 0,
            poisoned: false,
        })
    }

    /// Returns the next admissible absolute token position.
    #[must_use]
    pub const fn next_start(&self) -> usize {
        self.next_start
    }

    /// Reports whether the first-block state requires reset.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Clears the window and cursor, retaining the supplied immutable operands.
    pub fn reset(&mut self) -> Result<(), StartupSessionError> {
        self.attention.reset()?;
        self.next_start = 0;
        self.poisoned = false;
        Ok(())
    }

    /// Executes caller-supplied token IDs and positional frequencies.
    ///
    /// An out-of-order call is rejected without invalidating an otherwise healthy
    /// session. Any error during an admitted call invalidates the entire session.
    pub fn step(
        &mut self,
        start: usize,
        ids: &[u64],
        frequencies: &[RotaryFrequency],
    ) -> Result<StartupStepOutput, StartupSessionError> {
        if self.poisoned {
            return Err(StartupSessionError::Poisoned);
        }
        if start != self.next_start {
            return Err(StartupSessionError::UnexpectedStart {
                expected: self.next_start,
                actual: start,
            });
        }
        self.poisoned = true;
        let next_start = start
            .checked_add(ids.len())
            .ok_or(StartupSessionError::ElementLimit)?;
        let startup = startup_bf16_reference(ids, self.table, self.layout)?;
        let mut attention_input = reserve(ids.len().checked_mul(self.width))?;
        for (residual, pre) in startup
            .residual_bf16()
            .chunks_exact(self.width * self.copies)
            .zip(startup.identity_pre().chunks_exact(self.copies))
        {
            attention_input.extend_from_slice(self.input.forward(residual, pre)?.normalized_bf16());
        }
        let mut residual = reserve(Some(startup.residual_bf16().len()))?;
        let mut next_pre = reserve(Some(startup.identity_pre().len()))?;
        let mut tails = reserve(Some(ids.len()))?;
        let attention = self.attention.forward_window_only(
            &attention_input,
            start,
            frequencies,
            self.weights,
        )?;
        for (initial, attended) in startup
            .residual_bf16()
            .chunks_exact(self.width * self.copies)
            .zip(attention.final_output.chunks_exact(self.width))
        {
            let diagnostic = self.tail.forward_token(initial, attended)?;
            residual.extend_from_slice(diagnostic.ffn().output_bf16());
            next_pre.extend_from_slice(diagnostic.ffn().coefficients().pre());
            tails.push(diagnostic);
        }
        self.next_start = next_start;
        self.poisoned = false;
        Ok(StartupStepOutput {
            startup,
            attention_input,
            attention,
            tails,
            residual,
            next_pre,
        })
    }
}

/// Numerical boundaries from a completed first-block call.
#[derive(Debug)]
pub struct StartupStepOutput {
    startup: StartupOutput,
    attention_input: Vec<u16>,
    attention: LayerAttentionDiagnostic,
    tails: Vec<BlockTailDiagnostic>,
    residual: Vec<u16>,
    next_pre: Vec<f32>,
}

impl StartupStepOutput {
    #[must_use]
    pub const fn startup(&self) -> &StartupOutput {
        &self.startup
    }
    #[must_use]
    pub fn attention_input(&self) -> &[u16] {
        &self.attention_input
    }
    #[must_use]
    pub const fn attention(&self) -> &LayerAttentionDiagnostic {
        &self.attention
    }
    #[must_use]
    pub fn tails(&self) -> &[BlockTailDiagnostic] {
        &self.tails
    }
    #[must_use]
    pub fn residual(&self) -> &[u16] {
        &self.residual
    }
    #[must_use]
    pub fn next_pre(&self) -> &[f32] {
        &self.next_pre
    }
}

fn reserve<T>(elements: Option<usize>) -> Result<Vec<T>, StartupSessionError> {
    let elements = elements
        .filter(|&n| n <= MAX_ELEMENTS)
        .ok_or(StartupSessionError::ElementLimit)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| StartupSessionError::Allocation)?;
    Ok(output)
}

/// Invalid first-block operands or a failed numerical stage.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StartupSessionError {
    #[error("startup block geometry is incompatible")]
    Geometry,
    #[error("startup session requires reset after a failed call")]
    Poisoned,
    #[error("startup position {actual} differs from expected {expected}")]
    UnexpectedStart { expected: usize, actual: usize },
    #[error("startup session exceeds bounded element count")]
    ElementLimit,
    #[error("startup session allocation failed")]
    Allocation,
    #[error(transparent)]
    Startup(#[from] StartupError),
    #[error(transparent)]
    Input(#[from] AttentionInputError),
    #[error(transparent)]
    Attention(#[from] LayerAttentionError),
    #[error(transparent)]
    Tail(#[from] BlockTailError),
}
