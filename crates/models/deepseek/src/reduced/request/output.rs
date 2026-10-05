//! Owned numerical boundaries returned by one request step.

use crate::{
    attention::layer::LayerAttentionDiagnostic,
    reduced::{
        AttentionInputOutput, BlockTailDiagnostic, EngramStepOutput, FinalHeadOutput,
        LayerFourStepOutput, LayerOneStepOutput, LayerThreeStepOutput, StartupStepOutput,
    },
};

/// The attention stages of one scheduled layer, by kind.
#[derive(Debug)]
#[non_exhaustive]
pub enum ScheduledAttentionOutput {
    /// Window-only attention stages.
    WindowOnly(LayerAttentionDiagnostic),
    /// Ratio-two owner, direct-score, and attention stages.
    RatioTwoOwner(LayerOneStepOutput),
    /// Attention over the latest ratio-two publication.
    RatioTwoConsumer(LayerAttentionDiagnostic),
    /// Ratio-one owner, candidate, selection, and attention stages.
    RatioOneOwner(LayerThreeStepOutput),
    /// Candidate-restricted scores, selection, and attention stages.
    RatioOneIndexer(LayerFourStepOutput),
    /// Attention over the ratio-one owner's KV and latest ratio-one indices.
    RatioOneConsumer(LayerAttentionDiagnostic),
}

impl ScheduledAttentionOutput {
    /// Returns this layer's attention output rows.
    #[must_use]
    pub fn final_output(&self) -> &[u16] {
        match self {
            Self::WindowOnly(attention)
            | Self::RatioTwoConsumer(attention)
            | Self::RatioOneConsumer(attention) => &attention.final_output,
            Self::RatioTwoOwner(layer) => &layer.attention().final_output,
            Self::RatioOneOwner(layer) => &layer.attention().final_output,
            Self::RatioOneIndexer(layer) => &layer.attention().final_output,
        }
    }
}

/// Owned numerical boundaries of one scheduled layer in one request chunk.
#[derive(Debug)]
pub struct LayerStepOutput {
    pub(super) engram: Option<EngramStepOutput>,
    pub(super) attention_input: Vec<AttentionInputOutput>,
    pub(super) attention: ScheduledAttentionOutput,
    pub(super) tails: Vec<BlockTailDiagnostic>,
}

impl LayerStepOutput {
    /// Returns the Engram operation applied before this block, if scheduled.
    #[must_use]
    pub const fn engram(&self) -> Option<&EngramStepOutput> {
        self.engram.as_ref()
    }
    /// Returns HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_input(&self) -> &[AttentionInputOutput] {
        &self.attention_input
    }
    /// Returns this layer's attention stages.
    #[must_use]
    pub const fn attention(&self) -> &ScheduledAttentionOutput {
        &self.attention
    }
    /// Returns block tail diagnostics.
    #[must_use]
    pub fn tails(&self) -> &[BlockTailDiagnostic] {
        &self.tails
    }
}

/// Owned numerical boundaries from one scheduled request chunk.
///
/// The numbered accessors (`layer_one` through `tails_four`) name the
/// reduced schedule built by
/// [`RequestModel::new`](super::RequestModel::new); [`Self::layers`] covers
/// any schedule.
#[derive(Debug)]
pub struct RequestStepOutput {
    pub(super) startup: StartupStepOutput,
    pub(super) layers: Vec<LayerStepOutput>,
    pub(super) residual: Vec<u16>,
    pub(super) incoming_pre: Vec<f32>,
    pub(super) heads: Vec<FinalHeadOutput>,
}

impl RequestStepOutput {
    fn numbered(&self, number: usize) -> &LayerStepOutput {
        self.layers
            .get(number - 1)
            .expect("numbered accessors require the reduced schedule")
    }

    fn engram_at(&self, number: usize) -> &EngramStepOutput {
        self.numbered(number)
            .engram()
            .expect("numbered Engram accessors require the reduced schedule")
    }

    /// Returns startup stages and its first block tail.
    #[must_use]
    pub const fn startup(&self) -> &StartupStepOutput {
        &self.startup
    }
    /// Returns every scheduled layer's boundaries in execution order.
    #[must_use]
    pub fn layers(&self) -> &[LayerStepOutput] {
        &self.layers
    }
    /// Returns the first live Engram operation.
    ///
    /// # Panics
    ///
    /// Panics unless layer one applied an Engram, as in the reduced schedule.
    #[must_use]
    pub fn engram_one(&self) -> &EngramStepOutput {
        self.engram_at(1)
    }
    /// Returns L1 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer one.
    #[must_use]
    pub fn attention_one(&self) -> &[AttentionInputOutput] {
        self.numbered(1).attention_input()
    }
    /// Returns the L1 owner, direct-score, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer one is a ratio-two owner, as in the reduced schedule.
    #[must_use]
    pub fn layer_one(&self) -> &LayerOneStepOutput {
        match self.numbered(1).attention() {
            ScheduledAttentionOutput::RatioTwoOwner(layer) => layer,
            _ => panic!("reduced layer one is a ratio-two owner"),
        }
    }
    /// Returns numbered block-one tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer one.
    #[must_use]
    pub fn tails_one(&self) -> &[BlockTailDiagnostic] {
        self.numbered(1).tails()
    }
    /// Returns L2 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer two.
    #[must_use]
    pub fn attention_two(&self) -> &[AttentionInputOutput] {
        self.numbered(2).attention_input()
    }
    /// Returns L2 attention over the live L1 publication.
    ///
    /// # Panics
    ///
    /// Panics unless layer two is a ratio-two consumer, as in the reduced schedule.
    #[must_use]
    pub fn layer_two(&self) -> &LayerAttentionDiagnostic {
        match self.numbered(2).attention() {
            ScheduledAttentionOutput::RatioTwoConsumer(layer) => layer,
            _ => panic!("reduced layer two is a ratio-two consumer"),
        }
    }
    /// Returns numbered block-two tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer two.
    #[must_use]
    pub fn tails_two(&self) -> &[BlockTailDiagnostic] {
        self.numbered(2).tails()
    }
    /// Returns the second live Engram operation.
    ///
    /// # Panics
    ///
    /// Panics unless layer three applied an Engram, as in the reduced schedule.
    #[must_use]
    pub fn engram_three(&self) -> &EngramStepOutput {
        self.engram_at(3)
    }
    /// Returns L3 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer three.
    #[must_use]
    pub fn attention_three(&self) -> &[AttentionInputOutput] {
        self.numbered(3).attention_input()
    }
    /// Returns L3 owner, candidate, selection, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer three is a ratio-one owner, as in the reduced schedule.
    #[must_use]
    pub fn layer_three(&self) -> &LayerThreeStepOutput {
        match self.numbered(3).attention() {
            ScheduledAttentionOutput::RatioOneOwner(layer) => layer,
            _ => panic!("reduced layer three is a ratio-one owner"),
        }
    }
    /// Returns numbered block-three tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer three.
    #[must_use]
    pub fn tails_three(&self) -> &[BlockTailDiagnostic] {
        self.numbered(3).tails()
    }
    /// Returns L4 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer four.
    #[must_use]
    pub fn attention_four(&self) -> &[AttentionInputOutput] {
        self.numbered(4).attention_input()
    }
    /// Returns L4 scores, selection, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer four is a ratio-one indexer, as in the reduced schedule.
    #[must_use]
    pub fn layer_four(&self) -> &LayerFourStepOutput {
        match self.numbered(4).attention() {
            ScheduledAttentionOutput::RatioOneIndexer(layer) => layer,
            _ => panic!("reduced layer four is a ratio-one indexer"),
        }
    }
    /// Returns numbered block-four tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer four.
    #[must_use]
    pub fn tails_four(&self) -> &[BlockTailDiagnostic] {
        self.numbered(4).tails()
    }
    /// Returns the final copy-major residual rows.
    #[must_use]
    pub fn residual(&self) -> &[u16] {
        &self.residual
    }
    /// Returns final per-copy incoming coefficients.
    #[must_use]
    pub fn incoming_pre(&self) -> &[f32] {
        &self.incoming_pre
    }
    /// Returns one final-head result per token, or only the last token's
    /// under [`super::HeadPositions::Last`].
    #[must_use]
    pub fn heads(&self) -> &[FinalHeadOutput] {
        &self.heads
    }
}
