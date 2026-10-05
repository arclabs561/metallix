//! L1 lifecycle and preceding-publication policy with ordinary runtime operands.

use deepseek::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionLayout, LayerAttentionWeights},
    indexer::{
        cache::IndexKeyPublicationId,
        key::IndexKeyWeights,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
    },
    reduced::{
        LayerOneCall, LayerOneConfig, LayerOneSession, LayerOneSessionError, LayerOneStepOutput,
        PreviousLayerThreeKeys, RatioTwoOwnerLayout, RatioTwoOwnerWeights,
    },
};
use std::num::NonZeroUsize;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

struct Operands {
    zero_codes: Vec<u8>,
    scales: Vec<u8>,
    norm: Vec<u16>,
    zero_bf16: Vec<u16>,
    zero_f32: Vec<f32>,
    frequencies: Vec<RotaryFrequency>,
}

impl Operands {
    fn new() -> Self {
        Self {
            zero_codes: vec![0; 32 * 32],
            scales: vec![127],
            norm: vec![0x3f80; 32],
            zero_bf16: vec![0; 32 * 32],
            zero_f32: vec![0.0; 32 * 32],
            frequencies: vec![RotaryFrequency::new(1.0, 0.0).unwrap(); 8],
        }
    }

    fn session(&self) -> LayerOneSession {
        self.session_with(|config| config)
    }

    fn session_with(
        &self,
        configure: impl FnOnce(LayerOneConfig) -> LayerOneConfig,
    ) -> LayerOneSession {
        let owner =
            RatioTwoOwnerLayout::new(nz(1), nz(32), nz(32), nz(32), nz(1), nz(4), 1.0e-6).unwrap();
        let attention = LayerAttentionLayout::new(
            nz(1),
            nz(32),
            nz(1),
            nz(32),
            nz(1),
            nz(32),
            nz(6),
            nz(1),
            nz(32),
            1,
            nz(2),
            1.0e-6,
            0.125,
        )
        .unwrap();
        LayerOneSession::new(
            configure(LayerOneConfig::new(owner, attention, nz(2)).unwrap()),
            &self.norm,
        )
        .unwrap()
    }

    fn step(
        &self,
        session: &mut LayerOneSession,
        positions: usize,
        previous: Option<PreviousLayerThreeKeys<'_>>,
        bad_attention: bool,
    ) -> Result<LayerOneStepOutput, LayerOneSessionError> {
        let fp8 = Fp8Projection {
            codes: &self.zero_codes,
            scales: &self.scales,
        };
        let query = CandidateQueryWeights {
            wq_a: fp8,
            q_norm: &self.norm,
            index: IndexQueryWeights {
                wq_b_codes: &self.zero_codes,
                wq_b_scales: &self.scales,
                weights_proj: &self.zero_bf16[..32],
            },
        };
        let attention = LayerAttentionWeights {
            wq_a: fp8,
            q_norm: if bad_attention { &[] } else { &self.norm },
            wq_b: fp8,
            wkv: fp8,
            kv_norm: &self.norm,
            attn_sink: &self.zero_f32[..1],
            wo_a: &self.zero_bf16,
            wo_b: fp8,
        };
        session.step(LayerOneCall::new(
            &self.zero_bf16[..positions * 32],
            nz(positions),
            &self.frequencies,
            RatioTwoOwnerWeights::new(
                &self.zero_f32,
                &self.zero_f32,
                IndexKeyWeights::new(&self.zero_bf16, &self.norm),
            ),
            query,
            CandidateQueryLayout::new(
                IndexQueryLayout::new(nz(1), nz(32), nz(32), nz(1), nz(32), nz(1)).unwrap(),
                1.0e-6,
            )
            .unwrap(),
            attention,
            previous,
        ))
    }
}

#[test]
fn partial_group_scores_previous_layer_three_keys_but_keeps_own_kv() {
    let operands = Operands::new();
    let mut session = operands.session();
    let first = operands.step(&mut session, 4, None, false).unwrap();
    let prior_keys = vec![0x3f80; 4 * 32];
    let previous = PreviousLayerThreeKeys::new(IndexKeyPublicationId::new(3, 0, 0), &prior_keys);
    let partial = operands
        .step(&mut session, 1, Some(previous), false)
        .unwrap();
    assert!(partial.owner().latent().is_none());
    assert_eq!(partial.score_key_prefix(), &prior_keys[..2 * 32]);
    assert_ne!(partial.score_key_prefix(), partial.key_prefix());
    assert_eq!(partial.key_prefix(), first.key_prefix());
    assert_eq!(partial.kv_prefix(), first.kv_prefix());
    assert_eq!(partial.selected_indices(), &[6, 7]);
    assert_eq!(partial.publication(), IndexKeyPublicationId::new(1, 0, 1));
    assert_eq!(session.next_start(), 5);
}

#[test]
fn late_attention_failure_requires_reset_and_rejects_old_epoch_keys() {
    let operands = Operands::new();
    let mut session = operands.session();
    assert!(matches!(
        operands.step(&mut session, 4, None, true),
        Err(LayerOneSessionError::Attention(_))
    ));
    assert_eq!(
        session.next_start(),
        4,
        "owner committed before attention rejected"
    );
    assert!(session.is_poisoned());
    assert!(matches!(
        operands.step(&mut session, 4, None, false),
        Err(LayerOneSessionError::Poisoned)
    ));
    session.reset().unwrap();
    let recovered = operands.step(&mut session, 4, None, false).unwrap();
    assert_eq!(recovered.publication(), IndexKeyPublicationId::new(1, 1, 0));
    assert_eq!(recovered.attention().final_output, vec![0; 4 * 32]);
    let prior_keys = vec![0x3f80; 4 * 32];
    let stale = PreviousLayerThreeKeys::new(IndexKeyPublicationId::new(3, 0, 0), &prior_keys);
    assert!(matches!(
        operands.step(&mut session, 1, Some(stale), false),
        Err(LayerOneSessionError::PreviousLayerThreeIdentity {
            epoch: 0,
            expected_epoch: 1,
            ..
        })
    ));
    assert!(session.is_poisoned());
    session.reset().unwrap();
    operands.step(&mut session, 4, None, false).unwrap();
    let current = PreviousLayerThreeKeys::new(IndexKeyPublicationId::new(3, 2, 0), &prior_keys);
    let partial = operands
        .step(&mut session, 1, Some(current), false)
        .unwrap();
    assert_eq!(partial.publication(), IndexKeyPublicationId::new(1, 2, 1));
    assert_eq!(partial.score_key_prefix(), &prior_keys[..2 * 32]);
}

#[test]
fn incomplete_group_accepts_only_the_configured_previous_owner_layer() {
    let operands = Operands::new();
    let prior_keys = vec![0x3f80; 4 * 32];
    let from =
        |layer| PreviousLayerThreeKeys::new(IndexKeyPublicationId::new(layer, 0, 0), &prior_keys);

    let mut default = operands.session();
    operands.step(&mut default, 4, None, false).unwrap();
    assert!(matches!(
        operands.step(&mut default, 1, Some(from(20)), false),
        Err(LayerOneSessionError::PreviousLayerThreeIdentity {
            source_layer: 20,
            expected_source_layer: 3,
            ..
        })
    ));

    let mut configured = operands.session_with(|config| config.with_previous_owner_layer(20));
    operands.step(&mut configured, 4, None, false).unwrap();
    let partial = operands
        .step(&mut configured, 1, Some(from(20)), false)
        .unwrap();
    assert_eq!(partial.score_key_prefix(), &prior_keys[..2 * 32]);

    let mut configured = operands.session_with(|config| config.with_previous_owner_layer(20));
    operands.step(&mut configured, 4, None, false).unwrap();
    assert!(matches!(
        operands.step(&mut configured, 1, Some(from(3)), false),
        Err(LayerOneSessionError::PreviousLayerThreeIdentity {
            source_layer: 3,
            expected_source_layer: 20,
            ..
        })
    ));
}
