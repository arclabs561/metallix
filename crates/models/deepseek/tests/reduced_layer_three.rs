//! Runtime L3 construction rejects incompatible producer/consumer geometry.

use deepseek::{
    attention::layer::LayerAttentionLayout,
    indexer::key::IndexKeyLayout,
    reduced::{LayerThreeConfig, LayerThreeSession, LayerThreeSessionError},
};
use std::num::NonZeroUsize;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

#[test]
fn rejects_incompatible_publication_geometry_before_allocating_state() {
    let keys = IndexKeyLayout::new(nz(1), nz(64), nz(64), nz(16), 1.0e-20).unwrap();
    let attention = |source, ratio| {
        LayerAttentionLayout::new(
            nz(1),
            nz(128),
            nz(2),
            nz(64),
            nz(16),
            nz(32),
            nz(6),
            nz(1),
            nz(32),
            source,
            nz(ratio),
            1.0e-20,
            0.125,
        )
        .unwrap()
    };
    assert!(LayerThreeConfig::new(keys, nz(128), nz(16), attention(3, 1), nz(6), nz(1)).is_ok());
    // Any compressed source layer is accepted; the session checks it against its owner.
    assert!(LayerThreeConfig::new(keys, nz(128), nz(16), attention(20, 1), nz(6), nz(1)).is_ok());
    assert!(matches!(
        LayerThreeConfig::new(keys, nz(128), nz(16), attention(3, 2), nz(6), nz(1)),
        Err(LayerThreeSessionError::AttentionCompressionRatio)
    ));
    assert!(matches!(
        LayerThreeConfig::new(keys, nz(128), nz(16), attention(3, 1), nz(5), nz(1)),
        Err(LayerThreeSessionError::WindowMismatch { .. })
    ));
    assert!(matches!(
        LayerThreeConfig::new(keys, nz(64), nz(16), attention(3, 1), nz(6), nz(1)),
        Err(LayerThreeSessionError::InputDimensionMismatch { .. })
    ));
}

#[test]
fn session_source_layer_must_match_its_attention_layout() {
    let keys = IndexKeyLayout::new(nz(1), nz(64), nz(64), nz(16), 1.0e-20).unwrap();
    let attention = LayerAttentionLayout::new(
        nz(1),
        nz(128),
        nz(2),
        nz(64),
        nz(16),
        nz(32),
        nz(6),
        nz(1),
        nz(32),
        20,
        nz(1),
        1.0e-20,
        0.125,
    )
    .unwrap();
    let config = LayerThreeConfig::new(keys, nz(128), nz(16), attention, nz(6), nz(1)).unwrap();
    let norm = [0x3f80; 64];
    assert!(LayerThreeSession::new(config, 20, &norm, 1.0e-20).is_ok());
    assert!(matches!(
        LayerThreeSession::new(config, 3, &norm, 1.0e-20),
        Err(LayerThreeSessionError::SourceLayer { actual: 3 })
    ));
}
