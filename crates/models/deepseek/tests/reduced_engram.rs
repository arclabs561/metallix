use deepseek::{
    engram::EngramHashLayout,
    reduced::{EngramSession, EngramSessionConfig, EngramSessionError, EngramSessionWeights},
};

fn session() -> EngramSession {
    let hash_layout = EngramHashLayout::new(2, 1, 1, 1, vec![3], vec![0], vec![1, 1])
        .expect("bounded hash layout");
    let config =
        EngramSessionConfig::new(hash_layout, vec![0, 1], 0, 4, 1, 32, 2, 32, 1.0e-6, 1.0e-6)
            .expect("bounded session config");
    let weights = EngramSessionWeights::new(
        vec![0x38; 2 * 32],
        vec![127; 2],
        vec![0; 64 * 32],
        vec![127; 2],
        vec![0x3f80; 32],
        vec![0x3f80; 32],
    );
    EngramSession::new(config, weights).expect("bounded session weights")
}

#[test]
fn staged_late_failure_leaves_hash_history_retryable() {
    let mut failed = session();
    let mut fresh = session();
    let residual = vec![0x7f7f; 32];
    let error = failed
        .step(0, &[0], &residual)
        .expect_err("overflowing gate stream must fail after staged hashing");
    assert!(matches!(error, EngramSessionError::Gate(_)));
    assert_eq!(
        failed.next_start(),
        0,
        "failed gate must not advance cursor"
    );

    let valid = vec![0x3f80; 32];
    let retried = failed
        .step(0, &[0], &valid)
        .expect("retry after late gate failure");
    let control = fresh.step(0, &[0], &valid).expect("fresh control");
    assert_eq!(retried, control, "failure must not publish hash history");
}

#[test]
fn reset_reconstructs_fresh_history() {
    let mut used = session();
    let mut fresh = session();
    let residual = vec![0x3f80; 32];
    used.step(0, &[0, 1], &[residual.clone(), residual.clone()].concat())
        .expect("first request chunk");
    assert_eq!(used.next_start(), 2);
    used.reset().expect("fresh reset");
    assert_eq!(used.next_start(), 0);

    let restarted = used.step(0, &[0], &residual).expect("restarted request");
    let control = fresh.step(0, &[0], &residual).expect("fresh request");
    assert_eq!(restarted, control);
}

#[test]
fn constructor_accepts_partial_last_wkv_scale_group() {
    let hash_layout = EngramHashLayout::new(2, 1, 1, 1, vec![3], vec![0], vec![1, 1])
        .expect("bounded hash layout");
    let config =
        EngramSessionConfig::new(hash_layout, vec![0, 1], 0, 4, 1, 33, 2, 32, 1.0e-6, 1.0e-6)
            .expect("non-multiple WKV output width is valid");
    let weights = EngramSessionWeights::new(
        vec![0x38; 2 * 32],
        vec![127; 2],
        vec![0; 66 * 32],
        vec![127; 3],
        vec![0x3f80; 33],
        vec![0x3f80; 33],
    );

    let mut runtime = EngramSession::new(config, weights).expect("ceil-grouped WKV scales");
    let residual = vec![0x3f80; 33];
    let output = runtime
        .step(0, &[0], &residual)
        .expect("partial scale group executes");
    assert_eq!(
        output.output(),
        residual,
        "zero WKV preserves supplied residual"
    );
    assert_eq!(output.wkv().len(), 66);
}

#[test]
fn invalid_calls_preserve_the_contiguous_request_cursor() {
    let mut runtime = session();
    let residual = vec![0x3f80; 32];
    assert!(matches!(
        runtime.step(1, &[0], &residual),
        Err(EngramSessionError::UnexpectedStart { .. })
    ));
    assert!(matches!(
        runtime.step(0, &[], &[]),
        Err(EngramSessionError::EmptyChunk)
    ));
    assert!(matches!(
        runtime.step(0, &[0; 5], &[]),
        Err(EngramSessionError::ChunkExceedsCapacity { .. })
    ));
    assert!(matches!(
        runtime.step(0, &[-1], &residual),
        Err(EngramSessionError::TokenIdOutOfRange { .. })
    ));
    assert!(matches!(
        runtime.step(0, &[2], &residual),
        Err(EngramSessionError::TokenIdOutOfRange { .. })
    ));
    assert!(matches!(
        runtime.step(0, &[0], &residual[..31]),
        Err(EngramSessionError::Length { .. })
    ));
    assert!(matches!(
        runtime.step(0, &[0], &[0x7f80; 32]),
        Err(EngramSessionError::NonFiniteBf16 { .. })
    ));
    assert_eq!(runtime.next_start(), 0);
    let mut fresh = session();
    assert_eq!(
        runtime.step(0, &[0], &residual).unwrap(),
        fresh.step(0, &[0], &residual).unwrap()
    );
    assert!(matches!(
        runtime.step(0, &[0], &residual),
        Err(EngramSessionError::UnexpectedStart { .. })
    ));
    assert_eq!(
        runtime.step(1, &[1], &residual).unwrap(),
        fresh.step(1, &[1], &residual).unwrap()
    );
}
