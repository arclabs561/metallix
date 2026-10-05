//! MLX 0.32 keeps per-thread stream state, so a lazy graph built on one thread
//! and first evaluated on another can fail ("There is no Stream(cpu, N) in
//! current thread"). Serving builds and evaluates on one worker thread; this
//! pins what happens if a graph does cross threads: the right values or an
//! `Err` from `eval`, never an abort.
#![cfg(feature = "metal")]
#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::thread;

use mlx_rs::{Array, StreamOrDevice};

/// Builds `[1, 2, 3] * [4, 5, 6]` lazily on its own thread, then evaluates it
/// on this one.
fn built_elsewhere(stream: fn() -> StreamOrDevice) -> Result<Vec<f32>, String> {
    let product = thread::spawn(move || {
        let left = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let right = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]);
        left.multiply_device(&right, stream())
            .expect("lazy multiply")
    })
    .join()
    .expect("builder thread");
    product.eval().map_err(|error| error.to_string())?;
    Ok(product.as_slice::<f32>().to_vec())
}

#[test]
fn graphs_built_on_another_thread_evaluate_or_fail_cleanly() {
    for (name, stream) in [
        ("gpu", StreamOrDevice::gpu as fn() -> StreamOrDevice),
        ("cpu", StreamOrDevice::cpu as fn() -> StreamOrDevice),
    ] {
        match built_elsewhere(stream) {
            Ok(values) => assert_eq!(values, [4.0, 10.0, 18.0], "{name}"),
            Err(error) => eprintln!("{name}: cross-thread eval refused: {error}"),
        }
    }
}

#[test]
fn arrays_evaluated_on_another_thread_feed_new_graphs() {
    let (left, right) = thread::spawn(|| {
        let left = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3])
            .multiply_device(Array::from_slice(&[2.0_f32], &[1]), StreamOrDevice::gpu())
            .expect("lazy scale");
        let right = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]);
        left.eval().expect("evaluated where built");
        (left, right)
    })
    .join()
    .expect("builder thread");
    let product = left
        .multiply_device(&right, StreamOrDevice::gpu())
        .expect("lazy multiply");
    product
        .eval()
        .expect("graph built and evaluated on this thread");
    assert_eq!(product.as_slice::<f32>(), [8.0, 20.0, 36.0]);
}
