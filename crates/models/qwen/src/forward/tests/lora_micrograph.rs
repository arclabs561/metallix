//! Test-only MLX LoRA training micrograph.
//!
//! This intentionally exercises the smallest useful training boundary: a
//! frozen base projection, trainable low-rank A/B adapters, MLX
//! `value_and_grad`, one optimizer update, and adapter-only safetensors
//! round-tripping. It does not expose a training or fine-tuning API.

use std::{collections::HashMap, fs, path::PathBuf};

use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters, ModuleParametersExt, Param};
use mlx_rs::{
    Array, array, nn, ops,
    optimizers::{Optimizer, Sgd},
};

use crate::GPU_TEST_LOCK;

const INPUTS: [[f32; 3]; 4] = [
    [1.0, -2.0, 0.5],
    [0.0, 1.5, -1.0],
    [2.0, 0.25, -0.75],
    [-1.0, 0.5, 2.0],
];
const TARGETS: [[f32; 2]; 4] = [[0.5, -1.0], [1.0, 0.25], [-0.5, 0.75], [1.25, 0.5]];

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
struct LoraProjection {
    /// Only A and B are adapter parameters. The base projection is frozen by
    /// construction and therefore never appears in flattened parameters.
    #[param]
    a: Param<Array>,
    #[param]
    b: Param<Array>,
    base: Array,
    alpha: f32,
}

impl LoraProjection {
    fn new() -> Self {
        Self {
            a: Param::new(array!([[0.10_f32, -0.05, 0.20], [0.03, 0.08, -0.07]])),
            b: Param::new(array!([[0.04_f32, -0.02], [0.06, 0.01]])),
            base: array!([[0.7_f32, -0.3, 0.2], [0.1, 0.4, -0.6]]),
            alpha: 2.0,
        }
    }

    fn forward_array(&mut self, inputs: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let base = ops::matmul(inputs, self.base.t())?;
        let low_rank = ops::matmul(&ops::matmul(inputs, self.a.t())?, self.b.t())?;
        base.add(&low_rank.multiply(array!(self.alpha / 2.0))?)
    }
}

impl Module<&Array> for LoraProjection {
    type Error = mlx_rs::error::Exception;
    type Output = Array;

    fn forward(&mut self, inputs: &Array) -> Result<Array, Self::Error> {
        self.forward_array(inputs)
    }

    fn training_mode(&mut self, _: bool) {}
}

fn arrays() -> (Array, Array) {
    (
        Array::from_slice(&INPUTS.concat(), &[4, 3]),
        Array::from_slice(&TARGETS.concat(), &[4, 2]),
    )
}

fn loss(
    model: &mut LoraProjection,
    (inputs, targets): (&Array, &Array),
) -> Result<Array, mlx_rs::error::Exception> {
    model
        .forward_array(inputs)?
        .subtract(targets)?
        .square()?
        .sum(None)
}

fn scalar(array: &Array) -> f32 {
    array.eval().expect("materialize scalar");
    array.as_slice::<f32>()[0]
}

fn max_abs(array: &Array) -> f32 {
    let absolute = array.abs().expect("absolute error");
    absolute.eval().expect("materialize error");
    absolute
        .as_slice::<f32>()
        .iter()
        .copied()
        .fold(0.0, f32::max)
}

fn adapter_metadata() -> HashMap<String, String> {
    HashMap::from([
        ("format".to_owned(), "metallix-lora-test-v1".to_owned()),
        ("base".to_owned(), "frozen-linear-projection".to_owned()),
        ("rank".to_owned(), "2".to_owned()),
        ("alpha".to_owned(), "2.0".to_owned()),
        ("trainable".to_owned(), "a,b".to_owned()),
    ])
}

fn temporary_adapter_path() -> PathBuf {
    std::env::temp_dir().join(format!("metallix-lora-{}.safetensors", std::process::id()))
}

#[test]
fn lora_value_and_grad_matches_independent_gradient_and_updates_only_adapter() {
    let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let (inputs, targets) = arrays();
    let mut model = LoraProjection::new();
    let base_before = model.base.clone();

    let mut value_and_grad = nn::value_and_grad(loss);
    let (before, gradients) =
        value_and_grad(&mut model, (&inputs, &targets)).expect("MLX gradients");
    let before = scalar(&before);

    assert_eq!(gradients.len(), 2, "only A and B are trainable");
    assert!(gradients.contains_key("a"));
    assert!(gradients.contains_key("b"));

    let prediction = model.forward_array(&inputs).expect("forward prediction");
    let error = prediction.subtract(&targets).expect("prediction error");
    let delta = error
        .multiply(array!(2.0_f32))
        .expect("squared-loss derivative");
    let hidden = ops::matmul(&inputs, model.a.t()).expect("adapter activation");
    let independent_b = ops::matmul(&delta.t(), &hidden).expect("independent B gradient");
    let independent_a = ops::matmul(&delta, &model.b)
        .expect("backpropagated adapter gradient")
        .t()
        .matmul(&inputs)
        .expect("independent A gradient");
    let a_error = max_abs(
        &gradients["a"]
            .subtract(&independent_a)
            .expect("A gradient diff"),
    );
    assert!(a_error < 1e-5, "A gradient error {a_error}");
    let b_error = max_abs(
        &gradients["b"]
            .subtract(&independent_b)
            .expect("B gradient diff"),
    );
    assert!(b_error < 1e-5, "B gradient error {b_error}");

    let mut optimizer = Sgd::new(0.05_f32);
    optimizer
        .update(&mut model, gradients)
        .expect("adapter update");
    let after = scalar(&loss(&mut model, (&inputs, &targets)).expect("post-update loss"));
    assert!(after < before, "one adapter update must lower loss");
    assert!(
        max_abs(
            &model
                .base
                .subtract(&base_before)
                .expect("base immutability")
        ) == 0.0
    );
}

#[test]
fn lora_adapter_export_resume_preserves_metadata_and_output() {
    let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let (inputs, targets) = arrays();
    let mut source = LoraProjection::new();
    let mut value_and_grad = nn::value_and_grad(loss);
    let (_, gradients) = value_and_grad(&mut source, (&inputs, &targets)).expect("MLX gradients");
    let mut optimizer = Sgd::new(0.05_f32);
    optimizer
        .update(&mut source, gradients)
        .expect("adapter update");
    let expected = source.forward_array(&inputs).expect("source output");

    let path = temporary_adapter_path();
    let metadata = adapter_metadata();
    let params = source.parameters().flatten();
    Array::save_safetensors(params, Some(&metadata), &path).expect("adapter-only export");
    let mut resumed = LoraProjection::new();
    resumed
        .load_safetensors(&path)
        .expect("adapter-only resume");
    let actual = resumed.forward_array(&inputs).expect("resumed output");
    assert!(max_abs(&actual.subtract(&expected).expect("resume output diff")) == 0.0);
    assert_eq!(
        resumed.parameters().flatten().len(),
        2,
        "base is excluded from export"
    );
    fs::remove_file(path).expect("remove test adapter artifact");
}
