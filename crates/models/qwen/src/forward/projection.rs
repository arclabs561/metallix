//! Weight precisions and the one path from a weight tensor to a matmul.
//!
//! Projections and the token embedding are stored either dense, in a float
//! dtype, or as MLX affine-quantized triples (`weight` packed into `u32`,
//! plus per-group `scales` and `biases`). Every projection goes through
//! [`project`], which picks the matmul from the checkpoint's declared
//! quantization and checks the tensor dtypes match it, so a packed tensor
//! cannot reach a dense matmul and a dense one cannot reach a quantized one.
//! A family whose projection carries an output bias (Qwen2's Q, K and V)
//! adds it after either matmul, in the activation dtype.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{collections::HashMap, hash::BuildHasher};

use mlx_rs::{Array, Dtype, StreamOrDevice, ops};

use super::{Qwen3FloatPrecision, Qwen3ForwardConfig, Qwen3ForwardError, weight};
use crate::quantization::Qwen3AffineQuantization;

/// How resident weights are stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3WeightPrecision {
    /// Every tensor in one float dtype.
    Dense(Qwen3FloatPrecision),
    /// Projections and the token embedding affine-quantized; norms, scales
    /// and biases, and so activations and cached K/V, in `activations`.
    Affine {
        /// The packed layout.
        quantization: Qwen3AffineQuantization,
        /// Dtype of every unpacked tensor.
        activations: Qwen3FloatPrecision,
    },
}

impl From<Qwen3FloatPrecision> for Qwen3WeightPrecision {
    fn from(precision: Qwen3FloatPrecision) -> Self {
        Self::Dense(precision)
    }
}

impl Qwen3WeightPrecision {
    /// The dtype activations, and so cached K/V, take under these weights.
    #[must_use]
    pub const fn activations(self) -> Qwen3FloatPrecision {
        match self {
            Self::Dense(precision)
            | Self::Affine {
                activations: precision,
                ..
            } => precision,
        }
    }

    /// The packed layout, for quantized weights.
    #[must_use]
    pub const fn quantization(self) -> Option<Qwen3AffineQuantization> {
        match self {
            Self::Dense(_) => None,
            Self::Affine { quantization, .. } => Some(quantization),
        }
    }
}

/// One projection's tensors: the matrix, typed by how it is stored, and the
/// output bias its family declares.
struct Projection<'a> {
    matmul: Matmul<'a>,
    /// `{stem}.bias`, added to the matmul output. Unrelated to an affine
    /// layout's `biases`, which are per-group dequantization offsets.
    bias: Option<&'a Array>,
}

enum Matmul<'a> {
    Dense(&'a Array),
    Affine {
        packed: &'a Array,
        scales: &'a Array,
        biases: &'a Array,
        quantization: Qwen3AffineQuantization,
    },
}

/// Whether `stem` is one of the attention projections a family with Q/K/V
/// bias (Qwen2) adds a bias to. O, the MLP and the embedding never carry one.
fn has_output_bias(config: &Qwen3ForwardConfig, stem: &str) -> bool {
    config.family().has_qkv_bias()
        && matches!(
            stem.rsplit('.').next(),
            Some("q_proj" | "k_proj" | "v_proj")
        )
}

fn projection<'a, S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array, S>,
    stem: &str,
) -> Result<Projection<'a>, Qwen3ForwardError> {
    // The bias is read because the family declares it, never because a
    // `.bias` tensor happens to be present.
    let bias = if has_output_bias(config, stem) {
        Some(weight(weights, &format!("{stem}.bias"))?)
    } else {
        None
    };
    Ok(Projection {
        matmul: matmul(config, weights, stem)?,
        bias,
    })
}

fn matmul<'a, S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array, S>,
    stem: &str,
) -> Result<Matmul<'a>, Qwen3ForwardError> {
    let tensor = weight(weights, &format!("{stem}.weight"))?;
    match config.quantization() {
        None => {
            if !matches!(
                tensor.dtype(),
                Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32
            ) {
                return Err(Qwen3ForwardError::WeightLayoutMismatch(format!(
                    "{stem}.weight is {:?}, not a dense float tensor",
                    tensor.dtype()
                )));
            }
            Ok(Matmul::Dense(tensor))
        }
        Some(quantization) => {
            if tensor.dtype() != Dtype::Uint32 {
                return Err(Qwen3ForwardError::WeightLayoutMismatch(format!(
                    "{stem}.weight is {:?}, not packed affine words",
                    tensor.dtype()
                )));
            }
            Ok(Matmul::Affine {
                packed: tensor,
                scales: weight(weights, &format!("{stem}.scales"))?,
                biases: weight(weights, &format!("{stem}.biases"))?,
                quantization,
            })
        }
    }
}

/// `input @ W.T` for the projection `{stem}.weight`, plus `{stem}.bias` when
/// the family declares one, `[.., in] -> [.., out]`.
pub(crate) fn project<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    input: &Array,
    stem: &str,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let Projection { matmul, bias } = projection(config, weights, stem)?;
    let output = match matmul {
        Matmul::Dense(dense) => linear(input, dense)?,
        Matmul::Affine {
            packed,
            scales,
            biases,
            quantization,
        } => ops::quantized_matmul_device(
            input,
            packed,
            scales,
            biases,
            true,
            quantization.group_size(),
            quantization.bits(),
            &stream,
        )?,
    };
    match bias {
        Some(bias) => {
            // A checkpoint may store biases more precisely than its weights.
            // Preserve the projection dtype used to budget cached K/V.
            let bias = bias.as_dtype_device(output.dtype(), &stream)?;
            Ok(output.add_device(&bias, &stream)?)
        }
        None => Ok(output),
    }
}

/// Embedding rows for `ids` (`[n]`), as `[n, hidden]` in the activation dtype.
pub(crate) fn embed_rows<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    ids: &Array,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    match matmul(config, weights, "model.embed_tokens")? {
        Matmul::Dense(table) => Ok(table.take_axis_device(ids, 0, &stream)?),
        Matmul::Affine {
            packed,
            scales,
            biases,
            quantization,
        } => Ok(ops::dequantize_device(
            packed.take_axis_device(ids, 0, &stream)?,
            scales.take_axis_device(ids, 0, &stream)?,
            &biases.take_axis_device(ids, 0, &stream)?,
            quantization.group_size(),
            quantization.bits(),
            &stream,
        )?),
    }
}

/// The dense weight of `{stem}`, dequantized when stored packed; for
/// diagnostics that need the full matrix.
pub(crate) fn dense_weight<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    stem: &str,
) -> Result<Array, Qwen3ForwardError> {
    match matmul(config, weights, stem)? {
        Matmul::Dense(dense) => Ok(dense.clone()),
        Matmul::Affine {
            packed,
            scales,
            biases,
            quantization,
        } => Ok(ops::dequantize_device(
            packed,
            scales,
            biases,
            quantization.group_size(),
            quantization.bits(),
            StreamOrDevice::gpu(),
        )?),
    }
}

fn linear(input: &Array, weight: &Array) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    #[cfg(test)]
    let transpose_started = std::time::Instant::now();
    let transposed = weight.transpose_device(&stream)?;
    #[cfg(test)]
    super::record_decode_profile_transpose_node(transpose_started.elapsed());
    Ok(input.matmul_device(&transposed, &stream)?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::{Array, Dtype, StreamOrDevice, ops};

    use super::{dense_weight, embed_rows, project};
    use crate::{
        GPU_TEST_LOCK,
        forward::{Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor},
        quantization::{Qwen3AffineQuantization, is_quantizable},
    };

    /// One decoder layer whose every projection input is 64 wide, so 64-element
    /// groups divide each row.
    fn config(quantized: bool) -> Qwen3ForwardConfig {
        let quantization = if quantized {
            r#","quantization":{"group_size":64,"bits":4}"#
        } else {
            ""
        };
        Qwen3ForwardConfig::parse(&format!(
            r#"{{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
              "intermediate_size":128,"vocab_size":96,"num_attention_heads":2,
              "num_key_value_heads":1,"head_dim":32,"max_position_embeddings":64,
              "rms_norm_eps":0.000001,"rope_theta":1000000,"hidden_act":"silu",
              "tie_word_embeddings":true{quantization}}}"#
        ))
        .expect("small quantizable config")
    }

    /// The same layer as a `qwen2` checkpoint: Q/K/V carry a bias, no Q/K norm.
    fn qwen2_config(quantized: bool) -> Qwen3ForwardConfig {
        let quantization = if quantized {
            r#","quantization":{"group_size":64,"bits":4}"#
        } else {
            ""
        };
        Qwen3ForwardConfig::parse(&format!(
            r#"{{"model_type":"qwen2","num_hidden_layers":1,"hidden_size":64,
              "intermediate_size":128,"vocab_size":96,"num_attention_heads":2,
              "num_key_value_heads":1,"max_position_embeddings":64,
              "rms_norm_eps":0.000001,"rope_theta":1000000,"hidden_act":"silu",
              "sliding_window":32768,"use_sliding_window":false,
              "tie_word_embeddings":true{quantization}}}"#
        ))
        .expect("small quantizable Qwen2 config")
    }

    /// `dense_weights` with Q/K/V biases and without Q/K norms.
    fn qwen2_weights() -> HashMap<String, Array> {
        let mut weights = dense_weights();
        weights
            .retain(|name, _| !name.ends_with("q_norm.weight") && !name.ends_with("k_norm.weight"));
        for (name, width, salt) in [("q_proj", 64, 31), ("k_proj", 32, 37), ("v_proj", 32, 41)] {
            weights.insert(
                format!("model.layers.0.self_attn.{name}.bias"),
                matrix(1, width, salt).reshape(&[width]).expect("bias"),
            );
        }
        weights
    }

    /// `input @ weight.T + bias` on the host, from the same values.
    fn host_affine(input: &Array, weight: &Array, bias: Option<&Array>) -> Vec<f32> {
        let (input, weight) = (values(input), values(weight));
        let columns = 64;
        let rows = weight.len() / columns;
        let bias = bias.map_or_else(|| vec![0.0; rows], values);
        input
            .chunks_exact(columns)
            .flat_map(|x| {
                weight
                    .chunks_exact(columns)
                    .zip(&bias)
                    .map(|(w, b)| x.iter().zip(w).map(|(x, w)| x * w).sum::<f32>() + b)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn matrix(rows: i32, columns: i32, salt: i32) -> Array {
        let values: Vec<f32> = (0..rows * columns)
            .map(|index| {
                let bucket = (index * 7_919 + salt * 104_729) % 2_001;
                f32::from(i16::try_from(bucket).expect("bucket")) / 4_000.0 - 0.25
            })
            .collect();
        Array::from_slice(&values, &[rows, columns])
    }

    fn dense_weights() -> HashMap<String, Array> {
        let mut weights = HashMap::new();
        let mut put = |name: &str, rows: i32, columns: i32| {
            let salt = i32::try_from(weights.len()).expect("few tensors");
            weights.insert(name.to_owned(), matrix(rows, columns, salt));
        };
        put("model.embed_tokens.weight", 96, 64);
        for (name, rows, columns) in [
            ("self_attn.q_proj", 64, 64),
            ("self_attn.k_proj", 32, 64),
            ("self_attn.v_proj", 32, 64),
            ("self_attn.o_proj", 64, 64),
            ("mlp.gate_proj", 128, 64),
            ("mlp.up_proj", 128, 64),
            ("mlp.down_proj", 64, 128),
        ] {
            put(&format!("model.layers.0.{name}.weight"), rows, columns);
        }
        for (name, width) in [
            ("model.layers.0.input_layernorm.weight", 64),
            ("model.layers.0.post_attention_layernorm.weight", 64),
            ("model.layers.0.self_attn.q_norm.weight", 32),
            ("model.layers.0.self_attn.k_norm.weight", 32),
            ("model.norm.weight", 64),
        ] {
            weights.insert(
                name.to_owned(),
                Array::from_slice(
                    &[1.0_f32; 128][..width],
                    &[i32::try_from(width).expect("width")],
                ),
            );
        }
        weights
    }

    /// Packs every projection and the embedding with MLX's affine quantizer.
    fn quantize(dense: &HashMap<String, Array>) -> HashMap<String, Array> {
        let layout = Qwen3AffineQuantization::FOUR_BIT_G64;
        let mut packed = HashMap::new();
        for (name, tensor) in dense {
            if is_quantizable(name) {
                let stem = name.trim_end_matches(".weight");
                let (words, scales, biases) =
                    ops::quantize(tensor, layout.group_size(), layout.bits()).expect("quantize");
                packed.insert(name.clone(), words);
                packed.insert(format!("{stem}.scales"), scales);
                packed.insert(format!("{stem}.biases"), biases);
            } else {
                packed.insert(name.clone(), tensor.clone());
            }
        }
        packed
    }

    /// The dense weights the packed ones decode to.
    fn dequantized(packed: &HashMap<String, Array>) -> HashMap<String, Array> {
        let quantized = config(true);
        packed
            .iter()
            .filter(|(name, _)| !name.ends_with(".scales") && !name.ends_with(".biases"))
            .map(|(name, tensor)| {
                let tensor = if is_quantizable(name) {
                    dense_weight(&quantized, packed, name.trim_end_matches(".weight"))
                        .expect("dequantize")
                } else {
                    tensor.clone()
                };
                (name.clone(), tensor)
            })
            .collect()
    }

    fn values(array: &Array) -> Vec<f32> {
        array.eval().expect("eval");
        array.as_slice::<f32>().to_vec()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} differs from {expected}"
            );
        }
    }

    #[test]
    fn affine_projections_match_their_dequantized_dense_weights() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let packed = quantize(&dense_weights());
        let dense = dequantized(&packed);
        let input = matrix(3, 64, 9).reshape(&[1, 3, 64]).expect("input");
        for stem in ["model.layers.0.self_attn.q_proj", "model.embed_tokens"] {
            let quantized = project(&config(true), &packed, &input, stem).expect("affine");
            let reference = project(&config(false), &dense, &input, stem).expect("dense");
            assert_close(&values(&quantized), &values(&reference), 1e-4);
        }
        // A packed row decodes exactly as the decoded table's row.
        let ids = Array::from_slice(&[5_i32, 0, 95], &[3]);
        assert_eq!(
            values(&embed_rows(&config(true), &packed, &ids).expect("packed rows")),
            values(&embed_rows(&config(false), &dense, &ids).expect("dense rows")),
        );
    }

    #[test]
    fn packed_and_dense_tensors_only_reach_their_own_matmul() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let dense = dense_weights();
        let packed = quantize(&dense);
        let input = matrix(1, 64, 3).reshape(&[1, 1, 64]).expect("input");
        let stem = "model.layers.0.mlp.gate_proj";
        // Packed words under a dense config, and dense floats under a
        // quantized one, are refused before any matmul is built.
        assert!(matches!(
            project(&config(false), &packed, &input, stem),
            Err(Qwen3ForwardError::WeightLayoutMismatch(_))
        ));
        assert!(matches!(
            project(&config(true), &dense, &input, stem),
            Err(Qwen3ForwardError::WeightLayoutMismatch(_))
        ));
        let mut missing_scales = packed.clone();
        missing_scales.remove(&format!("{stem}.scales"));
        assert!(matches!(
            project(&config(true), &missing_scales, &input, stem),
            Err(Qwen3ForwardError::MissingWeight(_))
        ));
    }

    #[test]
    fn a_quantized_decoder_runs_like_its_dequantized_dense_twin() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let packed = quantize(&dense_weights());
        let dense = dequantized(&packed);
        let (quantized_config, dense_config) = (config(true), config(false));
        let mut quantized = Qwen3ForwardExecutor::new(&quantized_config, &packed);
        let mut reference = Qwen3ForwardExecutor::new(&dense_config, &dense);
        let prompt = [3, 17, 42, 8];
        assert_close(
            &quantized
                .prefill_last_logits(&prompt)
                .expect("quantized prefill"),
            &reference
                .prefill_last_logits(&prompt)
                .expect("dense prefill"),
            1e-3,
        );
        for token in [7, 60] {
            assert_close(
                &quantized
                    .decode_last_logits(token)
                    .expect("quantized decode"),
                &reference.decode_last_logits(token).expect("dense decode"),
                1e-3,
            );
        }
    }

    #[test]
    fn qwen2_float32_bias_preserves_bfloat16_projection_and_cache_budget() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let stream = StreamOrDevice::gpu();
        let config = qwen2_config(false);
        let weights: HashMap<_, _> = qwen2_weights()
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = if name
                    .rsplit_once('.')
                    .is_some_and(|(_, suffix)| suffix == "bias")
                {
                    tensor
                } else {
                    tensor
                        .as_dtype_device(Dtype::Bfloat16, &stream)
                        .expect("BF16")
                };
                (name, tensor)
            })
            .collect();
        let input = matrix(2, 64, 47)
            .as_dtype_device(Dtype::Bfloat16, &stream)
            .expect("BF16 input");
        for name in ["q_proj", "k_proj", "v_proj"] {
            let stem = format!("model.layers.0.self_attn.{name}");
            let weight = &weights[&format!("{stem}.weight")];
            let bias = &weights[&format!("{stem}.bias")];
            assert_eq!(weight.dtype(), Dtype::Bfloat16);
            assert_eq!(bias.dtype(), Dtype::Float32);
            let reference = input
                .matmul_device(
                    weight.transpose_device(&stream).expect("transpose"),
                    &stream,
                )
                .expect("reference product")
                .add_device(
                    bias.as_dtype_device(Dtype::Bfloat16, &stream)
                        .expect("rounded bias"),
                    &stream,
                )
                .expect("reference bias");
            let actual = project(&config, &weights, &input, &stem).expect("mixed bias projection");
            assert_eq!(actual.dtype(), Dtype::Bfloat16, "{name}");
            let widen = |array: &Array| {
                values(
                    &array
                        .as_dtype_device(Dtype::Float32, &stream)
                        .expect("widen"),
                )
            };
            assert_eq!(widen(&actual), widen(&reference), "{name}");
        }
        let precision =
            crate::forward::kv_precision(&config, &weights).expect("declared cache dtype");
        assert_eq!(precision, crate::forward::Qwen3FloatPrecision::BFloat16);
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        executor
            .prefill_last_logits(&[3, 17, 42])
            .expect("mixed bias prefill");
        for tokens in [3_usize, 4] {
            if tokens == 4 {
                executor.decode_last_logits(8).expect("mixed bias decode");
            }
            let cache = executor.cache[0].as_ref().expect("one live layer");
            assert_eq!(cache.keys.dtype(), Dtype::Bfloat16, "keys at {tokens}");
            assert_eq!(cache.values.dtype(), Dtype::Bfloat16, "values at {tokens}");
            // One layer, key + value, one KV head, width32, two bytes/element.
            let expected_bytes = 2 * tokens * 32 * 2;
            assert_eq!(executor.kv_bytes(), expected_bytes);
            assert_eq!(
                config
                    .cached_kv_bytes_at(tokens, precision)
                    .expect("budget"),
                u64::try_from(expected_bytes).expect("small cache")
            );
        }
    }

    #[test]
    fn qwen2_adds_qkv_biases_after_either_matmul_and_nowhere_else() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let dense = qwen2_weights();
        let input = matrix(3, 64, 9).reshape(&[1, 3, 64]).expect("input");
        let stem = |name: &str| format!("model.layers.0.{name}");
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
            let stem = stem(name);
            let expected = host_affine(
                &input,
                &dense[&format!("{stem}.weight")],
                Some(&dense[&format!("{stem}.bias")]),
            );
            let actual = project(&qwen2_config(false), &dense, &input, &stem).expect("dense");
            assert_close(&values(&actual), &expected, 1e-5);
        }

        // The affine path adds the same dense bias after the quantized matmul.
        let packed = quantize(&dense);
        let decoded = dequantized(&packed);
        let q_proj = stem("self_attn.q_proj");
        assert_close(
            &values(&project(&qwen2_config(true), &packed, &input, &q_proj).expect("affine")),
            &values(&project(&qwen2_config(false), &decoded, &input, &q_proj).expect("dense")),
            1e-4,
        );

        // O and the MLP never read a bias, even one present in the map, and
        // Qwen3 ignores a stray Q bias: only the family's declaration counts.
        let mut stray = dense.clone();
        for name in ["self_attn.o_proj", "mlp.gate_proj"] {
            stray.insert(
                format!("{}.bias", stem(name)),
                matrix(1, 64, 5).reshape(&[64]).expect("stray"),
            );
            let stem = stem(name);
            assert_close(
                &values(&project(&qwen2_config(false), &stray, &input, &stem).expect("biasless")),
                &host_affine(&input, &dense[&format!("{stem}.weight")], None),
                1e-5,
            );
        }
        assert_close(
            &values(&project(&config(false), &dense, &input, &q_proj).expect("qwen3")),
            &host_affine(&input, &dense[&format!("{q_proj}.weight")], None),
            1e-5,
        );

        // A Qwen2 checkpoint without the bias is refused, not run unbiased.
        let mut missing = dense;
        missing.remove(&format!("{q_proj}.bias"));
        assert!(matches!(
            project(&qwen2_config(false), &missing, &input, &q_proj),
            Err(Qwen3ForwardError::MissingWeight(name)) if name.ends_with("q_proj.bias")
        ));
    }

    #[test]
    fn a_quantized_qwen2_decoder_runs_like_its_dequantized_dense_twin() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let packed = quantize(&qwen2_weights());
        let dense = dequantized(&packed);
        let (quantized_config, dense_config) = (qwen2_config(true), qwen2_config(false));
        let mut quantized = Qwen3ForwardExecutor::new(&quantized_config, &packed);
        let mut reference = Qwen3ForwardExecutor::new(&dense_config, &dense);
        let prompt = [3, 17, 42, 8];
        assert_close(
            &quantized
                .prefill_last_logits(&prompt)
                .expect("quantized prefill"),
            &reference
                .prefill_last_logits(&prompt)
                .expect("dense prefill"),
            1e-3,
        );
        assert_close(
            &quantized.decode_last_logits(60).expect("quantized decode"),
            &reference.decode_last_logits(60).expect("dense decode"),
            1e-3,
        );
    }

    #[test]
    fn embedded_rows_preserve_dense_and_affine_qwen_and_qwen2_paths() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let ids = [1, 2, 3, 4];
        for qwen2 in [false, true] {
            for affine in [false, true] {
                let config = if qwen2 {
                    qwen2_config(affine)
                } else {
                    config(affine)
                };
                let dense = if qwen2 {
                    qwen2_weights()
                } else {
                    dense_weights()
                };
                let weights = if affine { quantize(&dense) } else { dense };
                let mut ordinary = Qwen3ForwardExecutor::resident(&config, &weights, 16, u64::MAX)
                    .expect("resident plan");
                let expected = ordinary
                    .prefill_last_logits(&ids)
                    .expect("ordinary prefill");
                let expected_next = ordinary.decode_last_logits(5).expect("ordinary decode");
                for replace in [false, true] {
                    let spans = if replace {
                        let rows =
                            embed_rows(&config, &weights, &Array::from_slice(&ids[1..3], &[2]))
                                .expect("decoded embedding rows");
                        vec![media::EmbeddedSpan::new(
                            1,
                            2,
                            media::SpanKind::Audio,
                            [0; 32],
                            rows,
                        )]
                    } else {
                        Vec::new()
                    };
                    let prompt = media::EmbeddedPrompt::new(&ids, spans).expect("prompt");
                    let mut embedded =
                        Qwen3ForwardExecutor::resident(&config, &weights, 16, u64::MAX)
                            .expect("embedded resident plan");
                    let actual = embedded
                        .prefill_embedded_last_logits(&prompt)
                        .expect("embedded prefill");
                    let next = embedded.decode_last_logits(5).expect("embedded decode");
                    let bits =
                        |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "qwen2={qwen2} affine={affine} replace={replace}"
                    );
                    assert_eq!(
                        bits(&next),
                        bits(&expected_next),
                        "qwen2={qwen2} affine={affine} replace={replace}"
                    );
                }
            }
        }
    }
}
