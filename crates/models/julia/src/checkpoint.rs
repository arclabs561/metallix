//! Strict loader for the published Julia-1 checkpoint directory.
//!
//! The safetensors header must name exactly the pinned 170 F32 tensors with
//! their published shapes and contiguous offsets; anything missing, extra,
//! retyped or reshaped fails before a payload byte is read.

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use serde_json::Value;
use thiserror::Error;

use crate::{
    ENCODER_FF_WIDTH, EncoderBlockWeights, FEED_FORWARD_WIDTH, FullEncoderWeights, HEAD_LAYERS,
    HeadLayerWeights, HeadWeights, JuliaEncoder, JuliaEncoderError, JuliaHeadError, ScorerWeights,
    WIDTH, head::DecisionHead,
};

const VOCAB: usize = 256_000;
const ENCODER_LAYERS: usize = 22;
const MAX_HEADER_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum JuliaCheckpointError {
    #[error("Julia checkpoint {path} could not be read: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Julia checkpoint is malformed: {0}")]
    Format(String),
    #[error(transparent)]
    Encoder(#[from] JuliaEncoderError),
    #[error(transparent)]
    Head(#[from] JuliaHeadError),
}

fn format_error<T>(message: impl Into<String>) -> Result<T, JuliaCheckpointError> {
    Err(JuliaCheckpointError::Format(message.into()))
}

/// Every tensor in the pinned `model.safetensors` with its published shape.
fn expected_tensors() -> Vec<(String, Vec<usize>)> {
    let mut tensors = vec![
        (
            "encoder.embeddings.tok_embeddings.weight".into(),
            vec![VOCAB, WIDTH],
        ),
        ("encoder.embeddings.norm.weight".into(), vec![WIDTH]),
        ("encoder.final_norm.weight".into(), vec![WIDTH]),
        ("type_emb.weight".into(), vec![3, WIDTH]),
        ("scorer.0.weight".into(), vec![WIDTH]),
        ("scorer.0.bias".into(), vec![WIDTH]),
        ("scorer.1.weight".into(), vec![WIDTH, WIDTH]),
        ("scorer.1.bias".into(), vec![WIDTH]),
        ("scorer.3.weight".into(), vec![1, WIDTH]),
        ("scorer.3.bias".into(), vec![1]),
        // Present in the checkpoint but unused by `forward(..., return_actions=False)`.
        ("temperature".into(), vec![3]),
        ("act_head.0.weight".into(), vec![256, WIDTH + 4]),
        ("act_head.0.bias".into(), vec![256]),
        ("act_head.2.weight".into(), vec![2, 256]),
        ("act_head.2.bias".into(), vec![2]),
    ];
    for layer in 0..ENCODER_LAYERS {
        let p = format!("encoder.layers.{layer}.");
        tensors.extend([
            (format!("{p}attn.Wqkv.weight"), vec![3 * WIDTH, WIDTH]),
            (format!("{p}attn.Wo.weight"), vec![WIDTH, WIDTH]),
            (
                format!("{p}mlp.Wi.weight"),
                vec![2 * ENCODER_FF_WIDTH, WIDTH],
            ),
            (format!("{p}mlp.Wo.weight"), vec![WIDTH, ENCODER_FF_WIDTH]),
            (format!("{p}mlp_norm.weight"), vec![WIDTH]),
        ]);
        // Layer 0 has an identity attention norm in the source, so no tensor.
        if layer > 0 {
            tensors.push((format!("{p}attn_norm.weight"), vec![WIDTH]));
        }
    }
    for layer in 0..HEAD_LAYERS {
        let p = format!("head.layers.{layer}.");
        tensors.extend([
            (
                format!("{p}self_attn.in_proj_weight"),
                vec![3 * WIDTH, WIDTH],
            ),
            (format!("{p}self_attn.in_proj_bias"), vec![3 * WIDTH]),
            (format!("{p}self_attn.out_proj.weight"), vec![WIDTH, WIDTH]),
            (format!("{p}self_attn.out_proj.bias"), vec![WIDTH]),
            (
                format!("{p}linear1.weight"),
                vec![FEED_FORWARD_WIDTH, WIDTH],
            ),
            (format!("{p}linear1.bias"), vec![FEED_FORWARD_WIDTH]),
            (
                format!("{p}linear2.weight"),
                vec![WIDTH, FEED_FORWARD_WIDTH],
            ),
            (format!("{p}linear2.bias"), vec![WIDTH]),
            (format!("{p}norm1.weight"), vec![WIDTH]),
            (format!("{p}norm1.bias"), vec![WIDTH]),
            (format!("{p}norm2.weight"), vec![WIDTH]),
            (format!("{p}norm2.bias"), vec![WIDTH]),
        ]);
    }
    tensors
}

/// Validated header: tensor name to absolute byte range in the file.
fn tensor_ranges(
    file: &mut File,
    path: &Path,
    expected: Vec<(String, Vec<usize>)>,
) -> Result<HashMap<String, (u64, u64)>, JuliaCheckpointError> {
    let io = |source| JuliaCheckpointError::Io {
        path: path.to_owned(),
        source,
    };
    let file_len = file.metadata().map_err(io)?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(io)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || 8 + header_len > file_len {
        return format_error(format!(
            "safetensors header length {header_len} is out of bounds"
        ));
    }
    let mut header = vec![0_u8; usize::try_from(header_len).expect("bounded by 1 MiB")];
    file.read_exact(&mut header).map_err(io)?;
    let header: Value = serde_json::from_slice(&header)
        .map_err(|error| JuliaCheckpointError::Format(format!("safetensors header: {error}")))?;
    let Some(entries) = header.as_object() else {
        return format_error("safetensors header is not an object");
    };
    let names = entries
        .keys()
        .filter(|name| *name != "__metadata__")
        .count();
    if names != expected.len() {
        return format_error(format!(
            "expected {} tensors, found {names}",
            expected.len()
        ));
    }
    let data_start = 8 + header_len;
    let mut ranges = HashMap::with_capacity(expected.len());
    let mut spans = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        let Some(info) = entries.get(&name) else {
            return format_error(format!("missing tensor {name}"));
        };
        if info["dtype"] != "F32" {
            return format_error(format!("{name} dtype is {}, expected F32", info["dtype"]));
        }
        let actual: Option<Vec<usize>> = info["shape"].as_array().and_then(|dims| {
            dims.iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                .collect()
        });
        if actual.as_ref() != Some(&shape) {
            return format_error(format!(
                "{name} shape is {}, expected {shape:?}",
                info["shape"]
            ));
        }
        let offsets: Option<Vec<u64>> = info["data_offsets"]
            .as_array()
            .and_then(|o| o.iter().map(Value::as_u64).collect());
        let bytes = shape.iter().product::<usize>() as u64 * 4;
        let Some([start, end]) = offsets
            .as_deref()
            .map(<[u64]>::to_vec)
            .and_then(|o| <[u64; 2]>::try_from(o).ok())
        else {
            return format_error(format!("{name} data offsets are malformed"));
        };
        if end.checked_sub(start) != Some(bytes) {
            return format_error(format!("{name} data offsets do not match its shape"));
        }
        spans.push((start, end));
        ranges.insert(name, (data_start + start, data_start + end));
    }
    // Offsets must tile the data section exactly, with no gaps, overlaps or trailing bytes.
    spans.sort_unstable();
    let mut cursor = 0;
    for (start, end) in spans {
        if start != cursor {
            return format_error("tensor data offsets are not contiguous");
        }
        cursor = end;
    }
    if data_start + cursor != file_len {
        return format_error("tensor data does not cover the file exactly");
    }
    Ok(ranges)
}

/// Checkpoint weights resident in memory, with the head built once.
pub struct JuliaCheckpoint {
    embeddings: Vec<f32>,
    embedding_norm_weight: Vec<f32>,
    layers: Vec<EncoderBlockWeights>,
    final_norm_weight: Vec<f32>,
    head: DecisionHead,
}

impl JuliaCheckpoint {
    /// Loads `dir` after requiring the published artifact layout.
    #[tracing::instrument(
        name = "julia.checkpoint.load",
        level = "info",
        skip_all,
        fields(dir = %dir.display())
    )]
    pub fn load(dir: &Path) -> Result<Self, JuliaCheckpointError> {
        for config in ["config.json", "julia_config.json"] {
            let path = dir.join(config);
            let text = std::fs::read(&path).map_err(|source| JuliaCheckpointError::Io {
                path: path.clone(),
                source,
            })?;
            let value: Value = serde_json::from_slice(&text)
                .map_err(|error| JuliaCheckpointError::Format(format!("{config}: {error}")))?;
            if value["architecture"] != "JuliaDecisionModel" || value["format_version"] != 1 {
                return format_error(format!("{config} is not a format-1 JuliaDecisionModel"));
            }
            if config == "julia_config.json" && value["head_layers"] != HEAD_LAYERS {
                return format_error("julia_config.json head_layers must be 2");
            }
        }
        for required in ["encoder/config.json", "tokenizer/tokenizer.json"] {
            if !dir.join(required).is_file() {
                return format_error(format!("missing {required}"));
            }
        }
        let path = dir.join("model.safetensors");
        let mut file = File::open(&path).map_err(|source| JuliaCheckpointError::Io {
            path: path.clone(),
            source,
        })?;
        let ranges = tensor_ranges(&mut file, &path, expected_tensors())?;
        let mut take = |name: &str| -> Result<Vec<f32>, JuliaCheckpointError> {
            let (start, end) = ranges
                .get(name)
                .copied()
                .ok_or_else(|| JuliaCheckpointError::Format(format!("missing tensor {name}")))?;
            let len = usize::try_from(end - start)
                .map_err(|_| JuliaCheckpointError::Format(format!("{name} is too large")))?;
            let mut bytes = vec![0_u8; len];
            file.seek(SeekFrom::Start(start))
                .and_then(|_| file.read_exact(&mut bytes))
                .map_err(|source| JuliaCheckpointError::Io {
                    path: path.clone(),
                    source,
                })?;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        let layers = (0..ENCODER_LAYERS)
            .map(|layer| {
                let p = format!("encoder.layers.{layer}.");
                Ok(EncoderBlockWeights {
                    wqkv_weight: take(&format!("{p}attn.Wqkv.weight"))?,
                    wo_weight: take(&format!("{p}attn.Wo.weight"))?,
                    wi_weight: take(&format!("{p}mlp.Wi.weight"))?,
                    wo_mlp_weight: take(&format!("{p}mlp.Wo.weight"))?,
                    // The native block skips layer 0's attention norm, matching the source identity.
                    attn_norm_weight: if layer == 0 {
                        vec![1.0; WIDTH]
                    } else {
                        take(&format!("{p}attn_norm.weight"))?
                    },
                    mlp_norm_weight: take(&format!("{p}mlp_norm.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, JuliaCheckpointError>>()?;
        let mut head_layer = |index: usize| -> Result<HeadLayerWeights, JuliaCheckpointError> {
            let p = format!("head.layers.{index}.");
            Ok(HeadLayerWeights {
                in_proj_weight: take(&format!("{p}self_attn.in_proj_weight"))?,
                in_proj_bias: take(&format!("{p}self_attn.in_proj_bias"))?,
                out_proj_weight: take(&format!("{p}self_attn.out_proj.weight"))?,
                out_proj_bias: take(&format!("{p}self_attn.out_proj.bias"))?,
                linear1_weight: take(&format!("{p}linear1.weight"))?,
                linear1_bias: take(&format!("{p}linear1.bias"))?,
                linear2_weight: take(&format!("{p}linear2.weight"))?,
                linear2_bias: take(&format!("{p}linear2.bias"))?,
                norm1_weight: take(&format!("{p}norm1.weight"))?,
                norm1_bias: take(&format!("{p}norm1.bias"))?,
                norm2_weight: take(&format!("{p}norm2.weight"))?,
                norm2_bias: take(&format!("{p}norm2.bias"))?,
            })
        };
        let head_layers = [head_layer(0)?, head_layer(1)?];
        let head = DecisionHead::new(HeadWeights {
            layers: head_layers,
            type_embedding: take("type_emb.weight")?,
            scorer: ScorerWeights {
                norm_weight: take("scorer.0.weight")?,
                norm_bias: take("scorer.0.bias")?,
                linear1_weight: take("scorer.1.weight")?,
                linear1_bias: take("scorer.1.bias")?,
                linear2_weight: take("scorer.3.weight")?,
                linear2_bias: take("scorer.3.bias")?,
            },
        })?;
        Ok(Self {
            embeddings: take("encoder.embeddings.tok_embeddings.weight")?,
            embedding_norm_weight: take("encoder.embeddings.norm.weight")?,
            layers,
            final_norm_weight: take("encoder.final_norm.weight")?,
            head,
        })
    }

    /// Builds an encoder holding the embedding rows for `token_ids` only.
    ///
    /// ponytail: clones the 22 layers (about 156 MB) per call; share them if
    /// per-question latency matters.
    pub fn encoder(&self, token_ids: &[u64]) -> Result<JuliaEncoder, JuliaCheckpointError> {
        let mut ids = token_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let mut token_rows = Vec::with_capacity(ids.len() * WIDTH);
        for &id in &ids {
            let row = usize::try_from(id)
                .ok()
                .filter(|&row| row < VOCAB)
                .ok_or(JuliaEncoderError::VocabularyId(id))?
                * WIDTH;
            token_rows.extend_from_slice(&self.embeddings[row..row + WIDTH]);
        }
        Ok(JuliaEncoder::new(FullEncoderWeights {
            token_ids: ids,
            token_rows,
            embedding_norm_weight: self.embedding_norm_weight.clone(),
            layers: self.layers.clone(),
            final_norm_weight: self.final_norm_weight.clone(),
        })?)
    }

    #[must_use]
    pub fn head(&self) -> &DecisionHead {
        &self.head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Writes `header` plus `data_bytes` zero bytes and validates it against `expected`.
    fn check(
        header: &serde_json::Value,
        data_bytes: usize,
        expected: &[(&str, Vec<usize>)],
    ) -> Result<HashMap<String, (u64, u64)>, JuliaCheckpointError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "julia-header-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _scratch = Scratch(path.clone());
        let header = serde_json::to_vec(header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + data_bytes, 0);
        std::fs::write(&path, bytes).unwrap();
        let expected = expected
            .iter()
            .map(|(n, s)| ((*n).to_owned(), s.clone()))
            .collect();
        tensor_ranges(&mut File::open(&path).unwrap(), &path, expected)
    }

    fn tensor(dtype: &str, shape: &[usize], start: u64, end: u64) -> serde_json::Value {
        serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]})
    }

    #[test]
    fn published_tensor_set_is_complete_and_unique() {
        let tensors = expected_tensors();
        let unique: std::collections::HashSet<_> = tensors.iter().map(|(n, _)| n).collect();
        assert_eq!((tensors.len(), unique.len()), (170, 170));
    }

    #[test]
    fn header_must_match_names_dtypes_shapes_and_tile_the_data() {
        let expected = [("a", vec![2]), ("b", vec![1, 3])];
        let good = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "a": tensor("F32", &[2], 0, 8),
            "b": tensor("F32", &[1, 3], 8, 20),
        });
        let ranges = check(&good, 20, &expected).unwrap();
        let header_end = 8 + serde_json::to_vec(&good).unwrap().len() as u64;
        assert_eq!(ranges["b"], (header_end + 8, header_end + 20));

        let reject = |header: serde_json::Value, data: usize, needle: &str| {
            let error = check(&header, data, &expected).unwrap_err().to_string();
            assert!(error.contains(needle), "{error} lacks {needle}");
        };
        reject(good.clone(), 24, "cover the file exactly");
        let mut h = good.clone();
        h["b"] = tensor("F16", &[1, 3], 8, 20);
        reject(h, 20, "dtype");
        let mut h = good.clone();
        h["b"] = tensor("F32", &[3, 1], 8, 20);
        reject(h, 20, "shape");
        let mut h = good.clone();
        h["b"] = tensor("F32", &[1, 3], 8, 16);
        reject(h, 20, "do not match its shape");
        let mut h = good.clone();
        h["b"] = tensor("F32", &[1, 3], 4, 16);
        reject(h, 20, "not contiguous");
        let mut h = good.clone();
        h["c"] = tensor("F32", &[1], 20, 24);
        reject(h, 24, "expected 2 tensors, found 3");
        let mut h = good.clone();
        h.as_object_mut().unwrap().remove("a");
        h["c"] = tensor("F32", &[2], 0, 8);
        reject(h, 20, "missing tensor a");
    }
}
