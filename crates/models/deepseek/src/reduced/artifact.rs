//! Bounded, self-contained numerical artifacts for the reduced request.
//!
//! This format contains configuration and named numerical tensors only. It is
//! distinct from a model checkpoint and never contains reference execution cases.

use std::collections::BTreeMap;

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, Visitor},
};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{FinalHeadExecution, RequestError, RequestStepOutput};
use crate::indexer::{key::IndexKeyRotaryExecution, query::IndexScoreExecution};

mod model;

/// Maximum encoded size of one reduced synthetic artifact (64 MiB).
pub const MAX_REDUCED_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 32 * 1024 * 1024;
const MAX_TENSOR_ELEMENTS: usize = 1 << 20;
const MAX_TENSORS: usize = 512;

/// Owned numerical operands for one fixed five-block reduced model.
///
/// Parsing validates the document, tensor encodings and allocation bounds.
/// `run` additionally constructs the numerical components and checks their
/// exact operand geometry before executing a request.
pub struct ReducedArtifact {
    config: ArtifactConfig,
    tensors: TensorStore,
}

impl ReducedArtifact {
    /// Parses a size-bounded, versioned artifact with checksummed tensors.
    pub fn parse(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() > MAX_REDUCED_ARTIFACT_BYTES {
            return Err(ArtifactError::SizeLimit);
        }
        let raw: RawArtifact = serde_json::from_slice(bytes)?;
        if raw.schema_version != 1 || raw.format != "metallix.deepseek.reduced" {
            return Err(ArtifactError::Format);
        }
        raw.config.validate()?;
        let expected_names = model::tensor_names(&raw.config);
        if !raw.tensors.keys().eq(expected_names.iter()) {
            return Err(ArtifactError::Invalid(String::from(
                "tensor inventory contains missing or unknown operands",
            )));
        }
        let mut decoded_bytes = 0_usize;
        let mut tensors = BTreeMap::new();
        for (name, raw_tensor) in raw.tensors {
            if name.is_empty() || name.len() > 128 || !name.is_ascii() {
                return Err(ArtifactError::Invalid(String::from("invalid tensor name")));
            }
            let tensor = raw_tensor.decode(&mut decoded_bytes)?;
            tensors.insert(name, tensor);
        }
        Ok(Self {
            config: raw.config,
            tensors: TensorStore(tensors),
        })
    }

    /// Runs a prefill followed by one-token decode calls over supplied token IDs.
    ///
    /// At least two prefill tokens are required by the ratio-two L1 owner.
    /// Every invocation constructs fresh request state; no state is retained
    /// in the artifact between calls. No token sampling or text decoding occurs.
    pub fn run(
        &self,
        ids: &[i64],
        prefill_tokens: usize,
    ) -> Result<Vec<RequestStepOutput>, ArtifactError> {
        self.run_with_execution(
            ids,
            prefill_tokens,
            IndexScoreExecution::Scalar,
            IndexKeyRotaryExecution::Scalar,
        )
    }

    /// Runs with an explicit index-score implementation across L1, L3 and L4.
    ///
    /// Metal scoring is mixed execution: all other arithmetic remains scalar.
    /// Every invocation constructs fresh state with the same admission limits as
    /// [`Self::run`]. The artifact does not retain an execution preference.
    pub fn run_with_score_execution(
        &self,
        ids: &[i64],
        prefill_tokens: usize,
        execution: IndexScoreExecution,
    ) -> Result<Vec<RequestStepOutput>, ArtifactError> {
        self.run_with_execution(
            ids,
            prefill_tokens,
            execution,
            IndexKeyRotaryExecution::Scalar,
        )
    }

    /// Runs with explicit index-score and index-key rotary implementations.
    ///
    /// Both device choices are bounded mixed-execution diagnostics; all other
    /// arithmetic remains scalar and the artifact retains neither preference.
    pub fn run_with_execution(
        &self,
        ids: &[i64],
        prefill_tokens: usize,
        score_execution: IndexScoreExecution,
        key_rotary_execution: IndexKeyRotaryExecution,
    ) -> Result<Vec<RequestStepOutput>, ArtifactError> {
        self.run_with_head_execution(
            ids,
            prefill_tokens,
            score_execution,
            key_rotary_execution,
            FinalHeadExecution::Scalar,
        )
    }

    /// Runs with explicit index-score, index-key rotary, and final-head choices.
    ///
    /// Every device choice is an independently selectable mixed-execution
    /// diagnostic. The artifact retains no execution preference between calls.
    pub fn run_with_head_execution(
        &self,
        ids: &[i64],
        prefill_tokens: usize,
        score_execution: IndexScoreExecution,
        key_rotary_execution: IndexKeyRotaryExecution,
        head_execution: FinalHeadExecution,
    ) -> Result<Vec<RequestStepOutput>, ArtifactError> {
        if prefill_tokens < 2 || prefill_tokens > ids.len() || ids.len() > self.config.max_tokens {
            return Err(ArtifactError::Invalid(String::from(
                "require 2 <= prefill_tokens <= input count <= configured max_tokens",
            )));
        }
        if ids
            .iter()
            .any(|&id| usize::try_from(id).map_or(true, |id| id >= self.config.vocabulary))
        {
            return Err(ArtifactError::Invalid(String::from(
                "input token is outside artifact vocabulary",
            )));
        }
        model::run(
            &self.config,
            &self.tensors,
            ids,
            prefill_tokens,
            score_execution,
            key_rotary_execution,
            head_execution,
        )
    }
}

/// An invalid reduced numerical artifact or failed reduced request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ArtifactError {
    /// The encoded document exceeds the fixed admission limit.
    #[error("reduced artifact exceeds 64 MiB encoded limit")]
    SizeLimit,
    /// The format identifier or schema version is unsupported.
    #[error("unsupported reduced artifact format or schema version")]
    Format,
    /// JSON did not match the numerical artifact schema.
    #[error("invalid reduced artifact JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Numerical metadata, tensor encoding or component geometry was invalid.
    #[error("invalid reduced artifact: {0}")]
    Invalid(String),
    /// A validly admitted request failed during numerical execution.
    #[error(transparent)]
    Request(#[from] RequestError),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    schema_version: u32,
    format: String,
    config: ArtifactConfig,
    #[serde(deserialize_with = "unique_tensors")]
    tensors: BTreeMap<String, RawTensor>,
}

fn unique_tensors<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, RawTensor>, D::Error> {
    struct TensorVisitor;
    impl<'de> Visitor<'de> for TensorVisitor {
        type Value = BTreeMap<String, RawTensor>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a bounded map of unique tensor names")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut tensors = BTreeMap::new();
            while let Some(name) = map.next_key::<String>()? {
                if tensors.len() >= MAX_TENSORS || tensors.contains_key(&name) {
                    return Err(serde::de::Error::custom(
                        "duplicate tensor name or tensor-count limit exceeded",
                    ));
                }
                tensors.insert(name, map.next_value()?);
            }
            Ok(tensors)
        }
    }
    deserializer.deserialize_map(TensorVisitor)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactConfig {
    max_tokens: usize,
    width: usize,
    copies: usize,
    vocabulary: usize,
    heads: usize,
    head_dimension: usize,
    rope_pairs: usize,
    query_rank: usize,
    output_groups: usize,
    output_rank: usize,
    window: usize,
    intermediate: usize,
    routed_experts: usize,
    active_experts: usize,
    gate_temperature: f32,
    normalize_topk: bool,
    route_scale: f32,
    swiglu_limit: f32,
    norm_epsilon: f32,
    hc_iterations: usize,
    hc_epsilon: f32,
    index_heads: usize,
    index_topk: usize,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
    engram_embedding_width: usize,
    engram_rows: [usize; 2],
    engram_ngram: usize,
    engram_heads: usize,
    engram_pad_id: i64,
    engram_gate_clamp: f32,
}

impl ArtifactConfig {
    fn validate(&self) -> Result<(), ArtifactError> {
        for (name, value, maximum) in [
            ("max_tokens", self.max_tokens, 16),
            ("width", self.width, 256),
            ("copies", self.copies, 4),
            ("vocabulary", self.vocabulary, 256),
            ("heads", self.heads, 4),
            ("head_dimension", self.head_dimension, 128),
            ("rope_pairs", self.rope_pairs, 64),
            ("query_rank", self.query_rank, 128),
            ("output_groups", self.output_groups, 4),
            ("output_rank", self.output_rank, 128),
            ("window", self.window, 16),
            ("intermediate", self.intermediate, 256),
            ("routed_experts", self.routed_experts, 16),
            ("active_experts", self.active_experts, 16),
            ("hc_iterations", self.hc_iterations, 32),
            ("index_heads", self.index_heads, 4),
            ("index_topk", self.index_topk, 16),
            ("candidate_topk_blocks", self.candidate_topk_blocks, 16),
            ("candidate_block_size", self.candidate_block_size, 16),
            ("engram_embedding_width", self.engram_embedding_width, 128),
            ("engram_rows[0]", self.engram_rows[0], 4096),
            ("engram_rows[1]", self.engram_rows[1], 4096),
            ("engram_ngram", self.engram_ngram, 8),
            ("engram_heads", self.engram_heads, 8),
        ] {
            if value == 0 || value > maximum {
                return Err(ArtifactError::Invalid(format!(
                    "{name} outside bounded synthetic range 1..={maximum}"
                )));
            }
        }
        if self.max_tokens < 2
            || self.engram_ngram < 2
            || self.active_experts > self.routed_experts
            || self.rope_pairs * 2 > self.head_dimension
        {
            return Err(ArtifactError::Invalid(String::from(
                "incompatible request, expert or rotary geometry",
            )));
        }
        for (name, value) in [
            ("norm_epsilon", self.norm_epsilon),
            ("hc_epsilon", self.hc_epsilon),
            ("gate_temperature", self.gate_temperature),
            ("route_scale", self.route_scale),
            ("engram_gate_clamp", self.engram_gate_clamp),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ArtifactError::Invalid(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        if !self.swiglu_limit.is_finite() || self.swiglu_limit < 0.0 {
            return Err(ArtifactError::Invalid(String::from(
                "swiglu_limit must be finite and nonnegative",
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Dtype {
    U8,
    Bf16,
    F32,
    I64,
}

impl Dtype {
    const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    storage_hex: String,
    storage_sha256: String,
}

struct Tensor {
    shape: Vec<usize>,
    data: TensorData,
}

enum TensorData {
    U8(Vec<u8>),
    Bf16(Vec<u16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}
struct TensorStore(BTreeMap<String, Tensor>);

impl RawTensor {
    fn decode(self, total: &mut usize) -> Result<Tensor, ArtifactError> {
        if self.shape.is_empty() || self.shape.len() > 4 || self.shape.contains(&0) {
            return Err(ArtifactError::Invalid(String::from(
                "tensor rank must be 1..=4 with nonzero dimensions",
            )));
        }
        let elements = self
            .shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| ArtifactError::Invalid(String::from("tensor shape overflow")))?;
        if elements > MAX_TENSOR_ELEMENTS {
            return Err(ArtifactError::Invalid(String::from(
                "tensor element limit exceeded",
            )));
        }
        let bytes = elements * self.dtype.bytes();
        *total = total
            .checked_add(bytes)
            .ok_or_else(|| ArtifactError::Invalid(String::from("tensor byte total overflow")))?;
        if *total > MAX_DECODED_BYTES || self.storage_hex.len() != bytes * 2 {
            return Err(ArtifactError::Invalid(String::from(
                "decoded byte limit or exact tensor length mismatch",
            )));
        }
        let hex = self.storage_hex.as_bytes();
        let raw: Vec<u8> = hex
            .chunks_exact(2)
            .map(|pair| {
                let digit = |value: u8| match value {
                    b'0'..=b'9' => Some(value - b'0'),
                    b'a'..=b'f' => Some(value - b'a' + 10),
                    _ => None,
                };
                match (digit(pair[0]), digit(pair[1])) {
                    (Some(hi), Some(lo)) => Ok(hi * 16 + lo),
                    _ => Err(ArtifactError::Invalid(String::from(
                        "tensor hex must use lowercase ASCII digits",
                    ))),
                }
            })
            .collect::<Result<_, _>>()?;
        if self.storage_sha256 != format!("{:x}", Sha256::digest(&raw)) {
            return Err(ArtifactError::Invalid(String::from(
                "tensor SHA-256 mismatch",
            )));
        }
        let data = match self.dtype {
            Dtype::U8 => TensorData::U8(raw),
            Dtype::Bf16 => {
                let values: Vec<_> = raw
                    .chunks_exact(2)
                    .map(|v| u16::from_le_bytes([v[0], v[1]]))
                    .collect();
                if values
                    .iter()
                    .any(|&v| !crate::precision::bf16_to_f32(v).is_finite())
                {
                    return Err(ArtifactError::Invalid(String::from(
                        "nonfinite BF16 weight",
                    )));
                }
                TensorData::Bf16(values)
            }
            Dtype::F32 => {
                let values: Vec<_> = raw
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                    .collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(ArtifactError::Invalid(String::from("nonfinite F32 weight")));
                }
                TensorData::F32(values)
            }
            Dtype::I64 => TensorData::I64(
                raw.chunks_exact(8)
                    .map(|v| i64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]))
                    .collect(),
            ),
        };
        Ok(Tensor {
            shape: self.shape,
            data,
        })
    }
}

impl TensorStore {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<&TensorData, ArtifactError> {
        let tensor = self
            .0
            .get(name)
            .ok_or_else(|| ArtifactError::Invalid(format!("missing tensor {name}")))?;
        if tensor.shape != shape {
            return Err(ArtifactError::Invalid(format!(
                "tensor {name} shape mismatch"
            )));
        }
        Ok(&tensor.data)
    }
    fn u8(&self, name: &str, shape: &[usize]) -> Result<&[u8], ArtifactError> {
        match self.tensor(name, shape)? {
            TensorData::U8(v) => Ok(v),
            _ => Err(ArtifactError::Invalid(format!("tensor {name} requires u8"))),
        }
    }
    fn u16(&self, name: &str, shape: &[usize]) -> Result<&[u16], ArtifactError> {
        match self.tensor(name, shape)? {
            TensorData::Bf16(v) => Ok(v),
            _ => Err(ArtifactError::Invalid(format!(
                "tensor {name} requires bf16"
            ))),
        }
    }
    fn f32(&self, name: &str, shape: &[usize]) -> Result<&[f32], ArtifactError> {
        match self.tensor(name, shape)? {
            TensorData::F32(v) => Ok(v),
            _ => Err(ArtifactError::Invalid(format!(
                "tensor {name} requires f32"
            ))),
        }
    }
    fn i64(&self, name: &str, shape: &[usize]) -> Result<&[i64], ArtifactError> {
        match self.tensor(name, shape)? {
            TensorData::I64(v) => Ok(v),
            _ => Err(ArtifactError::Invalid(format!(
                "tensor {name} requires i64"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn encoded(dtype: &str, shape: &[usize], bytes: &[u8]) -> RawTensor {
        use std::fmt::Write as _;
        let mut hex = String::new();
        for byte in bytes {
            write!(&mut hex, "{byte:02x}").unwrap();
        }
        serde_json::from_value(json!({
            "dtype": dtype, "shape": shape,
            "storage_hex": hex,
            "storage_sha256": format!("{:x}", Sha256::digest(bytes)),
        }))
        .unwrap()
    }

    #[test]
    fn exact_tensor_geometry_hash_and_finite_storage_are_admission_requirements() {
        let valid = encoded("bf16", &[2], &[0x80, 0x3f, 0, 0xc0]);
        let decoded = valid.decode(&mut 0).unwrap();
        let tensors = TensorStore(BTreeMap::from([("norm".to_owned(), decoded)]));
        assert_eq!(tensors.u16("norm", &[2]).unwrap(), &[0x3f80, 0xc000]);
        assert!(tensors.u16("norm", &[1, 2]).is_err());
        assert!(tensors.f32("norm", &[2]).is_err());

        let mut changed = encoded("u8", &[1], &[1]);
        changed.storage_hex = "02".to_owned();
        assert!(changed.decode(&mut 0).is_err());
        assert!(
            encoded("u8", &[usize::MAX, 2], &[1])
                .decode(&mut 0)
                .is_err()
        );
        assert!(
            encoded("f32", &[1], &f32::INFINITY.to_le_bytes())
                .decode(&mut 0)
                .is_err()
        );
        assert!(
            encoded("bf16", &[1], &0x7fc0_u16.to_le_bytes())
                .decode(&mut 0)
                .is_err()
        );
    }

    #[test]
    fn duplicate_tensor_keys_and_unknown_tensor_fields_fail_closed() {
        #[derive(Deserialize)]
        struct MapOnly {
            #[serde(deserialize_with = "unique_tensors")]
            #[allow(dead_code, reason = "deserialization rejection is the tested boundary")]
            tensors: BTreeMap<String, RawTensor>,
        }
        let tensor = json!({"dtype":"u8","shape":[1],"storage_hex":"00","storage_sha256":format!("{:x}",Sha256::digest([0]))});
        let repeated = format!("{{\"tensors\":{{\"x\":{tensor},\"x\":{tensor}}}}}");
        assert!(serde_json::from_str::<MapOnly>(&repeated).is_err());
        let mut with_oracle = tensor;
        with_oracle["expected_output"] = json!([0]);
        assert!(serde_json::from_value::<RawTensor>(with_oracle).is_err());
    }
}
