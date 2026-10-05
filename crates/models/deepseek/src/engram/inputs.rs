//! Checked loader for exported real V4.1 Engram hash inputs.
//!
//! `scripts/export_v41_engram_inputs.py` reads the compressed token map, bucket
//! primes, offsets, and multipliers from the pinned source's own
//! `NgramHashState` buffers. Storing those proven values avoids re-deriving
//! tokenizer normalization, the prime search, or `NumPy`'s RNG stream in Rust.
//!
//! Wire format, integers little-endian: 8-byte magic `MXENGRAM`, `u32` header
//! length, UTF-8 JSON header, then one `u32` compressed ID per raw token ID.
//! Parsing fails closed on identity, integrity, shape, and layout invariants.

use std::collections::BTreeSet;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{CompressedToken, EngramHashError, EngramHashLayout};

const MAGIC: &[u8; 8] = b"MXENGRAM";
const FORMAT: &str = "metallix-v41-engram-inputs";
const SCHEMA_VERSION: u32 = 1;
const PREAMBLE_BYTES: usize = MAGIC.len() + 4;
const MAX_HEADER_BYTES: usize = 1 << 16;
const MAX_TOKEN_MAP: usize = 1 << 20;

/// Provenance an artifact must carry before its values are trusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngramInputsIdentity {
    /// Model checkpoint revision the source and config belong to.
    pub revision: &'static str,
    /// SHA-256 of the pinned Engram source file.
    pub engram_source_sha256: &'static str,
    /// SHA-256 of the `tokenizer.json` the token map was built from.
    pub tokenizer_sha256: &'static str,
    /// SHA-256 of the inference config that set the Engram layout.
    pub inference_config_sha256: &'static str,
    /// SHA-256 of the complete artifact, when the exact bytes are pinned.
    pub artifact_sha256: Option<&'static str>,
}

/// The real `DeepSeek` V4.1 export, pinned to its exact bytes.
pub const V41_ENGRAM_INPUTS_IDENTITY: EngramInputsIdentity = EngramInputsIdentity {
    revision: "dba1be0a40aa45a94ad051997016db3960a90277",
    engram_source_sha256: "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    tokenizer_sha256: "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    inference_config_sha256: "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
    artifact_sha256: Some("1ec590ff51b508663fcc28af589d970b767c25c49832453b60b152aeb3fac6b6"),
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    schema_version: u32,
    identity: IdentityHeader,
    layer_ids: Vec<usize>,
    num_embeddings: Vec<i64>,
    engram_vocab_size: i64,
    max_ngram_size: usize,
    heads: usize,
    raw_pad_id: usize,
    compressed_pad_id: i64,
    compressed_vocab_size: i64,
    primes: Vec<Vec<i64>>,
    offsets: Vec<Vec<i64>>,
    multipliers: Vec<Vec<i64>>,
    token_map: TokenMapHeader,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityHeader {
    revision: String,
    engram_source_sha256: String,
    tokenizer_sha256: String,
    inference_config_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenMapHeader {
    count: usize,
    dtype: String,
    sha256: String,
}

/// Validated Engram hash inputs for every configured Engram layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramHashInputs {
    layer_ids: Vec<usize>,
    num_embeddings: Vec<i64>,
    max_ngram_size: usize,
    heads: usize,
    compressed_pad_id: i64,
    primes: Vec<i64>,
    offsets: Vec<i64>,
    multipliers: Vec<i64>,
    token_map: Vec<u32>,
}

impl EngramHashInputs {
    /// Parses and validates an exported artifact against `identity`.
    ///
    /// Beyond identity and the payload digest, this checks the source's layout
    /// invariants: bucket divisors start at `engram_vocab_size`, ascend within
    /// each n-gram, and are never reused; offsets are each layer's exclusive
    /// running sum and end at that layer's table rows; multipliers are odd and
    /// within the source's no-overflow bound; and the pad maps through the
    /// token map.
    pub fn parse(bytes: &[u8], identity: &EngramInputsIdentity) -> Result<Self, EngramInputsError> {
        if let Some(expected) = identity.artifact_sha256 {
            let actual = format!("{:x}", Sha256::digest(bytes));
            check_identity("artifact_sha256", expected, &actual)?;
        }
        if bytes.len() < PREAMBLE_BYTES {
            return Err(EngramInputsError::Truncated);
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(EngramInputsError::BadMagic);
        }
        let header_bytes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let header_bytes =
            usize::try_from(header_bytes).map_err(|_| EngramInputsError::Truncated)?;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(EngramInputsError::HeaderTooLarge {
                bytes: header_bytes,
            });
        }
        let payload_start = PREAMBLE_BYTES + header_bytes;
        let header = bytes
            .get(PREAMBLE_BYTES..payload_start)
            .ok_or(EngramInputsError::Truncated)?;
        let header: Header = serde_json::from_slice(header)
            .map_err(|error| EngramInputsError::Header(error.to_string()))?;
        if header.format != FORMAT {
            return Err(EngramInputsError::Format);
        }
        if header.schema_version != SCHEMA_VERSION {
            return Err(EngramInputsError::SchemaVersion {
                version: header.schema_version,
            });
        }
        check_identity("revision", identity.revision, &header.identity.revision)?;
        check_identity(
            "engram_source_sha256",
            identity.engram_source_sha256,
            &header.identity.engram_source_sha256,
        )?;
        check_identity(
            "tokenizer_sha256",
            identity.tokenizer_sha256,
            &header.identity.tokenizer_sha256,
        )?;
        check_identity(
            "inference_config_sha256",
            identity.inference_config_sha256,
            &header.identity.inference_config_sha256,
        )?;

        let token_map = token_map(&header.token_map, &bytes[payload_start..])?;
        let (primes, offsets, multipliers) = layout(&header)?;
        check_token_map(&header, &token_map)?;
        Ok(Self {
            layer_ids: header.layer_ids,
            num_embeddings: header.num_embeddings,
            max_ngram_size: header.max_ngram_size,
            heads: header.heads,
            compressed_pad_id: header.compressed_pad_id,
            primes,
            offsets,
            multipliers,
            token_map,
        })
    }

    /// Model layer IDs, in the order of the hash state's layer axis.
    #[must_use]
    pub fn layer_ids(&self) -> &[usize] {
        &self.layer_ids
    }

    /// Embedding-table rows per Engram layer.
    #[must_use]
    pub fn num_embeddings(&self) -> &[i64] {
        &self.num_embeddings
    }

    /// Compressed ID of the configured raw pad token.
    #[must_use]
    pub const fn compressed_pad_id(&self) -> i64 {
        self.compressed_pad_id
    }

    /// Compressed ID per raw token ID.
    #[must_use]
    pub fn token_map(&self) -> &[u32] {
        &self.token_map
    }

    /// Builds the hash layout covering every Engram layer.
    pub fn hash_layout(&self) -> Result<EngramHashLayout, EngramHashError> {
        EngramHashLayout::new(
            self.max_ngram_size,
            self.heads,
            self.layer_ids.len(),
            self.compressed_pad_id,
            self.primes.clone(),
            self.offsets.clone(),
            self.multipliers.clone(),
        )
    }

    /// Maps raw token IDs to live compressed tokens.
    pub fn compress(&self, raw_ids: &[i64]) -> Result<Vec<CompressedToken>, EngramInputsError> {
        raw_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| {
                usize::try_from(id)
                    .ok()
                    .and_then(|raw| self.token_map.get(raw))
                    .map(|&compressed| CompressedToken::Live(i64::from(compressed)))
                    .ok_or(EngramInputsError::UnknownRawId { index, id })
            })
            .collect()
    }
}

fn check_identity(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), EngramInputsError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EngramInputsError::Identity {
            field,
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        })
    }
}

fn invalid(field: &'static str, reason: &'static str) -> EngramInputsError {
    EngramInputsError::Invalid { field, reason }
}

fn token_map(header: &TokenMapHeader, payload: &[u8]) -> Result<Vec<u32>, EngramInputsError> {
    if header.dtype != "u32" {
        return Err(invalid("token_map.dtype", "must be u32"));
    }
    if header.count == 0 || header.count > MAX_TOKEN_MAP {
        return Err(invalid("token_map.count", "must be in 1..=1048576"));
    }
    if payload.len() != header.count * 4 {
        return Err(EngramInputsError::PayloadLength {
            expected: header.count * 4,
            actual: payload.len(),
        });
    }
    if format!("{:x}", Sha256::digest(payload)) != header.sha256 {
        return Err(EngramInputsError::PayloadDigest);
    }
    Ok(payload
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

type Layout = (Vec<i64>, Vec<i64>, Vec<i64>);

fn layout(header: &Header) -> Result<Layout, EngramInputsError> {
    let layers = header.layer_ids.len();
    if layers == 0 || !header.layer_ids.is_sorted_by(|a, b| a < b) {
        return Err(invalid(
            "layer_ids",
            "must be nonempty and strictly increasing",
        ));
    }
    if header.num_embeddings.len() != layers {
        return Err(invalid("num_embeddings", "needs one entry per layer"));
    }
    if header.max_ngram_size < 2 || header.heads == 0 {
        return Err(invalid(
            "max_ngram_size",
            "needs n-gram size at least 2 and nonzero heads",
        ));
    }
    let columns = (header.max_ngram_size - 1)
        .checked_mul(header.heads)
        .ok_or(invalid("heads", "column count overflows"))?;
    let shaped = |rows: &[Vec<i64>], width: usize| {
        rows.len() == layers && rows.iter().all(|row| row.len() == width)
    };
    if !shaped(&header.primes, columns) {
        return Err(invalid(
            "primes",
            "must be [layers, (max_ngram_size - 1) * heads]",
        ));
    }
    if !shaped(&header.offsets, columns) {
        return Err(invalid(
            "offsets",
            "must be [layers, (max_ngram_size - 1) * heads]",
        ));
    }
    if !shaped(&header.multipliers, header.max_ngram_size) {
        return Err(invalid("multipliers", "must be [layers, max_ngram_size]"));
    }
    if header.engram_vocab_size < 1 {
        return Err(invalid("engram_vocab_size", "must be positive"));
    }

    let mut seen = BTreeSet::new();
    for (layer, primes) in header.primes.iter().enumerate() {
        for group in primes.chunks_exact(header.heads) {
            if group[0] < header.engram_vocab_size || !group.is_sorted_by(|a, b| a < b) {
                return Err(invalid(
                    "primes",
                    "must start at engram_vocab_size and ascend per n-gram",
                ));
            }
        }
        if !primes.iter().all(|&prime| seen.insert(prime)) {
            return Err(invalid("primes", "a bucket divisor is reused"));
        }
        let mut running = 0_i64;
        for (&prime, &offset) in primes.iter().zip(&header.offsets[layer]) {
            if offset != running {
                return Err(invalid(
                    "offsets",
                    "must be the exclusive running sum of primes",
                ));
            }
            running = running
                .checked_add(prime)
                .ok_or(invalid("primes", "row count overflows"))?;
        }
        if running != header.num_embeddings[layer] {
            return Err(invalid(
                "num_embeddings",
                "must equal the sum of the layer's primes",
            ));
        }
    }

    if header.compressed_vocab_size < 1 {
        return Err(invalid("compressed_vocab_size", "must be positive"));
    }
    // The source draws `x` in `[0, bound)` and stores `2x + 1`, which keeps every
    // compressed-ID product inside i64.
    let bound = (i64::MAX / header.compressed_vocab_size / 2).max(1);
    for &multiplier in header.multipliers.iter().flatten() {
        if multiplier <= 0 || multiplier % 2 == 0 || multiplier / 2 >= bound {
            return Err(invalid(
                "multipliers",
                "must be odd and within the source bound",
            ));
        }
    }
    Ok((
        header.primes.concat(),
        header.offsets.concat(),
        header.multipliers.concat(),
    ))
}

fn check_token_map(header: &Header, token_map: &[u32]) -> Result<(), EngramInputsError> {
    let largest = token_map.iter().copied().max().map_or(-1, i64::from);
    if largest + 1 != header.compressed_vocab_size {
        return Err(invalid(
            "token_map",
            "largest compressed ID must be compressed_vocab_size - 1",
        ));
    }
    let pad = token_map
        .get(header.raw_pad_id)
        .ok_or(invalid("raw_pad_id", "outside the token map"))?;
    if i64::from(*pad) != header.compressed_pad_id {
        return Err(invalid(
            "compressed_pad_id",
            "must be token_map[raw_pad_id]",
        ));
    }
    Ok(())
}

/// An Engram inputs artifact failed its wire, identity, or layout contract.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngramInputsError {
    /// The artifact ends before its declared header or preamble.
    #[error("Engram inputs artifact is truncated")]
    Truncated,
    /// The artifact does not start with `MXENGRAM`.
    #[error("Engram inputs artifact has the wrong magic")]
    BadMagic,
    /// The declared header is larger than the fixed cap.
    #[error("Engram inputs header is {bytes} bytes, maximum is 65536")]
    HeaderTooLarge { bytes: usize },
    /// The header is not the expected JSON object.
    #[error("Engram inputs header is invalid: {0}")]
    Header(String),
    /// The header names a different format.
    #[error("Engram inputs header has the wrong format")]
    Format,
    /// The header uses an unsupported schema version.
    #[error("Engram inputs schema version {version} is unsupported")]
    SchemaVersion { version: u32 },
    /// A provenance field differs from the expected identity.
    #[error("Engram inputs {field} is {actual}, expected {expected}")]
    Identity {
        /// Identity field.
        field: &'static str,
        /// Expected value.
        expected: String,
        /// Artifact value.
        actual: String,
    },
    /// The token-map payload length differs from its declared count.
    #[error("Engram token map payload is {actual} bytes, expected {expected}")]
    PayloadLength { expected: usize, actual: usize },
    /// The token-map payload digest differs from the header.
    #[error("Engram token map payload digest mismatch")]
    PayloadDigest,
    /// A header value breaks a shape or layout invariant.
    #[error("Engram inputs {field} is invalid: {reason}")]
    Invalid {
        /// Header field.
        field: &'static str,
        /// Broken invariant.
        reason: &'static str,
    },
    /// A raw token ID is outside the token map.
    #[error("raw token ID {id} at index {index} is outside the Engram token map")]
    UnknownRawId { index: usize, id: i64 },
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::{
        CompressedToken, EngramHashInputs, EngramInputsError, EngramInputsIdentity,
        V41_ENGRAM_INPUTS_IDENTITY,
    };
    use crate::engram::EngramHashState;

    const TEST_IDENTITY: EngramInputsIdentity = EngramInputsIdentity {
        revision: "test-revision",
        engram_source_sha256: "test-source",
        tokenizer_sha256: "test-tokenizer",
        inference_config_sha256: "test-config",
        artifact_sha256: None,
    };

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/engram-hash-reference.json"
        ))
        .expect("checked-in Engram fixture JSON")
    }

    fn fixture_tensor(fixture: &Value, name: &str) -> Value {
        fixture["tensors"]
            .as_array()
            .expect("fixture tensors")
            .iter()
            .find(|tensor| tensor["name"] == name)
            .unwrap_or_else(|| panic!("fixture tensor {name}"))["values"]
            .clone()
    }

    fn flat(value: &Value) -> Vec<i64> {
        match value {
            Value::Number(number) => vec![number.as_i64().expect("integer")],
            Value::Array(values) => values.iter().flat_map(flat).collect(),
            _ => panic!("expected nested integers"),
        }
    }

    /// The pinned hash fixture's explicit layout, wrapped as an inputs artifact.
    fn tiny() -> (Value, Vec<u32>) {
        let fixture = fixture();
        let rows = |name| -> Vec<Vec<i64>> {
            fixture_tensor(&fixture, name)
                .as_array()
                .expect("layers")
                .iter()
                .map(flat)
                .collect()
        };
        let token_map = flat(&fixture_tensor(&fixture, "token_map"))
            .into_iter()
            .map(|id| u32::try_from(id).expect("u32 ID"))
            .collect();
        let header = json!({
            "format": "metallix-v41-engram-inputs",
            "schema_version": 1,
            "identity": {
                "revision": "test-revision",
                "engram_source_sha256": "test-source",
                "tokenizer_sha256": "test-tokenizer",
                "inference_config_sha256": "test-config",
            },
            "layer_ids": [1, 14],
            "num_embeddings": [156, 304],
            "engram_vocab_size": 17,
            "max_ngram_size": 4,
            "heads": 2,
            "raw_pad_id": fixture["explicit_state"]["raw_pad_id"],
            "compressed_pad_id": fixture["explicit_state"]["compressed_pad_id"],
            "compressed_vocab_size": 43,
            "primes": rows("primes"),
            "offsets": rows("offsets"),
            "multipliers": rows("multipliers"),
            "token_map": {"count": 0, "dtype": "u32", "sha256": ""},
        });
        (header, token_map)
    }

    fn encode(header: &Value, token_map: &[u32]) -> Vec<u8> {
        let payload: Vec<u8> = token_map.iter().flat_map(|id| id.to_le_bytes()).collect();
        let mut header = header.clone();
        if header["token_map"]["sha256"] == "" {
            header["token_map"]["count"] = json!(token_map.len());
            header["token_map"]["sha256"] = json!(format!("{:x}", Sha256::digest(&payload)));
        }
        let header = serde_json::to_vec(&header).expect("header JSON");
        let mut bytes = b"MXENGRAM".to_vec();
        bytes.extend_from_slice(
            &u32::try_from(header.len())
                .expect("header length")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Hashes the fixture's raw tokens, with masked positions dead.
    fn fixture_hashes(inputs: &EngramHashInputs) -> Vec<i64> {
        let fixture = fixture();
        let raw = flat(&fixture["input"]["raw_token_ids"]);
        let mask: Vec<bool> = fixture["input"]["token_mask"]
            .as_array()
            .expect("mask rows")
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .expect("mask row")
                    .iter()
                    .map(|v| v.as_bool().expect("bool"))
            })
            .collect();
        let tokens: Vec<CompressedToken> = inputs
            .compress(&raw)
            .expect("fixture raw IDs")
            .into_iter()
            .zip(mask)
            .map(|(token, live)| if live { token } else { CompressedToken::Dead })
            .collect();
        let mut state =
            EngramHashState::new(inputs.hash_layout().expect("layout"), 2, 6).expect("state");
        state.write_and_hash(&tokens, 6, 0).expect("hash")
    }

    #[test]
    fn tiny_artifact_reproduces_pinned_fixture_hashes() {
        let (header, token_map) = tiny();
        let inputs = EngramHashInputs::parse(&encode(&header, &token_map), &TEST_IDENTITY)
            .expect("tiny artifact");
        assert_eq!(inputs.layer_ids(), [1, 14]);
        assert_eq!(inputs.compressed_pad_id(), 42);
        assert_eq!(
            fixture_hashes(&inputs),
            flat(&fixture_tensor(&fixture(), "one_shot_hashes"))
        );
    }

    #[test]
    fn perturbed_multiplier_changes_only_its_layer() {
        let (mut header, token_map) = tiny();
        // Still odd and in bound, so the artifact stays valid.
        header["multipliers"][0][1] = json!(7);
        let inputs = EngramHashInputs::parse(&encode(&header, &token_map), &TEST_IDENTITY)
            .expect("perturbed artifact");
        let actual = fixture_hashes(&inputs);
        let expected = flat(&fixture_tensor(&fixture(), "one_shot_hashes"));
        let layer = |values: &[i64], layer: usize| -> Vec<i64> {
            values
                .chunks_exact(6)
                .skip(layer)
                .step_by(2)
                .flatten()
                .copied()
                .collect()
        };
        assert_ne!(layer(&actual, 0), layer(&expected, 0));
        assert_eq!(layer(&actual, 1), layer(&expected, 1));
    }

    #[test]
    fn rejects_broken_artifacts() {
        let (header, token_map) = tiny();
        let parse = |bytes: &[u8]| EngramHashInputs::parse(bytes, &TEST_IDENTITY);
        let edited = |edit: &dyn Fn(&mut Value)| {
            let mut header = header.clone();
            edit(&mut header);
            parse(&encode(&header, &token_map))
        };
        let good = encode(&header, &token_map);
        let invalid = |field| move |result: Result<EngramHashInputs, EngramInputsError>| matches!(result, Err(EngramInputsError::Invalid { field: actual, .. }) if actual == field);

        assert_eq!(parse(&good[..10]), Err(EngramInputsError::Truncated));
        let mut magic = good.clone();
        magic[0] = b'X';
        assert_eq!(parse(&magic), Err(EngramInputsError::BadMagic));
        assert!(matches!(
            parse(&good[..good.len() - 1]),
            Err(EngramInputsError::PayloadLength { .. })
        ));
        let mut corrupt = good.clone();
        *corrupt.last_mut().expect("payload") ^= 1;
        assert_eq!(parse(&corrupt), Err(EngramInputsError::PayloadDigest));
        let pinned = EngramInputsIdentity {
            artifact_sha256: Some("0"),
            ..TEST_IDENTITY
        };
        assert!(matches!(
            EngramHashInputs::parse(&good, &pinned),
            Err(EngramInputsError::Identity {
                field: "artifact_sha256",
                ..
            })
        ));
        assert!(matches!(
            EngramHashInputs::parse(&good, &V41_ENGRAM_INPUTS_IDENTITY),
            Err(EngramInputsError::Identity { .. })
        ));

        assert!(matches!(
            edited(&|h| h["extra"] = json!(1)),
            Err(EngramInputsError::Header(_))
        ));
        assert_eq!(
            edited(&|h| h["format"] = json!("other")),
            Err(EngramInputsError::Format)
        );
        assert_eq!(
            edited(&|h| h["schema_version"] = json!(2)),
            Err(EngramInputsError::SchemaVersion { version: 2 })
        );
        assert!(matches!(
            edited(&|h| h["identity"]["tokenizer_sha256"] = json!("other")),
            Err(EngramInputsError::Identity {
                field: "tokenizer_sha256",
                ..
            })
        ));
        assert!(invalid("layer_ids")(edited(
            &|h| h["layer_ids"] = json!([14, 1])
        )));
        assert!(invalid("primes")(
            edited(&|h| h["primes"][0][1] = json!(17))
        ));
        assert!(invalid("primes")(
            edited(&|h| h["primes"][1][0] = json!(37))
        ));
        assert!(invalid("primes")(edited(
            &|h| h["engram_vocab_size"] = json!(18)
        )));
        assert!(invalid("offsets")(edited(
            &|h| h["offsets"][0][1] = json!(18)
        )));
        assert!(invalid("num_embeddings")(edited(
            &|h| h["num_embeddings"][1] = json!(305)
        )));
        assert!(invalid("multipliers")(edited(
            &|h| h["multipliers"][1][0] = json!(12)
        )));
        assert!(invalid("multipliers")(edited(&|h| h["multipliers"][1]
            [0] =
            json!(i64::MAX))));
        assert!(invalid("multipliers")(edited(
            &|h| h["multipliers"][1] = json!([11, 13])
        )));
        assert!(invalid("token_map")(edited(
            &|h| h["compressed_vocab_size"] = json!(44)
        )));
        assert!(invalid("compressed_pad_id")(edited(
            &|h| h["compressed_pad_id"] = json!(9)
        )));
        assert!(invalid("raw_pad_id")(edited(
            &|h| h["raw_pad_id"] = json!(8)
        )));
        assert!(invalid("token_map.dtype")(edited(
            &|h| h["token_map"]["dtype"] = json!("u16")
        )));
    }

    #[test]
    fn compress_rejects_raw_ids_outside_the_map() {
        let (header, token_map) = tiny();
        let inputs = EngramHashInputs::parse(&encode(&header, &token_map), &TEST_IDENTITY)
            .expect("tiny artifact");
        assert_eq!(inputs.compress(&[5]), Ok(vec![CompressedToken::Live(42)]));
        assert_eq!(
            inputs.compress(&[0, 8]),
            Err(EngramInputsError::UnknownRawId { index: 1, id: 8 })
        );
        assert_eq!(
            inputs.compress(&[-1]),
            Err(EngramInputsError::UnknownRawId { index: 0, id: -1 })
        );
    }

    fn repo() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn int64s(path: &Path) -> Vec<i64> {
        std::fs::read(path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("8 bytes")))
            .collect()
    }

    /// Needs the local export and source captures; see `scripts/export_v41_engram_inputs.py`.
    #[test]
    #[ignore = "needs .agents/receipts/engram-hash/v41-engram-inputs.bin and route-trace captures"]
    fn real_artifact_reproduces_captured_source_hash_ids() {
        let repo = repo();
        let path = std::env::var_os("METALLIX_V41_ENGRAM_INPUTS").map_or_else(
            || repo.join(".agents/receipts/engram-hash/v41-engram-inputs.bin"),
            PathBuf::from,
        );
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let inputs =
            EngramHashInputs::parse(&bytes, &V41_ENGRAM_INPUTS_IDENTITY).expect("real artifact");
        assert_eq!(inputs.layer_ids(), [1, 14]);
        assert_eq!(inputs.num_embeddings(), [384_006_168, 384_016_682]);

        let trace = repo.join(".agents/receipts/route-trace");
        let shell: Value = serde_json::from_slice(
            &std::fs::read(trace.join("parity-shell.json")).expect("parity-shell.json"),
        )
        .expect("parity-shell JSON");
        let prompts = [
            (vec![42, 7, 42], trace.join("capture-parity2")),
            (
                flat(&shell["runs"][0]["prompt_ids"]),
                trace.join("capture-shell2"),
            ),
        ];
        let columns = 24;
        for (ids, capture) in prompts {
            let tokens = inputs.compress(&ids).expect("prompt IDs");
            let mut state =
                EngramHashState::new(inputs.hash_layout().expect("layout"), 1, ids.len())
                    .expect("state");
            let hashes = state.write_and_hash(&tokens, ids.len(), 0).expect("hash");
            for (layer, &layer_id) in inputs.layer_ids().iter().enumerate() {
                let native: Vec<i64> = hashes
                    .chunks_exact(columns)
                    .skip(layer)
                    .step_by(inputs.layer_ids().len())
                    .flatten()
                    .copied()
                    .collect();
                let captured = int64s(&capture.join(format!("engram{layer_id:02}.ids.int64.bin")));
                assert_eq!(native.len(), ids.len() * columns);
                assert_eq!(native, captured, "{} layer {layer_id}", capture.display());
            }
        }
    }
}
