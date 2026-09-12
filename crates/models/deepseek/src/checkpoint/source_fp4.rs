//! Selected source-I8 expert weight/scale metadata, without payload decoding.
//!
//! The pinned converter treats source `I8` expert weights as packed pairs: a
//! `[N, P]` source tensor describes a logical `[N, 2P]` matrix with one E8M0
//! scale per 32 logical reduction elements. This descriptor establishes only
//! that selected header and index metadata has that shape. It does not read a
//! file, identify a revision, decode payload bytes, establish nibble order, or
//! support other checkpoint layouts.

use thiserror::Error;

use super::{V41SafetensorsHeader, V41StorageDtype, V41TensorRange};
use crate::manifest::V41SafetensorsIndex;

const WEIGHT_GROUP: u64 = 32;

/// The canonical routed-expert projection named by a source checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V41ExpertProjection {
    /// The expert's first gate projection.
    W1,
    /// The expert's down projection.
    W2,
    /// The expert's up projection.
    W3,
}

/// A validated selected source-I8 routed-expert weight and its E8M0 scale.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExpertI8ScalePair {
    layer: u64,
    expert: u64,
    projection: V41ExpertProjection,
    shard: String,
    weight_name: String,
    scale_name: String,
    weight_range: V41TensorRange,
    scale_range: V41TensorRange,
    logical_shape: [u64; 2],
}

impl V41ExpertI8ScalePair {
    /// Parses one exact canonical expert weight name and its paired metadata.
    ///
    /// Only `layers.<digits>.ffn.experts.<digits>.w1|w2|w3.weight` names are
    /// accepted. Both the exact weight and its exact `.scale` sibling must be
    /// assigned to `shard` by `index` and present in `header`. This deliberately
    /// validates selected entries only, not full header/index agreement.
    /// Layer and expert identifiers are syntactic only. The caller establishes
    /// model-config bounds and binds the supplied metadata to an actual file and
    /// revision.
    ///
    /// # Errors
    ///
    /// Returns [`V41ExpertI8ScalePairError`] for a noncanonical name, missing
    /// or differently assigned selected entry, or incompatible selected dtype
    /// or shape metadata.
    pub fn parse(
        header: &V41SafetensorsHeader,
        index: &V41SafetensorsIndex,
        shard: &str,
        weight_name: &str,
    ) -> Result<Self, V41ExpertI8ScalePairError> {
        let (layer, expert, projection) = parse_weight_name(weight_name)?;
        let base = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid_name(weight_name))?;
        let scale_name = format!("{base}.scale");
        validate_index(index, shard, weight_name)?;
        validate_index(index, shard, &scale_name)?;
        let weight_range = header.tensor(weight_name).ok_or_else(|| {
            V41ExpertI8ScalePairError::MissingHeaderTensor {
                tensor: weight_name.to_owned(),
            }
        })?;
        let scale_range = header.tensor(&scale_name).ok_or_else(|| {
            V41ExpertI8ScalePairError::MissingHeaderTensor {
                tensor: scale_name.clone(),
            }
        })?;
        let logical_shape = validate_shapes(weight_range, scale_range)?;
        Ok(Self {
            layer,
            expert,
            projection,
            shard: shard.to_owned(),
            weight_name: weight_name.to_owned(),
            scale_name,
            weight_range: weight_range.clone(),
            scale_range: scale_range.clone(),
            logical_shape,
        })
    }

    /// Returns the parsed canonical layer number.
    #[must_use]
    pub const fn layer(&self) -> u64 {
        self.layer
    }

    /// Returns the parsed canonical routed-expert number.
    #[must_use]
    pub const fn expert(&self) -> u64 {
        self.expert
    }

    /// Returns the selected expert projection.
    #[must_use]
    pub const fn projection(&self) -> V41ExpertProjection {
        self.projection
    }

    /// Returns the caller-declared shard name.
    #[must_use]
    pub fn shard(&self) -> &str {
        &self.shard
    }

    /// Returns the exact selected source-I8 tensor name.
    #[must_use]
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the exact selected E8M0 scale tensor name.
    #[must_use]
    pub fn scale_name(&self) -> &str {
        &self.scale_name
    }

    /// Returns the cloned, validated source-I8 tensor interval metadata.
    #[must_use]
    pub fn weight_range(&self) -> &V41TensorRange {
        &self.weight_range
    }

    /// Returns the cloned, validated E8M0 scale interval metadata.
    #[must_use]
    pub fn scale_range(&self) -> &V41TensorRange {
        &self.scale_range
    }

    /// Returns the logical unpacked source matrix shape `[N, K]`.
    #[must_use]
    pub const fn logical_shape(&self) -> [u64; 2] {
        self.logical_shape
    }
}

/// A selected source-I8 expert pair does not meet the bounded metadata contract.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41ExpertI8ScalePairError {
    /// The supplied name is not exactly one supported canonical routed-expert weight.
    #[error("unsupported canonical source expert weight name {weight_name}")]
    InvalidWeightName {
        /// Caller-supplied source tensor name.
        weight_name: String,
    },
    /// A selected tensor is absent from the already-validated index.
    #[error("selected tensor {tensor} is absent from the safetensors index")]
    MissingIndexTensor {
        /// Exact selected tensor name.
        tensor: String,
    },
    /// A selected tensor is assigned to a shard other than the caller-declared shard.
    #[error("selected tensor {tensor} is assigned to {indexed_shard}, not {expected_shard}")]
    WrongIndexShard {
        /// Exact selected tensor name.
        tensor: String,
        /// Caller-declared shard.
        expected_shard: String,
        /// Index-declared shard.
        indexed_shard: String,
    },
    /// A selected tensor is absent from the already-validated header.
    #[error("selected tensor {tensor} is absent from the safetensors header")]
    MissingHeaderTensor {
        /// Exact selected tensor name.
        tensor: String,
    },
    /// The source weight dtype is not I8.
    #[error("source expert weight dtype is {actual:?}, expected I8")]
    WeightDtype {
        /// Header-declared dtype.
        actual: V41StorageDtype,
    },
    /// The source scale dtype is not `F8E8M0Fnu`.
    #[error("source expert scale dtype is {actual:?}, expected F8E8M0Fnu")]
    ScaleDtype {
        /// Header-declared dtype.
        actual: V41StorageDtype,
    },
    /// The source-I8 weight is not a nonzero rank-two `[N, P]` tensor.
    #[error("source expert weight shape must be nonzero rank-two [N, P]")]
    WeightShape,
    /// Doubling packed source columns does not fit u64.
    #[error("source expert logical reduction width overflows u64")]
    LogicalReductionOverflow,
    /// The logical reduction width is not divisible by 32.
    #[error("source expert logical reduction width {reduction} is not divisible by 32")]
    LogicalReductionNotGrouped {
        /// Doubled packed-column count.
        reduction: u64,
    },
    /// The E8M0 scale is not exactly rank-two `[N, K / 32]`.
    #[error("source expert scale shape must be exactly [N, K / 32]")]
    ScaleShape,
}

fn parse_weight_name(
    weight_name: &str,
) -> Result<(u64, u64, V41ExpertProjection), V41ExpertI8ScalePairError> {
    let mut parts = weight_name.split('.');
    let parsed = match (
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
    ) {
        (
            Some("layers"),
            Some(layer),
            Some("ffn"),
            Some("experts"),
            Some(expert),
            Some(projection),
            Some("weight"),
            None,
        ) => {
            let projection = match projection {
                "w1" => V41ExpertProjection::W1,
                "w2" => V41ExpertProjection::W2,
                "w3" => V41ExpertProjection::W3,
                _ => return Err(invalid_name(weight_name)),
            };
            match (canonical_u64(layer), canonical_u64(expert)) {
                (Some(layer), Some(expert)) => (layer, expert, projection),
                _ => return Err(invalid_name(weight_name)),
            }
        }
        _ => return Err(invalid_name(weight_name)),
    };
    Ok(parsed)
}

fn canonical_u64(text: &str) -> Option<u64> {
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    text.parse().ok()
}

fn invalid_name(weight_name: &str) -> V41ExpertI8ScalePairError {
    V41ExpertI8ScalePairError::InvalidWeightName {
        weight_name: weight_name.to_owned(),
    }
}

fn validate_index(
    index: &V41SafetensorsIndex,
    shard: &str,
    tensor: &str,
) -> Result<(), V41ExpertI8ScalePairError> {
    let indexed_shard = index.shard_for_tensor(tensor).ok_or_else(|| {
        V41ExpertI8ScalePairError::MissingIndexTensor {
            tensor: tensor.to_owned(),
        }
    })?;
    if indexed_shard != shard {
        return Err(V41ExpertI8ScalePairError::WrongIndexShard {
            tensor: tensor.to_owned(),
            expected_shard: shard.to_owned(),
            indexed_shard: indexed_shard.to_owned(),
        });
    }
    Ok(())
}

fn validate_shapes(
    weight: &V41TensorRange,
    scale: &V41TensorRange,
) -> Result<[u64; 2], V41ExpertI8ScalePairError> {
    if weight.dtype() != V41StorageDtype::I8 {
        return Err(V41ExpertI8ScalePairError::WeightDtype {
            actual: weight.dtype(),
        });
    }
    let &[rows, packed_columns] = weight.shape() else {
        return Err(V41ExpertI8ScalePairError::WeightShape);
    };
    if rows == 0 || packed_columns == 0 {
        return Err(V41ExpertI8ScalePairError::WeightShape);
    }
    let reduction = packed_columns
        .checked_mul(2)
        .ok_or(V41ExpertI8ScalePairError::LogicalReductionOverflow)?;
    if !reduction.is_multiple_of(WEIGHT_GROUP) {
        return Err(V41ExpertI8ScalePairError::LogicalReductionNotGrouped { reduction });
    }
    if scale.dtype() != V41StorageDtype::F8E8M0Fnu {
        return Err(V41ExpertI8ScalePairError::ScaleDtype {
            actual: scale.dtype(),
        });
    }
    if scale.shape() != [rows, reduction / WEIGHT_GROUP] {
        return Err(V41ExpertI8ScalePairError::ScaleShape);
    }
    Ok([rows, reduction])
}

#[cfg(test)]
mod tests {
    use super::{V41ExpertI8ScalePair, V41ExpertI8ScalePairError, V41ExpertProjection};
    use crate::{checkpoint::V41SafetensorsHeader, manifest::V41SafetensorsIndex};

    const SHARD: &str = "model-00009-of-00048.safetensors";

    fn header(
        weight_name: &str,
        weight_dtype: &str,
        weight_shape: &[u64],
        scale_dtype: &str,
        scale_shape: &[u64],
    ) -> V41SafetensorsHeader {
        let scale_name = weight_name
            .strip_suffix(".weight")
            .expect("weight name")
            .to_owned()
            + ".scale";
        let weight_bytes = elements(weight_shape) * bytes_per_element(weight_dtype);
        let scale_bytes = elements(scale_shape) * bytes_per_element(scale_dtype);
        let json = format!(
            r#"{{"{weight_name}":{{"dtype":"{weight_dtype}","shape":{weight_shape:?},"data_offsets":[0,{weight_bytes}]}},"{scale_name}":{{"dtype":"{scale_dtype}","shape":{scale_shape:?},"data_offsets":[{weight_bytes},{}]}}}}"#,
            weight_bytes + scale_bytes
        );
        V41SafetensorsHeader::parse(
            json.as_bytes(),
            8 + u64::try_from(json.len()).expect("small test header") + weight_bytes + scale_bytes,
        )
        .expect("bounded test header")
    }

    fn elements(shape: &[u64]) -> u64 {
        shape.iter().copied().product()
    }

    fn bytes_per_element(dtype: &str) -> u64 {
        match dtype {
            "F32" => 4,
            _ => 1,
        }
    }

    fn index(
        weight_name: &str,
        scale: bool,
        shard: &str,
        unrelated: Option<&str>,
    ) -> V41SafetensorsIndex {
        use std::fmt::Write as _;

        let scale_name = weight_name
            .strip_suffix(".weight")
            .expect("weight name")
            .to_owned()
            + ".scale";
        let mut mappings = format!(r#""{weight_name}":"{shard}""#);
        if scale {
            write!(mappings, r#", "{scale_name}":"{shard}""#).expect("write test mapping");
        }
        if let Some(unrelated_shard) = unrelated {
            write!(mappings, r#", "unrelated":"{unrelated_shard}""#).expect("write test mapping");
        }
        V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{{mappings}}}}}"#
        ))
        .expect("bounded test index")
    }

    #[test]
    fn parses_each_projection_without_requiring_unrelated_agreement() {
        for (projection_name, projection) in [
            ("w1", V41ExpertProjection::W1),
            ("w2", V41ExpertProjection::W2),
            ("w3", V41ExpertProjection::W3),
        ] {
            let name = format!("layers.12.ffn.experts.34.{projection_name}.weight");
            let header = header(&name, "I8", &[2, 16], "F8_E8M0FNU", &[2, 1]);
            let index = index(&name, true, SHARD, Some("another-shard.safetensors"));
            let pair = V41ExpertI8ScalePair::parse(&header, &index, SHARD, &name)
                .expect("selected pair ignores unrelated disagreement");
            assert_eq!(pair.layer(), 12);
            assert_eq!(pair.expert(), 34);
            assert_eq!(pair.projection(), projection);
            assert_eq!(pair.shard(), SHARD);
            assert_eq!(pair.weight_name(), name);
            assert_eq!(pair.scale_name(), name.replace(".weight", ".scale"));
            assert_eq!(pair.logical_shape(), [2, 32]);
            assert_eq!(pair.weight_range().shape(), [2, 16]);
            assert_eq!(pair.scale_range().shape(), [2, 1]);
        }
    }

    #[test]
    fn rejects_noncanonical_names_before_metadata_lookup() {
        let header = header(
            "layers.1.ffn.experts.2.w1.weight",
            "I8",
            &[2, 16],
            "F8_E8M0FNU",
            &[2, 1],
        );
        let index = index("layers.1.ffn.experts.2.w1.weight", true, SHARD, None);
        for name in [
            "model.layers.1.ffn.experts.2.w1.weight",
            "mtp.layers.1.ffn.experts.2.w1.weight",
            "layers.01.ffn.experts.2.w1.weight",
            "layers.1.ffn.experts.02.w1.weight",
            "layers.1.ffn.shared_experts.2.w1.weight",
            "layers.1.ffn.experts.2.w4.weight",
            "layers.1.ffn.experts.2.w1.scale",
            "vision.layers.1.ffn.experts.2.w1.weight",
            "layers.18446744073709551616.ffn.experts.2.w1.weight",
        ] {
            assert!(matches!(
                V41ExpertI8ScalePair::parse(&header, &index, SHARD, name),
                Err(V41ExpertI8ScalePairError::InvalidWeightName { .. })
            ));
        }
    }

    #[test]
    fn rejects_selected_index_and_header_absence_or_wrong_shard() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        let header = header(name, "I8", &[2, 16], "F8_E8M0FNU", &[2, 1]);
        let missing_scale = index(name, false, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &missing_scale, SHARD, name),
            Err(V41ExpertI8ScalePairError::MissingIndexTensor { .. })
        ));
        let wrong_shard = index(name, true, "other.safetensors", None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &wrong_shard, SHARD, name),
            Err(V41ExpertI8ScalePairError::WrongIndexShard { .. })
        ));
        let split_index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":34}},"weight_map":{{"{name}":"{SHARD}","layers.1.ffn.experts.2.w1.scale":"other.safetensors"}}}}"#
        ))
        .expect("selected scale alone is on another shard");
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &split_index, SHARD, name),
            Err(V41ExpertI8ScalePairError::WrongIndexShard { tensor, .. })
                if tensor == "layers.1.ffn.experts.2.w1.scale"
        ));

        let weight_bytes = 32_u64;
        let json = format!(
            r#"{{"{name}":{{"dtype":"I8","shape":[2,16],"data_offsets":[0,{weight_bytes}]}}}}"#
        );
        let only_weight = V41SafetensorsHeader::parse(
            json.as_bytes(),
            8 + u64::try_from(json.len()).expect("small test header") + weight_bytes,
        )
        .expect("bounded one-tensor header");
        let complete_index = index(name, true, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&only_weight, &complete_index, SHARD, name),
            Err(V41ExpertI8ScalePairError::MissingHeaderTensor { .. })
        ));
    }

    #[test]
    fn rejects_dtypes_ranks_shapes_and_ungrouped_reduction() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        for (weight_dtype, weight_shape, scale_dtype, scale_shape, expected) in [
            (
                "U8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 1][..],
                "weight dtype",
            ),
            (
                "I8",
                &[2, 16, 1][..],
                "F8_E8M0FNU",
                &[2, 1][..],
                "weight rank",
            ),
            ("I8", &[0, 16][..], "F8_E8M0FNU", &[0, 1][..], "weight zero"),
            ("I8", &[2, 0][..], "F8_E8M0FNU", &[2, 0][..], "weight zero"),
            ("I8", &[2, 15][..], "F8_E8M0FNU", &[2, 1][..], "group"),
            ("I8", &[2, 16][..], "U8", &[2, 1][..], "scale dtype"),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 1, 1][..],
                "scale rank",
            ),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[3, 1][..],
                "scale dimensions",
            ),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 2][..],
                "scale dimensions",
            ),
        ] {
            let header = header(name, weight_dtype, weight_shape, scale_dtype, scale_shape);
            let index = index(name, true, SHARD, None);
            let error =
                V41ExpertI8ScalePair::parse(&header, &index, SHARD, name).expect_err(expected);
            match expected {
                "weight dtype" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::WeightDtype { .. }
                )),
                "weight rank" | "weight zero" => {
                    assert!(matches!(error, V41ExpertI8ScalePairError::WeightShape));
                }
                "group" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::LogicalReductionNotGrouped { .. }
                )),
                "scale dtype" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::ScaleDtype { .. }
                )),
                "scale rank" | "scale dimensions" => {
                    assert!(matches!(error, V41ExpertI8ScalePairError::ScaleShape));
                }
                _ => unreachable!("fixed test labels"),
            }
        }
    }

    #[test]
    fn rejects_logical_reduction_overflow_without_allocating_payload() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        let packed_columns = u64::MAX / 2 + 1;
        let header = header(name, "I8", &[1, packed_columns], "F8_E8M0FNU", &[1, 1]);
        let index = index(name, true, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &index, SHARD, name),
            Err(V41ExpertI8ScalePairError::LogicalReductionOverflow)
        ));
    }
}
