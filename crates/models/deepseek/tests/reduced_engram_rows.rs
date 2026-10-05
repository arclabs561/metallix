//! The row-source Engram path against the owned-table path on the pinned
//! reduced layer-one and layer-three Engram fixtures.
#![allow(dead_code)]

#[path = "support/runtime_engram.rs"]
mod runtime_engram;

use std::cell::RefCell;

use deepseek::{
    engram::embedding::{EngramEmbeddingError, EngramRowSource},
    reduced::{EngramSession, EngramSessionError, EngramSessionWeights},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Serves rows from a full owned table, refusing any row in `missing`.
struct TableRows {
    codes: Vec<u8>,
    scales: Vec<u8>,
    width: usize,
    missing: Option<usize>,
    requested: RefCell<Vec<usize>>,
}

impl EngramRowSource for TableRows {
    fn read_rows(
        &self,
        rows: &[usize],
        codes: &mut [u8],
        scales: &mut [u8],
    ) -> Result<(), EngramEmbeddingError> {
        let scale_width = self.width / 32;
        for (index, &row) in rows.iter().enumerate() {
            if Some(row) == self.missing {
                return Err(EngramEmbeddingError::RowsUnavailable {
                    reason: format!("row {row} absent"),
                });
            }
            codes[index * self.width..(index + 1) * self.width]
                .copy_from_slice(&self.codes[row * self.width..(row + 1) * self.width]);
            scales[index * scale_width..(index + 1) * scale_width]
                .copy_from_slice(&self.scales[row * scale_width..(row + 1) * scale_width]);
        }
        self.requested.borrow_mut().extend_from_slice(rows);
        Ok(())
    }
}

fn fixture(name: &str) -> Value {
    let raw = match name {
        "layer1" => include_str!("../../../../fixtures/deepseek-v41/layer1-engram-reference.json"),
        _ => include_str!("../../../../fixtures/deepseek-v41/layer3-engram-reference.json"),
    };
    serde_json::from_str(raw).expect("Engram fixture JSON")
}

fn tensor(value: &Value) -> Vec<u8> {
    let raw: Vec<u8> = value["storage_hex"]
        .as_str()
        .expect("hex")
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        value["storage_sha256"].as_str().unwrap()
    );
    raw
}

fn words(value: &Value, width: usize) -> Vec<i64> {
    tensor(value)
        .chunks_exact(width)
        .map(|word| match width {
            2 => i64::from(u16::from_le_bytes(word.try_into().unwrap())),
            _ => i64::from_le_bytes(word.try_into().unwrap()),
        })
        .collect()
}

fn bf16(value: &Value) -> Vec<u16> {
    words(value, 2)
        .into_iter()
        .map(|word| u16::try_from(word).unwrap())
        .collect()
}

/// Owned-table, row-source-over-owned, and table-free sessions plus the source.
fn sessions(root: &Value, layer: u64) -> ([EngramSession; 3], TableRows) {
    let (config, owned) = runtime_engram::definition(root, layer);
    let parameter =
        |name: &str| &root["encoded_parameters"][format!("layers.{layer}.engram.{name}")];
    let rows = TableRows {
        codes: tensor(parameter("embed.weight")),
        scales: tensor(parameter("embed.scale")),
        width: usize::try_from(root["model"]["embedding_dim"].as_u64().unwrap()).unwrap(),
        missing: None,
        requested: RefCell::new(Vec::new()),
    };
    let table_free = EngramSessionWeights::without_embedding_table(
        tensor(parameter("wkv.weight")),
        tensor(parameter("wkv.scale")),
        bf16(parameter("q_weight")),
        bf16(parameter("k_weight")),
    );
    let sessions = [
        EngramSession::new(config.clone(), owned.clone()).unwrap(),
        EngramSession::new(config.clone(), owned).unwrap(),
        EngramSession::new(config, table_free).unwrap(),
    ];
    (sessions, rows)
}

fn case_inputs(case: &Value) -> (usize, Vec<i64>, Vec<u16>) {
    (
        usize::try_from(case["start_pos"].as_u64().unwrap()).unwrap(),
        words(&case["input_ids"], 8),
        bf16(&case["stream"]),
    )
}

#[test]
fn row_source_steps_are_bit_identical_to_owned_table_steps() {
    for (name, layer) in [("layer1", 1), ("layer3", 3)] {
        let root = fixture(name);
        let ([mut owned, mut over_owned, mut table_free], rows) = sessions(&root, layer);
        for case in root["cases"].as_array().unwrap() {
            let (start, ids, stream) = case_inputs(case);
            let expected = owned.step(start, &ids, &stream).unwrap();
            runtime_engram::assert_output(case, &expected);
            assert_eq!(
                over_owned.step_with(start, &ids, &stream, &rows).unwrap(),
                expected,
                "{name}: row source over an owned-table session"
            );
            assert_eq!(
                table_free.step_with(start, &ids, &stream, &rows).unwrap(),
                expected,
                "{name}: table-free session"
            );
            // Each step requests exactly its distinct in-table hash rows.
            let table_rows = rows.codes.len() / rows.width;
            let mut selected: Vec<usize> = expected
                .hash_ids()
                .iter()
                .filter_map(|&id| usize::try_from(id).ok())
                .filter(|&row| row < table_rows)
                .collect();
            selected.sort_unstable();
            selected.dedup();
            assert!(!selected.is_empty());
            assert_eq!(
                *rows.requested.borrow(),
                [selected.clone(), selected].concat()
            );
            rows.requested.borrow_mut().clear();
        }
    }
}

#[test]
fn missing_rows_fail_closed_without_publishing_history() {
    let root = fixture("layer1");
    let ([mut control, _, mut table_free], mut rows) = sessions(&root, 1);
    let case = &root["cases"][0];
    let (start, ids, stream) = case_inputs(case);

    assert!(matches!(
        table_free.step(start, &ids, &stream),
        Err(EngramSessionError::MissingEmbeddingRows)
    ));
    let expected = control.step(start, &ids, &stream).unwrap();
    let hashed_row = usize::try_from(expected.hash_ids()[0]).unwrap();
    rows.missing = Some(hashed_row);
    assert!(matches!(
        table_free.step_with(start, &ids, &stream, &rows),
        Err(EngramSessionError::Embedding(
            EngramEmbeddingError::RowsUnavailable { .. }
        ))
    ));
    assert_eq!(table_free.next_start(), 0, "failed lookup must not advance");

    rows.missing = None;
    assert_eq!(
        table_free.step_with(start, &ids, &stream, &rows).unwrap(),
        expected
    );
}
