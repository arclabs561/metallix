//! Teacher-forced scoring against the all-position logits it replaces.

use super::prefix_extend::{sensitive_weights, two_layer_long_config};
use super::*;
use crate::forward::{Qwen3Scores, SCORE_CHUNK_ROWS};

const IDS: [i32; 11] = [1, 5, 2, 7, 3, 6, 4, 2, 7, 0, 5];
const FROM: usize = 4;

/// Log-softmax of one host row.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row
        .iter()
        .map(|&value| f64::from(value - maximum).exp())
        .sum();
    row.iter()
        .map(|&value| f64::from(value - maximum) - sum.ln())
        .collect()
}

/// The chunked head gives the log-softmax of the full rows, whatever the
/// chunk size, and chunking changes no gathered value.
#[test]
fn scoring_matches_all_position_logits_at_any_chunk_size() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = || {
        config
            .resident_chat_plan(600, u64::MAX, crate::forward::Qwen3FloatPrecision::Float32)
            .expect("tiny resident plan")
    };
    let rows = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan())
        .prefill_all_logits(&IDS)
        .expect("all-position logits");
    let scored_by_chunk: Vec<Qwen3Scores> = [1, 3, SCORE_CHUNK_ROWS]
        .into_iter()
        .map(|chunk| {
            Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan())
                .score(&IDS, FROM, 3, chunk)
                .expect("scored")
        })
        .collect();
    let top_three = |row: usize| {
        let expected = log_softmax(rows.row(row).expect("row"));
        let mut order: Vec<usize> = (0..expected.len()).collect();
        order.sort_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
        order[..3]
            .iter()
            .map(|&id| i32::try_from(id).expect("id"))
            .collect::<Vec<i32>>()
    };
    for scores in &scored_by_chunk {
        assert_eq!(scores.tokens.len(), IDS.len() - FROM);
        let next: Vec<i32> = scores.next.iter().map(|&(id, _)| id).collect();
        assert_eq!(next, top_three(IDS.len() - 1), "the row after the sequence");
        for (offset, token) in scores.tokens.iter().enumerate() {
            let position = FROM + offset;
            let expected = log_softmax(rows.row(position - 1).expect("row"));
            let target = usize::try_from(IDS[position]).expect("token id");
            assert!(
                (f64::from(token.logprob) - expected[target]).abs() < 1e-5,
                "position {position}: {} vs {}",
                token.logprob,
                expected[target]
            );
            let top: Vec<i32> = token.top.iter().map(|&(id, _)| id).collect();
            assert_eq!(top, top_three(position - 1), "position {position}");
        }
    }
    // Two prefills, as a cached prefix would be, move values only by
    // reduction order.
    let split = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan())
        .score_split(&IDS, FROM, FROM, 3, SCORE_CHUNK_ROWS)
        .expect("scored in two appends");
    for (a, b) in split.tokens.iter().zip(&scored_by_chunk[2].tokens) {
        assert!(
            (a.logprob - b.logprob).abs() < 1e-5,
            "{} vs {}",
            a.logprob,
            b.logprob
        );
    }
    // Chunking only bounds memory: the gathered values agree across sizes.
    let unchunked = &scored_by_chunk[2];
    for chunked in &scored_by_chunk[..2] {
        for (a, b) in chunked.tokens.iter().zip(&unchunked.tokens) {
            assert!(
                (a.logprob - b.logprob).abs() <= 1e-6,
                "{} vs {}",
                a.logprob,
                b.logprob
            );
        }
    }
}

#[test]
fn a_scoring_range_needs_a_prefix_within_the_sequence() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = config
        .resident_chat_plan(600, u64::MAX, crate::forward::Qwen3FloatPrecision::Float32)
        .expect("tiny resident plan");
    let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    for from in [0, IDS.len() + 1] {
        assert!(matches!(
            executor.score(&IDS, from, 0, 4),
            Err(Qwen3ForwardError::InvalidScoreRange { .. })
        ));
    }
}

/// For a greedy continuation each scored token is its row's top
/// alternative, and its log probability is that alternative's value bit for
/// bit, in every chunk layout and on the rows either side of a chunk
/// boundary. lm-eval calls a continuation greedy when each
/// `token_logprobs[i]` equals `max(top_logprobs[i].values())` exactly.
#[test]
fn a_greedy_tokens_logprob_is_bit_equal_to_its_top_alternative() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = || {
        config
            .resident_chat_plan(600, u64::MAX, crate::forward::Qwen3FloatPrecision::Float32)
            .expect("tiny resident plan")
    };
    let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan());
    let mut ids = IDS[..FROM].to_vec();
    for _ in 0..8 {
        let next = executor
            .score(&ids, ids.len(), 1, SCORE_CHUNK_ROWS)
            .expect("next token")
            .next;
        ids.push(next[0].0);
    }
    for chunk in [1, 3, SCORE_CHUNK_ROWS] {
        let scores = executor
            .score(&ids, FROM, 3, chunk)
            .expect("scored continuation");
        assert_eq!(scores.tokens.len(), ids.len() - FROM);
        for (offset, token) in scores.tokens.iter().enumerate() {
            let (best_id, best) = token.top[0];
            assert_eq!(
                best_id,
                ids[FROM + offset],
                "chunk {chunk}, offset {offset}"
            );
            assert_eq!(
                token.logprob.to_bits(),
                best.to_bits(),
                "chunk {chunk}, offset {offset}: {} vs {}",
                token.logprob,
                best
            );
        }
    }
}
