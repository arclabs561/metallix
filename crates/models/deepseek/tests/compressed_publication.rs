//! Test-only consumer for the pinned compressed-KV publication trace.
//!
//! It checks scheduling and visibility boundaries from source-captured events,
//! not compressor/indexer/rotary/FP4 numerical implementations.

use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressed-publication-reference.json"
    ))
    .expect("checked-in compressed publication fixture")
}

fn case<'a>(fixture: &'a Value, name: &str) -> &'a Value {
    fixture["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .find(|candidate| candidate["name"] == name)
        .unwrap_or_else(|| panic!("missing case {name}"))
}

fn event_names(case: &Value) -> Vec<&str> {
    case["events"]
        .as_array()
        .expect("case events")
        .iter()
        .map(|event| event["event"].as_str().expect("event name"))
        .collect()
}

fn values(value: &Value) -> Vec<u64> {
    value["values"]
        .as_array()
        .expect("tensor values")
        .iter()
        .map(|word| word.as_u64().expect("word"))
        .collect()
}

#[test]
fn source_trace_publishes_pre_mutation_index_then_post_write_prefix() {
    let fixture = fixture();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["receipt"]["device"], "cpu");
    assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
    assert_eq!(
        fixture["source"]["revision"],
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture["source"]["sha256"],
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture["source"]["symbols"],
        serde_json::json!(["Attention._compress_kv", "Attention._compress_topk_idxs"])
    );
    let prefill = case(&fixture, "prefill_ratio2_s3");
    assert_eq!(
        event_names(prefill),
        [
            "compressor",
            "indexer",
            "rotary",
            "quant",
            "cache_write",
            "cache_read"
        ]
    );
    let events = prefill["events"].as_array().expect("prefill events");
    assert_eq!(
        events[0]["latent_bf16"],
        events[1]["latent_before_mutation_bf16"]
    );
    assert_ne!(
        events[1]["latent_before_mutation_bf16"],
        events[4]["value_bf16"]
    );
    assert_eq!(events[2]["frequency_indices"], serde_json::json!([0]));
    assert_eq!(events[3]["block_size"], 16);
    assert_eq!(events[3]["inplace"], true);
    assert_eq!(events[3]["scale_dtype"], "torch.float8_e4m3fn");
    assert_eq!(
        events[4]["key"],
        "(slice(None, 1, None), slice(0, 1, None))"
    );
    assert_eq!(
        values(&prefill["returned_prefix"]),
        events[4]["value_bf16"]
            .as_array()
            .expect("written latent")
            .iter()
            .map(|word| word.as_u64().expect("BF16 word"))
            .collect::<Vec<_>>()
    );
}

#[test]
fn completion_nonboundary_and_consumer_share_follow_source_schedule() {
    let fixture = fixture();
    let completion = case(&fixture, "singleton_completion_start3");
    let completion_events = completion["events"].as_array().expect("completion events");
    assert_eq!(
        event_names(completion),
        [
            "compressor",
            "indexer",
            "rotary",
            "quant",
            "cache_write",
            "cache_read"
        ]
    );
    assert_eq!(
        completion_events[2]["frequency_indices"],
        serde_json::json!([2])
    );
    assert_eq!(
        completion_events[4]["key"],
        "(slice(None, 1, None), slice(1, 2, None))"
    );

    let nonboundary = case(&fixture, "nonboundary_start4");
    assert_eq!(
        event_names(nonboundary),
        ["compressor", "indexer", "cache_read"]
    );
    assert!(nonboundary["events"][1]["latent_before_mutation_bf16"].is_null());

    let zero = case(&fixture, "initial_s1_zero_compress");
    assert_eq!(event_names(zero), ["compressor", "cache_read"]);
    assert_eq!(values(&zero["returned_prefix"]), Vec::<u64>::new());

    let consumer = case(&fixture, "consumer_reuses_shared_cache");
    assert_eq!(event_names(consumer), ["cache_read"]);
    assert_eq!(consumer["start_pos"], nonboundary["start_pos"]);
    assert_eq!(
        consumer["returned_indices_i32"],
        nonboundary["returned_indices_i32"]
    );
    assert_eq!(consumer["returned_indices_i32"], serde_json::json!([3]));
    assert_eq!(
        values(&consumer["returned_prefix"]),
        values(&completion["returned_prefix"])
    );
}
