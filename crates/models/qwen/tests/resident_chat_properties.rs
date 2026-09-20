//! CPU-only property coverage for resident-chat admission.
//!
//! The generated configurations exercise the public parsing and planning API.
//! They intentionally construct no MLX arrays or executor graphs.

#![cfg(feature = "metal")]

use std::collections::HashMap;

use mlx_rs::Array;
use proptest::prelude::*;
use qwen::forward::{
    MAX_DENSE_DEBUG_TOKENS, MAX_RESIDENT_CHAT_TOKENS, Qwen3ForwardConfig, Qwen3ForwardError,
    Qwen3ForwardExecutor,
};

fn config_json(
    hidden_layers: usize,
    key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
) -> String {
    format!(
        r#"{{
          "model_type":"qwen3",
          "num_hidden_layers":{hidden_layers},
          "hidden_size":1,
          "intermediate_size":1,
          "vocab_size":1,
          "num_attention_heads":{key_value_heads},
          "num_key_value_heads":{key_value_heads},
          "head_dim":{head_dim},
          "max_position_embeddings":{max_position_embeddings},
          "rms_norm_eps":0.000001,
          "rope_theta":1000000,
          "hidden_act":"silu",
          "tie_word_embeddings":true,
          "attention_bias":false,
          "mlp_bias":false,
          "sliding_window":null,
          "use_sliding_window":false
        }}"#,
    )
}

fn logical_kv_bytes(
    hidden_layers: usize,
    key_value_heads: usize,
    head_dim: usize,
    context_tokens: usize,
) -> Option<u64> {
    let bytes = u128::try_from(hidden_layers)
        .ok()?
        .checked_mul(2)?
        .checked_mul(u128::try_from(key_value_heads).ok()?)?
        .checked_mul(u128::try_from(context_tokens).ok()?)?
        .checked_mul(u128::try_from(head_dim).ok()?)?
        .checked_mul(4)?;
    u64::try_from(bytes).ok()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Admission exactly follows its documented logical f32 K/V formula and
    /// rejects every boundary before any executor/MLX graph can be created.
    #[test]
    fn resident_plan_matches_the_independent_checked_formula(
        hidden_layers in prop_oneof![1_usize..=64, Just(usize::MAX)],
        key_value_heads in 1_usize..=64,
        head_dim_halves in 1_usize..=512,
        model_context in 1_usize..=4_096,
        requested_context in 0_usize..=4_096,
        budget_delta in -1_i8..=1,
    ) {
        let head_dim = head_dim_halves * 2;
        let config = Qwen3ForwardConfig::parse(&config_json(
            hidden_layers,
            key_value_heads,
            head_dim,
            model_context,
        ))
        .expect("generated Qwen3 layout is valid");
        let maximum = MAX_RESIDENT_CHAT_TOKENS.min(model_context);
        let required = logical_kv_bytes(
            hidden_layers,
            key_value_heads,
            head_dim,
            requested_context,
        );
        let budget = required.map_or(u64::MAX, |required| match budget_delta {
            -1 => required.saturating_sub(1),
            0 => required,
            1 => required.saturating_add(1),
            _ => unreachable!("generated budget delta is bounded"),
        });
        let result = config.resident_chat_plan(requested_context, budget);

        if requested_context == 0 || requested_context > maximum {
            prop_assert!(matches!(
                result,
                Err(Qwen3ForwardError::ResidentChatContextLimit {
                    requested,
                    maximum: actual_maximum,
                }) if requested == requested_context && actual_maximum == maximum
            ), "context limit must be enforced before KV planning");
        } else if let Some(required) = required {
            if required > budget {
                prop_assert!(matches!(
                    result,
                    Err(Qwen3ForwardError::ResidentChatKvBudget {
                        required: actual_required,
                        maximum,
                    }) if actual_required == required && maximum == budget
                ), "a budget below the independent byte count must be rejected");
            } else {
                let plan = result.expect("formula-admitted plan");
                prop_assert_eq!(plan.maximum_context_tokens(), requested_context);
                prop_assert_eq!(plan.planned_kv_bytes(), required);
            }
        } else {
            prop_assert!(matches!(result, Err(Qwen3ForwardError::ShapeOverflow)));
        }
    }

    /// The admitted context is the smaller of the model and resident-chat
    /// caps, with the first out-of-range token rejected independently of KV.
    #[test]
    fn resident_context_boundary_uses_the_smaller_model_or_chat_cap(
        hidden_layers in 1_usize..=8,
        key_value_heads in 1_usize..=8,
        head_dim_halves in 1_usize..=64,
        model_context in 1_usize..=4_096,
    ) {
        let config = Qwen3ForwardConfig::parse(&config_json(
            hidden_layers,
            key_value_heads,
            head_dim_halves * 2,
            model_context,
        ))
        .expect("generated Qwen3 layout is valid");
        let maximum = MAX_RESIDENT_CHAT_TOKENS.min(model_context);
        let admitted = config
            .resident_chat_plan(maximum, u64::MAX)
            .expect("maximum context is admitted with an unbounded logical budget");
        prop_assert_eq!(admitted.maximum_context_tokens(), maximum);
        prop_assert!(matches!(
            config.resident_chat_plan(maximum + 1, u64::MAX),
            Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested,
                maximum: actual_maximum,
            }) if requested == maximum + 1 && actual_maximum == maximum
        ), "first token beyond the model or chat cap must be rejected");
    }

    /// Each K/V-cache dimension contributes positively to a resident plan.
    /// This checks the public plan's monotonicity without duplicating its byte
    /// calculation, and holds the model position limit fixed above every
    /// generated request.
    #[test]
    fn resident_plan_grows_when_any_kv_dimension_grows(
        hidden_layers in 1_usize..=32,
        key_value_heads in 1_usize..=32,
        head_dim_halves in 1_usize..=256,
        context_tokens in 1_usize..=MAX_RESIDENT_CHAT_TOKENS,
    ) {
        let head_dim = head_dim_halves * 2;
        let plan_for = |hidden_layers, key_value_heads, head_dim| {
            Qwen3ForwardConfig::parse(&config_json(
                hidden_layers,
                key_value_heads,
                head_dim,
                MAX_RESIDENT_CHAT_TOKENS,
            ))
            .expect("generated Qwen3 layout is valid")
            .resident_chat_plan(context_tokens, u64::MAX)
            .expect("small generated layout fits an unbounded budget")
            .planned_kv_bytes()
        };
        let baseline = plan_for(hidden_layers, key_value_heads, head_dim);

        prop_assert!(
            plan_for(hidden_layers + 1, key_value_heads, head_dim) > baseline,
            "one additional decoder layer must increase retained K/V bytes"
        );
        prop_assert!(
            plan_for(hidden_layers, key_value_heads + 1, head_dim) > baseline,
            "one additional K/V head must increase retained K/V bytes"
        );
        prop_assert!(
            plan_for(hidden_layers, key_value_heads, head_dim + 2) > baseline,
            "one larger even K/V head dimension must increase retained K/V bytes"
        );
    }

    /// Any budget at or above an admitted plan retains exactly that plan;
    /// budget only controls admission and cannot alter its context or estimate.
    #[test]
    fn resident_budget_is_monotone_without_changing_the_admitted_plan(
        hidden_layers in 1_usize..=32,
        key_value_heads in 1_usize..=32,
        head_dim_halves in 1_usize..=256,
        context_tokens in 1_usize..=MAX_RESIDENT_CHAT_TOKENS,
        surplus in 0_u64..=1_000_000,
    ) {
        let config = Qwen3ForwardConfig::parse(&config_json(
            hidden_layers,
            key_value_heads,
            head_dim_halves * 2,
            MAX_RESIDENT_CHAT_TOKENS,
        ))
        .expect("generated Qwen3 layout is valid");
        let exact = config
            .resident_chat_plan(context_tokens, u64::MAX)
            .expect("small generated layout fits an unbounded budget");
        let larger = config
            .resident_chat_plan(context_tokens, exact.planned_kv_bytes().saturating_add(surplus))
            .expect("a budget at least as large as the exact plan is admitted");

        prop_assert_eq!(larger, exact);
        prop_assert!(matches!(
            config.resident_chat_plan(context_tokens, exact.planned_kv_bytes() - 1),
            Err(Qwen3ForwardError::ResidentChatKvBudget { required, maximum })
                if required == exact.planned_kv_bytes()
                    && maximum == exact.planned_kv_bytes() - 1
        ), "one byte below a nonzero plan is rejected");
    }

    /// K/V arithmetic overflow is refused for either untrusted multiplicand,
    /// while an invalid context is rejected first without entering that math.
    #[test]
    fn resident_plan_handles_kv_dimension_overflow_and_context_precedence(
        overflow_in_layers in any::<bool>(),
        model_context in 1_usize..=MAX_RESIDENT_CHAT_TOKENS,
    ) {
        let (hidden_layers, key_value_heads) = if overflow_in_layers {
            (usize::MAX, 1)
        } else {
            (1, usize::MAX)
        };
        let config = Qwen3ForwardConfig::parse(&config_json(
            hidden_layers,
            key_value_heads,
            2,
            model_context,
        ))
        .expect("overflowing K/V dimensions remain syntactically valid Qwen3 layout");
        let maximum = MAX_RESIDENT_CHAT_TOKENS.min(model_context);

        prop_assert!(matches!(
            config.resident_chat_plan(1, u64::MAX),
            Err(Qwen3ForwardError::ShapeOverflow)
        ), "checked K/V sizing must reject overflow before producing a plan");
        prop_assert!(matches!(
            config.resident_chat_plan(maximum + 1, 0),
            Err(Qwen3ForwardError::ResidentChatContextLimit { requested, maximum: actual_maximum })
                if requested == maximum + 1 && actual_maximum == maximum
        ), "the public context cap must reject before impossible K/V sizing");
    }

    /// Resident planning must not widen the existing default executor cap.
    #[test]
    fn default_executor_keeps_the_512_token_diagnostic_limit(
        hidden_layers in 1_usize..=8,
        key_value_heads in 1_usize..=8,
        head_dim_halves in 1_usize..=64,
        model_context in (MAX_DENSE_DEBUG_TOKENS + 1)..=4_096,
    ) {
        let config = Qwen3ForwardConfig::parse(&config_json(
            hidden_layers,
            key_value_heads,
            head_dim_halves * 2,
            model_context,
        ))
        .expect("generated Qwen3 layout is valid");
        let weights = HashMap::<String, Array>::new();
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        prop_assert!(matches!(
            executor.prefill_last_logits(&vec![0; MAX_DENSE_DEBUG_TOKENS + 1]),
            Err(Qwen3ForwardError::PromptTooLong {
                actual,
                maximum,
            }) if actual == MAX_DENSE_DEBUG_TOKENS + 1 && maximum == MAX_DENSE_DEBUG_TOKENS
        ), "default diagnostics must still reject token 513");
    }
}
