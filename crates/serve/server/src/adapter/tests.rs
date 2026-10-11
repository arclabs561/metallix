use std::path::PathBuf;

use super::Adapter;

/// The `docs/arch` diagram that draws an adapter's checkpoints, or `None`
/// while it is undrawn. A new adapter does not compile until it is placed
/// here, and only adapters in [`UNDRAWN`] may return `None`.
fn diagram(adapter: Adapter) -> Option<&'static str> {
    match adapter {
        Adapter::Qwen3 | Adapter::Llama | Adapter::Qwen2 | Adapter::PplxQwen3 => Some("qwen3"),
        Adapter::Julia
        | Adapter::Qwen35
        | Adapter::Qwen3Asr
        | Adapter::Gemma4
        | Adapter::DeepseekV41 => None,
    }
}

/// Every adapter, in declaration order.
const ADAPTERS: [Adapter; 9] = [
    Adapter::Qwen3,
    Adapter::Llama,
    Adapter::Qwen2,
    Adapter::PplxQwen3,
    Adapter::Julia,
    Adapter::Qwen35,
    Adapter::Qwen3Asr,
    Adapter::Gemma4,
    Adapter::DeepseekV41,
];

/// Adapters not yet drawn. Remove one in the change that draws it; never add
/// one.
const UNDRAWN: [Adapter; 5] = [
    Adapter::Julia,
    Adapter::Qwen35,
    Adapter::Qwen3Asr,
    Adapter::Gemma4,
    Adapter::DeepseekV41,
];

#[test]
fn every_adapter_has_an_architecture_diagram() {
    let arch = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch");
    for adapter in ADAPTERS {
        let Some(stem) = diagram(adapter) else {
            assert!(
                UNDRAWN.contains(&adapter),
                "{} has no diagram under docs/arch",
                adapter.name()
            );
            continue;
        };
        assert!(
            !UNDRAWN.contains(&adapter),
            "{} is drawn in {stem}.typ: remove it from UNDRAWN",
            adapter.name()
        );
        for suffix in ["typ", "svg", "facts.json"] {
            let path = arch.join(format!("{stem}.{suffix}"));
            assert!(
                path.is_file(),
                "{} is missing; it draws {}",
                path.display(),
                adapter.name()
            );
        }
    }
}
