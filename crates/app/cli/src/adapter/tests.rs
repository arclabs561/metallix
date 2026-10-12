use std::path::PathBuf;

use super::Adapter;

/// The `docs/arch` diagram that draws an adapter's checkpoints, or `None`
/// for an adapter listed in [`UNDRAWN`]. A new adapter does not compile
/// until it is placed here.
#[allow(
    clippy::unnecessary_wraps,
    reason = "None marks an UNDRAWN adapter, though none is undrawn today"
)]
fn diagram(adapter: Adapter) -> Option<&'static str> {
    match adapter {
        Adapter::Qwen3 | Adapter::Llama | Adapter::Qwen2 | Adapter::PplxQwen3 => Some("qwen3"),
        Adapter::Julia => Some("julia"),
        Adapter::Qwen35 => Some("qwen35"),
        Adapter::Qwen3Asr => Some("qwen3_asr"),
        Adapter::Gemma4 => Some("gemma4"),
        Adapter::DeepseekV41 => Some("deepseek_v41"),
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

/// Adapters without a diagram yet, each with why and who will draw it.
/// Adding one takes a reviewed reason; `scripts/check_arch_diagrams.py`
/// prints every entry in each check, and the change that draws an adapter
/// removes its entry.
const UNDRAWN: [(Adapter, &str); 0] = [];

/// Every way `adapters` (each with its diagram, if any) and `undrawn`
/// disagree: an adapter with neither a diagram nor an entry, an entry for a
/// drawn adapter, an empty reason, or a repeated entry.
fn diagram_problems(
    adapters: &[(Adapter, Option<&str>)],
    undrawn: &[(Adapter, &str)],
) -> Vec<String> {
    let mut problems = Vec::new();
    for (index, &(adapter, reason)) in undrawn.iter().enumerate() {
        if reason.trim().is_empty() {
            problems.push(format!("{} is undrawn without a reason", adapter.name()));
        }
        if undrawn[..index]
            .iter()
            .any(|&(earlier, _)| earlier == adapter)
        {
            problems.push(format!("{} is listed twice in UNDRAWN", adapter.name()));
        }
    }
    for &(adapter, stem) in adapters {
        let listed = undrawn.iter().any(|&(entry, _)| entry == adapter);
        match (stem, listed) {
            (Some(stem), true) => problems.push(format!(
                "{} is drawn in {stem}.typ: remove it from UNDRAWN",
                adapter.name()
            )),
            (None, false) => problems.push(format!(
                "{} has no diagram and no UNDRAWN entry",
                adapter.name()
            )),
            _ => {}
        }
    }
    problems
}

#[test]
fn every_adapter_has_an_architecture_diagram() {
    let adapters = ADAPTERS.map(|adapter| (adapter, diagram(adapter)));
    assert_eq!(diagram_problems(&adapters, &UNDRAWN), Vec::<String>::new());
    let arch = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch");
    for (adapter, stem) in adapters {
        let Some(stem) = stem else { continue };
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

#[test]
fn undrawn_entries_need_a_reason_and_appear_once() {
    let adapters = [(Adapter::Qwen3, Some("qwen3")), (Adapter::Julia, None)];
    assert_eq!(
        diagram_problems(&adapters, &[]),
        ["julia has no diagram and no UNDRAWN entry"]
    );
    assert!(
        diagram_problems(
            &adapters,
            &[(Adapter::Julia, "drawn after its parser lands")]
        )
        .is_empty()
    );
    assert_eq!(
        diagram_problems(&adapters, &[(Adapter::Julia, " ")]),
        ["julia is undrawn without a reason"]
    );
    assert_eq!(
        diagram_problems(&adapters, &[(Adapter::Julia, "a"), (Adapter::Julia, "b")]),
        ["julia is listed twice in UNDRAWN"]
    );
    assert_eq!(
        diagram_problems(&adapters, &[(Adapter::Julia, "a"), (Adapter::Qwen3, "b")]),
        ["qwen3 is drawn in qwen3.typ: remove it from UNDRAWN"]
    );
}
