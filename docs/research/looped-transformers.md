# Looped Transformers: state and execution boundaries

**Decision:** treat recurrent depth as an adapter-owned execution contract.
Weight sharing does not establish cache sharing, cheaper decoding, or compatibility
with an existing checkpoint. DeepSeek's selected-weight consumer remains the
implementation priority.

## What the research establishes

[Shared-memory LPT](https://arxiv.org/html/2610.02383v1) trains repeated passes
through one stack: the first retains full-context KV; later passes attend that
memory and their own short windows with one softmax. This trained architecture
is not a demonstrated retrofit for ordinary checkpoints. Its hybrid variant
also retains per-recursion recurrent state. The study covers models up to 1B
parameters at 4,096-token contexts; larger batch capacity and throughput gains
do not imply lower single-sequence latency.

Related designs exercise different contracts:

- [Huginn](https://arxiv.org/html/2502.05171v2) studies adaptive stopping,
  bounded recurrent caches and latent warm starts.
- [Ouro](https://arxiv.org/html/2510.25741v1) evaluates depth-specific caches
  and quality-changing reuse policies.
- [Mixture-of-Recursions](https://arxiv.org/html/2507.10524v3) routes tokens
  across depths, so cache membership need not be dense.

An exit gate is not proof of skipped computation. The
[pinned Ouro implementation](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py#L592)
runs every configured pass before its causal-LM wrapper selects outputs using
the gates. Measure executed blocks, not just the reported exit depth.

## Local-backend consequence

Token position and within-token pass depth are different axes. EOS and output
limits govern emitted tokens; internal stopping needs its own qualified policy.
Admission must count per-depth caches, recurrent state, scratch and work, not
only unique parameter bytes. Repeated SSD reads can erase a weight-sharing
advantage.

The existing consumer is
[Qwen35's snapshot boundary](../../crates/models/qwen35/src/forward.rs): snapshots
bind convolution, recurrent and KV state to a weight load and exact token
position. A future loop adapter would additionally preserve depth-local state,
window cursors and routing/stopping policy. This does not make Qwen35 a looped
model. Compilation must preserve those state transitions and report actual
retrace/evaluation behavior.

## Qualification gate

For a named reference implementation and checkpoint, first compare tiny
full-forward and incremental decoding at fixed depth, including window wrap,
complete snapshot/restore and suffix replay. Preserve the source numerical
contract before testing adaptive exits or compilation. Then measure matched-batch
latency separately from maximum-batch throughput and memory. This note declares
neither a new supported model nor a serving API.
