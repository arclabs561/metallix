# V4.1 Engram hash and residual-gate contracts

These are narrow contracts for token-to-row addresses and the residual gate
with supplied, preprojected tensors. They do not establish table payload
layout, loader ranges, complete Engram parity, CED behavior, or a Metal
implementation.

## Provenance and reading coverage

The complete 184-line upstream `inference/engram.py` was read from the
DeepSeek V4.1 Flash revision
`dba1be0a40aa45a94ad051997016db3960a90277`, retained locally as
`artifacts/v41-engram-pinned.py`. Its SHA-256 is
`11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`.
The source URL is
<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py>.
The adjacent use site was cross-checked in the pinned
`artifacts/v41-reference-model.py:328-365,1241-1267`.

## Address derivation

1. Decode every original token ID with the tokenizer's raw backend and
   `skip_special_tokens=False`; do not enable cleanup. Normalization is NFKC,
   NFD, accent stripping, lower-casing, whitespace-run collapse to one space,
   and trimming. A single space is temporarily replaced with U+E000 before
   trimming, then restored, so it does not collapse to empty. Tokens whose
   decoded text contains U+FFFD instead use the backend's raw token spelling.
   If normalization still gives empty text, the unnormalized decoded text is
   the key. Equal keys share a compressed ID in increasing original-ID order.

2. The resulting compressed-vocabulary count is an algorithm input, not just
   a bounds check: it must equal `engram_compressed_vocab_size`. The pad value
   is the compressed ID of `engram_pad_id`, rather than that raw token ID.

3. For every configured Engram layer, each n-gram size (2 through maximum),
   and each head, choose the next previously unused prime strictly above the
   preceding value, initially `engram_vocab_size - 1`. Primes are globally
   unique across all layers. Each `(n-gram, head)` owns that prime-sized bucket;
   offsets restart at zero for each layer and accumulate in n-gram, then head
   order within that layer. Prime uniqueness does not make offsets global.

4. Per layer, generate one multiplier per lookback with NumPy
   `default_rng(10007 * layer_id)`. Draw signed 64-bit integers in
   `[0, max(1, (i64::MAX / compressed_vocab_size) / 2))`, then map each to the
   odd multiplier `2x + 1`. Rust must not substitute a different RNG or infer
   a random sequence from the seed alone without a version-qualified parity
   fixture; storing the proven multiplier values is the safe first boundary.

5. Cache compressed IDs at absolute positions. A masked input token becomes
   `DEAD = -1`. At every output position, gather the current token and each
   older lookback, clamping negative source positions to zero. `blocked` is
   cumulative: crossing sequence start or a `DEAD` token pads that lookback
   and every older lookback with compressed `pad_id`. Thus an n-gram never
   crosses an image/dead span, while the current non-dead token can still
   participate in its own shorter history.

6. For every layer, multiply compressed IDs by its multipliers; XOR the
   products incrementally. The running XOR after lookback 1 addresses the
   2-gram, then successively larger n-grams. Modulo by the matching prime and
   add the bucket offset. The output order is
   `[batch, tokens, engram_layers, (max_ngram_size-1)*heads]`, grouped by
   increasing n-gram then head.

## Integration boundary

Engram hashes are computed once per input chunk before embedding expansion.
At configured layer IDs, `Engram.forward` fetches these rows, transforms them
to per-HC-copy keys plus one value, and gates the value into the residual
before that block executes. Image token types mask Engram; text-only calls pass
no mask. This establishes placement and hash ownership, not row decoding or
the gated residual's weights.

## Compressed-token reference

`crates/models/deepseek/src/engram.rs` now consumes explicit compressed
`Live`/`Dead` tokens and captured hash tensors. Its state update computes
addresses against a private candidate history and commits only on success.
The cumulative blocked flag makes the lookback work linear in n-gram width;
history, tensor, output and work limits are checked. Products and addresses
must remain in non-negative signed-64-bit range.

`scripts/v41-engram-reference.py` hash-checks and extracts only the pinned
`NgramHashState.forward`, then captures CPU Torch 2.13.0 results using explicit
synthetic token maps, multipliers, divisors and offsets. The checked-in
`fixtures/deepseek-v41/engram-hash-reference.json` covers two independent batch
histories, two layers, multiple heads, full/split/tokenwise calls and DEAD
boundaries. Rust tests compare the addresses directly with this capture.
No upstream constructors, tokenizer normalization or RNG setup execute.
The local capture emitted Torch's optional missing-NumPy warning; serialization
uses explicit little-endian integer packing and does not require NumPy.

Start-position zero invalidates unwritten prior tail slots in the Rust
reference. This is a stricter lifecycle guard than the upstream caller-owned
cache: a short reset cannot make a later skipped append read an earlier
sequence's tail. Failure preserves the previous state.

Regenerate after acquiring the source identified above:

```sh
uv run scripts/v41-engram-reference.py > artifacts/v41-engram-reference.json
cargo test -p deepseek engram
```

## Preprojected residual gate

The source capture `scripts/v41-engram-gate-reference.py` now exercises the
pinned `Engram.forward` with explicit BF16 key/value tensors supplied by
embedding/projection stubs. Its CPU receipt covers signed dots, per-HC-copy
normalization, q/k factorization, shared-value broadcasting, masked
passthrough and the unmasked zero-dot clamp floor. Recomputing the same
Torch expression is a consistency check, not an independent numerical
oracle. Deliberately wrong joint-HC normalization and copy-specific value
broadcasting distinguish the fixture's intended boundaries.

Run `uv run scripts/v41-engram-gate-reference.py` to emit the bit-preserving
capture. The local run passed its assertions; Torch warned about optional
NumPy initialization, which this script does not use.

`crates/models/deepseek/src/engram/gate.rs` consumes these preprojected tensors
as a bounded scalar BF16 reference. Tests require exact captured BF16 output
and absolute FP32 gate error at most `2e-6`; this is not a claim of bitwise
FP32 reduction parity. Inputs and intermediate results must remain finite,
and failures leave the caller's output unchanged.

Two source-order details matter: form `q_weight * k_weight` before multiplying
the stream, and mask the gate rather than copying the residual. A masked row
still computes `h + 0 * value`, which can change the sign of zero. The capture
includes this signed-zero case. Per-copy reductions start from positive zero.
This reference does not qualify FP8 rows/scales, table sharding, `wkv` weights
or Metal.

## Remaining gates

The next fixture must cover the exact tokenizer backend/version and a
small original-ID-to-decoded-text projection (including case, whitespace,
accent, U+FFFD, pad, and image/dead cases), compressed IDs, explicit primes,
multipliers, offsets, and final addresses. The existing explicit-map address
fixture does not qualify these derivations. A separate fixture with fetched
rows and `wkv`/gate weights is required before
claiming Engram residual or full-forward parity.
