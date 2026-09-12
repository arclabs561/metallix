# V4.1 Engram hash contract

This is a narrow contract for the token-to-row-address portion of Engram.  It
does not establish table payload layout, loader ranges, residual numerical
parity, CED behavior, or a Metal implementation.

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

## Next gate

Add a tiny pinned fixture containing the exact tokenizer backend/version, a
small original-ID-to-decoded-text projection (including case, whitespace,
accent, U+FFFD, pad, and image/dead cases), compressed IDs, explicit primes,
multipliers, offsets, and final addresses. It must compare one-shot prefill
with an equivalent split prefill/decode call and verify that a dead span breaks
all longer histories. Only then is a pure Rust address oracle justified. A
separate fixture with fetched rows and `wkv`/gate weights is required before
claiming Engram residual or full-forward parity.
