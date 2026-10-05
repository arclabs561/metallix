# Embeddings

`mx serve` answers `POST /v1/embeddings` for registry entries of kind
`qwen_embedding`, for `pplx_context` (see [Contextual chunks](#contextual-chunks)),
and for `pplx_late` (see [Multi-vector](#multi-vector)).
The qualified `qwen_embedding` checkpoint is
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) at
revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

```json
{"models": [{"id": "embed", "kind": "qwen_embedding", "path": "/path/to/Qwen3-Embedding-0.6B"}]}
```

```sh
mx serve --registry models.json
curl -s localhost:8321/v1/embeddings -d '{"model": "embed", "input": "Explain gravity",
  "input_type": "query", "dimensions": 256}'
```

## Request

| Field | Meaning |
| --- | --- |
| `model` | Registered ID. |
| `input` | One text or a list of 1 to 64 nonempty texts. Token arrays are rejected. |
| `input_type` | `document` (default) embeds each text as written; `query` prefixes it with a task instruction. |
| `instruction` | Query task sentence. Defaults to the model card's "Given a web search query, retrieve relevant passages that answer the query". Rejected for documents. |
| `dimensions` | Optional truncation from 1024 down to at least 32, applied before normalization. |
| `encoding_format` | Only `float`. |
| `user` | Accepted and ignored. |

A query is formatted exactly as the model card does,
`Instruct: {instruction}\nQuery:{text}`, with no space after `Query:`. Embed
queries and the documents they search with the same instruction convention;
mixing conventions moves scores. Any other field returns 400, so a client
asking for an output this server does not produce gets an error rather than a
pooled vector. Texts over 512 tokens also return 400; nothing is truncated.

## Response

```json
{"object": "list", "model": "embed",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, ...]}],
 "usage": {"prompt_tokens": 27, "total_tokens": 27},
 "metallix": {"pooling": "last_token", "normalized": true, "dimensions": 256,
              "input_type": "query", "instruction": "...", "precision": "float32",
              "embed_ms": 31, "tokenizer_json_sha256": "..."}}
```

Each text is tokenized with the tokenizer's special-token template, which
appends `<|endoftext|>`. The final-norm hidden state at that last position is
the embedding, truncated to `dimensions` if requested and L2-normalized, so the
dot product of two embeddings is their cosine similarity. `data` keeps input
order. Inputs in one request are grouped by length, each group padded to at
most 1.25 times its real tokens, and each group runs as one forward pass.

## Precision and qualification

Served weights are float32. An opt-in test sends the 68 inputs of
`fixtures/qwen3-embedding-0.6b/embedding-reference.json` (the card's four
retrieval inputs plus stress inputs up to 500 tokens) through `mx serve`.
Every vector matches the CPU float32 source oracle at cosine at least
1 - 1e-5, the fixture's declared policy, at full width and at 256 dimensions
(worst 0.99999999999).

BF16 weights would halve weight memory (about 1.2 GB instead of 2.4 GB). They
are measured against a looser tolerance (cosine at least 1 - 1e-3; worst
1 - 5.1e-4 on the first seven inputs) because the pinned MLX computes BF16 sigmoid imprecisely.
Serving stays float32 until the MLX upgrade lands and BF16 is requalified.

```sh
METALLIX_QWEN_EMBEDDING_MODEL=/path/to/Qwen3-Embedding-0.6B \
  cargo test --release -p server --features metal --test serve_embeddings -- --ignored
```

Measured on an Apple M3 Max under heavy unrelated load, so upper bounds: a
resident model is ready about 0.6 to 1.2 s after launch; an on-demand model's
first request takes about 0.6 s; inputs of 6, 66 and 258 tokens take about 14,
28 and 51 ms; the front process adds under 1 ms per request.

## Contextual chunks

Registry entries of kind `pplx_context` embed each chunk of a document with the
whole document as context. The qualified checkpoint is
[pplx-embed-context-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b)
at revision `b42df969d4d78d1840769e45c68c4e4ce763768b` (MIT), a Qwen3-0.6B
backbone with bidirectional attention. A registry entry pointing at a causal
checkpoint fails at load.

```json
{"models": [{"id": "ctx", "kind": "pplx_context", "path": "/path/to/pplx-embed-context-v1-0.6b"}]}
```

```sh
curl -s localhost:8321/v1/embeddings -d '{"model": "ctx",
  "input": [["Marie Curie was born in Warsaw.", "She moved to Paris."], ["One chunk."]],
  "encoding_format": "int8"}'
```

| Field | Meaning |
| --- | --- |
| `model` | Registered ID. |
| `input` | A list of 1 to 64 documents, each a list of 1 to 256 chunk strings. A chunk may be empty; one containing `<\|endoftext\|>` is rejected. A flat text or list of texts is rejected. |
| `encoding_format` | `float` (default): the pooled vector before quantization. `int8`: the model's `clamp(round(tanh(x) * 127), -128, 127)`. `binary`: `1` where the pooled value is at least 0, else `-1`. |
| `user` | Accepted and ignored. |

Following the model card, a document's chunks are joined with `<|endoftext|>`
and encoded as one sequence with no prefix or instruction. Each chunk's vector
is the mean of the final-norm hidden states over its tokens, excluding the
separators; an empty chunk gives zeros. There are no `instruction`,
`input_type` or `dimensions` fields; sending one returns 400. A document over
512 tokens, separators included, returns 400.

```json
{"object": "list", "model": "ctx",
 "data": [{"object": "embedding", "index": 0, "document": 0, "chunk": 0, "embedding": [12, -40, ...]},
          {"object": "embedding", "index": 1, "document": 0, "chunk": 1, "embedding": [...]},
          {"object": "embedding", "index": 2, "document": 1, "chunk": 0, "embedding": [...]}],
 "usage": {"prompt_tokens": 19, "total_tokens": 19},
 "metallix": {"pooling": "mean_per_chunk", "context": "document", "normalized": false,
              "encoding": "int8", "dimensions": 1024, "precision": "float32",
              "embed_ms": 23, "tokenizer_json_sha256": "..."}}
```

`data` has one entry per chunk, in document then chunk order. Vectors are not
normalized, as the model card publishes them; compare them with cosine
similarity.

Served weights are float32, as the checkpoint stores them. An opt-in test sends
the six documents of `fixtures/pplx-embed-context-v1-0.6b/context-reference.json`
(16 chunks, including an empty last chunk and a multilingual document) through
`mx serve` once per encoding. Against the source's own `encode` path the worst
float chunk is at 1 - 8.0e-12 cosine, 16,383 of 16,384 int8 codes match exactly
and the other differs by one code, and binary signs match. The fixture's
declared policy is cosine at least 1 - 1e-5, int8 codes at most one apart, and
binary signs equal except where the pooled value is within 1e-4 of zero.

```sh
METALLIX_PPLX_CONTEXT_MODEL=/path/to/pplx-embed-context-v1-0.6b \
  cargo test --release -p server --features metal --test serve_pplx_context -- --ignored --nocapture
```

A 4-chunk, 53-token document takes 23 to 25 ms over HTTP (warm median over two
runs, of which 21 to 23 ms is the forward pass and pooling); the first request
after load takes 23 to 26 ms.

## Multi-vector

Registry entries of kind `pplx_late` return one vector per token for
late-interaction (ColBERT-style) retrieval. The qualified checkpoint is
[pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b)
at revision `4d28cf627d225552cfc29fb7df6cb0705ea0f1b3` (MIT), the same
bidirectional Qwen3-0.6B backbone with a 1024-to-128 projection head. A causal
checkpoint fails at load.

```json
{"models": [{"id": "late", "kind": "pplx_late", "path": "/path/to/pplx-embed-v1-late-0.6b"}]}
```

```sh
curl -s localhost:8321/v1/embeddings -d '{"model": "late",
  "input": "What motivates scientific discovery?", "input_type": "query"}'
```

| Field | Meaning |
| --- | --- |
| `model` | Registered ID. |
| `input` | One text or a list of 1 to 64 nonempty texts. |
| `input_type` | `document` (default) or `query`. |
| `encoding_format` | Only `float`. |
| `user` | Accepted and ignored. |

Following the checkpoint's sentence-transformers pipeline, a query is `[Q] `
followed by its text, truncated or padded with the tokenizer's mask token to
exactly 32 tokens; every position, padding included, is attended and returned.
A document is `[D] ` followed by its text, truncated to 512 tokens, and its
ASCII punctuation tokens are left out. Each returned token's final-norm hidden
state is projected to 128 values and L2-normalized. There are no
`instruction` or `dimensions` fields; sending one returns 400.

```json
{"object": "list", "model": "late",
 "data": [{"object": "embedding", "index": 0,
           "embedding": [[0.031, -0.112, ...], ...],
           "positions": [0, 1, 2, ...],
           "offsets": [null, [0, 4], [4, 14], ..., null],
           "truncated": false}],
 "usage": {"prompt_tokens": 32, "total_tokens": 32},
 "metallix": {"pooling": "none", "vectors": "per_token", "similarity": "maxsim",
              "normalized": true, "dimensions": 128, "input_type": "query",
              "query_length": 32, "document_length": 512, "precision": "float32",
              "embed_ms": 23, "tokenizer_json_sha256": "..."}}
```

`embedding` holds one 128-value vector per returned token. `positions` gives
each vector's index in the model's input sequence, and `offsets` its
`[start, end)` character span in the input text, or `null` for the prefix
token and for query padding, which have no text. `truncated` says whether the
text was longer than the query or document length. Score a query against a
document with MaxSim: for each query vector, take the largest dot product with
any document vector, and sum those. The server does not score; clients do.

Served weights are float32, as the checkpoint stores them. An opt-in test sends
the 11 inputs of `fixtures/pplx-embed-v1-late-0.6b/late-reference.json` (four
queries, including one over 32 tokens, and seven documents, including one over
512 tokens, punctuation, code and Japanese) through `mx serve`. Against the
source's `MultiVectorEncoder`, positions match exactly, the worst token vector
is at 1 - 1.6e-11 cosine, MaxSim scores over eight query and document pairs are
within 3.2e-6, and the card's three published scores (31.4841, 31.2462,
31.4041) are reproduced to their printed precision. The fixture's declared
policy is cosine at least 1 - 1e-5 per token and MaxSim within 1e-3.

```sh
METALLIX_PPLX_LATE_MODEL=/path/to/pplx-embed-v1-late-0.6b \
  cargo test --release -p server --features metal --test serve_pplx_late -- --ignored --nocapture
```

Over HTTP (warm median over two runs), a query takes about 25 ms, a 9-token
document about 10 ms and a 512-token document about 116 ms. For that document
about 6 ms falls outside the forward pass, mostly writing and sending its 456
vectors as JSON.

## Not supported yet

Pooling over caller-chosen spans is kept out: no served model is trained for
it, so arbitrary spans of token states are not qualified retrieval vectors.
Qwen3-Embedding's and pplx-embed-context's own token states are not returned
for the same reason; per-token vectors come only from `pplx_late`. Every
request shape rejects unknown fields so new modes can be added as fields
without changing existing responses.
