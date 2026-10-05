# Embeddings

`mx serve` answers `POST /v1/embeddings` for registry entries of kind
`qwen_embedding`. The qualified checkpoint is
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
order. Inputs in one request run one after another; a list is a convenience,
not a batched forward pass.

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

## Not supported yet

One pooled vector per text is the only output. Two further scopes are kept
out deliberately:

- Multi-vector output: per-token vectors for late-interaction scoring, or
  pooling over caller-chosen spans. Qwen3-Embedding is trained only for
  last-token pooling, so its other token states are not qualified retrieval
  vectors.
- Contextual chunk embeddings: embedding chunks of one document with the whole
  document as context. That needs a model trained to do it.

Each will be offered per model, only for a model trained for that output and
qualified against its source. The request rejects unknown fields so these
modes can be added as new fields without changing the default response.
