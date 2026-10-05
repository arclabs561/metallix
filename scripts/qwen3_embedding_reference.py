# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
# ]
# ///
"""Capture a CPU float32 Qwen3-Embedding-0.6B reference from the model card's recipe.

Each input is tokenized with the pinned tokenizer's special-token template (which
appends <|endoftext|>), run alone through `AutoModel` in float32 eager mode on one
CPU thread, pooled at its last position and L2-normalized, as in the card's
Transformers snippet. The card's four retrieval inputs are also run as the card
does (one left-padded batch) to check that batching does not change the result.

The tolerance policy below is declared before any comparison is run; the Rust
opt-in test (crates/models/qwen/tests/embedding_checkpoint.rs) asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PINNED_SHA256 = {
    "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "model.safetensors": (
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
    ),
}
TASK = "Given a web search query, retrieve relevant passages that answer the query"
# The card's published Transformers scores: queries (rows) by documents (columns).
CARD_SCORES = [
    [0.7645568251609802, 0.14142508804798126],
    [0.13549736142158508, 0.5999549627304077],
]
TOLERANCE_POLICY = {
    "token_ids": "exact equality with this reference",
    "embedding_cosine_min": 1 - 1e-5,
    "score_abs_vs_reference": 1e-4,
    "score_abs_vs_card": 1e-3,
    "native_precision": "weights promoted to float32 (prepare_float32)",
    "rationale": (
        "Declared before running. Native f32 and this f32 oracle differ only in "
        "reduction order, so per-vector cosine and pairwise scores must agree "
        "closely. The card's precision and hardware are unstated (its own vLLM "
        "numbers differ from its Transformers numbers by about 2.5e-3), so the "
        "card anchor uses a looser bound and checks recipe, not arithmetic."
    ),
    "bf16": {
        "embedding_cosine_min": 1 - 4e-4,
        "score_abs_vs_reference": 5e-3,
        "native_precision": "checkpoint BF16 weights and activations",
        "rationale": (
            "Measured on the full stress set with mlx-rs 0.32.0 (MLX 0.32.2): worst "
            "1 - cosine 2.9e-4 (ko), worst score delta 3.3e-3, BF16 runs "
            "bit-identical run to run. The bounds leave about 1.4x and 1.5x headroom "
            "for kernel or hardware changes. The earlier bounds (1e-3 and 1e-2) "
            "covered MLX 0.25.1, whose imprecise BF16 sigmoid in SiLU gave worst "
            "1 - cosine 8.0e-4 (hi) and score delta 5.2e-3; MLX 0.29.3 fixed it. "
            "The HF BF16 source's worst is 4.7e-4."
        ),
    },
}
LONG_SENTENCE = (
    "The river carried silt from the mountains to the delta, where farmers "
    "planted rice in the rich soil every spring. "
)
MAX_TOKENS = 512  # the native uncached path's limit


def query(text: str, task: str = TASK) -> str:
    return f"Instruct: {task}\nQuery:{text}"


CODE_TASK = "Given a programming question, retrieve code that answers it"
MEDICAL_TASK = "Given a medical question, retrieve relevant passages"
MATH_TASK = "Retrieve the mathematical statement that answers the question"
EN_PASSAGE = (
    "The printing press changed how ideas moved across Europe. Before it, books "
    "were copied by hand, which made them rare and expensive. Within fifty years "
    "of Gutenberg's first Bible, presses operated in more than two hundred towns. "
    "Photosynthesis converts light, water and carbon dioxide into sugar and "
    "oxygen; chlorophyll absorbs mostly red and blue light, which is why leaves "
    "look green. A good bread dough needs flour, water, salt and yeast, and time: "
    "a long, cool fermentation develops flavor that a quick rise cannot. The "
    "Pacific Ocean covers about a third of the planet's surface and holds its "
    "deepest point, the Challenger Deep, nearly eleven kilometers down. In 1969 "
    "two astronauts walked on the Moon while a third orbited above, waiting to "
    "bring them home. Modern compilers translate high-level code into machine "
    "instructions, applying optimizations such as inlining, loop unrolling and "
    "dead-code elimination. Migratory birds navigate using the sun, the stars and "
    "the Earth's magnetic field, sometimes flying thousands of kilometers without "
    "rest. The violin family evolved in sixteenth-century Italy, and instruments "
    "by Stradivari are still prized for their tone. Glaciers carve valleys into a "
    "U shape, while rivers cut narrower V-shaped valleys. Vaccines train the "
    "immune system to recognize a pathogen without causing the disease itself. "
    "Chess engines evaluate millions of positions per second, yet human players "
    "still find creative plans that machines misjudge. Coffee was first "
    "cultivated in Yemen and spread through the Ottoman Empire before reaching "
    "Europe. Tides rise and fall twice a day because of the Moon's gravity and the "
    "rotation of the Earth. Volcanic soil is often fertile, which is why farms "
    "cluster on the slopes of active volcanoes despite the danger. Libraries in "
    "the ancient world, such as the one in Alexandria, gathered scrolls from "
    "across the Mediterranean, and scholars there measured the circumference of "
    "the Earth with surprising accuracy using shadows and geometry. "
    "Bees communicate the direction and distance of flowers through a waggle "
    "dance, and a colony can decide on a new nest site by a kind of vote among "
    "its scouts. The first public railway carried passengers between Stockton "
    "and Darlington in 1825, and within a few decades rail lines had reshaped "
    "trade, time zones and the size of cities. Sourdough starters are mixtures of "
    "wild yeast and lactic acid bacteria, and bakers keep some alive for decades "
    "by feeding them flour and water. Lighthouses once used whale oil, then "
    "kerosene, and finally electric lamps focused by large Fresnel lenses that "
    "could throw a beam more than twenty nautical miles. Octopuses can change the "
    "color and texture of their skin in a fraction of a second, even though they "
    "are thought to be color-blind. The Silk Road was never a single road but a "
    "web of routes across deserts and mountains, along which merchants traded "
    "goods, languages and ideas for more than a thousand years. "
)
ZH_PASSAGE = (
    "长城是中国古代的军事防御工程，修建历史可以追溯到春秋战国时期。秦始皇统一六国后，"
    "把各国的城墙连接起来。明朝时又进行了大规模的修缮和扩建，我们今天看到的长城大多是"
    "明代修建的。茶叶起源于中国，最早被当作药物使用，后来逐渐成为日常饮品。唐代陆羽著"
    "有《茶经》，系统地介绍了茶的种植、采摘和饮用方法。丝绸之路连接了东方和西方，商人"
    "们沿着这条路线运输丝绸、瓷器和香料，同时也传播了宗教、技术和艺术。中国的四大发明"
    "包括造纸术、印刷术、火药和指南针，它们对世界文明的发展产生了深远的影响。长江是亚"
    "洲最长的河流，全长六千三百多公里，流经十一个省级行政区，最后注入东海。"
    "黄河被称为中华民族的母亲河，流域内孕育了灿烂的古代文明。由于河水携带大量泥沙，"
    "下游河床不断抬高，历史上多次发生改道和洪水。京剧是中国的国粹之一，融合了唱、念、"
    "做、打等多种表演形式，脸谱的颜色象征着人物的性格。春节是中国最重要的传统节日，"
    "人们会贴春联、放鞭炮、吃饺子，并与家人团聚。大熊猫主要生活在四川、陕西和甘肃的"
    "山区，以竹子为主要食物，是世界上最受欢迎的动物之一。书法讲究笔画的轻重和结构的"
    "平衡，历代书法家留下了大量珍贵的作品。中医认为人体是一个整体，强调阴阳平衡和"
    "经络的作用。"
)
CODE_PASSAGE = """use std::collections::HashMap;

/// Counts how often each word appears in `text`, ignoring case.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let word = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if word.is_empty() {
            continue;
        }
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

def merge_sorted(left, right):
    \"\"\"Merge two sorted lists into one sorted list.\"\"\"
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result + left[i:] + right[j:]

SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE created_at >= '2026-01-01'
GROUP BY customer_id
HAVING SUM(amount) > 1000
ORDER BY total DESC;
"""
# Long inputs are token-count prefixes of a passage, resolved with the tokenizer.
PREFIXES = {"en": EN_PASSAGE, "zh": ZH_PASSAGE, "code": CODE_PASSAGE}


INPUTS = [
    ("card_query_0", query("What is the capital of China?")),
    ("card_query_1", query("Explain gravity")),
    ("card_document_0", "The capital of China is Beijing."),
    (
        "card_document_1",
        (
            "Gravity is a force that attracts two bodies towards each other. It gives "
            "weight to physical objects and is responsible for the movement of planets "
            "around the sun."
        ),
    ),
    ("single_word", "hello"),
    ("non_ascii", "北京是中国的首都。Café naïve résumé — 東京"),
    ("long", LONG_SENTENCE * 17),
    # Queries with task instructions, and the same queries raw.
    ("q_capital_fr", query("What is the capital of France?")),
    ("q_code", query("how to reverse a list in python", CODE_TASK)),
    ("q_med", query("symptoms of iron deficiency", MEDICAL_TASK)),
    ("q_zh", query("如何学习编程？")),
    ("q_es", query("¿Cuál es la montaña más alta del mundo?")),
    ("q_math", query("derivative of x squared", MATH_TASK)),
    ("raw_q_capital_fr", "What is the capital of France?"),
    ("raw_q_code", "how to reverse a list in python"),
    ("raw_q_zh", "如何学习编程？"),
    # Documents for those queries.
    ("d_paris", "Paris is the capital and largest city of France."),
    ("d_lyon", "Lyon is a city in France known for its cuisine."),
    ("d_reverse_py", "my_list.reverse()  # or: reversed_list = my_list[::-1]"),
    (
        "d_iron",
        (
            "Iron deficiency can cause fatigue, pale skin, shortness of breath, "
            "and brittle nails."
        ),
    ),
    ("d_zh_code", "学习编程最好的方法是多写代码，并阅读优秀的开源项目。"),
    ("d_everest_es", "El monte Everest es la montaña más alta del mundo, con 8.849 m."),
    ("d_deriv", "The derivative of x^2 with respect to x is 2x."),
    # Languages and scripts.
    ("ja", "東京は日本の首都で、世界有数の大都市です。"),
    ("ko", "서울은 대한민국의 수도이며 한강이 도시를 가로지른다."),
    ("ar", "القاهرة هي عاصمة مصر وأكبر مدنها."),
    ("hi", "नई दिल्ली भारत की राजधानी है।"),
    ("ru", "Москва — столица России и крупнейший город страны."),
    ("el", "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."),
    ("he", "ירושלים היא עיר עתיקה עם היסטוריה ארוכה."),
    ("th", "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"),
    ("de", "Die Straße führt über die Brücke zum größten Platz der Stadt."),
    ("fr", "L'été dernier, nous avons visité les châteaux de la Loire."),
    ("vi", "Hà Nội là thủ đô của Việt Nam, nổi tiếng với phố cổ."),
    ("emoji", "Great job 🎉🚀 see you soon 👋😀"),
    # Code and structured text.
    ("rust_fn", "fn add(a: i32, b: i32) -> i32 { a + b }"),
    ("sql", "SELECT name, age FROM users WHERE age > 30 ORDER BY name;"),
    ("json", '{"id": 42, "tags": ["a", "b"], "active": true, "score": 0.97}'),
    ("shell", "find . -name '*.rs' -newer Cargo.toml | xargs wc -l | sort -n"),
    ("html", '<div class="card"><h2>Title</h2><p>Body &amp; more</p></div>'),
    # Numbers.
    ("pi", "3.14159265358979323846264338327950288"),
    ("timestamp", "2026-10-04T20:38:00Z"),
    ("phone", "+1 (555) 010-9999 ext. 42"),
    ("money", "1,234,567.89 USD"),
    ("algebra", "(a + b)^2 = a^2 + 2ab + b^2"),
    ("digits", "0123456789" * 8),
    # Near-duplicates of one sentence.
    ("nd_base", "The quick brown fox jumps over the lazy dog."),
    ("nd_no_period", "The quick brown fox jumps over the lazy dog"),
    ("nd_tense", "The quick brown fox jumped over the lazy dog."),
    ("nd_upper", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG."),
    ("nd_trailing_space", "The quick brown fox jumps over the lazy dog. "),
    ("nd_double_space", "The quick brown fox  jumps over the lazy dog."),
    # Edge cases.
    ("period", "."),
    ("spaces", "   "),
    ("repeated", "a" * 200),
    ("url", "https://example.com/path/to/page?query=1&x=y#fragment"),
    ("mixed_scripts", "Hello世界 مرحبا мир 🌍"),
    # Length sweep: prefixes of a passage, by token count before <|endoftext|>.
    ("long_en_16", ("en", 16)),
    ("long_en_64", ("en", 64)),
    ("long_en_128", ("en", 128)),
    ("long_en_256", ("en", 256)),
    ("long_en_384", ("en", 384)),
    ("long_en_500", ("en", 500)),
    ("long_zh_64", ("zh", 64)),
    ("long_zh_128", ("zh", 128)),
    ("long_zh_256", ("zh", 256)),
    ("long_code_128", ("code", 128)),
    ("long_code_256", ("code", 256)),
]
# Query-document and near-duplicate pairs whose scores are compared.
SCORE_PAIRS = [
    ("q_capital_fr", "d_paris"),
    ("q_capital_fr", "d_lyon"),
    ("raw_q_capital_fr", "d_paris"),
    ("q_code", "d_reverse_py"),
    ("raw_q_code", "d_reverse_py"),
    ("q_med", "d_iron"),
    ("q_zh", "d_zh_code"),
    ("raw_q_zh", "d_zh_code"),
    ("q_es", "d_everest_es"),
    ("q_math", "d_deriv"),
    ("q_capital_fr", "d_iron"),
    ("q_med", "d_paris"),
    ("q_zh", "card_document_0"),
    ("card_query_0", "d_zh_code"),
    ("nd_base", "nd_no_period"),
    ("nd_base", "nd_tense"),
    ("nd_base", "nd_upper"),
    ("nd_base", "nd_trailing_space"),
    ("nd_base", "nd_double_space"),
    ("long_en_500", "long_en_384"),
    ("long_zh_256", "long_zh_128"),
]


def resolve_text(tokenizer, text: str | tuple[str, int]) -> str:
    if isinstance(text, str):
        return text
    passage, tokens = text
    ids = tokenizer(PREFIXES[passage], add_special_tokens=False)["input_ids"]
    if len(ids) < tokens:
        raise SystemExit(f"{passage} passage has only {len(ids)} tokens")
    return tokenizer.decode(ids[:tokens])


def f32le_hex(values: torch.Tensor) -> str:
    return struct.pack(f"<{values.numel()}f", *values.tolist()).hex()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_dir(model_dir: Path | None) -> Path:
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION))
    for name, expected in PINNED_SHA256.items():
        actual = sha256_file(model_dir / name)
        if actual != expected:
            raise SystemExit(f"{name} sha256 {actual} is not the pinned {expected}")
    return model_dir


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The card's pooling: the last position, or the last unpadded one."""
    if bool(mask[:, -1].sum() == mask.shape[0]):
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0]), lengths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/qwen3-embedding-0.6b/embedding-reference.json",
    )
    args = parser.parse_args()
    model_dir = resolve_model_dir(args.model_dir)

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, padding_side="left"
    )
    model = transformers.AutoModel.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    records, embeddings = [], {}
    for name, spec in INPUTS:
        text = resolve_text(tokenizer, spec)
        batch = tokenizer([text], return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        if ids[-1] != tokenizer.pad_token_id or len(ids) > MAX_TOKENS:
            raise SystemExit(f"{name}: unexpected template or length ({len(ids)})")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        embedding = F.normalize(last_token_pool(hidden, batch["attention_mask"]), dim=1)
        embeddings[name] = embedding[0]
        records.append(
            {
                "name": name,
                "text": text,
                "input_ids": ids,
                "embedding_f32le": f32le_hex(embedding[0]),
            }
        )

    queries = torch.stack([embeddings["card_query_0"], embeddings["card_query_1"]])
    documents = torch.stack(
        [embeddings["card_document_0"], embeddings["card_document_1"]]
    )
    scores = (queries @ documents.T).tolist()
    pair_scores = [
        {
            "left": left,
            "right": right,
            "score": float(embeddings[left] @ embeddings[right]),
        }
        for left, right in SCORE_PAIRS
    ]

    # The card's own batched, left-padded run of the same four inputs.
    card_texts = [text for name, text in INPUTS if name.startswith("card_")]
    assert all(isinstance(text, str) for text in card_texts)
    batch = tokenizer(card_texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
    batched = F.normalize(last_token_pool(hidden, batch["attention_mask"]), dim=1)
    batched_scores = (batched[:2] @ batched[2:].T).tolist()

    def max_diff(left: list[list[float]], right: list[list[float]]) -> float:
        return max(abs(a - b) for la, lb in zip(left, right) for a, b in zip(la, lb))

    fixture = {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": PINNED_SHA256,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "pooling": "last token (appended <|endoftext|>), then L2 normalize",
            "task": TASK,
        },
        "tolerance_policy": TOLERANCE_POLICY,
        "inputs": records,
        "scores": scores,
        "pair_scores": pair_scores,
        "card_scores": CARD_SCORES,
        "observed": {
            "scores_max_abs_vs_card": max_diff(scores, CARD_SCORES),
            "batched_scores_max_abs_vs_single": max_diff(batched_scores, scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), **fixture["observed"]}))


if __name__ == "__main__":
    main()
