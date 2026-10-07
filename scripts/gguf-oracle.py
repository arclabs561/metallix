# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf==0.19.0", "numpy"]
# ///
"""Writes llama.cpp's outputs on a GGUF file for qwen35's opt-in GGUF test.

For each fixed prompt it runs `llama-results`, which writes the prompt's
token IDs and the F32 logits at every position to `oracle-<name>.gguf`, and
asks `llama-server` for 64 greedy tokens after the prompt. `prompts.json`
holds the prompts, those greedy tokens and the llama.cpp version.

Usage:
  uv run scripts/gguf-oracle.py MODEL.gguf OUT_DIR [--llama-bin DIR]

Then run, with the Hugging Face checkpoint the file was converted from:
  METALLIX_QWEN35_GGUF=MODEL.gguf METALLIX_QWEN35_GGUF_ORACLE=OUT_DIR \\
  METALLIX_QWEN35_GGUF_BASE=BASE_DIR \\
  cargo test -p qwen35 --features metal --test gguf_checkpoint \
      gguf_logits_agree_with_llama_cpp -- --exact --nocapture --test-threads=1
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GREEDY_TOKENS = 64

# No special-token spellings: llama-results tokenizes without parsing them.
# No trailing newline either: llama.cpp drops it when reading a prompt file.
PROMPTS = {
    "prose": (
        "The lighthouse keeper kept a ledger of every ship that passed the point. "
        "Most entries were short: a name, a time, the direction of travel, and a note "
        "about the weather. In the winter of the storm year the entries grew longer. "
        "He wrote about the colour of the water before a squall, about the way gulls "
        "flew low and quiet over the rocks, and about a schooner that anchored in the "
        "bay for three days while its crew mended a torn sail. On the fourth morning "
        "the schooner was gone, and the keeper wrote only that the wind had turned to "
        "the north overnight. Years later, when the station was automated and the "
        "ledgers were sent to the county archive, a librarian read them from start to "
        "finish. She noticed that the handwriting changed in the spring after the "
        "storm, becoming smaller and more careful, as if the keeper had decided that "
        "every line deserved the same attention whether it recorded a passing ferry "
        "or a shipwreck. She catalogued the volumes, wrote a short description for the "
        "finding aid, and then, against the archive's usual practice, added one more "
        "sentence: the ledgers are worth reading in order, because the keeper's "
        "attention is itself the subject. The finding aid was printed and bound with "
        "the others, and for a long time nobody asked for the ledgers at all. "
        "Then a historian writing about coastal trade found the reference, drove "
        "out on a wet afternoon, and spent a week at the reading-room table with "
        "the volumes stacked beside her. She copied the schooner's entries into a "
        "notebook and compared them with the harbour registers of three nearby "
        "towns. None of them recorded the schooner arriving or leaving, and the "
        "name the keeper had written did not match any vessel in the insurance "
        "lists of that decade. In her book she gave the story a single paragraph "
        "and a footnote, noting that the most careful witness on the coast had "
        "seen a ship that no other record could find, and that she had decided to "
        "believe him."
    ),
    "code": (
        "use std::collections::HashMap;\n\n"
        "/// Counts how often each word appears, ignoring case and punctuation.\n"
        "pub fn word_counts(text: &str) -> HashMap<String, usize> {\n"
        "    let mut counts = HashMap::new();\n"
        "    for word in text\n"
        "        .split(|c: char| !c.is_alphanumeric())\n"
        "        .filter(|word| !word.is_empty())\n"
        "    {\n"
        "        *counts.entry(word.to_lowercase()).or_insert(0) += 1;\n"
        "    }\n"
        "    counts\n"
        "}\n\n"
        "/// The `n` most frequent words, most frequent first; ties break by word.\n"
        "pub fn top_words(counts: &HashMap<String, usize>, n: usize) -> Vec<(&str, usize)> {\n"
        "    let mut entries: Vec<(&str, usize)> =\n"
        "        counts.iter().map(|(word, count)| (word.as_str(), *count)).collect();\n"
        "    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));\n"
        "    entries.truncate(n);\n"
        "    entries\n"
        "}\n\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n\n"
        "    #[test]\n"
        "    fn counts_words_case_insensitively() {\n"
        '        let counts = word_counts("The cat and the hat. THE end!");\n'
        '        assert_eq!(counts["the"], 3);\n'
        '        assert_eq!(counts["cat"], 1);\n'
        '        assert_eq!(top_words(&counts, 1), vec![("the", 3)]);\n'
        "    }\n"
        "}\n\n"
        "fn main() {\n"
        '    let text = std::fs::read_to_string("input.txt").expect("input");\n'
        "    for (word, count) in top_words(&word_counts(&text), 10) {"
    ),
    "dialogue": (
        "Customer: Hi, I ordered a bookshelf last week and one of the side panels "
        "arrived cracked along the bottom edge. The rest of the parts look fine.\n"
        "Agent: I'm sorry to hear that. Could you tell me the order number and send a "
        "photo of the damaged panel?\n"
        "Customer: Sure, the order number is 48213. I've attached two photos, one of "
        "the crack and one of the label on the box.\n"
        "Agent: Thank you, I can see the damage clearly. I can send a replacement "
        "panel, which usually arrives within five working days, or I can arrange a "
        "full refund if you'd prefer to return the whole bookshelf.\n"
        "Customer: A replacement panel would be great. Do I need to send the cracked "
        "one back?\n"
        "Agent: No, you can keep it or recycle it. I've created the replacement order "
        "and you'll get a tracking link by email once it ships. Is there anything "
        "else I can help with today?\n"
        "Customer: Actually, yes. The assembly instructions mention a wall anchor kit, "
        "but I couldn't find it in the box. Is it sold separately?\n"
        "Agent: Good question. The anchor kit should have been included, so I'll add "
        "one to the replacement shipment at no charge.\n"
        "Customer: That's very helpful, thank you. One last thing: the shelf is "
        "going in a room with a sloping ceiling. Can I leave off the top panel "
        "and still anchor it safely?\n"
        "Agent: The top panel holds the two sides square, so leaving it off would "
        "make the frame wobble even with the anchors fitted. Some customers in the "
        "same situation buy the shorter model, which has four shelves instead of "
        "five and stands about thirty centimetres lower. I can check whether it "
        "would fit if you tell me the height of the wall where the ceiling starts "
        "to slope.\n"
        "Customer: It's about one hundred and sixty centimetres at that point.\n"
        "Agent:"
    ),
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def read_tokens(oracle: Path) -> list[int]:
    """The `tokens` tensor of a llama-results file."""
    from gguf import GGUFReader

    tensors = {tensor.name: tensor for tensor in GGUFReader(oracle).tensors}
    return [int(token) for token in tensors["tokens"].data]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--llama-bin", type=Path, default=Path("/opt/homebrew/bin"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [args.llama_bin / "llama-cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    record = {
        "model": args.model.name,
        "llama_cpp": (version.stdout + version.stderr).strip().splitlines()[0],
        "prompts": [],
    }
    for name, text in PROMPTS.items():
        assert not text.endswith("\n"), name
        prompt_file = args.out / f"prompt-{name}.txt"
        prompt_file.write_text(text)
        oracle = args.out / f"oracle-{name}.gguf"
        subprocess.run(
            [
                args.llama_bin / "llama-results",
                "-m",
                args.model,
                "-o",
                oracle,
                "-f",
                prompt_file,
                "-c",
                "4096",
            ],
            check=True,
            capture_output=True,
        )
        record["prompts"].append(
            {"name": name, "text": text, "tokens": read_tokens(oracle)}
        )

    port = free_port()
    server = subprocess.Popen(
        [
            args.llama_bin / "llama-server",
            "-m",
            args.model,
            "--port",
            str(port),
            "-c",
            "4096",
            "--parallel",
            "1",
            "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ):
                    break
            except OSError:
                time.sleep(0.5)
        for prompt in record["prompts"]:
            reply = post(
                f"http://127.0.0.1:{port}/completion",
                {
                    "prompt": prompt["tokens"],
                    "n_predict": GREEDY_TOKENS,
                    "temperature": 0,
                    "top_k": 1,
                    "samplers": ["top_k"],
                    "cache_prompt": False,
                    "return_tokens": True,
                },
            )
            prompt["llama_greedy"] = reply["tokens"]
    finally:
        server.terminate()
        server.wait(timeout=30)

    (args.out / "prompts.json").write_text(json.dumps(record, indent=1) + "\n")
    for prompt in record["prompts"]:
        print(
            f"{prompt['name']}: {len(prompt['tokens'])} prompt tokens, "
            f"{len(prompt['llama_greedy'])} greedy tokens"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
