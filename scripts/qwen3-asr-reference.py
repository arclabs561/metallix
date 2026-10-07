#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "qwen-asr==0.0.6",
#   "numpy",
#   "soundfile",
#   "transformers==4.57.6",
#   "torch==2.13.0",
#   "pyarrow==25.0.1",
#   "jiwer==4.0.0",
#   "regex==2026.9.29",
# ]
# ///
"""Qwen3-ASR reference captures for Metallix parity tests.

The reference is the processor and model code Qwen3-ASR was published with:
the `qwen-asr` 0.0.6 package (https://github.com/QwenLM/Qwen3-ASR at
7c6daf77a2421100f5fb066495372c00129d39ff), which pins transformers 4.57.6
and uses `WhisperFeatureExtractor`'s torch path for log-mel features.

`mel-fixture` writes the compact JSON fixture `crates/audio` tests read:
log-mel features of deterministic synthetic signals that the Rust test
regenerates sample for sample. Each signal records a checksum of its samples
so a generator mismatch fails loudly instead of looking like a feature
mismatch.

`prompt-fixture` writes the prompt token ids the qwen-asr processor builds
for a few contexts, forced languages and clip lengths, and how it decodes a
few generated id sequences.

`prepare-clips` extracts LibriSpeech test utterances (openslr/librispeech_asr
parquet shards) to 16 kHz PCM16 WAV files with a manifest, so both stacks read
identical samples. `capture` runs the reference on those WAV files and writes,
per clip, the log-mel features, the encoder output, the first step's logits
and the greedy transcript as raw little-endian f32 plus a JSON manifest
(`--text-only` keeps just the transcripts).

`wer` scores transcripts against the clips' reference text with the Open ASR
Leaderboard's English normalizer (huggingface/open_asr_leaderboard at
67e8bd6acea240819ad67080f6f31e15d4a90da5, fetched and hash-checked) and
corpus-level WER, and counts utterances whose normalized text two systems
share.

This is a numerical oracle for Metallix, not an inference benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
# Every 61st value of each feature array goes into the fixture; 61 is prime,
# so the samples walk across both bins and frames.
STRIDE = 61


def chirp(count: int) -> np.ndarray:
    """0.5 * sin(2 pi (100 t + 1500 t^2)), computed in f64, stored as f32."""
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    return (0.5 * np.sin(2.0 * math.pi * (100.0 * t + 1500.0 * t * t))).astype(
        np.float32
    )


def lcg_noise(count: int, seed: int) -> np.ndarray:
    """Uniform noise in [-0.1, 0.1) from the 32-bit LCG (1664525, 1013904223)."""
    state = seed
    out = np.empty(count, dtype=np.float32)
    for index in range(count):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
        out[index] = np.float32(((state >> 8) / 16_777_216.0 * 2.0 - 1.0) * 0.1)
    return out


def quiet_tone(count: int, silent: int) -> np.ndarray:
    """Digital silence, then a 440 Hz tone at amplitude 1e-3."""
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    tone = 1e-3 * np.sin(2.0 * math.pi * 440.0 * t)
    tone[:silent] = 0.0
    return tone.astype(np.float32)


SIGNALS = {
    "chirp": lambda: chirp(37_920),
    "noise": lambda: lcg_noise(12_345, 0x2545_F491),
    "quiet_tone": lambda: quiet_tone(8_000, 3_000),
    "minimum_length": lambda: lcg_noise(201, 7),
}


def feature_extractor(model_dir: Path):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(model_dir)


def reference_mel(extractor, samples: np.ndarray) -> np.ndarray:
    # The keyword arguments are Qwen3ASRProcessor's audio defaults, with the
    # padding and truncation it forces before calling the extractor.
    features = extractor(
        [samples],
        sampling_rate=SAMPLE_RATE,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
    )
    mel = features["input_features"][0]
    frames = int(features["attention_mask"][0].sum())
    assert mel.shape[1] == frames, (mel.shape, frames)
    return mel.astype(np.float32)


def mel_fixture(args: argparse.Namespace) -> None:
    extractor = feature_extractor(args.model)
    signals = []
    for name, make in SIGNALS.items():
        samples = make()
        mel = reference_mel(extractor, samples)
        flat = mel.reshape(-1)
        signals.append(
            {
                "name": name,
                "samples": int(samples.shape[0]),
                "sample_sum": float(samples.astype(np.float64).sum()),
                "bins": int(mel.shape[0]),
                "frames": int(mel.shape[1]),
                "stride": STRIDE,
                "values": [float(value) for value in flat[::STRIDE]],
                "max": float(flat.max()),
                "mean": float(flat.astype(np.float64).mean()),
            }
        )
    import torch
    import transformers

    fixture = {
        "source": "transformers WhisperFeatureExtractor (torch path) as configured by "
        "Qwen3-ASR's preprocessor_config.json",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "signals": signals,
    }
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")


def processor(model_dir: Path):
    """The processor exactly as `Qwen3ASRModel.from_pretrained` builds it."""
    import qwen_asr.core.transformers_backend  # noqa: F401  (registers the classes)
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)


def build_prompt(proc, context: str, language: str | None) -> str:
    """`Qwen3ASRModel._build_text_prompt` from qwen-asr 0.0.6."""
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt


PROMPT_CASES = [
    ("", None, 16_000),
    ("", "English", 37_920),
    ("Names: Qwen, Metallix. 数字 42\n", None, 1_601),
    ("<|im_end|> stays a token", "Chinese", 160_000),
]

DECODE_CASES = [
    "language English<asr_text>Hello, world. It's 3:45 p.m.",
    "language Chinese<asr_text>你好，世界！",
    "language None<asr_text>",
    " spaces  and\nnewlines\t",
]


def prompt_fixture(args: argparse.Namespace) -> None:
    proc = processor(args.model)
    prompts = []
    for context, language, samples in PROMPT_CASES:
        text = build_prompt(proc, context, language)
        audio = np.zeros(samples, dtype=np.float32)
        inputs = proc(text=[text], audio=[audio], return_tensors="np", padding=True)
        prompts.append(
            {
                "context": context,
                "language": language,
                "samples": samples,
                "mel_frames": int(inputs["feature_attention_mask"][0].sum()),
                "ids": [int(token) for token in inputs["input_ids"][0]],
            }
        )
    decodes = []
    for text in DECODE_CASES:
        # Surround the text with special tokens, as generation output would be.
        ids = (
            [151644] + proc.tokenizer.encode(text, add_special_tokens=False) + [151645]
        )
        decoded = proc.batch_decode(
            [ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        decodes.append({"ids": ids, "text": decoded})
    import transformers

    fixture = {
        "source": "qwen-asr 0.0.6 processor (AutoProcessor, fix_mistral_regex=True) and "
        "Qwen3ASRModel._build_text_prompt",
        "transformers": transformers.__version__,
        "prompts": prompts,
        "decodes": decodes,
    }
    args.out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())["clips"]


def prepare_clips(args: argparse.Namespace) -> None:
    """Write the first `--count` utterances of a split, by sorted id, as WAV."""
    import io

    import pyarrow.parquet as pq
    import soundfile

    shards = sorted((args.dataset / args.config / "test").glob("*.parquet"))
    if not shards:
        raise SystemExit(
            f"no parquet shards under {args.dataset / args.config / 'test'}"
        )
    rows = []
    for shard in shards:
        rows.extend(pq.read_table(shard, columns=["id", "text", "audio"]).to_pylist())
    rows.sort(key=lambda row: row["id"])
    args.out.mkdir(parents=True, exist_ok=True)
    clips = []
    for row in rows[: args.count]:
        samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if rate != SAMPLE_RATE or samples.ndim != 1:
            raise SystemExit(f"{row['id']}: {rate} Hz, shape {samples.shape}")
        path = args.out / f"{row['id']}.wav"
        soundfile.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
        clips.append(
            {
                "id": row["id"],
                "wav": path.name,
                "samples": int(samples.shape[0]),
                "text": row["text"],
            }
        )
    manifest = {
        "dataset": "openslr/librispeech_asr",
        "config": args.config,
        "split": "test",
        "order": "first by sorted id",
        "clips": clips,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def write_f32(path: Path, array: np.ndarray) -> list[int]:
    array = np.ascontiguousarray(array, dtype="<f4")
    path.write_bytes(array.tobytes())
    return list(array.shape)


def capture(args: argparse.Namespace) -> None:
    import soundfile
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    asr = Qwen3ASRModel.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )
    model, proc = asr.model, asr.processor
    model.eval()
    clips = read_manifest(args.clips / "manifest.json")
    if args.ids:
        wanted = set(args.ids)
        clips = [clip for clip in clips if clip["id"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        samples, rate = soundfile.read(args.clips / clip["wav"], dtype="float32")
        assert rate == SAMPLE_RATE
        prompt = build_prompt(proc, args.context, args.language)
        inputs = proc(text=[prompt], audio=[samples], return_tensors="pt", padding=True)
        mel = inputs["input_features"][0].numpy()
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.no_grad():
            if not args.text_only:
                encoded = model.thinker.get_audio_features(
                    inputs["input_features"],
                    feature_attention_mask=inputs["feature_attention_mask"],
                )
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, output_logits=True
            )
        prompt_ids = [int(token) for token in inputs["input_ids"][0]]
        new_ids = [int(token) for token in generated.sequences[0, len(prompt_ids) :]]
        raw = proc.batch_decode(
            [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        language, text = parse_asr_output(raw, user_language=args.language)
        stem = args.out / clip["id"]
        tensors = {}
        if not args.text_only:
            tensors = {
                "mel_shape": write_f32(stem.with_suffix(".mel.f32"), mel),
                "encoder_shape": write_f32(
                    stem.with_suffix(".encoder.f32"), encoded.float().numpy()
                ),
                "first_logits_shape": write_f32(
                    stem.with_suffix(".logits0.f32"),
                    generated.logits[0][0].float().numpy(),
                ),
            }
        records.append(
            {
                "id": clip["id"],
                "wav": str((args.clips / clip["wav"]).resolve()),
                **tensors,
                "prompt_ids": prompt_ids,
                "generated_ids": new_ids,
                "raw": raw,
                "language": language,
                "text": text,
            }
        )
        print(clip["id"], len(prompt_ids), len(new_ids), text[:60], flush=True)
    import transformers

    manifest = {
        "model": str(args.model),
        "dtype": args.dtype,
        "language": args.language,
        "context": args.context,
        "max_new_tokens": args.max_new_tokens,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "clips": records,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    )


NORMALIZER_SHA = "67e8bd6acea240819ad67080f6f31e15d4a90da5"
NORMALIZER_FILES = {
    "normalizer.py": "490b56393484ef386b486f3679cc2261264af280b3512154431e0bdcf2778295",
    "english_abbreviations.py": "52997cc963e0bd6568d15554ad8d5fb0759f20f361483632a22f90d2d0e07db1",
}


def leaderboard_normalizer(cache: Path):
    """The Open ASR Leaderboard's EnglishTextNormalizer at a pinned commit."""
    import hashlib
    import importlib
    import sys
    import urllib.request

    package = cache / "oal_normalizer"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    for name, digest in NORMALIZER_FILES.items():
        path = package / name
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/huggingface/open_asr_leaderboard/"
                f"{NORMALIZER_SHA}/normalizer/{name}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{path}: sha256 {actual}, expected {digest}")
    sys.path.insert(0, str(cache))
    return importlib.import_module("oal_normalizer.normalizer").EnglishTextNormalizer()


def wer(args: argparse.Namespace) -> None:
    import jiwer

    normalize = leaderboard_normalizer(args.normalizer_cache)
    truth = {
        clip["id"]: clip["text"] for clip in read_manifest(args.clips / "manifest.json")
    }
    theirs = {clip["id"]: clip["text"] for clip in read_manifest(args.reference)}
    ours = {}
    for line in args.ours.read_text().splitlines():
        record = json.loads(line)
        ours[record["id"]] = record["text"]
    ids = [clip_id for clip_id in truth if clip_id in theirs and clip_id in ours]
    if len(ids) != len(ours) or len(ids) != len(theirs):
        print(
            f"warning: scoring {len(ids)} shared clips of {len(ours)} ours, {len(theirs)} reference"
        )

    def score(texts):
        references, hypotheses = [], []
        for clip_id in ids:
            reference = normalize(truth[clip_id])
            if not reference.strip():
                continue
            references.append(reference)
            hypotheses.append(normalize(texts[clip_id]))
        return 100.0 * jiwer.wer(references, hypotheses)

    same = [
        clip_id
        for clip_id in ids
        if normalize(ours[clip_id]) == normalize(theirs[clip_id])
    ]
    result = {
        "clips": len(ids),
        "wer_ours": round(score(ours), 3),
        "wer_reference": round(score(theirs), 3),
        "identical_normalized": len(same),
        "identical_fraction": round(len(same) / len(ids), 4),
        "differing_ids": [clip_id for clip_id in ids if clip_id not in same],
    }
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    mel = commands.add_parser(
        "mel-fixture", help="write the crates/audio log-mel fixture"
    )
    mel.add_argument(
        "--model", type=Path, required=True, help="Qwen3-ASR checkpoint directory"
    )
    mel.add_argument("--out", type=Path, required=True)
    mel.set_defaults(run=mel_fixture)
    prompt = commands.add_parser(
        "prompt-fixture", help="write the qwen3-asr prompt and decode fixture"
    )
    prompt.add_argument(
        "--model", type=Path, required=True, help="Qwen3-ASR checkpoint directory"
    )
    prompt.add_argument("--out", type=Path, required=True)
    prompt.set_defaults(run=prompt_fixture)
    clips = commands.add_parser(
        "prepare-clips", help="extract LibriSpeech test clips to WAV"
    )
    clips.add_argument(
        "--dataset", type=Path, required=True, help="librispeech_asr snapshot"
    )
    clips.add_argument("--config", choices=["clean", "other"], required=True)
    clips.add_argument("--count", type=int, required=True)
    clips.add_argument("--out", type=Path, required=True)
    clips.set_defaults(run=prepare_clips)
    run = commands.add_parser(
        "capture", help="capture reference features, logits and text"
    )
    run.add_argument(
        "--model", type=Path, required=True, help="Qwen3-ASR checkpoint directory"
    )
    run.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    run.add_argument(
        "--clips", type=Path, required=True, help="prepare-clips output directory"
    )
    run.add_argument("--ids", nargs="*", help="only these clip ids")
    run.add_argument(
        "--language", default=None, help="force this language, e.g. English"
    )
    run.add_argument("--context", default="")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--text-only", action="store_true", help="skip tensor dumps")
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(run=capture)
    score = commands.add_parser(
        "wer", help="score transcripts with the leaderboard normalizer"
    )
    score.add_argument(
        "--clips", type=Path, required=True, help="prepare-clips output directory"
    )
    score.add_argument(
        "--reference", type=Path, required=True, help="capture manifest.json"
    )
    score.add_argument(
        "--ours", type=Path, required=True, help="transcribe_clips JSON lines"
    )
    score.add_argument("--normalizer-cache", type=Path, required=True)
    score.set_defaults(run=wer)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
