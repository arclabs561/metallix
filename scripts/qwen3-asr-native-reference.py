#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers==5.18.0",
#   "torch==2.13.0",
#   "soundfile==0.14.0",
# ]
# ///
"""Qwen3-ASR encoder, logits and transcript captures from transformers-native
Qwen3ASR, for Metallix parity tests.

transformers 5.18.0 native `Qwen3ASRForConditionalGeneration` runs the audio
encoder with 8 s attention windows on every attention backend, as the model's
`n_window_infer` and the qwen-asr vLLM backend do. (The qwen-asr 0.0.6
transformers backend applies full attention under sdpa and eager, so it is
not used as the encoder reference.) It loads the Qwen/Qwen3-ASR-*-hf
checkpoints, whose tensors are byte-identical renames of the original ones.

`capture` reads clips written by `qwen3-asr-reference.py prepare-clips` and
writes, per clip, the log-mel features, the projected encoder output, the
first step's logits and the greedy transcript as raw little-endian f32 plus a
JSON manifest, in the same layout as `qwen3-asr-reference.py capture`.

This is a numerical oracle for Metallix, not an inference benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
STOP_TOKENS = {151643, 151645}


def write_f32(path: Path, array: np.ndarray) -> list[int]:
    array = np.ascontiguousarray(array, dtype="<f4")
    path.write_bytes(array.tobytes())
    return list(array.shape)


def capture(args: argparse.Namespace) -> None:
    import soundfile
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.eval()
    clips = json.loads((args.clips / "manifest.json").read_text())["clips"]
    if args.ids:
        wanted = set(args.ids)
        clips = [clip for clip in clips if clip["id"] in wanted]
    if args.min_mel_frames:
        clips = [clip for clip in clips if clip["samples"] // 160 > args.min_mel_frames]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        samples, rate = soundfile.read(args.clips / clip["wav"], dtype="float32")
        assert rate == SAMPLE_RATE
        inputs = processor.apply_transcription_request(
            audio=[samples], language=args.language, prompt=args.context or None
        )
        frames = int(inputs["input_features_mask"][0].sum())
        mel = inputs["input_features"][0, :, :frames].float().numpy()
        inputs = inputs.to(model.device, model.dtype)
        with torch.no_grad():
            encoded = model.model.get_audio_features(
                inputs["input_features"], inputs["input_features_mask"]
            ).pooler_output
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )
        prompt_ids = [int(token) for token in inputs["input_ids"][0]]
        new_ids = [int(token) for token in generated.sequences[0, len(prompt_ids) :]]
        while new_ids and new_ids[-1] in STOP_TOKENS:
            new_ids.pop()
        raw = processor.decode(torch.tensor([new_ids]))[0]
        text = processor.decode(
            torch.tensor([new_ids]), return_format="transcription_only"
        )[0]
        stem = args.out / clip["id"]
        tensors = {}
        if not args.text_only:
            tensors = {
                "mel_shape": write_f32(stem.with_suffix(".mel.f32"), mel),
                "encoder_shape": write_f32(
                    stem.with_suffix(".encoder.f32"),
                    encoded.reshape(-1, encoded.shape[-1]).float().numpy(),
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
                "language": args.language,
                "text": text,
            }
        )
        print(clip["id"], len(prompt_ids), len(new_ids), text[:60], flush=True)
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


def layers(args: argparse.Namespace) -> None:
    """Dump the encoder's input to its first layer, each layer's output and
    the final norm, as [frames, d_model] f32, for locating precision drift."""
    import soundfile
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.eval()
    tower = model.model.audio_tower
    stages: list[torch.Tensor] = []
    hooks = [
        tower.layers[0].register_forward_pre_hook(
            lambda module, inputs: stages.append(inputs[0].detach().clone())
        )
    ]
    for layer in tower.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output: stages.append(
                    (output[0] if isinstance(output, tuple) else output)
                    .detach()
                    .clone()
                )
            )
        )
    hooks.append(
        tower.ln_post.register_forward_hook(
            lambda module, inputs, output: stages.append(output.detach().clone())
        )
    )
    clips = json.loads((args.clips / "manifest.json").read_text())["clips"]
    clips = [clip for clip in clips if clip["id"] in set(args.ids)]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        samples, rate = soundfile.read(args.clips / clip["wav"], dtype="float32")
        assert rate == SAMPLE_RATE
        inputs = processor.apply_transcription_request(
            audio=[samples], language="English"
        )
        inputs = inputs.to(model.device, model.dtype)
        stages.clear()
        with torch.no_grad():
            model.model.get_audio_features(
                inputs["input_features"], inputs["input_features_mask"]
            )
        shapes = []
        for index, stage in enumerate(stages):
            stage = stage.reshape(-1, stage.shape[-1])
            shapes.append(
                write_f32(
                    args.out / f"{clip['id']}.stage{index}.f32", stage.float().numpy()
                )
            )
        records.append({"id": clip["id"], "stages": shapes})
        print(clip["id"], len(shapes), shapes[0], flush=True)
    for hook in hooks:
        hook.remove()
    (args.out / "manifest.json").write_text(
        json.dumps(
            {"dtype": args.dtype, "model": str(args.model), "clips": records}, indent=1
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "capture", help="capture reference features, logits and text"
    )
    run.add_argument(
        "--model", type=Path, required=True, help="Qwen3-ASR -hf checkpoint directory"
    )
    run.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    run.add_argument(
        "--clips", type=Path, required=True, help="prepare-clips output directory"
    )
    run.add_argument("--ids", nargs="*", help="only these clip ids")
    run.add_argument(
        "--min-mel-frames", type=int, default=0, help="only clips longer than this"
    )
    run.add_argument(
        "--language", default=None, help="force this language, e.g. English"
    )
    run.add_argument("--context", default="")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--text-only", action="store_true", help="skip tensor dumps")
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(run=capture)
    stage = commands.add_parser("layers", help="dump encoder stage outputs")
    stage.add_argument("--model", type=Path, required=True)
    stage.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    stage.add_argument("--clips", type=Path, required=True)
    stage.add_argument("--ids", nargs="+", required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.set_defaults(run=layers)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
