#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers==0.41.0",
#   "numpy>=2",
#   "pillow",
#   "safetensors",
#   "torch==2.14.1",
#   "transformers==5.18.0",
# ]
# ///
"""Capture CPU FLUX.2 [klein] references for Metallix component parity.

Every subcommand loads only the components it needs from a local diffusers
checkpoint directory and writes one safetensors file plus a JSON sidecar
(versions, dtype, inputs). Noise is drawn here with a seeded torch generator
and saved, so Metallix consumes the same latents rather than reproducing
torch's random stream.

  text-encoder  prompt -> padded ids, mask, hidden states at 9/18/27, embeds
  transformer   embeds + saved latents -> double block 0, single block 0, output
  vae           saved latents -> decoded image in [0, 1]
  pipeline      prompt + seed -> initial latents, per-step latents, image, PNG

Peak memory is large in float32 (each 4B component is ~16 GB); run it under
the heavy lease.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16}
TEXT_LAYERS = (9, 18, 27)
MAX_SEQUENCE_LENGTH = 512


def versions() -> dict[str, str]:
    return {
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def write(out: Path, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, str(out))
    meta = {"versions": versions(), **meta}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def text_encoder(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model / "text_encoder", torch_dtype=dtype
    )
    model.eval()

    # Same rendering as Flux2KleinPipeline._get_qwen3_prompt_embeds.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    # Hooks pin down the hidden_states indexing convention: index k must be
    # the output of decoder layer k - 1 (0-based), before the final norm.
    captured: dict[int, torch.Tensor] = {}

    def hook(index: int):
        def record(_module, _inputs, output):
            captured[index] = output[0] if isinstance(output, tuple) else output

        return record

    handles = [
        model.model.layers[k - 1].register_forward_hook(hook(k)) for k in TEXT_LAYERS
    ]
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    for handle in handles:
        handle.remove()
    for k in TEXT_LAYERS:
        if not torch.equal(output.hidden_states[k], captured[k]):
            raise SystemExit(f"hidden_states[{k}] is not the output of layer {k - 1}")

    stacked = torch.stack([output.hidden_states[k] for k in TEXT_LAYERS], dim=1)
    batch, channels, seq_len, hidden = stacked.shape
    embeds = stacked.permute(0, 2, 1, 3).reshape(batch, seq_len, channels * hidden)

    tensors = {
        "input_ids": inputs["input_ids"].to(torch.int32),
        "attention_mask": inputs["attention_mask"].to(torch.int32),
        "prompt_embeds": embeds,
    }
    for k in TEXT_LAYERS:
        tensors[f"hidden_states.{k}"] = output.hidden_states[k]
    write(
        args.out,
        tensors,
        {
            "kind": "text-encoder",
            "dtype": args.dtype,
            "prompt": args.prompt,
            "rendered": text,
            "real_tokens": int(inputs["attention_mask"].sum()),
            "layers": list(TEXT_LAYERS),
        },
    )


def noise(seed: int, height: int, width: int, channels: int) -> torch.Tensor:
    # Flux2KleinPipeline.prepare_latents' unpacked shape: [B, 128, H/16, W/16].
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(
        (1, channels * 4, height // 16, width // 16), generator=generator
    )


def transformer(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    model = Flux2Transformer2DModel.from_pretrained(
        args.model / "transformer", torch_dtype=dtype
    )
    model.eval()
    embeds = load_file(str(args.text))["prompt_embeds"].to(dtype)
    latents = noise(args.seed, args.height, args.width, model.config.in_channels // 4)

    packed = Flux2KleinPipeline._pack_latents(latents).to(dtype)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents)
    text_ids = Flux2KleinPipeline._prepare_text_ids(embeds)
    # The pipeline casts the scheduler's float32 timestep to the latent dtype
    # before dividing by 1000.
    timestep = torch.tensor([args.timestep], dtype=torch.float32).to(dtype) / 1000

    captured: dict[str, torch.Tensor] = {}

    def keep(name: str):
        def record(_module, _inputs, output):
            if isinstance(output, tuple):
                for i, part in enumerate(output):
                    captured[f"{name}.{i}"] = part
            else:
                captured[name] = output

        return record

    handles = [
        model.transformer_blocks[0].register_forward_hook(keep("double_block_0")),
        model.single_transformer_blocks[0].register_forward_hook(
            keep("single_block_0")
        ),
    ]
    with torch.no_grad():
        velocity = model(
            hidden_states=packed,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
    for handle in handles:
        handle.remove()

    write(
        args.out,
        {
            "latents": latents,
            "packed_latents": packed,
            "latent_ids": latent_ids.to(torch.int32),
            "text_ids": text_ids.to(torch.int32),
            "timestep": timestep,
            "velocity": velocity,
            **captured,
        },
        {
            "kind": "transformer",
            "dtype": args.dtype,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "scheduler_timestep": args.timestep,
            "text_reference": str(args.text.name),
        },
    )


def vae(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    model = AutoencoderKLFlux2.from_pretrained(args.model / "vae", torch_dtype=dtype)
    model.eval()
    latents = load_file(str(args.latents))[args.key].to(dtype)
    with torch.no_grad():
        image = model.decode(latents, return_dict=False)[0]
    write(
        args.out,
        {"latents": latents, "decoded": image, "image": (image / 2 + 0.5).clamp(0, 1)},
        {
            "kind": "vae",
            "dtype": args.dtype,
            "source": f"{args.latents.name}:{args.key}",
        },
    )


def pipeline(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype).to(
        args.device
    )
    latents = noise(
        args.seed, args.height, args.width, pipe.transformer.config.in_channels // 4
    )
    steps: list[torch.Tensor] = []

    def record(_pipe, _index, _timestep, kwargs):
        steps.append(kwargs["latents"].detach().to("cpu", copy=True))
        return {}

    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            latents=latents.to(device=args.device, dtype=dtype),
            output_type="pt",
            callback_on_step_end=record,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
    image = image.cpu()
    tensors = {"initial_latents": latents, "image": image}
    for index, step in enumerate(steps):
        tensors[f"latents.step{index}"] = step

    # The VAE's input, rebuilt from the last step the way the pipeline does
    # after its loop: scatter tokens back to a grid, undo the latent
    # BatchNorm, unpatchify 2x2. `vae --key vae_input` decodes this.
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents)
    grid = Flux2KleinPipeline._unpack_latents_with_ids(
        steps[-1], latent_ids, args.height // 16, args.width // 16
    )
    bn = pipe.vae.bn
    mean = bn.running_mean.cpu().view(1, -1, 1, 1).to(grid.dtype)
    std = torch.sqrt(
        bn.running_var.cpu().view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(grid.dtype)
    tensors["vae_input"] = Flux2KleinPipeline._unpatchify_latents(grid * std + mean)
    write(
        args.out,
        tensors,
        {
            "kind": "pipeline",
            "dtype": args.dtype,
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "device": args.device,
        },
    )
    # diffusers' numpy_to_pil quantization: round(x * 255) as uint8.
    pixels = (image[0].permute(1, 2, 0).float().numpy() * 255).round().astype("uint8")
    from PIL import Image

    Image.fromarray(pixels).save(args.out.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="diffusers checkpoint directory"
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="pipeline only; CPU bf16 runs ~7 min per 1024x1024 step",
    )
    parser.add_argument("--out", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    te = sub.add_parser("text-encoder")
    te.add_argument("--prompt", required=True)
    te.set_defaults(run=text_encoder)

    dit = sub.add_parser("transformer")
    dit.add_argument(
        "--text", type=Path, required=True, help="a text-encoder reference"
    )
    dit.add_argument("--seed", type=int, default=0)
    dit.add_argument("--height", type=int, default=512)
    dit.add_argument("--width", type=int, default=512)
    dit.add_argument(
        "--timestep",
        type=float,
        default=1000.0,
        help="scheduler timestep, sigma * 1000",
    )
    dit.set_defaults(run=transformer)

    dec = sub.add_parser("vae")
    dec.add_argument("--latents", type=Path, required=True)
    dec.add_argument("--key", default="latents")
    dec.set_defaults(run=vae)

    full = sub.add_parser("pipeline")
    full.add_argument("--prompt", required=True)
    full.add_argument("--seed", type=int, default=0)
    full.add_argument("--height", type=int, default=1024)
    full.add_argument("--width", type=int, default=1024)
    full.add_argument("--steps", type=int, default=4)
    full.set_defaults(run=pipeline)

    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="CPU threads; the default leaves room for builds on a shared machine",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    args.run(args)


if __name__ == "__main__":
    main()
