# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""CPU numerical backend for DeepSeek-V4.1-Flash's pinned kernel interface.

This module intentionally mirrors the six imports from ``inference/kernel.py``
so a controlled CPU source-forward capture can inject it in place of TileLang.
It preserves the logical FP8 / E2M1 / scale boundaries and the source's
per-group scale placement.  It does *not* execute TileLang, CUDA, Tensor Core,
or hardware-specific reduction sequences, so matching results are graph
composition evidence rather than upstream GPU numerical parity.

CPU Torch 2.13 exposes ``float4_e2m1fn_x2`` storage but not ordinary FP4
arithmetic. ``fp4_gemm`` accepts that storage dtype or an explicit ``uint8``
view, then expands the pinned low-nibble-first runtime layout. Callers that
cannot provide that representation receive a hard error rather than an
identity or silently dequantized substitute.
"""

from __future__ import annotations

import math
from typing import Final

import torch

FP8_MAX: Final = 448.0
FP4_MAX: Final = 6.0
FP8_MAX_INV: Final = 1.0 / FP8_MAX
FP4_MAX_INV: Final = 1.0 / FP4_MAX
_E2M1_MAGNITUDES: Final = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class CpuKernelError(RuntimeError):
    """A pinned-kernel input has no faithful CPU representation here."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CpuKernelError(message)


def _require_cpu(tensor: torch.Tensor, name: str) -> None:
    _require(tensor.device.type == "cpu", f"{name} must be a CPU tensor")


def _require_contiguous(tensor: torch.Tensor, name: str) -> None:
    _require(tensor.is_contiguous(), f"{name} must be contiguous")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(tensor.float()).all()), f"{name} must be finite")


def _rows(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    _require(x.ndim >= 1, "input must have a reduction dimension")
    width = x.size(-1)
    _require(width > 0, "reduction dimension must be positive")
    return x.reshape(-1, width), tuple(x.shape[:-1]), width


def _scale_storage(scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Narrow scales exactly at the source storage boundary."""
    if dtype not in (torch.float32, torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise CpuKernelError(f"unsupported scale dtype {dtype}")
    return scales.to(dtype)


def _source_pow2_ceil(values: torch.Tensor) -> torch.Tensor:
    """Pinned ``fast_log2_ceil`` / ``fast_pow2`` for positive FP32 values.

    This deliberately uses the source's raw IEEE exponent/mantissa rule rather
    than ``log2``.  The two differ at power-of-two neighbours, which is a
    scale-boundary change rather than an implementation detail.
    """
    _require_finite(values, "scale inputs")
    _require(bool((values > 0).all()), "scale inputs must be positive")
    values = values.float().contiguous()
    bits = values.view(torch.int32).to(torch.int64)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7F_FFFF
    ceil_log2 = exponent - 127 + (mantissa != 0).to(torch.int64)
    result_bits = ((ceil_log2 + 127) << 23).to(torch.int32)
    return result_bits.view(torch.float32)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pinned blockwise E4M3 activation quantization on CPU.

    The returned activation is logical ``float8_e4m3fn`` and scales retain the
    requested storage dtype.  With ``inplace=True`` the source's quantize then
    dequantize-to-BF16 behavior is reproduced and the original tensor is
    updated.
    """
    _require_cpu(x, "x")
    _require(x.dtype == torch.bfloat16, "act_quant requires BF16 activations")
    _require(block_size > 0, "block_size must be positive")
    matrix, leading, width = _rows(x.contiguous())
    _require(width % block_size == 0, "reduction dimension must divide block_size")
    _require_finite(matrix, "x")
    groups = width // block_size
    grouped = matrix.float().reshape(-1, groups, block_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    raw_scales = (
        _source_pow2_ceil(amax * FP8_MAX_INV)
        if scale_fmt is not None
        else amax * FP8_MAX_INV
    )
    scales = _scale_storage(raw_scales, scale_dtype).reshape(-1, groups)
    scale_values = scales.float().unsqueeze(-1)
    quantized = (
        (grouped / scale_values).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    )
    if inplace:
        reconstructed = (
            (quantized.float() * scale_values).reshape_as(matrix).to(torch.bfloat16)
        )
        x.copy_(reconstructed.reshape_as(x))
        return x
    return quantized.reshape_as(x), scales.reshape(*leading, groups)


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Software E2M1 RNE, including the source tie-to-even code rule."""
    _require_cpu(values, "FP4 values")
    _require_finite(values, "FP4 values")
    flat = values.float().reshape(-1)
    magnitudes = flat.abs().unsqueeze(-1)
    levels = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32).unsqueeze(0)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    code_range = torch.arange(8)
    parity = torch.where(candidates, code_range % 2, 2)
    preferred_parity = parity.amin(dim=-1, keepdim=True)
    chosen = torch.where(candidates & (parity == preferred_parity), code_range, 8).amin(
        dim=-1
    )
    # An exact odd code is still selected when it is the sole nearest code;
    # parity only resolves a genuine midpoint tie.
    _require(bool((chosen < 8).all()), "could not choose an E2M1 code")
    signed = chosen | torch.where(flat.signbit(), 8, 0)
    return signed.to(torch.uint8).reshape_as(values)


def unpack_fp4_e2m1x2(packed: torch.Tensor) -> torch.Tensor:
    """Expand packed low-nibble-first E2M1x2 bytes into logical FP32 values."""
    _require_cpu(packed, "packed FP4")
    _require(packed.dtype == torch.uint8, "packed FP4 weights must be uint8")
    _require(packed.ndim >= 1 and packed.size(-1) > 0, "packed FP4 must be nonempty")
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)[
        (codes & 0x07).long()
    ]
    return torch.where((codes & 0x08) != 0, -magnitudes, magnitudes).flatten(-2)


def _packed_fp4_bytes(weight: torch.Tensor) -> torch.Tensor:
    """Return a byte view of the only two accepted CPU FP4 representations."""
    _require_cpu(weight, "packed FP4 weight")
    _require_contiguous(weight, "packed FP4 weight")
    if weight.dtype == torch.float4_e2m1fn_x2:
        return weight.view(torch.uint8)
    _require(
        weight.dtype == torch.uint8, "CPU FP4 weights must be E2M1x2 or packed uint8"
    )
    return weight


def pack_fp4_e2m1x2(values: torch.Tensor) -> torch.Tensor:
    """Pack logical values using software E2M1 RNE for controlled fixtures only."""
    _require_cpu(values, "logical FP4 values")
    _require(values.ndim >= 1 and values.size(-1) % 2 == 0, "FP4 width must be even")
    codes = _e2m1_codes(values)
    pairs = codes.reshape(*values.shape[:-1], -1, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pinned FP4 activation quantization, with a faithful CPU inplace path.

    Non-inplace packed-FP4 output cannot be represented by CPU Torch 2.13 and
    deliberately raises.  The V4.1 source-forward path only uses inplace mode.
    """
    _require_cpu(x, "x")
    _require(x.dtype == torch.bfloat16, "fp4_act_quant requires BF16 activations")
    _require(
        scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn),
        "unsupported FP4 scale dtype",
    )
    _require(inplace, "CPU backend supports fp4_act_quant only in inplace mode")
    _require(block_size > 0, "block_size must be positive")
    matrix, _, width = _rows(x.contiguous())
    _require(width % block_size == 0, "reduction dimension must divide block_size")
    _require_finite(matrix, "x")
    groups = width // block_size
    grouped = matrix.float().reshape(-1, groups, block_size)
    if scale_dtype == torch.float8_e4m3fn:
        raw_scales = grouped.abs().amax(dim=-1).clamp_min(6 * 2**-9) / FP4_MAX
    else:
        raw_scales = _source_pow2_ceil(
            grouped.abs().amax(dim=-1).clamp_min(6 * 2**-126) * FP4_MAX_INV
        )
    scales = _scale_storage(raw_scales, scale_dtype)
    scale_values = scales.float().unsqueeze(-1)
    codes = _e2m1_codes((grouped / scale_values).clamp(-FP4_MAX, FP4_MAX))
    reconstructed = unpack_fp4_e2m1x2(
        codes.reshape(-1, groups, block_size // 2, 2)[..., 0]
        | (codes.reshape(-1, groups, block_size // 2, 2)[..., 1] << 4)
    ).reshape_as(grouped)
    x.copy_((reconstructed * scale_values).reshape_as(x).to(torch.bfloat16))
    return x


def _matrix_output_dtype() -> torch.dtype:
    output_dtype = torch.get_default_dtype()
    _require(
        output_dtype in (torch.float32, torch.bfloat16),
        "default dtype must be FP32 or BF16",
    )
    return output_dtype


def _validate_gemm_activation(
    a: torch.Tensor, a_s: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    _require_cpu(a, "a")
    _require_cpu(a_s, "a_s")
    _require_contiguous(a, "a")
    _require_contiguous(a_s, "a_s")
    _require(a.dtype == torch.float8_e4m3fn, "a must be logical E4M3 FP8")
    _require(block_size in (32, 128), "block_size must be 32 or 128")
    matrix, leading, width = _rows(a)
    _require(width % block_size == 0, "activation width must divide block_size")
    rows = matrix.size(0)
    _require(
        a_s.numel() == rows * (width // block_size), "activation scale shape mismatch"
    )
    _require_finite(matrix, "a")
    _require_finite(a_s, "a_s")
    return matrix.float(), a_s.float().reshape(rows, -1), leading, rows, width


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """FP8 activation × FP8 weight GEMM with source per-group scaling."""
    del scale_dtype  # Scale storage is already represented by `a_s`/`b_s`.
    _require_cpu(b, "b")
    _require_cpu(b_s, "b_s")
    _require_contiguous(b, "b")
    _require_contiguous(b_s, "b_s")
    _require(b.dtype == torch.float8_e4m3fn, "b must be logical E4M3 FP8")
    a_values, a_scales, leading, _, width = _validate_gemm_activation(
        a, a_s, block_size
    )
    _require(
        b.ndim == 2 and b.size(1) == width,
        "FP8 weight shape must be [outputs, reduction]",
    )
    outputs = b.size(0)
    expected_scales = ((outputs + block_size - 1) // block_size, width // block_size)
    _require(tuple(b_s.shape) == expected_scales, "FP8 weight scale shape mismatch")
    _require_finite(b, "b")
    _require_finite(b_s, "b_s")
    result = torch.zeros((a_values.size(0), outputs), dtype=torch.float32)
    b_values = b.float()
    b_scales = b_s.float()
    for group in range(width // block_size):
        start = group * block_size
        stop = start + block_size
        dot = a_values[:, start:stop] @ b_values[:, start:stop].transpose(0, 1)
        result += (
            dot
            * a_scales[:, group : group + 1]
            * b_scales[torch.arange(outputs) // block_size, group]
        )
    _require_finite(result, "FP8 GEMM result")
    return result.reshape(*leading, outputs).to(_matrix_output_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """FP8 activation × packed E2M1 weight GEMM with source scaling placement."""
    del scale_dtype
    _require_cpu(b, "packed FP4 weight")
    _require_cpu(b_s, "FP4 weight scales")
    _require_contiguous(b_s, "FP4 weight scales")
    a_values, a_scales, leading, _, width = _validate_gemm_activation(
        a, a_s, act_block_size
    )
    _require(width % 32 == 0, "FP4 reduction width must divide 32")
    packed = _packed_fp4_bytes(b)
    _require(
        packed.ndim == 2 and packed.size(1) * 2 == width,
        "packed FP4 weight shape mismatch",
    )
    outputs = packed.size(0)
    _require(
        tuple(b_s.shape) == (outputs, width // 32), "FP4 weight scale shape mismatch"
    )
    _require_finite(b_s, "FP4 weight scales")
    b_values = unpack_fp4_e2m1x2(packed)
    b_scales = b_s.float()
    result = torch.zeros((a_values.size(0), outputs), dtype=torch.float32)
    for block in range(width // 32):
        start = block * 32
        stop = start + 32
        activation_group = start // act_block_size
        dot = a_values[:, start:stop] @ b_values[:, start:stop].transpose(0, 1)
        result += (
            dot
            * a_scales[:, activation_group : activation_group + 1]
            * b_scales[:, block]
        )
    _require_finite(result, "FP4 GEMM result")
    return result.reshape(*leading, outputs).to(_matrix_output_dtype())


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse gathered attention with duplicate slots and denominator-only sink."""
    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
    ):
        _require_cpu(tensor, name)
    _require(
        q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16,
        "attention q and kv must be BF16",
    )
    _require(topk_idxs.dtype == torch.int32, "topk_idxs must be int32")
    _require(
        q.ndim == 4 and kv.ndim == 3, "attention ranks must be q=[B,S,H,D], kv=[B,N,D]"
    )
    batches, queries, heads, width = q.shape
    _require(
        tuple(kv.shape[:1]) == (batches,) and kv.size(2) == width, "KV shape mismatch"
    )
    _require(
        tuple(topk_idxs.shape[:2]) == (batches, queries), "topk index shape mismatch"
    )
    _require(tuple(attn_sink.shape) == (heads,), "attention sink shape mismatch")
    _require(math.isfinite(softmax_scale), "softmax_scale must be finite")
    _require_finite(q, "q")
    _require_finite(kv, "kv")
    _require_finite(attn_sink, "attn_sink")
    key_count = kv.size(1)
    _require(
        bool(((topk_idxs >= -1) & (topk_idxs < key_count)).all()),
        "topk index out of range",
    )
    output = torch.zeros_like(q)
    for batch in range(batches):
        for query in range(queries):
            indices = topk_idxs[batch, query]
            maximum = torch.full((heads,), -1e30, dtype=torch.float32)
            sum_exp = torch.zeros((heads,), dtype=torch.float32)
            accumulator = torch.zeros((heads, width), dtype=torch.float32)
            for start in range(0, indices.numel(), 64):
                block_indices = indices[start : start + 64]
                valid = block_indices >= 0
                gathered = torch.zeros(
                    (block_indices.numel(), width), dtype=torch.bfloat16
                )
                if bool(valid.any()):
                    gathered[valid] = kv[batch, block_indices[valid]]
                scores = (
                    q[batch, query].float() @ gathered.float().transpose(0, 1)
                ) * softmax_scale
                scores[:, ~valid] = -torch.inf
                previous_maximum = maximum
                maximum = torch.maximum(maximum, scores.amax(dim=-1))
                rescale = torch.exp(previous_maximum - maximum)
                weights = torch.exp(scores - maximum.unsqueeze(-1))
                sum_exp = sum_exp * rescale + weights.sum(dim=-1)
                # The source explicitly stages probabilities through BF16 before
                # the value GEMM, while the denominator remains FP32.
                accumulator = accumulator * rescale.unsqueeze(-1) + (
                    weights.to(torch.bfloat16).float() @ gathered.float()
                )
            denominator = sum_exp + torch.exp(attn_sink.float() - maximum)
            output[batch, query] = (accumulator / denominator.unsqueeze(-1)).to(
                torch.bfloat16
            )
    return output


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split hyper-connection coefficients and run the pinned Sinkhorn sequence."""
    for name, tensor in (
        ("mixes", mixes),
        ("hc_scale", hc_scale),
        ("hc_base", hc_base),
    ):
        _require_cpu(tensor, name)
        _require(tensor.dtype == torch.float32, f"{name} must be FP32")
        _require_finite(tensor, name)
    _require(mixes.ndim == 3, "mixes must have shape [B,S,(2+HC)*HC]")
    _require(
        hc_mult > 0 and sinkhorn_iters > 0 and eps > 0 and math.isfinite(eps),
        "invalid Sinkhorn parameters",
    )
    width = (2 + hc_mult) * hc_mult
    _require(
        mixes.size(-1) == width and hc_base.numel() == width,
        "hyper-connection shape mismatch",
    )
    _require(hc_scale.numel() == 3, "hc_scale must have three values")
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(*mixes.shape[:2], hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    _require_finite(pre, "hyper-connection pre")
    _require_finite(post, "hyper-connection post")
    _require_finite(comb, "hyper-connection comb")
    return pre, post, comb
