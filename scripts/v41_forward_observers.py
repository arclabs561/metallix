"""Scoped source-graph observers for the bounded V4.1 forward capture.

All wrappers call the pinned source methods or injected numerical kernels
unchanged.  They retain exact storage at selected boundaries and restore every
patched instance/global binding when the capture scope exits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

TensorRecord = Callable[..., dict[str, Any]]
ObjectRecord = Callable[..., object]


class _IndexerState:
    """Per-call storage used only while layer four's source Indexer executes."""

    def __init__(self, tensor_record: TensorRecord) -> None:
        self.tensor_record = tensor_record
        self.active = False
        self.recording = False
        self.weights_proj_output: torch.Tensor | None = None
        self.einsum_output: torch.Tensor | None = None
        self.weighted_per_head_output: torch.Tensor | None = None
        self.candidate_mask: torch.Tensor | None = None
        self.operations: dict[str, object] = {}

    def begin(self) -> None:
        if self.active:
            raise RuntimeError("layer-four Indexer observer re-entered")
        self.active = True
        self.weights_proj_output = None
        self.einsum_output = None
        self.weighted_per_head_output = None
        self.candidate_mask = None
        self.operations = {}

    def record(self, name: str, value: torch.Tensor) -> None:
        if not self.active or self.recording or name in self.operations:
            return
        self.recording = True
        try:
            self.operations[name] = self.tensor_record(value, include_storage=True)
        finally:
            self.recording = False

    def end(self) -> dict[str, object]:
        if not self.active:
            raise RuntimeError("layer-four Indexer observer ended while inactive")
        self.active = False
        return self.operations


class _IndexerDispatch(TorchDispatchMode):
    """Observe source tensor operators without copying Indexer.forward logic."""

    def __init__(self, state: _IndexerState) -> None:
        super().__init__()
        self.state = state

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        result = func(*args, **({} if kwargs is None else kwargs))
        if not self.state.active or self.state.recording:
            return result
        name = str(func)
        if "relu_" in name and result is self.state.einsum_output:
            # In-place source ReLU changes the einsum output.  The pre-ReLU
            # value was captured by the graph-local einsum proxy already.
            self.state.record("scores_after_relu", result)
        elif "mul" in name and isinstance(result, torch.Tensor) and args:
            if args[0] is self.state.weights_proj_output:
                self.state.record("scaled_weights", result)
            elif args[0] is self.state.einsum_output and result.ndim == 4:
                self.state.weighted_per_head_output = result
                self.state.record("scores_weighted_per_head", result)
        elif "sum.dim_IntList" in name and isinstance(result, torch.Tensor):
            if args and args[0] is self.state.weighted_per_head_output:
                self.state.record("scores_after_head_sum", result)
        elif "masked_fill_" in name and isinstance(result, torch.Tensor):
            self.state.record("scores_after_causal_mask", result)
        elif "masked_fill" in name and isinstance(result, torch.Tensor):
            candidate = self.state.candidate_mask
            mask = args[1] if len(args) > 1 else None
            if (
                candidate is not None
                and isinstance(mask, torch.Tensor)
                and mask.shape == candidate.shape
            ):
                self.state.record("scores_after_candidate_mask", result)
        return result


class _GraphTorchProxy:
    """Observe the source module's einsum name without mutating global torch."""

    def __init__(self, real_torch: ModuleType, state: _IndexerState) -> None:
        self._real_torch = real_torch
        self._state = state

    def __getattr__(self, name: str) -> object:
        return getattr(self._real_torch, name)

    def einsum(self, equation: str, *operands: torch.Tensor) -> torch.Tensor:
        output = self._real_torch.einsum(equation, *operands)
        if self._state.active and equation == "bshd,btd->bsht":
            self._state.einsum_output = output
            self._state.record("scores_einsum", output)
        return output


def tracing_kernel_bundle(
    kernels: ModuleType,
    kernel_names: tuple[str, ...],
    tensor_record: TensorRecord,
) -> tuple[ModuleType, list[dict[str, object]], list[dict[str, object]]]:
    """Expose the CPU backend while observing HC and sparse-attention calls."""
    bundle = ModuleType("_metallix_v41_tracing_kernels")
    hc_records: list[dict[str, object]] = []
    sparse_records: list[dict[str, object]] = []
    for name in kernel_names:
        setattr(bundle, name, getattr(kernels, name))

    def traced_hc_split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = kernels.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        pre, post, comb = output
        identity = torch.eye(hc_mult, dtype=comb.dtype, device=comb.device)
        hc_records.append(
            {
                "inputs": {
                    "mixes": tensor_record(mixes, include_storage=True),
                    "hc_scale": tensor_record(hc_scale, include_storage=True),
                    "hc_base": tensor_record(hc_base, include_storage=True),
                    "hc_mult": hc_mult,
                    "sinkhorn_iters": sinkhorn_iters,
                    "eps": eps,
                },
                "outputs": {
                    "pre": tensor_record(pre, include_storage=True),
                    "post": tensor_record(post, include_storage=True),
                    "comb": tensor_record(comb, include_storage=True),
                },
                "nontrivial": {
                    "pre_varies": bool((pre.float().amax() - pre.float().amin()) > 0),
                    "post_varies": bool(
                        (post.float().amax() - post.float().amin()) > 0
                    ),
                    "comb_not_identity": not bool((comb == identity).all()),
                },
            }
        )
        return output

    def traced_sparse_attn(
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        idxs: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        output = kernels.sparse_attn(q, kv, sink, idxs, scale)
        # Attention.forward inverse-rotates this result in place after return.
        sparse_records.append(
            {
                "inputs": {
                    "q_after_rope": tensor_record(q, include_storage=True),
                    "kv": tensor_record(kv, include_storage=True),
                    "sink": tensor_record(sink, include_storage=True),
                    "indices": tensor_record(idxs, include_storage=True),
                    "softmax_scale": scale,
                },
                "output_pre_inverse_rope": tensor_record(
                    output.clone(), include_storage=True
                ),
            }
        )
        return output

    bundle.hc_split_sinkhorn = traced_hc_split_sinkhorn
    bundle.sparse_attn = traced_sparse_attn
    return bundle, hc_records, sparse_records


@contextmanager
def hooks_for(
    model: torch.nn.Module,
    graph: ModuleType,
    tensor_record: TensorRecord,
    object_record: ObjectRecord,
    max_hook_records: int,
) -> Iterator[dict[str, object]]:
    """Capture selected source boundaries and restore all observer bindings."""
    records: dict[str, object] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layer_four = model.layers[4]
    layer_four_attention = layer_four.attn
    layer_four_indexer = layer_four_attention.indexer
    if layer_four_indexer is None:
        raise RuntimeError("layer four must have an Indexer in the fixed capture graph")
    index_state = _IndexerState(tensor_record)

    def instance_method(
        module: object, name: str
    ) -> tuple[bool, object | None, object]:
        return name in module.__dict__, module.__dict__.get(name), getattr(module, name)

    had_hc_mixes, prior_hc_mixes, original_hc_mixes = instance_method(
        layer_four, "hc_mixes"
    )
    had_window, prior_window, original_window = instance_method(
        layer_four_attention, "_window_kv"
    )
    had_compress, prior_compress, original_compress = instance_method(
        layer_four_attention, "_compress_kv"
    )
    had_indexer_forward, prior_indexer_forward, original_indexer_forward = (
        instance_method(layer_four_indexer, "forward")
    )
    original_torch = graph.torch
    original_rotary = graph.apply_rotary_emb
    original_fp4_quant = graph.fp4_act_quant

    def cap_guard(name: str) -> None:
        if name not in records and len(records) >= max_hook_records:
            raise RuntimeError("hook receipt cap reached")

    def capture(name: str):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            cap_guard(name)
            records[name] = object_record(output, include_storage=True)

        return hook

    def capture_input(name: str, *, exactly_one: bool):
        def hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            cap_guard(name)
            if not inputs or (exactly_one and len(inputs) != 1):
                raise RuntimeError(
                    f"unexpected source inputs for {name}: got {len(inputs)}"
                )
            records[name] = object_record(inputs[0], include_storage=True)

        return hook

    def capture_block_input(
        _module: torch.nn.Module, inputs: tuple[object, ...]
    ) -> None:
        cap_guard("layers.4.block_input")
        if len(inputs) != 4:
            raise RuntimeError(
                f"expected four source inputs for layers.4 Block.forward, got {len(inputs)}"
            )
        records["layers.4.block_input"] = {
            "residual": object_record(inputs[0], include_storage=True),
            "incoming_pre": object_record(inputs[2], include_storage=True),
        }

    for name, module in model.named_modules():
        pieces = name.split(".")
        selected = (
            name in {"embed", "engram_hash", "norm", "head"}
            or (len(pieces) == 2 and pieces[0] == "layers")
            or (
                len(pieces) == 3
                and pieces[0] == "layers"
                and pieces[2] in {"engram", "attn", "ffn"}
            )
            or (
                len(pieces) == 4
                and pieces[:3] == ["layers", pieces[1], "attn"]
                and pieces[3] in {"compressor", "indexer"}
            )
            or (
                len(pieces) == 4
                and pieces[:3] == ["layers", pieces[1], "ffn"]
                and pieces[3] == "gate"
            )
        )
        if selected:
            handles.append(module.register_forward_hook(capture(name)))
        if name == "norm":
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("norm_input", exactly_one=True)
                )
            )
        if name == "layers.4.ffn":
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("layers.4.ffn_input", exactly_one=False)
                )
            )
        if name == "layers.4.attn":
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("layers.4.attention_input", exactly_one=False)
                )
            )
        if name in {"layers.4.attn.wq_a", "layers.4.attn.q_norm", "layers.4.attn.wq_b"}:
            handles.append(module.register_forward_hook(capture(name)))
        if name == "layers.4.attn.wo_b":
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("layers.4.attn.wo_b_input", exactly_one=True)
                )
            )
        if name == "layers.4.ffn_norm":
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("layers.4.ffn_collapsed", exactly_one=True)
                )
            )
        if name == "layers.4.attn.indexer.weights_proj":

            def capture_weights(
                _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "layer-four Indexer weights projection did not return a tensor"
                    )
                index_state.weights_proj_output = output
                index_state.record("weights_proj_output", output)

            handles.append(module.register_forward_hook(capture_weights))
        if name == "layers.4":
            handles.append(module.register_forward_pre_hook(capture_block_input))

    def observed_hc_mixes(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hc_fn is layer_four.hc_ffn_fn:
            cap_guard("layers.4.after_attention_residual")
            if "layers.4.after_attention_residual" in records:
                raise RuntimeError("layer-four FFN HC residual was observed twice")
            records["layers.4.after_attention_residual"] = object_record(
                x, include_storage=True
            )
        return original_hc_mixes(x, hc_fn, hc_scale, hc_base)

    def observed_window_kv(
        x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cap_guard("layers.4.attn.window")
        if "layers.4.attn.window" in records:
            raise RuntimeError("layer-four window KV was observed twice")
        window_kv, window_indices = original_window(x, freqs_cis, start_pos)
        prepared = (
            window_kv
            if start_pos == 0
            else layer_four_attention.window_kv_cache[
                : x.size(0),
                start_pos % layer_four_attention.window_size : start_pos
                % layer_four_attention.window_size
                + 1,
            ]
        )
        records["layers.4.attn.window"] = {
            "prepared_window_kv": object_record(prepared, include_storage=True),
            "window_kv": object_record(window_kv, include_storage=True),
            "indices": object_record(window_indices, include_storage=True),
            "ring_after": object_record(
                layer_four_attention.window_kv_cache, include_storage=True
            ),
        }
        return window_kv, window_indices

    def observed_compress_kv(
        x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cap_guard("layers.4.attn.compressed")
        if "layers.4.attn.compressed" in records:
            raise RuntimeError("layer-four compressed KV was observed twice")
        compressed_kv, compressed_indices = original_compress(x, qr, start_pos, offset)
        records["layers.4.attn.compressed"] = {
            "borrowed_kv": object_record(compressed_kv, include_storage=True),
            "indices": object_record(compressed_indices, include_storage=True),
        }
        return compressed_kv, compressed_indices

    def observed_indexer_forward(
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        cap_guard("layers.4.attn.indexer_observation")
        if "layers.4.attn.indexer_observation" in records:
            raise RuntimeError("layer-four Indexer was observed twice")
        ratio = layer_four_indexer.compress_ratio
        end_pos = start_pos + x.size(1)
        shared_index_k = graph.shared_attn.index_k
        candidate_mask = graph.shared_attn.candidates
        if shared_index_k is None:
            raise RuntimeError(
                "layer-four Indexer has no source-published shared index K"
            )
        if candidate_mask is None:
            raise RuntimeError("layer-four Indexer has no source candidate mask")
        # Clone receipt storage before the source Indexer can produce any later
        # call's cache state.  This is the prefix it actually reads, not compressed KV.
        inputs = {
            "x": object_record(x, include_storage=True),
            "qr": object_record(qr, include_storage=True),
            "latent": object_record(latent, include_storage=True),
            "start_pos": start_pos,
            "offset": offset,
            "shared_index_k_prefix": object_record(
                shared_index_k[: x.size(0), : end_pos // ratio], include_storage=True
            ),
            "candidate_mask": object_record(candidate_mask, include_storage=True),
        }
        index_state.begin()
        index_state.candidate_mask = candidate_mask
        try:
            with _IndexerDispatch(index_state):
                output = original_indexer_forward(x, qr, latent, start_pos, offset)
        finally:
            operations = index_state.end()
        records["layers.4.attn.indexer_observation"] = {
            "inputs": inputs,
            "operations": operations,
            "output_indices": object_record(output, include_storage=True),
        }
        return output

    def observed_rotary(
        x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
    ) -> None:
        original_rotary(x, freqs_cis, inverse)

    def observed_fp4_quant(
        x: torch.Tensor,
        block_size: int = 32,
        inplace: bool = False,
        scale_dtype: torch.dtype = torch.float8_e8m0fnu,
    ) -> torch.Tensor:
        output = original_fp4_quant(x, block_size, inplace, scale_dtype)
        if index_state.active:
            if not inplace:
                raise RuntimeError("source Indexer query quantization must be in place")
            index_state.record("q_after_rope_fp4", output)
        return output

    layer_four.hc_mixes = observed_hc_mixes
    layer_four_attention._window_kv = observed_window_kv
    layer_four_attention._compress_kv = observed_compress_kv
    layer_four_indexer.forward = observed_indexer_forward
    graph.torch = _GraphTorchProxy(original_torch, index_state)
    graph.apply_rotary_emb = observed_rotary
    graph.fp4_act_quant = observed_fp4_quant
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()
        _restore_instance(layer_four, "hc_mixes", had_hc_mixes, prior_hc_mixes)
        _restore_instance(layer_four_attention, "_window_kv", had_window, prior_window)
        _restore_instance(
            layer_four_attention, "_compress_kv", had_compress, prior_compress
        )
        _restore_instance(
            layer_four_indexer,
            "forward",
            had_indexer_forward,
            prior_indexer_forward,
        )
        graph.torch = original_torch
        graph.apply_rotary_emb = original_rotary
        graph.fp4_act_quant = original_fp4_quant


def _restore_instance(
    module: object, name: str, had_instance_attr: bool, prior: object | None
) -> None:
    if had_instance_attr:
        setattr(module, name, prior)
    else:
        delattr(module, name)


def hc_step_receipt(
    records: list[dict[str, object]], n_layers: int
) -> list[dict[str, object]]:
    expected = 2 * n_layers
    if len(records) != expected:
        raise RuntimeError(
            f"expected {expected} HC kernel calls for {n_layers} blocks, got {len(records)}"
        )
    labeled: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not all(record["nontrivial"].values()):
            raise RuntimeError(
                f"HC kernel call {index} lacks nontrivial coefficient evidence"
            )
        labeled.append(
            {
                "layer_id": index // 2,
                "sublayer": "attention" if index % 2 == 0 else "ffn",
                **record,
            }
        )
    return labeled


def sparse_step_receipt(
    records: list[dict[str, object]], n_layers: int
) -> list[dict[str, object]]:
    if len(records) != n_layers:
        raise RuntimeError(
            f"expected {n_layers} sparse attention calls, got {len(records)}"
        )
    return [{"layer_id": index, **record} for index, record in enumerate(records)]
