"""Scoped source-graph observers for the bounded V4.1 forward capture.

All wrappers call the pinned source methods or injected numerical kernels
unchanged.  They retain exact storage at selected boundaries and restore every
patched instance/global binding when the capture scope exits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum, auto
from types import ModuleType
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

TensorRecord = Callable[..., dict[str, Any]]
ObjectRecord = Callable[..., object]


class IndexerRole(Enum):
    """The two fixed source Indexers observed by this bounded capture."""

    PRODUCER = auto()
    CONSUMER = auto()


class _QuantizationPhase(Enum):
    """Source identity of the active Indexer FP4 operation."""

    KEY = auto()
    QUERY = auto()


class _IndexerState:
    """Per-call storage for one fixed source Indexer role."""

    def __init__(self, role: IndexerRole, tensor_record: TensorRecord) -> None:
        self.role = role
        self.tensor_record = tensor_record
        self.active = False
        self.recording = False
        self.weights_proj_output: torch.Tensor | None = None
        self.einsum_output: torch.Tensor | None = None
        self.weighted_per_head_output: torch.Tensor | None = None
        self.summed_score_output: torch.Tensor | None = None
        self.causal_score_output: torch.Tensor | None = None
        self.index_key_operand: torch.Tensor | None = None
        self.shared_index_k_prefix: dict[str, Any] | None = None
        self.quantization_phase: _QuantizationPhase | None = None
        self.operations: dict[str, object] = {}

    def begin(self) -> None:
        if self.active:
            raise RuntimeError(f"{self.role.name.lower()} Indexer observer re-entered")
        self.active = True
        self.weights_proj_output = None
        self.einsum_output = None
        self.weighted_per_head_output = None
        self.summed_score_output = None
        self.causal_score_output = None
        self.index_key_operand = None
        self.shared_index_k_prefix = None
        self.quantization_phase = None
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
            raise RuntimeError(
                f"{self.role.name.lower()} Indexer observer ended while inactive"
            )
        self.active = False
        return self.operations

    def set_quantization_phase(self, phase: _QuantizationPhase) -> None:
        if not self.active:
            return
        if phase is _QuantizationPhase.KEY and self.role is not IndexerRole.PRODUCER:
            raise RuntimeError(
                "only the fixed producer Indexer may quantize index keys"
            )
        self.quantization_phase = phase

    def record_score_einsum(
        self, output: torch.Tensor, index_key_operand: torch.Tensor
    ) -> None:
        self.einsum_output = output
        self.index_key_operand = index_key_operand
        self.recording = True
        try:
            self.shared_index_k_prefix = self.tensor_record(
                index_key_operand, include_storage=True
            )
        finally:
            self.recording = False
        self.record("scores_einsum", output)


class _IndexerStates:
    """Fixed producer/consumer state registry for one graph proxy."""

    def __init__(self, tensor_record: TensorRecord) -> None:
        self._states = {
            IndexerRole.PRODUCER: _IndexerState(IndexerRole.PRODUCER, tensor_record),
            IndexerRole.CONSUMER: _IndexerState(IndexerRole.CONSUMER, tensor_record),
        }

    def for_role(self, role: IndexerRole) -> _IndexerState:
        return self._states[role]

    def active(self) -> _IndexerState | None:
        active = [state for state in self._states.values() if state.active]
        if len(active) > 1:
            raise RuntimeError(
                "fixed producer and consumer Indexer observers overlapped"
            )
        return active[0] if active else None


class _IndexerDispatch(TorchDispatchMode):
    """Observe source tensor operators without copying Indexer.forward logic."""

    def __init__(self, states: _IndexerStates) -> None:
        super().__init__()
        self.states = states

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        result = func(*args, **({} if kwargs is None else kwargs))
        state = self.states.active()
        if state is None or state.recording:
            return result
        name = str(func)
        if "relu_" in name and result is state.einsum_output:
            # In-place source ReLU changes the einsum output.  The pre-ReLU
            # value was captured by the graph-local einsum proxy already.
            state.record("scores_after_relu", result)
        elif "mul" in name and isinstance(result, torch.Tensor) and args:
            if args[0] is state.weights_proj_output:
                state.record("scaled_weights", result)
            elif args[0] is state.einsum_output and result.ndim == 4:
                state.weighted_per_head_output = result
                state.record("scores_weighted_per_head", result)
        elif "sum.dim_IntList" in name and isinstance(result, torch.Tensor):
            if args and args[0] is state.weighted_per_head_output:
                state.summed_score_output = result
                state.record("scores_after_head_sum", result)
        elif "masked_fill_" in name and isinstance(result, torch.Tensor):
            if args and args[0] is state.summed_score_output:
                state.causal_score_output = result
                state.record("scores_after_causal_mask", result)
        elif (
            "masked_fill" in name
            and isinstance(result, torch.Tensor)
            and state.role is IndexerRole.CONSUMER
            and args
            and args[0]
            is (
                state.causal_score_output
                if state.causal_score_output is not None
                else state.summed_score_output
            )
        ):
            state.record("scores_after_candidate_mask", result)
        return result


class _GraphTorchProxy:
    """Observe the source module's einsum name without mutating global torch."""

    def __init__(self, real_torch: ModuleType, states: _IndexerStates) -> None:
        self._real_torch = real_torch
        self._states = states

    def __getattr__(self, name: str) -> object:
        return getattr(self._real_torch, name)

    def einsum(self, equation: str, *operands: torch.Tensor) -> torch.Tensor:
        output = self._real_torch.einsum(equation, *operands)
        state = self._states.active()
        if state is not None and equation == "bshd,btd->bsht":
            if len(operands) != 2:
                raise RuntimeError("source Indexer score einsum must have two operands")
            state.record_score_einsum(output, operands[1])
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
    layer_three = model.layers[3]
    layer_three_attention = layer_three.attn
    layer_three_indexer = layer_three_attention.indexer
    layer_four = model.layers[4]
    layer_four_attention = layer_four.attn
    layer_four_indexer = layer_four_attention.indexer
    if layer_three_indexer is None or layer_four_indexer is None:
        raise RuntimeError(
            "fixed layer three/four capture graph requires both Indexers"
        )
    states = _IndexerStates(tensor_record)
    producer_state = states.for_role(IndexerRole.PRODUCER)
    consumer_state = states.for_role(IndexerRole.CONSUMER)

    def instance_method(
        module: object, name: str
    ) -> tuple[bool, object | None, object]:
        return name in module.__dict__, module.__dict__.get(name), getattr(module, name)

    had_hc_mixes, prior_hc_mixes, original_hc_mixes = instance_method(
        layer_four, "hc_mixes"
    )
    had_producer_hc_mixes, prior_producer_hc_mixes, _ = instance_method(
        layer_three, "hc_mixes"
    )
    had_window, prior_window, original_window = instance_method(
        layer_four_attention, "_window_kv"
    )
    had_compress, prior_compress, original_compress = instance_method(
        layer_four_attention, "_compress_kv"
    )
    (
        had_producer_window,
        prior_producer_window,
        original_producer_window,
    ) = instance_method(layer_three_attention, "_window_kv")
    (
        had_producer_compress,
        prior_producer_compress,
        original_producer_compress,
    ) = instance_method(layer_three_attention, "_compress_kv")
    had_indexer_forward, prior_indexer_forward, original_indexer_forward = (
        instance_method(layer_four_indexer, "forward")
    )
    (
        had_producer_indexer_forward,
        prior_producer_indexer_forward,
        original_producer_indexer_forward,
    ) = instance_method(layer_three_indexer, "forward")
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

    def capture_block_input(layer_id: int):
        """Retain the source operands whose HC pre-mix feeds attention norm."""
        record_name = f"layers.{layer_id}.block_input"

        def hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            cap_guard(record_name)
            if len(inputs) != 4:
                raise RuntimeError(
                    f"expected four source inputs for {record_name} Block.forward, "
                    f"got {len(inputs)}"
                )
            records[record_name] = {
                "residual": object_record(inputs[0], include_storage=True),
                "incoming_pre": object_record(inputs[2], include_storage=True),
            }

        return hook

    def capture_weights_for(state: _IndexerState):
        def capture_weights(
            _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{state.role.name.lower()} Indexer weights projection did not return a tensor"
                )
            state.weights_proj_output = output
            state.record("weights_proj_output", output)

        return capture_weights

    def mark_quantization_phase(state: _IndexerState, phase: _QuantizationPhase):
        def mark_phase(
            _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{state.role.name.lower()} Indexer quantization boundary did not return a tensor"
                )
            state.set_quantization_phase(phase)

        return mark_phase

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
        if name in {"layers.3.ffn", "layers.4.ffn"}:
            handles.append(
                module.register_forward_pre_hook(
                    capture_input(f"{name}_input", exactly_one=False)
                )
            )
        if name in {"layers.3.attn", "layers.4.attn"}:
            handles.append(
                module.register_forward_pre_hook(
                    capture_input(
                        f"{name.removesuffix('.attn')}.attention_input",
                        exactly_one=False,
                    )
                )
            )
        if name == "layers.3.attn.compressor.wkv":
            handles.append(module.register_forward_hook(capture(name)))
        if name in {
            "layers.3.attn.wq_a",
            "layers.3.attn.q_norm",
            "layers.3.attn.wq_b",
            "layers.4.attn.wq_a",
            "layers.4.attn.q_norm",
            "layers.4.attn.wq_b",
        }:
            handles.append(module.register_forward_hook(capture(name)))
        if name in {"layers.3.attn.wo_b", "layers.4.attn.wo_b"}:
            handles.append(
                module.register_forward_pre_hook(
                    capture_input(
                        f"{name.removesuffix('.wo_b')}.wo_b_input",
                        exactly_one=True,
                    )
                )
            )
        if name in {"layers.3.ffn_norm", "layers.4.ffn_norm"}:
            handles.append(
                module.register_forward_pre_hook(
                    capture_input(
                        f"{name.removesuffix('_norm')}_collapsed", exactly_one=True
                    )
                )
            )
        if name == "layers.3.attn.indexer.weights_proj":
            handles.append(
                module.register_forward_hook(capture_weights_for(producer_state))
            )
        if name == "layers.4.attn.indexer.weights_proj":
            handles.append(
                module.register_forward_hook(capture_weights_for(consumer_state))
            )
        if name == "layers.3.attn.indexer.k_norm":
            handles.append(
                module.register_forward_hook(
                    mark_quantization_phase(producer_state, _QuantizationPhase.KEY)
                )
            )
        if name == "layers.3.attn.indexer.wq_b":
            handles.append(
                module.register_forward_hook(
                    mark_quantization_phase(producer_state, _QuantizationPhase.QUERY)
                )
            )
        if name == "layers.4.attn.indexer.wq_b":
            handles.append(
                module.register_forward_hook(
                    mark_quantization_phase(consumer_state, _QuantizationPhase.QUERY)
                )
            )
        if name in {"layers.3", "layers.4"}:
            layer_id = int(name.removeprefix("layers."))
            handles.append(
                module.register_forward_pre_hook(capture_block_input(layer_id))
            )

    def observed_hc_mixes(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_id = (
            3
            if hc_fn is layer_three.hc_ffn_fn
            else 4
            if hc_fn is layer_four.hc_ffn_fn
            else None
        )
        if layer_id is not None:
            record_name = f"layers.{layer_id}.after_attention_residual"
            cap_guard(record_name)
            if record_name in records:
                raise RuntimeError(
                    f"layer-{layer_id} FFN HC residual was observed twice"
                )
            records[record_name] = object_record(x, include_storage=True)
        return original_hc_mixes(x, hc_fn, hc_scale, hc_base)

    def observed_window_kv_for(
        layer_id: int,
        attention: torch.nn.Module,
        original_window: Callable[..., object],
    ) -> Callable[..., object]:
        record_name = f"layers.{layer_id}.attn.window"

        def observed_window_kv(
            x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cap_guard(record_name)
            if record_name in records:
                raise RuntimeError(f"layer-{layer_id} window KV was observed twice")
            window_kv, window_indices = original_window(x, freqs_cis, start_pos)
            prepared = (
                window_kv
                if start_pos == 0
                else attention.window_kv_cache[
                    : x.size(0),
                    start_pos % attention.window_size : start_pos
                    % attention.window_size
                    + 1,
                ]
            )
            records[record_name] = {
                "prepared_window_kv": object_record(prepared, include_storage=True),
                "window_kv": object_record(window_kv, include_storage=True),
                "indices": object_record(window_indices, include_storage=True),
                "ring_after": object_record(
                    attention.window_kv_cache, include_storage=True
                ),
            }
            return window_kv, window_indices

        return observed_window_kv

    def observed_compress_kv_for(
        layer_id: int, original_compress: Callable[..., object]
    ) -> Callable[..., object]:
        record_name = f"layers.{layer_id}.attn.compressed"

        def observed_compress_kv(
            x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cap_guard(record_name)
            if record_name in records:
                raise RuntimeError(f"layer-{layer_id} compressed KV was observed twice")
            compressed_kv, compressed_indices = original_compress(
                x, qr, start_pos, offset
            )
            records[record_name] = {
                "borrowed_kv": object_record(compressed_kv, include_storage=True),
                "indices": object_record(compressed_indices, include_storage=True),
            }
            return compressed_kv, compressed_indices

        return observed_compress_kv

    def observed_indexer_forward_for(
        role: IndexerRole,
        original_forward: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        state = states.for_role(role)
        layer_id = 3 if role is IndexerRole.PRODUCER else 4
        observation_name = f"layers.{layer_id}.attn.indexer_observation"

        def observed_indexer_forward(
            x: torch.Tensor,
            qr: torch.Tensor,
            latent: torch.Tensor | None,
            start_pos: int,
            offset: int,
        ) -> torch.Tensor:
            cap_guard(observation_name)
            if observation_name in records:
                raise RuntimeError(
                    f"fixed {role.name.lower()} Indexer was observed twice"
                )
            inputs: dict[str, object] = {
                "x": object_record(x, include_storage=True),
                "qr": object_record(qr, include_storage=True),
                "latent": object_record(latent, include_storage=True),
                "start_pos": start_pos,
                "offset": offset,
            }
            if role is IndexerRole.CONSUMER:
                candidate_mask = graph.shared_attn.candidates
                if candidate_mask is None:
                    raise RuntimeError(
                        "layer-four Indexer has no source candidate mask"
                    )
                # Preserve the historical consumer field: it is the source
                # candidate operand present before the consumer scores.
                inputs["candidate_mask"] = object_record(
                    candidate_mask, include_storage=True
                )
            state.begin()
            try:
                with _IndexerDispatch(states):
                    output = original_forward(x, qr, latent, start_pos, offset)
            finally:
                operations = state.end()
            if state.shared_index_k_prefix is None:
                raise RuntimeError(
                    f"fixed {role.name.lower()} Indexer did not execute its score einsum"
                )
            # This receipt comes from the exact second operand at score-einsum
            # time, not from a predicted cache slice before the source call.
            inputs["shared_index_k_prefix"] = state.shared_index_k_prefix
            candidate_mask_after = graph.shared_attn.candidates
            if candidate_mask_after is None:
                raise RuntimeError(
                    f"fixed {role.name.lower()} Indexer did not publish a candidate mask"
                )
            observation = {
                "inputs": inputs,
                "operations": operations,
                "output_indices": object_record(output, include_storage=True),
                "candidate_mask_after": object_record(
                    candidate_mask_after, include_storage=True
                ),
            }
            records[observation_name] = observation
            return output

        return observed_indexer_forward

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
        state = states.active()
        if state is not None:
            if not inplace:
                raise RuntimeError("source Indexer FP4 quantization must be in place")
            if state.quantization_phase is _QuantizationPhase.KEY:
                state.record("k_after_rope_fp4", output)
            elif state.quantization_phase is _QuantizationPhase.QUERY:
                state.record("q_after_rope_fp4", output)
            else:
                raise RuntimeError(
                    "source Indexer FP4 quantization lacked a k_norm/wq_b phase boundary"
                )
        return output

    try:
        layer_three.hc_mixes = observed_hc_mixes
        layer_four.hc_mixes = observed_hc_mixes
        layer_three_attention._window_kv = observed_window_kv_for(
            3, layer_three_attention, original_producer_window
        )
        layer_three_attention._compress_kv = observed_compress_kv_for(
            3, original_producer_compress
        )
        layer_four_attention._window_kv = observed_window_kv_for(
            4, layer_four_attention, original_window
        )
        layer_four_attention._compress_kv = observed_compress_kv_for(
            4, original_compress
        )
        layer_three_indexer.forward = observed_indexer_forward_for(
            IndexerRole.PRODUCER, original_producer_indexer_forward
        )
        layer_four_indexer.forward = observed_indexer_forward_for(
            IndexerRole.CONSUMER, original_indexer_forward
        )
        graph.torch = _GraphTorchProxy(original_torch, states)
        graph.apply_rotary_emb = observed_rotary
        graph.fp4_act_quant = observed_fp4_quant
        yield records
    finally:
        for handle in handles:
            handle.remove()
        _restore_instance(
            layer_three,
            "hc_mixes",
            had_producer_hc_mixes,
            prior_producer_hc_mixes,
        )
        _restore_instance(layer_four, "hc_mixes", had_hc_mixes, prior_hc_mixes)
        _restore_instance(layer_four_attention, "_window_kv", had_window, prior_window)
        _restore_instance(
            layer_four_attention, "_compress_kv", had_compress, prior_compress
        )
        _restore_instance(
            layer_three_attention,
            "_window_kv",
            had_producer_window,
            prior_producer_window,
        )
        _restore_instance(
            layer_three_attention,
            "_compress_kv",
            had_producer_compress,
            prior_producer_compress,
        )
        _restore_instance(
            layer_three_indexer,
            "forward",
            had_producer_indexer_forward,
            prior_producer_indexer_forward,
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
    elif name in module.__dict__:
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
