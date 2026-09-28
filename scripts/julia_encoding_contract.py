#!/usr/bin/env python3
"""Pure-stdlib reference for Julia-1's pinned request encoding contract.

This models only ``julia/data.py`` at Hugging Face revision
``a85b127321d580d65176c89ced8273f305745d85``.  It deliberately does not load
the Julia checkpoint, published tokenizer, or any remote code, and it does not
claim native Julia support.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

SOURCE_REPOSITORY = "SupersonicLabs/Julia-1"
SOURCE_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
SOURCE_DATA_PATH = "julia/data.py"
QTYPES = {"choice": 0, "score": 1, "noul": 2}


class ContractError(ValueError):
    """A request cannot be represented by the pinned source contract."""


class StubTokenizer:
    """Minimal callable tokenizer adapter for local encoding fixtures.

    ``mapping`` supplies already-known IDs for every exact string used by the
    fixture.  It is intentionally not a substitute for the published tokenizer.
    """

    def __init__(
        self,
        mapping: Mapping[str, Sequence[int]],
        *,
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        sep_token_id: int = 1,
        mask_token_id: int = 4,
        mask_token: str = "[MASK]",
    ) -> None:
        self.mapping = {text: list(ids) for text, ids in mapping.items()}
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.mask_token = mask_token

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise ContractError("fixture tokenizer only accepts no-special-token calls")
        try:
            return {"input_ids": list(self.mapping[text])}
        except KeyError as error:
            raise ContractError(f"fixture tokenizer has no IDs for {text!r}") from error


Tokenizer = Callable[..., Mapping[str, Sequence[int]]]


def validate_row(row: object, line: int = 1) -> dict[str, Any]:
    """Validate the JSONL row shape accepted by pinned ``julia/data.py``."""
    prefix = f"JSONL line {line}: "
    if not isinstance(row, dict):
        raise ContractError(prefix + "request must be a JSON object")
    if not isinstance(row.get("state"), (str, dict, list)) or not isinstance(
        row.get("question"), str
    ):
        raise ContractError(
            prefix + "state must be text/JSON and question must be text"
        )
    options = row.get("options")
    if (
        not isinstance(options, list)
        or not 2 <= len(options) <= 20
        or not all(isinstance(option, str) and option for option in options)
    ):
        raise ContractError(
            prefix + "options must contain 2–20 nonempty rendered descriptions"
        )
    kind = row.get("type", "choice")
    if kind not in QTYPES:
        raise ContractError(prefix + "type must be choice, score, or noul")
    if kind == "noul" and len(options) != 2:
        raise ContractError(prefix + "noul options must be ordered [false, true]")
    if "target" in row and (
        type(row["target"]) is not int or not 0 <= row["target"] < len(options)
    ):
        raise ContractError(prefix + "target must index the supplied option list")
    teacher = row.get("teacher_logits")
    if teacher is not None and (
        not isinstance(teacher, list)
        or len(teacher) != len(options)
        or not all(
            type(value) in (int, float) and math.isfinite(value) for value in teacher
        )
    ):
        raise ContractError(
            prefix + "teacher logits must be finite and match option count/order"
        )
    return row


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise ContractError("tokenizer must return input_ids")
    if any(type(token_id) is not int for token_id in ids):
        raise ContractError("fixture tokenizer input_ids must be integers")
    return list(ids)


def sequence(
    tokenizer: Tokenizer,
    row: Mapping[str, Any],
    max_length: int = 8192,
    head_length: int = 256,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Return the pinned source serialization for one already-validated row."""
    if head_length + 4 >= max_length:
        raise ContractError("max_length must leave room beyond the question head")
    token_fields = ("mask_token_id", "cls_token_id", "sep_token_id")
    if any(getattr(tokenizer, field, None) is None for field in token_fields):
        raise ContractError("tokenizer must define MASK, CLS and SEP IDs")

    state_value = row["state"]
    state = (
        state_value
        if isinstance(state_value, str)
        else json.dumps(state_value, ensure_ascii=False)
    )
    mask_token = getattr(tokenizer, "mask_token", None)
    if not isinstance(mask_token, str):
        raise ContractError("tokenizer must define a mask token string")
    texts = [state, row["question"], *row["options"]]
    if strict and any(mask_token in text for text in texts):
        raise ContractError("Reserved model marker in request")
    clean = lambda text: text.replace(mask_token, " ")

    kind = row.get("type", "choice")
    head = _encode(tokenizer, f"{kind} question: {clean(row['question'])}")
    option_ids = [_encode(tokenizer, " " + clean(option)) for option in row["options"]]
    if strict and any(len(option) > 48 for option in option_ids):
        raise ContractError("Option exceeds 48-token model contract")
    options = [[tokenizer.mask_token_id] + option[:48] for option in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [option[:per_option] for option in options]
        budget = head_length - sum(map(len, options))
    if strict and (
        len(head) > budget
        or any(
            len(option) != len(original) + 1
            for option, original in zip(options, option_ids)
        )
    ):
        raise ContractError("Question/options exceed lossless head budget")

    ids = [tokenizer.cls_token_id] + head[: max(8, budget)] + [tokenizer.sep_token_id]
    markers: list[int] = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    state_ids = _encode(tokenizer, clean(state))
    room = max_length - len(ids) - 1
    if room < 1:
        raise ContractError(
            "Question/options exceed sequence budget; shorten descriptions"
        )
    if strict and len(state_ids) > room:
        raise ContractError("Game state exceeds lossless context budget")
    return {
        "ids": ids + state_ids[:room] + [tokenizer.sep_token_id],
        "markers": markers,
        "qtype": QTYPES[kind],
        "truncated": len(state_ids) > room,
        **({"option_tokens": list(map(len, option_ids))} if strict else {}),
    }


def collate(
    tokenizer: Tokenizer,
    rows: Sequence[Mapping[str, Any]],
    max_length: int = 8192,
    head_length: int = 256,
    *,
    include_targets: bool = True,
) -> dict[str, Any]:
    """Represent the source collator's tensors as nested Python lists."""
    if not rows:
        raise ContractError("collation requires at least one row")
    encoded = [
        row["_encoded"]
        if "_encoded" in row
        else sequence(tokenizer, row, max_length, head_length)
        for row in rows
    ]
    length = min(max_length, ((max(len(item["ids"]) for item in encoded) + 7) // 8) * 8)
    count = max(len(item["markers"]) for item in encoded)
    ids = [[tokenizer.pad_token_id] * length for _ in rows]
    attention = [[0] * length for _ in rows]
    positions = [[0] * count for _ in rows]
    marker_mask = [[False] * count for _ in rows]
    has_teacher = include_targets and all("teacher_logits" in row for row in rows)
    teacher = [[0.0] * count for _ in rows] if has_teacher else None
    for index, (row, item) in enumerate(zip(rows, encoded)):
        token_count, marker_count = len(item["ids"]), len(item["markers"])
        ids[index][:token_count] = item["ids"]
        attention[index][:token_count] = [1] * token_count
        positions[index][:marker_count] = item["markers"]
        marker_mask[index][:marker_count] = [True] * marker_count
        if teacher is not None:
            teacher[index][:marker_count] = row["teacher_logits"]
    batch: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": positions,
        "marker_mask": marker_mask,
        "qtype": [item["qtype"] for item in encoded],
    }
    if include_targets and all("target" in row for row in rows):
        batch["labels"] = [row["target"] for row in rows]
    if teacher is not None:
        batch["teacher_logits"] = teacher
    return batch
