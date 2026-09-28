#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Qualify local ``mx decide`` receipts and bridge supplied JevBench JSONL.

The default is a dry run: it validates an operator-supplied JevBench JSONL and
prints the equivalent bounded requests.  ``--execute`` is deliberately opt-in;
it only invokes the supplied local binary and model directory.  It neither
downloads a checkpoint nor calls a service.
"""

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

CALIBRATION_STATUS = "uncalibrated"
CALIBRATION_METHOD = "temperature_scaled_option_softmax"
OPERATION = "qwen_typed_decision_prefill"
MAX_TASKS = 256
MAX_LINE_BYTES = 1 << 20
EPSILON = 1e-6
OPTION_LABELS = tuple(chr(ord("A") + index) for index in range(16))


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be an object")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _finite_probability(value: object, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{where} must be in [0, 1]")
    return result


def _labels(task: Mapping[str, Any]) -> list[str]:
    labels = task.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 16:
        raise ValueError("task.labels must contain 2..16 labels")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("task.labels must contain non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("task.labels must be unique")
    return labels


def _noul_labels(labels: Sequence[str]) -> tuple[str, str]:
    """Return the task labels for false and true without guessing semantics."""
    if len(labels) != 2:
        raise ValueError("noul tasks must have exactly two labels")
    normalized = {label.casefold(): label for label in labels}
    if set(normalized) == {"false", "true"}:
        return normalized["false"], normalized["true"]
    if set(normalized) == {"no", "yes"}:
        return normalized["no"], normalized["yes"]
    raise ValueError(
        "noul labels must be false/true or no/yes for an unambiguous bridge"
    )


def built_in_tasks() -> list[dict[str, Any]]:
    """Three contract fixtures, deliberately not claims about model accuracy."""
    return [
        {
            "id": "qualification-choice",
            "state": "A bounded choice fixture.",
            "labels": ["alpha", "beta"],
            "expected": "alpha",
            "question": {
                "type": "choice",
                "instructions": "Choose one.",
                "criteria": {"alpha": "first", "beta": "second"},
            },
        },
        {
            "id": "qualification-score",
            "state": "A bounded score fixture.",
            "labels": ["0", "1"],
            "expected": 0,
            "question": {
                "type": "score",
                "instructions": "Rate it.",
                "criteria": ["low", "high"],
            },
        },
        {
            "id": "qualification-noul",
            "state": "A bounded Boolean fixture.",
            "labels": ["false", "true"],
            "expected": "true",
            "question": {"type": "noul", "instructions": "Is this a Boolean fixture?"},
        },
    ]


def task_to_request(task: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one public JevBench-shaped task and form one mx request.

    The second return value is bridge-only metadata used to map the typed result
    back to the benchmark's labels.  It never changes or fits a prediction.
    """
    task_id = _string(task.get("id"), "task.id")
    state = task.get("state")
    if not isinstance(state, (str, dict, list)):
        raise TypeError("task.state must be a string, JSON array, or JSON object")
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("task.state must be a finite JSON value") from error
    question = _object(task.get("question"), "task.question")
    kind = _string(question.get("type"), "task.question.type")
    if kind not in {"choice", "score", "noul"}:
        raise ValueError("task.question.type must be choice, score, or noul")
    instructions = _string(question.get("instructions"), "task.question.instructions")
    labels = _labels(task)
    expected = task.get("expected")
    split = task.get("split")
    if split is not None and split != "public":
        raise ValueError("the local JevBench bridge accepts only public-split tasks")
    provenance = task.get("provenance", {})
    if provenance is not None:
        provenance = _object(provenance, "task.provenance")

    base = {"type": kind, "instructions": instructions}
    bridge: dict[str, Any] = {
        "id": task_id,
        "type": kind,
        "labels": labels,
        "expected": expected,
        "eligible": expected is not None and not provenance.get("exclude_reason"),
    }
    criteria = question.get("criteria")
    if kind == "choice":
        criteria = _object(criteria, "choice criteria")
        if set(criteria) != set(labels) or any(
            not isinstance(value, str) or not value for value in criteria.values()
        ):
            raise ValueError("choice criteria must map every label to a description")
        base["criteria"] = {label: criteria[label] for label in sorted(labels)}
    elif kind == "score":
        if (
            not isinstance(criteria, list)
            or len(criteria) != len(labels)
            or any(not isinstance(value, str) or not value for value in criteria)
        ):
            raise ValueError("score criteria must be one description per label")
        base["criteria"] = criteria
        if expected is not None and (
            type(expected) is not int or not 0 <= expected < len(labels)
        ):
            raise ValueError("score expected must be a zero-based level index or null")
        bridge["expected_index"] = expected
        bridge["expected"] = labels[expected] if expected is not None else None
    else:
        if criteria is None:
            criteria = {"false": "false", "true": "true"}
        criteria = _object(criteria, "noul criteria")
        if set(criteria) != {"false", "true"} or any(
            not isinstance(value, str) or not value for value in criteria.values()
        ):
            raise ValueError("noul criteria must explicitly describe false and true")
        false_label, true_label = _noul_labels(labels)
        base["criteria"] = {"false": criteria["false"], "true": criteria["true"]}
        bridge["false_label"] = false_label
        bridge["true_label"] = true_label
    if kind != "score" and expected is not None and expected not in labels:
        raise ValueError("task.expected must be one of task.labels or null")
    return {"state": state, "questions": {task_id: base}}, bridge


def _expected_options(question: Mapping[str, Any]) -> list[object]:
    kind = question["type"]
    criteria = question["criteria"]
    if kind == "score":
        return [str(index) for index in range(len(criteria))]
    return sorted(criteria)


def _wire_key(option: object) -> str:
    return str(option)


def _validate_distribution(
    answer: Mapping[str, Any], expected_options: Sequence[object]
) -> list[float]:
    option_order = answer.get("option_order")
    if option_order != list(expected_options):
        raise ValueError("answer.option_order does not match the request")
    probabilities = _object(answer.get("probabilities"), "answer.probabilities")
    expected_keys = {_wire_key(option) for option in expected_options}
    if set(probabilities) != expected_keys:
        raise ValueError("answer.probabilities must cover every option once")
    option_tokens = answer.get("option_tokens")
    if not isinstance(option_tokens, list) or len(option_tokens) != len(
        expected_options
    ):
        raise ValueError("answer.option_tokens has the wrong cardinality")
    values: list[float] = []
    token_ids: set[int] = set()
    for index, row in enumerate(option_tokens):
        row = _object(row, f"answer.option_tokens[{index}]")
        option = row.get("option")
        if option != expected_options[index]:
            raise ValueError("answer option token order does not match the request")
        if row.get("label") != OPTION_LABELS[index]:
            raise ValueError(
                "answer option token label does not match its option order"
            )
        token_id = row.get("token_id")
        if type(token_id) is not int:
            raise ValueError("answer probability token_id must be an integer")
        if token_id in token_ids:
            raise ValueError("answer option token IDs must be distinct")
        token_ids.add(token_id)
    for option in expected_options:
        values.append(
            _finite_probability(probabilities[_wire_key(option)], "answer probability")
        )
    if abs(sum(values) - 1.0) > EPSILON:
        raise ValueError("answer probabilities must already sum to one")
    return values


def _first_argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def check_receipt(report: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Check structural and probability invariants without scoring model quality."""
    if report.get("operation") != OPERATION:
        raise ValueError("unexpected decision operation")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported decision receipt schema version")
    _string(report.get("model"), "receipt.model")
    _string(report.get("backend"), "receipt.backend")
    session_load_ms = report.get("session_load_ms")
    if (
        type(session_load_ms) not in (int, float)
        or not math.isfinite(session_load_ms)
        or session_load_ms < 0
    ):
        raise ValueError("receipt.session_load_ms must be finite and non-negative")
    calibration = _object(report.get("calibration"), "receipt.calibration")
    if (
        calibration.get("status") != CALIBRATION_STATUS
        or calibration.get("method") != CALIBRATION_METHOD
    ):
        raise ValueError("receipt must state its uncalibrated provenance")
    if (
        type(calibration.get("temperature")) not in (int, float)
        or not math.isfinite(calibration["temperature"])
        or calibration["temperature"] <= 0
    ):
        raise ValueError("receipt calibration temperature must be finite and positive")
    _string(calibration.get("note"), "receipt.calibration.note")
    provenance = _object(report.get("provenance"), "receipt.provenance")
    if not provenance:
        raise ValueError("receipt.provenance must not be empty")
    usage = _object(report.get("usage"), "receipt.usage")
    if usage.get("output_tokens") != 0:
        raise ValueError("direct option scoring must report zero output tokens")
    if type(usage.get("input_tokens")) is not int or usage["input_tokens"] < 0:
        raise ValueError("receipt.usage.input_tokens must be non-negative")
    questions = _object(request.get("questions"), "request.questions")
    answers = _object(report.get("answers"), "receipt.answers")
    if set(answers) != set(questions):
        raise ValueError("receipt answers do not match requested questions")
    total_prompt_tokens = 0
    for question_id, question in questions.items():
        question = _object(question, f"request.questions.{question_id}")
        answer = _object(answers[question_id], f"receipt.answers.{question_id}")
        kind = question.get("type")
        if answer.get("type") != kind:
            raise ValueError("answer kind does not match request")
        for timing in ("prompt_tokens", "render_ms", "prefill_ms"):
            if (
                type(answer.get(timing)) not in (int, float)
                or not math.isfinite(answer[timing])
                or answer[timing] < 0
            ):
                raise ValueError(f"answer {timing} must be finite and non-negative")
        total_prompt_tokens += answer["prompt_tokens"]
        expected_options = _expected_options(question)
        values = _validate_distribution(answer, expected_options)
        expected_index = _first_argmax(values)
        if kind == "choice":
            if answer.get("choice") != expected_options[expected_index]:
                raise ValueError("choice is not the deterministic probability argmax")
        elif kind == "score":
            reported = answer.get("score")
            if type(reported) not in (int, float) or not math.isfinite(reported):
                raise ValueError("score result must be finite")
            expected_value = sum(index * value for index, value in enumerate(values))
            if (
                not 0.0 <= reported <= len(values) - 1
                or abs(reported - expected_value) > EPSILON
            ):
                raise ValueError("score result must equal the probability expectation")
        elif kind == "noul":
            p_true = _finite_probability(answer.get("noul"), "noul result")
            true_index = expected_options.index("true")
            if abs(p_true - values[true_index]) > EPSILON:
                raise ValueError(
                    "noul result does not equal its true-option probability"
                )
        else:
            raise ValueError("unknown decision kind")
    if usage["input_tokens"] != total_prompt_tokens:
        raise ValueError("receipt input_tokens must equal the answer prompt-token sum")


def result_for_task(
    task: Mapping[str, Any], bridge: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    """Create local transparent per-task metrics; this is not an official score."""
    answer = report["answers"][bridge["id"]]
    probabilities = answer["probabilities"]
    kind = bridge["type"]
    if kind == "choice":
        selected = answer["choice"]
        by_label = probabilities
    elif kind == "score":
        score_values = [
            probabilities[str(index)] for index in range(len(bridge["labels"]))
        ]
        selected = bridge["labels"][_first_argmax(score_values)]
        by_label = {
            label: probabilities[str(index)]
            for index, label in enumerate(bridge["labels"])
        }
    else:
        selected = (
            bridge["true_label"] if answer["noul"] > 0.5 else bridge["false_label"]
        )
        by_label = {
            bridge["false_label"]: 1.0 - answer["noul"],
            bridge["true_label"]: answer["noul"],
        }
    expected = bridge["expected"]
    return {
        "id": bridge["id"],
        "family": task.get("family")
        if isinstance(task.get("family"), str)
        else "unclassified",
        "type": kind,
        "expected": expected,
        "selected": selected,
        "eligible": bridge["eligible"],
        "correct": selected == expected if bridge["eligible"] else None,
        "expected_probability": by_label[expected] if bridge["eligible"] else None,
        "probabilities": by_label,
        "score": answer["score"] if kind == "score" else None,
    }


def _number_summary(values: Sequence[object]) -> dict[str, float | int | None]:
    finite = [
        float(value)
        for value in values
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
    ]
    if not finite:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(finite),
        "mean_ms": sum(finite) / len(finite),
        "median_ms": median(finite),
        "min_ms": min(finite),
        "max_ms": max(finite),
    }


def _source_eligible(task: Mapping[str, Any]) -> bool:
    provenance = task.get("provenance") or {}
    return (
        task.get("expected") is not None
        and isinstance(provenance, dict)
        and provenance.get("exclude_reason") is None
    )


def _record_metadata(
    entry: Mapping[str, Any], source_tasks: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    cached_result = (
        entry.get("result") if isinstance(entry.get("result"), dict) else None
    )
    task_id = entry.get("id")
    source_task = source_tasks.get(task_id) if isinstance(task_id, str) else None
    source_eligible = _source_eligible(source_task) if source_task else None
    entry_eligible = (
        entry.get("eligible") if type(entry.get("eligible")) is bool else None
    )
    family = (
        entry.get("family") if isinstance(entry.get("family"), str) else "unclassified"
    )
    decision_type = (
        entry.get("type") if isinstance(entry.get("type"), str) else "unknown"
    )
    if source_task:
        family = (
            source_task.get("family")
            if isinstance(source_task.get("family"), str)
            else "unclassified"
        )
    report = entry.get("report")
    request = entry.get("request")
    if isinstance(report, dict) and isinstance(request, dict):
        try:
            check_receipt(report, request)
            if source_task:
                projected_request, bridge = task_to_request(source_task)
                if projected_request != request:
                    raise ValueError(
                        "saved request differs from the verified source task"
                    )
                result = result_for_task(source_task, bridge, report)
                return {
                    "type": result["type"],
                    "family": family,
                    "eligible": source_eligible,
                    "result": result,
                    "metrics_source": "recomputed_from_verified_dataset",
                }
            if cached_result:
                return {
                    "type": cached_result.get("type", decision_type),
                    "family": cached_result.get("family", family),
                    "eligible": cached_result.get("eligible", entry_eligible),
                    "result": cached_result,
                    "metrics_source": "saved_derived_result_unverified_dataset",
                }
        except (TypeError, ValueError) as error:
            return {
                "type": decision_type,
                "family": family,
                "eligible": source_eligible if source_task else entry_eligible,
                "result": None,
                "verification_error": str(error),
            }
    if cached_result:
        return {
            "type": cached_result.get("type", decision_type),
            "family": cached_result.get("family", family),
            "eligible": cached_result.get("eligible", entry_eligible),
            "result": cached_result,
            "metrics_source": "saved_derived_result_without_receipt",
        }
    if source_task:
        try:
            _, bridge = task_to_request(source_task)
            decision_type = bridge["type"]
        except (TypeError, ValueError):
            decision_type = "unsupported"
        return {
            "type": decision_type,
            "family": family,
            "eligible": source_eligible,
            "result": None,
        }
    questions = request.get("questions") if isinstance(request, dict) else None
    question = next(iter(questions.values()), {}) if isinstance(questions, dict) else {}
    return {
        "type": question.get("type", "unknown")
        if isinstance(question, dict)
        else "unknown",
        "family": family,
        "eligible": entry_eligible,
        "result": None,
    }


def _latencies(entry: Mapping[str, Any]) -> dict[str, object]:
    report = entry.get("report") if isinstance(entry.get("report"), dict) else {}
    answers = report.get("answers") if isinstance(report.get("answers"), dict) else {}
    answer = next(iter(answers.values()), {}) if answers else {}
    return {
        "process_elapsed_ms": entry.get("elapsed_ms"),
        "session_load_ms": report.get("session_load_ms"),
        "render_ms": answer.get("render_ms") if isinstance(answer, dict) else None,
        "prefill_ms": answer.get("prefill_ms") if isinstance(answer, dict) else None,
    }


def _bucket(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["result"] is not None]
    eligible = [record for record in records if record["eligible"] is True]
    eligible_successful = [
        record for record in successful if record["eligible"] is True
    ]
    correct = sum(record["result"]["correct"] is True for record in eligible_successful)
    brier_values: list[float] = []
    nll_values: list[float] = []
    zero_support = 0
    for record in eligible_successful:
        result = record["result"]
        probabilities = result["probabilities"]
        expected = result["expected"]
        brier_values.append(
            sum(
                (float(probability) - float(label == expected)) ** 2
                for label, probability in probabilities.items()
            )
        )
        expected_probability = float(result["expected_probability"])
        if expected_probability == 0.0:
            zero_support += 1
        else:
            nll_values.append(-math.log(expected_probability))
    latency = {
        key: _number_summary([record["latency"][key] for record in records])
        for key in ("process_elapsed_ms", "session_load_ms", "render_ms", "prefill_ms")
    }
    nll_status = (
        "no_data"
        if not eligible_successful
        else "infinite_zero_support"
        if zero_support
        else "finite"
    )
    return {
        "total": len(records),
        "successful": len(successful),
        "failures_or_unsupported": len(records) - len(successful),
        "coverage": len(successful) / len(records) if records else None,
        "eligible": len(eligible),
        "excluded_or_unknown": len(records) - len(eligible),
        "correct": correct,
        "accuracy": correct / len(eligible) if eligible else None,
        "conditional_accuracy": correct / len(eligible_successful)
        if eligible_successful
        else None,
        "probability_quality": {
            "multiclass_brier_mean": sum(brier_values) / len(brier_values)
            if brier_values
            else None,
            "multiclass_brier_count": len(brier_values),
            "mean_nll": sum(nll_values) / len(eligible_successful)
            if eligible_successful and not zero_support
            else None,
            "nll_count": len(eligible_successful) - zero_support,
            "zero_expected_probability_count": zero_support,
            "nll_status": nll_status,
            "nll_semantics": "zero expected-label probability gives infinite NLL; no_data has no eligible successful decisions",
            "calibration": "not_claimed",
        },
        "latency_ms": latency,
    }


def summarize_qualification(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute transparent metrics from saved local receipts without inference."""
    source_tasks: dict[str, Mapping[str, Any]] = {}
    source = receipt.get("source")
    expected_source_hash = receipt.get("source_sha256")
    source_verification = "not_available"
    if (
        isinstance(source, str)
        and isinstance(expected_source_hash, str)
        and Path(source).is_file()
    ):
        with Path(source).open("rb") as source_handle:
            actual_source_hash = hashlib.file_digest(
                source_handle, "sha256"
            ).hexdigest()
        if actual_source_hash == expected_source_hash:
            source_tasks = {
                task["id"]: task for task in load_tasks(Path(source), MAX_TASKS)
            }
            source_verification = "sha256_verified"
        else:
            source_verification = "sha256_mismatch_fail_closed"
    elif source is None:
        source_verification = "not_applicable_builtin_fixture"
    elif isinstance(source, str):
        source_verification = "not_verified_missing_or_unreadable_dataset"
    records = []
    for entry in receipt.get("tasks", []):
        if not isinstance(entry, dict):
            continue
        record = _record_metadata(entry, source_tasks)
        record["latency"] = _latencies(entry)
        records.append(record)
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(str(record["type"]), []).append(record)
        by_family.setdefault(str(record["family"]), []).append(record)
    return {
        "summary_schema_version": 1,
        "source_dataset": source,
        "source_dataset_sha256": expected_source_hash,
        "source_dataset_verification": source_verification,
        "overall": _bucket(records),
        "by_type": {name: _bucket(items) for name, items in sorted(by_type.items())},
        "by_family": {
            name: _bucket(items) for name, items in sorted(by_family.items())
        },
        "calibration": "not_claimed; normalized option probabilities require held-out calibration evaluation",
        "hierarchy": "flattened leaf labels are supported; tree routing and parent aggregation are not implemented",
    }


def same_decision(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Compare the deterministic decision content, excluding timing metadata."""

    def semantic_answers(report):
        return {
            name: {
                key: value
                for key, value in answer.items()
                if key not in {"render_ms", "prefill_ms"}
            }
            for name, answer in report.get("answers", {}).items()
        }

    return (
        first.get("operation") == second.get("operation")
        and first.get("calibration") == second.get("calibration")
        and semantic_answers(first) == semantic_answers(second)
        and first.get("usage") == second.get("usage")
        and first.get("provenance") == second.get("provenance")
    )


def load_tasks(path: Path, maximum: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(
                    f"JSONL line {line_number} exceeds {MAX_LINE_BYTES} bytes"
                )
            if not line.strip():
                continue
            if len(tasks) >= maximum:
                break
            try:
                task = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            tasks.append(_object(task, f"task at line {line_number}"))
    if not tasks:
        raise ValueError("JSONL contained no tasks")
    if len({task.get("id") for task in tasks}) != len(tasks):
        raise ValueError("JSONL task ids must be unique")
    return tasks


def _command(
    binary: Path,
    model: Path,
    request_path: Path,
    context_tokens: int,
    kv_budget_mib: int,
) -> list[str]:
    return [
        str(binary.resolve()),
        "decide",
        "--model",
        str(model.resolve()),
        "--request",
        str(request_path),
        "--temperature",
        "1",
        "--context-tokens",
        str(context_tokens),
        "--kv-budget-mib",
        str(kv_budget_mib),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--jevbench-jsonl", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--summarize-receipt",
        type=Path,
        help="recompute metrics from a saved qualifier JSON report without model inference",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="run each supplied task twice and require identical decision content",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=MAX_TASKS,
        help=f"read at most this many JSONL tasks (1..{MAX_TASKS})",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=2048,
        help="per-question rendered-token bound (1..16384; default: 2048)",
    )
    parser.add_argument(
        "--kv-budget-mib",
        type=int,
        default=512,
        help="logical K/V admission budget (1..8192; default: 512)",
    )
    args = parser.parse_args()
    if not 1 <= args.max_tasks <= MAX_TASKS:
        parser.error(f"--max-tasks must be in 1..{MAX_TASKS}")
    if not 1 <= args.context_tokens <= 16384:
        parser.error("--context-tokens must be in 1..16384")
    if not 1 <= args.kv_budget_mib <= 8192:
        parser.error("--kv-budget-mib must be in 1..8192")
    if args.summarize_receipt:
        if args.execute or args.jevbench_jsonl:
            parser.error("--summarize-receipt cannot be combined with execution inputs")
        try:
            receipt = json.loads(args.summarize_receipt.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    summarize_qualification(_object(receipt, "receipt")),
                    indent=2,
                    sort_keys=True,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"could not summarize receipt: {error}")
        return 0
    if args.binary is None or args.model is None:
        parser.error(
            "--binary and --model are required unless --summarize-receipt is used"
        )
    if args.execute and (not args.binary.is_file() or not args.model.is_dir()):
        parser.error("--execute requires an existing --binary and --model directory")
    if args.jevbench_jsonl:
        tasks = load_tasks(args.jevbench_jsonl, args.max_tasks)
        with args.jevbench_jsonl.open("rb") as source_handle:
            source_hash: str | None = hashlib.file_digest(
                source_handle, "sha256"
            ).hexdigest()
        source: str | None = str(args.jevbench_jsonl)
    else:
        tasks = built_in_tasks()
        source = None
        source_hash = None
    output: dict[str, Any] = {
        "executed": args.execute,
        "source": source,
        "source_sha256": source_hash,
        "total": len(tasks),
        "selection": {
            "max_tasks": args.max_tasks,
            "policy": "first nonempty JSONL rows; total describes this selected subset",
        },
        "configuration": {
            "temperature": 1.0,
            "context_tokens": args.context_tokens,
            "kv_budget_mib": args.kv_budget_mib,
        },
        "tasks": [],
        "official_score": None,
    }
    outcomes: list[dict[str, Any]] = []
    failures = 0
    unsupported = 0
    eligible_total = 0
    excluded = 0
    for task in tasks:
        # Unsupported requests still count against accuracy when labelled and
        # otherwise eligible; coverage failures must not improve the headline.
        eligible = _source_eligible(task)
        eligible_total += int(eligible)
        excluded += int(not eligible)
        try:
            request, bridge = task_to_request(task)
        except (TypeError, ValueError) as error:
            unsupported += 1
            question = task.get("question")
            output["tasks"].append(
                {
                    "id": task.get("id"),
                    "type": question.get("type")
                    if isinstance(question, dict)
                    else "unknown",
                    "family": task.get("family")
                    if isinstance(task.get("family"), str)
                    else "unclassified",
                    "eligible": eligible,
                    "unsupported": str(error),
                }
            )
            continue
        entry: dict[str, Any] = {
            "id": bridge["id"],
            "type": bridge["type"],
            "family": task.get("family")
            if isinstance(task.get("family"), str)
            else "unclassified",
            "eligible": bridge["eligible"],
            "request": request,
        }
        if args.execute:
            with tempfile.TemporaryDirectory(prefix="metallix-decide-") as directory:
                request_path = Path(directory) / "request.json"
                request_path.write_text(json.dumps(request), encoding="utf-8")
                completed = None
                try:
                    command = _command(
                        args.binary,
                        args.model,
                        request_path,
                        args.context_tokens,
                        args.kv_budget_mib,
                    )
                    started = time.monotonic()
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    elapsed_ms = (time.monotonic() - started) * 1000
                    entry["command"] = command
                    entry["elapsed_ms"] = elapsed_ms
                    report = json.loads(completed.stdout)
                    if completed.returncode != 0:
                        raise ValueError(f"mx decide exited {completed.returncode}")
                    check_receipt(report, request)
                    entry["report"] = report
                    if args.replay:
                        replay_started = time.monotonic()
                        replay = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            check=False,
                        )
                        entry["replay_elapsed_ms"] = (
                            time.monotonic() - replay_started
                        ) * 1000
                        replay_report = json.loads(replay.stdout)
                        if replay.returncode != 0:
                            raise ValueError(
                                f"mx decide replay exited {replay.returncode}"
                            )
                        check_receipt(replay_report, request)
                        entry["replay_report"] = replay_report
                        if not same_decision(report, replay_report):
                            raise ValueError(
                                "deterministic replay changed decision content"
                            )
                except (
                    OSError,
                    json.JSONDecodeError,
                    ValueError,
                    subprocess.TimeoutExpired,
                ) as error:
                    failures += 1
                    entry["failure"] = str(error)
                    entry["exit_code"] = completed.returncode if completed else None
                else:
                    entry["exit_code"] = completed.returncode
                    entry["result"] = result_for_task(task, bridge, report)
                    outcomes.append(entry["result"])
        output["tasks"].append(entry)
    output["attempted"] = len(outcomes) + failures
    output["unsupported"] = unsupported
    output["eligible"] = eligible_total
    output["excluded"] = excluded
    output["coverage"] = len(outcomes) / len(tasks) if args.execute else None
    output["failures"] = failures
    output["abstentions"] = 0
    successful_eligible = [item for item in outcomes if item["eligible"]]
    output["accuracy"] = (
        sum(item["correct"] for item in successful_eligible) / eligible_total
        if args.execute and eligible_total
        else None
    )
    output["conditional_accuracy"] = (
        sum(item["correct"] for item in successful_eligible) / len(successful_eligible)
        if successful_eligible
        else None
    )
    output["analysis"] = summarize_qualification(output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if failures or unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
