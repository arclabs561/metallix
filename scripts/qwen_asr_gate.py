"""Fail-closed coverage and numerical bars for the declared ASR G3 split gates."""

import math


def transcripts(rows):
    """Index one manifest without silently replacing duplicate or malformed rows."""
    result = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("text"), str)
        ):
            raise ValueError("malformed transcript row")
        if row["id"] in result:
            raise ValueError(f"duplicate transcript ID: {row['id']}")
        result[row["id"]] = row["text"]
    return result


def coverage(split, truth, reference, records):
    """Require the complete declared split and completed, unique candidate rows."""
    expected = {"clean": 200, "other": 100}[split]
    if len(truth) != expected:
        raise ValueError(f"{split}: expected {expected} truth clips, got {len(truth)}")
    ours = transcripts(records)
    for row in records:
        clip_id = row["id"]
        if row.get("stopped") is not True or not isinstance(row.get("text"), str):
            raise ValueError(f"incomplete candidate clip: {clip_id}")
    if set(reference) != set(truth) or set(ours) != set(truth):
        raise ValueError(
            "G3 requires identical complete clip IDs in truth, reference and candidate"
        )
    return ours


def passes(ours, reference, identical_fraction, *, resampled=False):
    """Use unrounded percentage-point WER and normalized agreement values."""
    if not all(math.isfinite(value) for value in (ours, reference, identical_fraction)):
        return False
    if min(ours, reference) < 0 or not 0 <= identical_fraction <= 1:
        return False
    margin = 0.5 if resampled else 0.2
    return abs(ours - reference) <= margin and (resampled or identical_fraction >= 0.97)
