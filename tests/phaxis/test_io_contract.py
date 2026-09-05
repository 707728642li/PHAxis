from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaxis.io import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_json,
)


def test_canonical_json_and_digest_are_order_independent():
    left = {"z": 1, "中文": [True, None], "a": {"b": 2}}
    right = {"a": {"b": 2}, "中文": [True, None], "z": 1}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)
    assert b"\\u" not in canonical_json_bytes(left)


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(ValueError):
        canonical_json_bytes({"invalid": float("nan")})


def test_atomic_write_is_replace_only_and_cleans_failed_temporary_file(tmp_path: Path):
    destination = tmp_path / "nested" / "prediction.json"
    atomic_write_json(destination, {"generation": 1})
    original = destination.read_bytes()
    assert original.endswith(b"\n")

    with pytest.raises(ValueError):
        atomic_write_json(destination, {"invalid": float("nan")})

    assert destination.read_bytes() == original
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_read_json_requires_an_object(tmp_path: Path):
    path = tmp_path / "array.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object expected"):
        read_json(path)
