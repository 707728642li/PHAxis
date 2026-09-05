from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phaxis.root_provider.bundle import (
    BundleError,
    HYBRID_POLARITY_QC_LOCK_BUNDLE_PATH,
    HYBRID_POLARITY_QC_LOCK_SHA256,
    verify_bundle,
)
from phaxis.root_provider.reference import ndarray_sha256
from phaxis.io import sha256_json


def _registry(root: Path, artifact: Path) -> None:
    record = {
        "path": "payload.bin",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": artifact.stat().st_size,
        "roles": ["test"],
    }
    identity = {
        "schema_version": "PHAxis-root-provider-model-bundle-1.0",
        "bundle_id": "PHAXIS-V1.0-FROZEN-V1-V20-Q8-HYBRID-ROOT-20260828",
        "files": [record],
        "contracts": {"full_legacy_hair_branch_required_for_root_equivalence": True},
    }
    payload = {
        **identity,
        "status": "materialized_unverified",
        "root_effect_slice_files": 0,
        "bundle_identity_sha256": sha256_json(identity),
    }
    (root / "root_provider_bundle.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_bundle_verification_rejects_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"locked")
    _registry(tmp_path, artifact)
    assert verify_bundle(tmp_path)["files_verified"] == 1
    artifact.write_bytes(b"tampered")
    with pytest.raises(BundleError, match="verification failed"):
        verify_bundle(tmp_path)


def test_ndarray_hash_binds_shape_dtype_and_bytes() -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert ndarray_sha256(array) == ndarray_sha256(array.copy())
    assert ndarray_sha256(array) != ndarray_sha256(array.reshape(2, 6))
    assert ndarray_sha256(array) != ndarray_sha256(array.astype(np.uint16))


def test_hybrid_polarity_qc_lock_is_an_explicit_hash_pinned_runtime_asset() -> None:
    assert HYBRID_POLARITY_QC_LOCK_SHA256 == (
        "ffd71fc025d58312375bc854a55f3c9512fe0c56b9771eb3ca848a6b41b9085d"
    )
    assert HYBRID_POLARITY_QC_LOCK_BUNDLE_PATH.as_posix() == (
        "hybrid_candidate/model/runtime_locks/hair_polarity_qcdev44_evaluation.json"
    )
