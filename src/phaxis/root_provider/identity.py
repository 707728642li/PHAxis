"""Label-free deployment identity adapter for the frozen Q8/Hybrid runtime.

The frozen deployment modules only need the HumanCurated443 dataset and split
identities to authenticate their weights.  Their historical loader nevertheless
opens the complete 443-image training package.  A portable model package must
not carry training imagery merely to compare two hashes, so this module builds a
minimal, immutable :class:`DatasetContract` after independently checking every
bundled JSON provenance record that declares those identities.

This adapter is deployment-only.  It deliberately contains no samples and must
never be used by training/evaluation code or by Q8 without an explicit
label-free deployment manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

from phaxis.io import read_json, sha256_file
from phaxis.root_provider.bundle import BundleError, verify_bundle


EXPECTED_DATASET_IDENTITY = (
    "1f80f9b9602fd82cf45d57886800981e6c7d657ebec0eb2f2fcabe28e52ac905"
)
EXPECTED_SPLIT_IDENTITY = (
    "7cb3367432fefbf5fb425b830b813fb37bdf8de692be06372f5687d4e476fad7"
)
EXPECTED_SPLIT_LOCK_SHA256 = (
    "8f77896c2e093ffddfa87520b4e9c7aed170bde2773c48e76c34ef2aaf968d69"
)


@dataclass(frozen=True)
class IdentityOnlyDatasetContract:
    """The narrow attribute surface consumed by frozen deployment modules."""

    root: Path
    samples: tuple[Any, ...]
    train: tuple[Any, ...]
    val: tuple[Any, ...]
    file_sha256: Mapping[str, str]
    identity_sha256: str
    split_file_sha256: Mapping[str, str]
    split_identity_sha256: str
    split_override_path: Path

    def by_task_id(self) -> dict[str, Any]:
        return {}


def _walk_declared_identities(value: Any, source: Path) -> Iterable[tuple[str, str, Path]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"dataset_identity_sha256", "source_dataset_identity_sha256"}:
                yield "dataset", str(item).casefold(), source
            elif key == "split_identity_sha256":
                yield "split", str(item).casefold(), source
            yield from _walk_declared_identities(item, source)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_declared_identities(item, source)


def _identity_provenance_files(model_root: Path) -> tuple[Path, ...]:
    weight_root = model_root / "weights"
    selected: list[Path] = []
    for path in sorted(weight_root.rglob("*.json")):
        # Parsing only compact provenance/lock files avoids unrelated reports;
        # all bytes were already authenticated by verify_bundle.
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "dataset_identity_sha256" in text or "source_dataset_identity_sha256" in text:
            selected.append(path)
    if not selected:
        raise BundleError("bundled model declares no training-identity provenance")
    return tuple(selected)


def load_identity_only_contract(
    bundle_root: str | Path,
    *,
    bundle_already_verified: bool = False,
) -> IdentityOnlyDatasetContract:
    """Authenticate the bundle and return its label-free identity projection."""

    bundle = Path(bundle_root).resolve()
    if not bundle_already_verified:
        verify_bundle(bundle)
    model_root = bundle / "hybrid_candidate" / "model"
    contract_path = bundle / "contracts" / "root_provider_contract.json"
    contract = read_json(contract_path)
    declared = contract.get("identity_only_training_contract", {})
    if declared.get("dataset_identity_sha256") != EXPECTED_DATASET_IDENTITY:
        raise BundleError("root-provider dataset identity contract drift")
    if declared.get("split_identity_sha256") != EXPECTED_SPLIT_IDENTITY:
        raise BundleError("root-provider split identity contract drift")
    if declared.get("canonical_annotations_read_during_deployment") is not False:
        raise BundleError("identity-only contract does not forbid annotation reads")

    split_lock = (
        model_root
        / "runtime/configs/rhaxis_nextgen/splits/qc_development_v1_0/split_lock.json"
    )
    if sha256_file(split_lock) != EXPECTED_SPLIT_LOCK_SHA256:
        raise BundleError("bundled QC-development split lock drift")
    split_payload = read_json(split_lock)
    if split_payload.get("source_dataset_identity_sha256") != EXPECTED_DATASET_IDENTITY:
        raise BundleError("split lock dataset identity mismatch")
    if split_payload.get("split_identity_sha256") != EXPECTED_SPLIT_IDENTITY:
        raise BundleError("split lock identity mismatch")
    if split_payload.get("blind_images_used") != 0:
        raise BundleError("split lock is blind-tainted")

    observed = list(
        identity
        for path in _identity_provenance_files(model_root)
        for identity in _walk_declared_identities(read_json(path), path)
    )
    bad_dataset = sorted(
        {str(path) for kind, value, path in observed if kind == "dataset" and value != EXPECTED_DATASET_IDENTITY}
    )
    bad_split = sorted(
        {str(path) for kind, value, path in observed if kind == "split" and value != EXPECTED_SPLIT_IDENTITY}
    )
    if bad_dataset or bad_split:
        raise BundleError(
            "model training-identity provenance drift: "
            f"dataset={bad_dataset}, split={bad_split}"
        )
    kinds = {kind for kind, _value, _path in observed}
    if kinds != {"dataset", "split"}:
        raise BundleError("model provenance does not cover dataset and split identities")

    semantic_provenance = read_json(
        model_root / "weights/q8/semantic_e0/dataset_provenance.json"
    )
    return IdentityOnlyDatasetContract(
        root=bundle / "identity_only_training_payload_not_packaged",
        samples=(),
        train=(),
        val=(),
        file_sha256=dict(semantic_provenance.get("dataset_contract_file_sha256", {})),
        identity_sha256=EXPECTED_DATASET_IDENTITY,
        split_file_sha256=dict(semantic_provenance.get("split_file_sha256", {})),
        split_identity_sha256=EXPECTED_SPLIT_IDENTITY,
        split_override_path=split_lock,
    )


def install_deployment_identity_adapter(
    module: ModuleType,
    bundle_root: str | Path,
    *,
    bundle_already_verified: bool = False,
) -> IdentityOnlyDatasetContract:
    """Patch one already imported frozen deployment module, and only that module."""

    contract = load_identity_only_contract(
        bundle_root, bundle_already_verified=bundle_already_verified
    )
    if not hasattr(module, "load_dataset_contract"):
        raise BundleError(f"deployment module has no loader hook: {module.__name__}")

    def identity_loader(
        _dataset_root: str | Path,
        _split_override: str | Path | None = None,
    ) -> IdentityOnlyDatasetContract:
        return contract

    module.load_dataset_contract = identity_loader
    return contract


__all__ = [
    "EXPECTED_DATASET_IDENTITY",
    "EXPECTED_SPLIT_IDENTITY",
    "IdentityOnlyDatasetContract",
    "install_deployment_identity_adapter",
    "load_identity_only_contract",
]
