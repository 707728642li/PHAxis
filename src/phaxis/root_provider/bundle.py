"""Build and verify the external PHAxis 1.0.0 root-provider model bundle.

This module never edits the historical RHAxis/RhizoWeave tree.  It copies or
hard-links a byte-exact frozen closure into a new directory, records every
file by SHA-256, and supplies portable wrappers from the PHAxis wheel.  The
historical V20 exporter contains a dormant Z-drive default; the mandatory
read-only adapter replaces it with an explicit compatibility root before any
sample is processed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json


BUNDLE_SCHEMA = "PHAxis-root-provider-model-bundle-1.0"
BUNDLE_ID = "PHAXIS-V1.0-FROZEN-V1-V20-Q8-HYBRID-ROOT-20260828"
V1_RUNTIME_CONFIG = Path("configs/rhpheno_dual_mode_runtime_v6_v19_prospective.json")
V1_RUNTIME_CONFIG_SHA256 = (
    "1711f038bbf1a6cf5d156596e71616a5a0a353154d56e8fb1bcffe1777997fe7"
)
V12_ROOT_CHECKPOINT_SHA256 = (
    "d242a4e50c88c658a0ac081c6e38f4dd8dfcfbfacd9bdb763fb769da2e90f3db"
)
V20_CONFIG = Path("configs/rhaxis_v20_assisted_review_rootcap_recall_dev_v20.json")
V20_CONFIG_SHA256 = (
    "09a97a09e3b9833ac634ff9a1eb3dec009eda9637224623f7789da7fbd1d7dad"
)
V20_ADAPTER_SHA256 = (
    "ca51b6e2f4acd3959d2db0f94559bbd0aa643709581925b8fd50a9ece6b127bb"
)
HYBRID_REUSE_ROOT = Path("release/RHAxis_Hybrid_Max_Final_Candidate_Reuse_20260825")
HYBRID_PACKAGE_IDENTITY = (
    "51133b022f89e4d611e14ec288f13b2f545ce15e375ab4a654df34e74418e679"
)
HYBRID_RUNTIME_BENCHMARK_ASSETS = (
    Path("benchmark/raw/final_axis_ridge_qcdev44_evaluation.json"),
)
HYBRID_POLARITY_QC_LOCK = Path(
    "outputs/rhaxis_nextgen_hybrid_max_hair_polarity_verifier_qcdev44_run2_biological20um/"
    "hair_polarity_verifier_evaluation.json"
)
HYBRID_POLARITY_QC_LOCK_SHA256 = (
    "ffd71fc025d58312375bc854a55f3c9512fe0c56b9771eb3ca848a6b41b9085d"
)
HYBRID_POLARITY_QC_LOCK_BUNDLE_PATH = Path(
    "hybrid_candidate/model/runtime_locks/hair_polarity_qcdev44_evaluation.json"
)

# Root-only V20 effect slice identified by source audit.  The complete V20
# hair closure is additionally bundled because seven final root masks use
# attachment-supported axis extension; this smaller list remains an explicit
# scientific audit target, not a substitute for the full executable closure.
V20_ROOT_EFFECT_SLICE = tuple(
    Path("src/rhizoweave") / name
    for name in (
        "__init__.py",
        "prepare.py",
        "acquisition_quality.py",
        "curvilinear.py",
        "image_geometry.py",
        "rootguide_pseudo.py",
        "scale_bar.py",
        "valid_canvas.py",
        "model.py",
        "v7_root_contour.py",
        "root_boundary_conservative.py",
        "root_boundary_learned_calibrator.py",
        "dense_hair_field_v20.py",
        "dense_shape_prior_v20.py",
        "source_root_continuation_v20.py",
    )
)

FULL_CHAIN_PYTHON_ROOTS = (
    Path("src/rhizoweave/prepare.py"),
    Path("scripts/build_six_condition_v1_preparation_inputs.py"),
    Path("scripts/run_six_condition_v1_sharded.py"),
    Path("scripts/run_rhpheno_dual_mode_v5.py"),
    Path("scripts/merge_six_condition_v1_shards.py"),
    Path("scripts/materialize_six_condition_v1_prefill_adapter.py"),
    Path("scripts/run_six_condition_v20_12_readonly_adapter.py"),
    Path("scripts/run_six_condition_v20_12_sharded.py"),
    Path("scripts/merge_six_condition_v20_12_shards.py"),
    Path("scripts/export_rhaxis_v20_assisted_review_prefills.py"),
)

V20_CONFIG_PATH_KEYS = frozenset(
    {
        "base_config",
        "config",
        "calibrator",
        "candidate_scorer",
        "path_refiner",
        "ridge_completion_config",
    }
)


class BundleError(RuntimeError):
    """Raised when a root-provider closure or hash contract is invalid."""


@dataclass(frozen=True)
class SourceArtifact:
    source: Path
    bundle_relpath: Path
    sha256: str
    bytes: int
    roles: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "path": self.bundle_relpath.as_posix(),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "roles": list(self.roles),
        }


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BundleError(f"artifact escapes project root: {resolved}") from error


def _module_candidates(project_root: Path, module: str) -> list[Path]:
    relative = Path(*module.split("."))
    return [
        project_root / "src" / relative.with_suffix(".py"),
        project_root / "src" / relative / "__init__.py",
        project_root / "scripts" / relative.with_suffix(".py"),
        project_root / relative.with_suffix(".py"),
        project_root / relative / "__init__.py",
    ]


def _package_name(project_root: Path, path: Path) -> str | None:
    relative = _inside(project_root, path)
    if relative.parts[0] == "src":
        parts = list(relative.parts[1:])
    elif relative.parts[0] == "scripts":
        # Historical scripts sometimes import peers as top-level modules.
        return None
    else:
        parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def _resolve_imports(project_root: Path, path: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise BundleError(f"cannot parse local dependency: {path}") from error
    modules: set[str] = set()
    package = _package_name(project_root, path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if not package:
                    continue
                package_parts = package.split(".")
                anchor = package_parts[: max(0, len(package_parts) - node.level)]
                base = ".".join((*anchor, *([base] if base else [])))
            if base:
                modules.add(base)
                modules.update(f"{base}.{alias.name}" for alias in node.names)
            elif package:
                modules.update(
                    f"{package.rsplit('.', 1)[0]}.{alias.name}"
                    if "." in package
                    else alias.name
                    for alias in node.names
                )
    dependencies: set[Path] = set()
    for module in modules:
        for candidate in _module_candidates(project_root, module):
            if candidate.is_file():
                dependencies.add(candidate.resolve())
                break
        if path.parent == (project_root / "scripts").resolve():
            peer = project_root / "scripts" / f"{module.split('.')[-1]}.py"
            if peer.is_file():
                dependencies.add(peer.resolve())
    return dependencies


def discover_local_python_closure(
    project_root: Path, roots: Iterable[Path]
) -> tuple[Path, ...]:
    project_root = project_root.resolve()
    pending = [
        (project_root / root).resolve() if not root.is_absolute() else root.resolve()
        for root in roots
    ]
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        _inside(project_root, path)
        if path in closure:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        closure.add(path)
        relative = _inside(project_root, path)
        if relative.parts[:1] == ("src",) and len(relative.parts) >= 3:
            parent = path.parent
            src_root = project_root / "src"
            while parent != src_root:
                initializer = parent / "__init__.py"
                if initializer.is_file() and initializer not in closure:
                    pending.append(initializer.resolve())
                parent = parent.parent
        pending.extend(_resolve_imports(project_root, path) - closure)
    return tuple(sorted(closure))


def _verified_file(project_root: Path, relative: str | Path, expected: str | None = None) -> Path:
    path = (project_root / Path(relative)).resolve()
    _inside(project_root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual.casefold() != expected.casefold():
        raise BundleError(f"SHA-256 mismatch: {relative}: {actual} != {expected}")
    return path


def _v1_registry_files(project_root: Path) -> set[Path]:
    registry_path = _verified_file(
        project_root, V1_RUNTIME_CONFIG, V1_RUNTIME_CONFIG_SHA256
    )
    payload = read_json(registry_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or len(artifacts) != 67:
        raise BundleError("V19 runtime must contain exactly 67 registered artifacts")
    result = {registry_path}
    root = (registry_path.parent / str(payload.get("artifact_root", ".."))).resolve()
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            raise BundleError(f"malformed V1 artifact record: {name}")
        path = (root / str(raw["path"])).resolve()
        _inside(project_root, path)
        if sha256_file(path) != str(raw["sha256"]).casefold():
            raise BundleError(f"V1 registry artifact drift: {name}")
        result.add(path)
    checkpoint = project_root / "outputs/v12_root_contour_research_grade_v3/best.pt"
    if sha256_file(checkpoint) != V12_ROOT_CHECKPOINT_SHA256:
        raise BundleError("V12 root checkpoint identity drift")
    return result


def _scan_v20_config(
    project_root: Path, relative: Path, seen: set[Path], result: set[Path]
) -> None:
    path = _verified_file(project_root, relative)
    if path in seen:
        return
    seen.add(path)
    result.add(path)
    payload = read_json(path)

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in V20_CONFIG_PATH_KEYS and isinstance(item, str):
                    candidate = Path(item)
                    resolved = _verified_file(project_root, candidate)
                    result.add(resolved)
                    declared = value.get(f"{key}_sha256")
                    if declared is not None and sha256_file(resolved) != str(declared):
                        raise BundleError(f"V20 config dependency drift: {key}")
                    if candidate.suffix.casefold() == ".json" and key in {
                        "base_config",
                        "config",
                        "ridge_completion_config",
                    }:
                        _scan_v20_config(project_root, candidate, seen, result)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)


def _v20_files(project_root: Path) -> set[Path]:
    result: set[Path] = set()
    _scan_v20_config(project_root, V20_CONFIG, set(), result)
    if sha256_file(project_root / V20_CONFIG) != V20_CONFIG_SHA256:
        raise BundleError("V20 final overlay config identity drift")
    adapter = project_root / "scripts/run_six_condition_v20_12_readonly_adapter.py"
    if sha256_file(adapter) != V20_ADAPTER_SHA256:
        raise BundleError("V20 read-only adapter identity drift")
    # Some frozen components (notably ``v7_root_contour``) are reached through
    # configuration/dynamic dispatch rather than a statically visible import.
    # Seed the audited effect slice explicitly, then close over its imports.
    closure = discover_local_python_closure(
        project_root, (*FULL_CHAIN_PYTHON_ROOTS, *V20_ROOT_EFFECT_SLICE)
    )
    result.update(closure)
    missing_slice = [
        relative.as_posix()
        for relative in V20_ROOT_EFFECT_SLICE
        if (project_root / relative).resolve() not in result
    ]
    if missing_slice:
        raise BundleError(f"V20 root effect slice is incomplete: {missing_slice}")
    return result


def _hybrid_files(project_root: Path) -> set[Path]:
    release_root = (project_root / HYBRID_REUSE_ROOT).resolve()
    package_lock = read_json(release_root / "PACKAGE_LOCK.json")
    if package_lock.get("identity_sha256") != HYBRID_PACKAGE_IDENTITY:
        raise BundleError("Hybrid reuse package identity drift")
    model_root = release_root / "model"
    selected = {
        path.resolve()
        for path in model_root.rglob("*")
        if path.is_file() and "benchmark" not in path.relative_to(model_root).parts
    }
    for relative in HYBRID_RUNTIME_BENCHMARK_ASSETS:
        runtime_asset = (model_root / relative).resolve()
        if not runtime_asset.is_file():
            raise BundleError(f"Hybrid runtime benchmark-lock asset missing: {relative}")
        selected.add(runtime_asset)
    if not selected:
        raise BundleError("Hybrid candidate model closure is empty")
    return selected


def collect_bundle_artifacts(project_root: str | Path) -> tuple[SourceArtifact, ...]:
    root = Path(project_root).resolve()
    roles: dict[tuple[str, str], set[str]] = {}
    paths: dict[tuple[str, str], Path] = {}

    def add(path: Path, bundle_relpath: Path, role: str) -> None:
        key = (path.resolve().as_posix().casefold(), bundle_relpath.as_posix())
        paths[key] = path.resolve()
        roles.setdefault(key, set()).add(role)

    for path in _v1_registry_files(root):
        relative = _inside(root, path)
        add(path, Path("legacy_project") / relative, "frozen_v1_full_runtime")
    for path in _v20_files(root):
        relative = _inside(root, path)
        role = (
            "v20_root_effect_slice"
            if relative in V20_ROOT_EFFECT_SLICE
            else "frozen_v20_full_attachment_preserving_closure"
        )
        add(path, Path("legacy_project") / relative, role)
    hybrid_model = (root / HYBRID_REUSE_ROOT / "model").resolve()
    for path in _hybrid_files(root):
        relative = path.relative_to(hybrid_model)
        add(path, Path("hybrid_candidate/model") / relative, "q8_hybrid_final_candidate")
    # The handover model package omitted the QC-development evidence lock that
    # the frozen final deployment actually validates at runtime.  Preserve the
    # exact accepted lock as an explicit, hash-pinned runtime asset instead of
    # silently substituting the similarly named axis-ridge evaluation file.
    polarity_qc_lock = _verified_file(
        root, HYBRID_POLARITY_QC_LOCK, HYBRID_POLARITY_QC_LOCK_SHA256
    )
    add(
        polarity_qc_lock,
        HYBRID_POLARITY_QC_LOCK_BUNDLE_PATH,
        "hybrid_hair_polarity_qc_runtime_lock",
    )
    contract_path = root / "configs/phaxis/v1_0/root_provider_contract.json"
    if contract_path.is_file():
        add(
            contract_path,
            Path("contracts/root_provider_contract.json"),
            "phaxis_root_provider_contract",
        )
    artifacts = []
    destination_hashes: dict[str, str] = {}
    for key, path in paths.items():
        bundle_relpath = Path(key[1])
        digest = sha256_file(path)
        existing = destination_hashes.get(bundle_relpath.as_posix())
        if existing is not None and existing != digest:
            raise BundleError(f"bundle destination collision: {bundle_relpath}")
        destination_hashes[bundle_relpath.as_posix()] = digest
        artifacts.append(
            SourceArtifact(
                source=path,
                bundle_relpath=bundle_relpath,
                sha256=digest,
                bytes=path.stat().st_size,
                roles=tuple(sorted(roles[key])),
            )
        )
    return tuple(sorted(artifacts, key=lambda item: item.bundle_relpath.as_posix()))


def _materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("materialization mode must be 'hardlink' or 'copy'")


def build_bundle(
    project_root: str | Path,
    output: str | Path,
    *,
    mode: str = "hardlink",
    plan_only: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output).resolve()
    artifacts = collect_bundle_artifacts(root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"root-provider bundle output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if not plan_only:
        for artifact in artifacts:
            _materialize(
                artifact.source, destination / artifact.bundle_relpath, mode
            )
    records = [artifact.record() for artifact in artifacts]
    identity_payload = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": BUNDLE_ID,
        "files": records,
        "contracts": {
            "v1_runtime_config_sha256": V1_RUNTIME_CONFIG_SHA256,
            "v12_root_checkpoint_sha256": V12_ROOT_CHECKPOINT_SHA256,
            "v20_config_sha256": V20_CONFIG_SHA256,
            "v20_readonly_adapter_sha256": V20_ADAPTER_SHA256,
            "hybrid_reuse_package_identity_sha256": HYBRID_PACKAGE_IDENTITY,
            "hybrid_hair_polarity_qc_lock_sha256": HYBRID_POLARITY_QC_LOCK_SHA256,
            "full_legacy_hair_branch_required_for_root_equivalence": True,
        },
    }
    payload = {
        **identity_payload,
        "status": "planned" if plan_only else "materialized_unverified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "materialization_mode": "none" if plan_only else mode,
        "files_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "root_effect_slice_files": sum(
            "v20_root_effect_slice" in record["roles"] for record in records
        ),
        "bundle_identity_sha256": sha256_json(identity_payload),
        "portable_execution_contract": {
            "python_executable": "caller supplied / sys.executable",
            "physical_gpu": "caller supplied; mapped through CUDA_VISIBLE_DEVICES",
            "logical_gpu": "cuda:0",
            "v20_compatibility_data_root": "caller supplied and injected before processing",
            "legacy_project_root": "derived from bundle location",
            "implicit_download": False,
            "blind_images_used": 0,
        },
    }
    atomic_write_json(destination / "root_provider_bundle.json", payload)
    return payload


def verify_bundle(
    bundle_root: str | Path, *, require_exact_closure: bool = False
) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    registry_path = root / "root_provider_bundle.json"
    payload = read_json(registry_path)
    if payload.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleError("unexpected root-provider bundle schema")
    if payload.get("bundle_id") != BUNDLE_ID:
        raise BundleError("unexpected root-provider bundle identity")
    if payload.get("status") == "planned":
        raise BundleError("a plan-only registry is not an executable model bundle")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise BundleError("root-provider bundle file registry is empty")
    failures: list[dict[str, str]] = []
    for record in files:
        relative = Path(str(record["path"]))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise BundleError(f"bundle path escape: {relative}") from error
        if not path.is_file():
            failures.append({"path": relative.as_posix(), "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            failures.append(
                {
                    "path": relative.as_posix(),
                    "reason": "sha256_mismatch",
                    "actual": actual,
                }
            )
    identity_payload = {
        "schema_version": payload["schema_version"],
        "bundle_id": payload["bundle_id"],
        "files": files,
        "contracts": payload["contracts"],
    }
    identity_matches = sha256_json(identity_payload) == payload.get(
        "bundle_identity_sha256"
    )
    if failures or not identity_matches:
        raise BundleError(
            f"root-provider bundle verification failed: files={len(failures)}, identity={identity_matches}"
        )
    expected_paths = {
        "root_provider_bundle.json",
        *(Path(str(record["path"])).as_posix() for record in files),
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    unlisted = sorted(actual_paths - expected_paths)
    missing_from_closure = sorted(expected_paths - actual_paths)
    if require_exact_closure and (unlisted or missing_from_closure):
        raise BundleError(
            "root-provider bundle exact closure failed: "
            f"unlisted={len(unlisted)}, missing={len(missing_from_closure)}"
        )
    return {
        "schema_version": "PHAxis-root-provider-model-bundle-verification-1.0",
        "status": "pass",
        "bundle_id": payload["bundle_id"],
        "bundle_identity_sha256": payload["bundle_identity_sha256"],
        "files_verified": len(files),
        "bytes_verified": sum(int(record["bytes"]) for record in files),
        "root_effect_slice_files": int(payload["root_effect_slice_files"]),
        "exact_file_closure_required": bool(require_exact_closure),
        "exact_file_closure_passed": not unlisted and not missing_from_closure,
        "unlisted_file_count": len(unlisted),
        "missing_closure_file_count": len(missing_from_closure),
        "full_legacy_hair_branch_required_for_root_equivalence": True,
        "blind_images_used": 0,
    }
