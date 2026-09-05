#!/usr/bin/env python3
"""Verify a formal PHAxis wheel in a new offline environment (plan-only by default)."""

from __future__ import annotations

import argparse
import base64
import csv
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid
import zipfile


# The clean-install producer can itself be launched from a sealed source tree;
# prevent its contract imports from creating unmanifested bytecode there.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if (SOURCE_ROOT / "phaxis").is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import sha256_file, sha256_json  # noqa: E402
from phaxis.contracts import ContractError  # noqa: E402
from phaxis.public_identity import validate_proposal_public_identity  # noqa: E402


RECEIPT_SCHEMA = "PHAxis-clean-install-verification-1.0"
PLAN_SCHEMA = "PHAxis-clean-install-verification-plan-1.0"
EXPECTED_SCHEMA = "PHAxis-clean-install-example-expected-identity-1.0"
SOURCE_SCHEMA = "PHAxis-source-release-manifest-2.0"
MODEL_BUNDLE_SCHEMA = "PHAxis-model-bundle-release-manifest-1.0"
WORKFLOW_SCHEMA = "PHAxis-analysis-workflow-manifest-1.0"
VERSION = "1.0.0"
REQUIRED_PEP639_LICENSE_FILES = (
    "LICENSE",
    "src/phaxis/_vendor/tomli/LICENSE.txt",
)
CAPSULE_SCHEMA = "PHAxis-portable-model-runtime-capsule-1.0"
REQUIRED_DEPLOYMENT_DISTRIBUTIONS = frozenset(
    {
        "imageio",
        "joblib",
        "matplotlib",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "pillow",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "tifffile",
        "timm",
        "torch",
        "torchvision",
    }
)
DEPLOYMENT_IMPORT_MODULES = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "imageio": "imageio",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "tifffile": "tifffile",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_REQUIRED_OUTPUTS = {
    "traits/traits.csv",
    "traits/image_traits.csv",
    "traits/detailed_root_statistics.csv",
    "traits/hair_instances.csv",
    "traits/analysis_metadata.csv",
    "distal_axis_profiles/distal_axis_profiles.csv",
}


class CleanInstallError(RuntimeError):
    """The formal clean-install contract failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanInstallError(message)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _read_json(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanInstallError(f"{role}: cannot read JSON: {error}") from error
    _require(isinstance(payload, dict), f"{role}: JSON object required")
    return payload


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(_is_sha(observed), f"{role}: {field} is invalid")
    _require(sha256_json(unsigned) == observed, f"{role}: {field} does not seal the receipt")
    return str(observed)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _project_file(root: Path, value: str | Path, role: str) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    lexical = candidate.absolute()
    _require(_inside(lexical, root), f"{role}: path must stay inside project root")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{role}: symlink traversal is forbidden")
    resolved = candidate.resolve()
    _require(_inside(resolved, root) and resolved.is_file(), f"{role}: file is absent")
    return resolved


def _project_directory(root: Path, value: str | Path, role: str) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    lexical = candidate.absolute()
    _require(_inside(lexical, root), f"{role}: path must stay inside project root")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{role}: symlink traversal is forbidden")
    resolved = candidate.resolve()
    _require(_inside(resolved, root) and resolved.is_dir(), f"{role}: directory is absent")
    return resolved


def _new_project_path(root: Path, value: str | Path, role: str) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    lexical = candidate.absolute()
    _require(_inside(lexical, root), f"{role}: path must stay inside project root")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{role}: symlink traversal is forbidden")
    resolved = candidate.resolve()
    _require(_inside(resolved, root), f"{role}: resolved path escapes project root")
    return resolved


def _safe_relative(value: Any, role: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{role}: relative path is absent")
    path = PurePosixPath(value.replace("\\", "/"))
    _require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts, f"{role}: unsafe relative path")
    _require(all(part and ":" not in part and not part.endswith((" ", ".")) for part in path.parts), f"{role}: unsafe Windows path")
    return path.as_posix()


def _source_release(root: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest = _read_json(manifest_path, "source-release manifest")
    _require(
        manifest.get("schema_version") == SOURCE_SCHEMA
        and manifest.get("distribution") == "phaxis"
        and manifest.get("version") == VERSION
        and manifest.get("release_mode") == "formal"
        and manifest.get("source_policy") == "explicit_path_bounded_allowlist",
        "source release is not formal PHAxis 1.0.0",
    )
    records = manifest.get("files")
    _require(isinstance(records, list) and records, "source-release file inventory is absent")
    _require(manifest.get("tree_identity_sha256") == sha256_json(records), "source-release tree identity mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"source-release row {index} is invalid")
        relative = _safe_relative(record.get("path"), f"source-release row {index}")
        _require(relative not in indexed and _is_sha(record.get("sha256")), f"source-release row {index} identity is invalid")
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"source-release file is absent: {relative}")
        _require(path.stat().st_size == record.get("bytes") and sha256_file(path) == record.get("sha256"), f"source-release file drift: {relative}")
        indexed[relative] = dict(record)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _require(actual == set(indexed), "source-release tree closure differs from manifest")
    return manifest_path, manifest, indexed


def _b64_sha(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _audit_wheel(wheel: Path, source_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(
        all(relative in source_records for relative in REQUIRED_PEP639_LICENSE_FILES),
        "source release omits a required PEP 639 license file",
    )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)) and names, "wheel contains duplicate or absent members")
        for name in names:
            _safe_relative(name.rstrip("/"), "wheel member")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        _require(len(metadata_names) == 1, "wheel must contain one dist-info/METADATA")
        dist_info = metadata_names[0].removesuffix("METADATA")
        metadata = BytesParser(policy=email_policy).parsebytes(archive.read(metadata_names[0]))
        _require(metadata.get("Name", "").casefold() == "phaxis" and metadata.get("Version") == VERSION, "wheel metadata is not phaxis 1.0.0")
        license_file_headers = metadata.get_all("License-File", [])
        _require(
            license_file_headers == list(REQUIRED_PEP639_LICENSE_FILES),
            "wheel METADATA License-File headers are not the exact PHAxis and vendored Tomli set",
        )
        expected_dist_info_licenses = {
            dist_info + "licenses/" + relative: (
                str(source_records[relative]["sha256"]),
                int(source_records[relative]["bytes"]),
            )
            for relative in REQUIRED_PEP639_LICENSE_FILES
        }
        for name, (expected_sha256, expected_bytes) in expected_dist_info_licenses.items():
            _require(name in names, f"wheel lacks required PEP 639 license member: {name}")
            data = archive.read(name)
            _require(
                hashlib.sha256(data).hexdigest() == expected_sha256
                and len(data) == expected_bytes,
                f"wheel PEP 639 license member differs from source authority: {name}",
            )
        observed_dist_info_licenses = {
            name for name in names if name.startswith(dist_info + "licenses/")
        }
        _require(
            observed_dist_info_licenses == set(expected_dist_info_licenses),
            "wheel PEP 639 license member closure is not the exact PHAxis and vendored Tomli set",
        )
        entry_name = dist_info + "entry_points.txt"
        _require(entry_name in names and "phaxis = phaxis.cli:main" in archive.read(entry_name).decode("utf-8"), "wheel does not install the canonical phaxis CLI")
        record_name = dist_info + "RECORD"
        _require(record_name in names, "wheel RECORD is absent")
        rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
        record_map = {row[0]: row[1:] for row in rows if len(row) == 3}
        _require(set(record_map) == set(names), "wheel RECORD member closure mismatch")
        for name in names:
            digest, size = record_map[name]
            if name == record_name:
                _require(digest == "" and size == "", "wheel RECORD self-row must be unhashed")
                continue
            data = archive.read(name)
            _require(digest == "sha256=" + _b64_sha(data) and size == str(len(data)), f"wheel RECORD mismatch: {name}")
        expected_code = {
            relative.removeprefix("src/"): (str(record["sha256"]), int(record["bytes"]))
            for relative, record in source_records.items()
            if relative.startswith("src/phaxis/")
        }
        observed_code = {
            name: (hashlib.sha256(archive.read(name)).hexdigest(), len(archive.read(name)))
            for name in names
            if name.startswith("phaxis/") and not name.endswith("/")
        }
        _require(observed_code == expected_code, "formal wheel PHAxis code is not the exact source-manifest package tree")
        unexpected_code = [name for name in names if not name.startswith(("phaxis/", dist_info))]
        _require(not unexpected_code, "wheel contains code/payload outside phaxis and dist-info")
    return {
        "filename": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": sha256_file(wheel),
        "distribution": "phaxis",
        "version": VERSION,
        "entry_point": "phaxis = phaxis.cli:main",
        "source_package_file_count": len(expected_code),
        "source_package_identity_sha256": sha256_json(
            [{"path": path, "sha256": digest, "bytes": size} for path, (digest, size) in sorted(expected_code.items())]
        ),
        "source_package_hashes_verified": True,
        "record_verified": True,
        "metadata_license_files": list(REQUIRED_PEP639_LICENSE_FILES),
        "pep639_license_member_count": len(expected_dist_info_licenses),
        "license_file_hashes_verified": True,
    }


def _dependency_lock(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    logical: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current += (" " if current else "") + stripped.rstrip("\\").strip()
        if not stripped.endswith("\\"):
            logical.append(current)
            current = ""
    _require(not current and logical, "dependency lock is empty or has an unterminated continuation")
    requirements: list[dict[str, Any]] = []
    for row in logical:
        lowered = row.casefold()
        hashes = sorted(set(re.findall(r"--hash=sha256:([0-9a-f]{64})", lowered)))
        pinned = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", row)
        _require(pinned is not None and hashes, "dependency lock must pin every requirement with SHA-256")
        _require(not any(token in lowered for token in ("-e ", "--editable", "--index-url", "--extra-index-url", " @ ", "http://", "https://")), "dependency lock contains an unaudited source/index directive")
        assert pinned is not None
        requirements.append(
            {
                "name": re.sub(r"[-_.]+", "-", pinned.group(1)).casefold(),
                "version": pinned.group(2),
                "allowed_sha256": hashes,
            }
        )
    _require(
        len({record["name"] for record in requirements}) == len(requirements),
        "dependency lock repeats a distribution",
    )
    return {
        "sha256": sha256_file(path),
        "requirements": requirements,
        "requirement_count": len(requirements),
        "pip_require_hashes": True,
    }


def _wheelhouse(path: Path) -> dict[str, Any]:
    records = []
    for item in sorted(path.iterdir(), key=lambda value: value.name.casefold()):
        _require(item.is_file() and not item.is_symlink() and item.suffix.casefold() == ".whl", "offline wheelhouse may contain only wheel files")
        try:
            with zipfile.ZipFile(item) as archive:
                _require(archive.testzip() is None, f"offline wheel is corrupt: {item.name}")
                metadata_names = [
                    name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
                ]
                _require(len(metadata_names) == 1, f"offline wheel metadata is invalid: {item.name}")
                metadata = BytesParser(policy=email_policy).parsebytes(
                    archive.read(metadata_names[0])
                )
        except (OSError, zipfile.BadZipFile) as error:
            raise CleanInstallError(f"offline wheel is unreadable: {item.name}: {error}") from error
        name = metadata.get("Name")
        version = metadata.get("Version")
        _require(isinstance(name, str) and bool(name) and isinstance(version, str) and bool(version), f"offline wheel name/version is absent: {item.name}")
        records.append(
            {
                "filename": item.name,
                "distribution": re.sub(r"[-_.]+", "-", name).casefold(),
                "version": version,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    _require(records, "offline wheelhouse is empty")
    return {"files": records, "wheelhouse_identity_sha256": sha256_json(records)}


def _canonical_expected_files(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and value, "expected canonical output files are absent")
    records = []
    for index, record in enumerate(value):
        _require(isinstance(record, Mapping), f"expected output row {index} is invalid")
        relative = _safe_relative(record.get("path"), f"expected output row {index}")
        _require(_is_sha(record.get("sha256")) and isinstance(record.get("bytes"), int) and record["bytes"] >= 0, f"expected output row {index} identity is invalid")
        records.append({"path": relative, "bytes": int(record["bytes"]), "sha256": str(record["sha256"])})
    _require(records == sorted(records, key=lambda row: row["path"]) and len({row["path"] for row in records}) == len(records), "expected output rows must be unique and path-sorted")
    paths = {row["path"] for row in records}
    _require(CANONICAL_REQUIRED_OUTPUTS.issubset(paths), "expected identity omits canonical traits/profiles outputs")
    _require(any(path.startswith("fusion/predictions/") and path.endswith(".json") for path in paths), "expected identity omits final inference prediction")
    return records


def _portable_capsule(
    *,
    project_root: Path,
    portable_capsule_root: Path,
    example_manifest: Path,
    proposal: Path,
    applied: Path,
    model_bundle: Path,
) -> dict[str, Any]:
    capsule = _project_directory(
        project_root, portable_capsule_root, "portable model/runtime capsule"
    )
    canonical = {
        "example": capsule
        / "model/examples/clean_install/release_example_manifest.json",
        "proposal": capsule / "model/assets/runtime/model_contract_proposal.json",
        "applied": capsule / "model/assets/runtime/applied_model_contract.json",
        "bundle": capsule / "model/assets/MODEL_BUNDLE_MANIFEST.json",
        "receipt": capsule / "model/examples/clean_install/receipt.json",
    }
    for role, path in canonical.items():
        _require(
            path.is_file() and not path.is_symlink(),
            f"portable capsule canonical {role} is absent or symlinked",
        )
    for role, external in (
        ("example", example_manifest),
        ("proposal", proposal),
        ("applied", applied),
        ("bundle", model_bundle),
    ):
        _require(
            sha256_file(canonical[role]) == sha256_file(external),
            f"portable capsule {role} differs from the formal authority",
        )

    bundle = _read_json(canonical["bundle"], "portable capsule model bundle")
    members = bundle.get("members")
    _require(isinstance(members, list) and bool(members), "portable capsule member inventory is absent")
    expected_paths = {"model/assets/MODEL_BUNDLE_MANIFEST.json"}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(members):
        _require(isinstance(row, Mapping), f"portable capsule member {index} is invalid")
        relative = _safe_relative(row.get("path"), f"portable capsule member {index}")
        path = (capsule / relative).resolve()
        _require(
            _inside(path, capsule)
            and path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == row.get("sha256")
            and path.stat().st_size == row.get("bytes"),
            f"portable capsule member {index} differs from the model-bundle seal",
        )
        expected_paths.add(relative)
        normalized.append(
            {
                "path": relative,
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
        )
    observed_paths = {
        path.relative_to(capsule).as_posix()
        for path in capsule.rglob("*")
        if path.is_file()
    }
    _require(
        observed_paths == expected_paths,
        "portable capsule contains unlisted or missing files",
    )

    receipt = _read_json(canonical["receipt"], "portable capsule receipt")
    capsule_identity = _sealed(
        receipt,
        "portable_capsule_identity_sha256",
        "portable capsule receipt",
    )
    _require(
        receipt.get("schema_version") == CAPSULE_SCHEMA
        and receipt.get("status")
        == "completed_self_contained_raw_to_profiles_runtime"
        and receipt.get("authoring_workspace_paths_required") is False
        and receipt.get("root_subprocess_python_rebound_to_active_interpreter")
        is True
        and receipt.get("blind_images_used") == 0,
        "portable capsule receipt is not release eligible",
    )

    workflow = _read_json(canonical["example"], "portable workflow manifest")
    base = canonical["example"].parent

    def local_path(value: Any, role: str, *, directory: bool = False) -> Path:
        _require(isinstance(value, str) and bool(value), f"{role} path is absent")
        supplied = Path(value)
        _require(not supplied.is_absolute(), f"{role} is authoring-host absolute")
        path = (base / supplied).resolve()
        _require(_inside(path, capsule), f"{role} escapes portable capsule")
        _require(
            path.is_dir() if directory else path.is_file(),
            f"{role} is absent from portable capsule",
        )
        return path

    def local_ref(value: Any, role: str) -> Path:
        _require(isinstance(value, Mapping), f"{role} locked reference is absent")
        path = local_path(value.get("path"), role)
        _require(sha256_file(path) == value.get("sha256"), f"{role} hash differs")
        return path

    root_section = workflow.get("root_provider")
    stageb_section = workflow.get("stageb")
    _require(
        isinstance(root_section, Mapping) and isinstance(stageb_section, Mapping),
        "portable workflow root/Stage-B sections are absent",
    )
    _require(
        "python_executable" not in root_section,
        "portable workflow pins an authoring Python executable",
    )
    local_path(str(root_section.get("project") or "."), "root project", directory=True)
    local_path(root_section.get("image_root"), "root image root", directory=True)
    local_path(stageb_section.get("image_root"), "Stage-B image root", directory=True)
    bundle_ref = root_section.get("bundle")
    _require(isinstance(bundle_ref, Mapping), "portable workflow bundle ref is absent")
    root_bundle = local_path(bundle_ref.get("path"), "root bundle", directory=True)
    _require(
        sha256_file(root_bundle / "root_provider_bundle.json")
        == bundle_ref.get("registry_sha256"),
        "portable root bundle registry differs",
    )
    local_ref(workflow.get("model_contract_proposal"), "model-contract proposal")
    for field in (
        "input_manifest",
        "acquisition_gate",
        "deployment_metadata",
        "canonical_manifest",
        "deployment_manifest",
        "deployment_lock",
    ):
        local_ref(root_section.get(field), f"root {field}")
    _require(
        root_section.get("reference_registry") is None,
        "one-task clean install cannot execute the exact283 reference audit",
    )
    for field in (
        "input_manifest",
        "candidate_manifest",
        "selected_model_metadata",
        "selection_receipt",
    ):
        local_ref(stageb_section.get(field), f"Stage-B {field}")
    checkpoint_refs = stageb_section.get("checkpoints")
    _require(
        isinstance(checkpoint_refs, list) and len(checkpoint_refs) == 5,
        "portable workflow checkpoint closure is not exactly five",
    )
    for index, ref in enumerate(checkpoint_refs):
        local_ref(ref, f"Stage-B checkpoint {index}")
    traits = workflow.get("traits")
    profiles = workflow.get("distal_axis_profiles", workflow.get("profiles"))
    _require(isinstance(traits, Mapping) and isinstance(profiles, Mapping), "portable traits/profile sections are absent")
    local_ref(traits.get("metadata_csv"), "traits metadata")
    local_ref(profiles.get("contract_json"), "distal-axis profile contract")
    release = workflow.get("release_example")
    _require(isinstance(release, Mapping), "portable release-example block is absent")
    source = local_path(release.get("source_image_relpath"), "release-example source image")
    _require(
        sha256_file(source) == release.get("source_image_sha256"),
        "portable source image differs from its lock",
    )
    return {
        "root": capsule,
        "identity": capsule_identity,
        "tree_identity_sha256": sha256_json(normalized),
        "members": len(normalized),
        "canonical": canonical,
    }


def _authority_context(
    *, project_root: Path, wheel: Path, source_release_root: Path,
    applied_model_contract: Path, model_contract_proposal: Path,
    model_bundle_manifest: Path, example_manifest: Path,
    portable_capsule_root: Path,
    expected_example_identity: Path, dependency_lock: Path, wheelhouse: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    source_root = _project_directory(root, source_release_root, "source release")
    manifest_path, source, source_records = _source_release(source_root)
    wheel_path = _project_file(root, wheel, "formal wheel")
    wheel_audit = _audit_wheel(wheel_path, source_records)
    proposal_path = _project_file(root, model_contract_proposal, "model-contract proposal")
    proposal = _read_json(proposal_path, "model-contract proposal")
    proposal_identity = _sealed(proposal, "model_contract_identity_sha256", "model-contract proposal")
    promotion = proposal.get("promotion")
    red_lines = proposal.get("red_lines")
    stageb_binding = promotion.get("stageb_binding") if isinstance(promotion, Mapping) else None
    checkpoints = stageb_binding.get("checkpoint_sha256") if isinstance(stageb_binding, Mapping) else None
    formal_sources = promotion.get("formal_gate_source_sha256") if isinstance(promotion, Mapping) else None
    formal_identities = promotion.get("formal_gate_identity_sha256") if isinstance(promotion, Mapping) else None
    _require(
        proposal.get("schema_version") == "PHAxis-model-contract-1.0.0"
        and proposal.get("product") == "PHAxis"
        and proposal.get("product_version") == VERSION
        and proposal.get("formal_release_status") == "passed_proposal_not_official"
        and isinstance(promotion, Mapping)
        and promotion.get("schema_version") == "PHAxis-model-contract-promotion-1.0"
        and promotion.get("status") == "validated_proposal_not_applied"
        and promotion.get("official_apply_performed") is False,
        "proposal is not the passed unapplied PHAxis 1.0.0 authority",
    )
    _require(
        isinstance(red_lines, Mapping)
        and red_lines.get("blind_images_used") == 0
        and red_lines.get("canonical_annotations_read_during_inference") is False
        and red_lines.get("condition_metadata_used_for_routing") is False
        and red_lines.get("root_cap_region_statistics_included") is False,
        "proposal inference red lines changed",
    )
    _require(
        isinstance(stageb_binding, Mapping)
        and isinstance(checkpoints, list)
        and len(checkpoints) == 5
        and len(set(checkpoints)) == 5
        and all(_is_sha(value) for value in checkpoints)
        and isinstance(formal_sources, Mapping)
        and all(
            _is_sha(formal_sources.get(role))
            for role in ("train399_candidate", "train399_selection", "train399_evaluation", "root_exact283")
        )
        and isinstance(formal_identities, Mapping)
        and all(
            _is_sha(formal_identities.get(field))
            for field in (
                "candidate_bundle_identity_sha256",
                "selection_receipt_identity_sha256",
                "selected_model_metadata_identity_sha256",
                "root_exact283_audit_identity_sha256",
            )
        )
        and all(
            formal_identities.get(field) == stageb_binding.get(field)
            for field in (
                "candidate_bundle_identity_sha256",
                "selection_receipt_identity_sha256",
                "selected_model_metadata_identity_sha256",
            )
        ),
        "proposal Stage-B/formal Gate bindings are incomplete",
    )
    try:
        proposal_public = validate_proposal_public_identity(proposal)
    except ContractError as error:
        raise CleanInstallError(f"model-contract proposal public identity is invalid: {error}") from error
    applied_path = _project_file(root, applied_model_contract, "applied model contract")
    applied = _read_json(applied_path, "applied model contract")
    applied_identity = _sealed(applied, "model_contract_identity_sha256", "applied model contract")
    applied_promotion = applied.get("promotion")
    _require(applied.get("schema_version") == "PHAxis-model-contract-1.0.0" and applied.get("product") == "PHAxis" and applied.get("product_version") == VERSION and applied.get("formal_release_status") == "passed", "applied contract is not formal PHAxis 1.0.0")
    _require(isinstance(applied_promotion, Mapping) and applied_promotion.get("status") == "applied_formal_release" and applied_promotion.get("official_apply_performed") is True, "model contract was not formally applied")
    _require(applied_promotion.get("proposal_file_sha256") == sha256_file(proposal_path) and applied_promotion.get("proposal_identity_sha256") == proposal_identity, "applied contract/proposal binding mismatch")
    try:
        applied_public = validate_proposal_public_identity(applied)
    except ContractError as error:
        raise CleanInstallError(f"applied model-contract public identity is invalid: {error}") from error
    _require(applied_public == proposal_public, "applied/proposal public identity changed")
    root_expert = applied.get("root_expert")
    hair_expert = applied.get("hair_identity_count_expert")
    _require(isinstance(root_expert, Mapping) and isinstance(hair_expert, Mapping) and _is_sha(root_expert.get("bundle_identity_sha256")), "applied expert identities are incomplete")
    bundle_path = _project_file(root, model_bundle_manifest, "model bundle manifest")
    bundle = _read_json(bundle_path, "model bundle manifest")
    bundle_identity = _sealed(bundle, "model_bundle_manifest_identity_sha256", "model bundle manifest")
    _require(
        bundle.get("schema_version") == MODEL_BUNDLE_SCHEMA
        and bundle.get("status") == "completed_final_immutable_bundle"
        and bundle.get("product") == "PHAxis"
        and bundle.get("product_version") == VERSION,
        "model bundle manifest is not the completed final PHAxis 1.0.0 bundle",
    )
    expected_public = {
        "model_bundle_id": applied.get("model_bundle_id"),
        "root_expert_id": root_expert.get("expert_id"),
        "root_bundle_identity_sha256": root_expert.get("bundle_identity_sha256"),
        "hair_identity_count_expert": hair_expert.get("expert_id"),
    }
    root_bundle = bundle.get("root_provider_bundle")
    checkpoints = bundle.get("stageb_checkpoints")
    members = bundle.get("members")
    _require(
        bundle.get("applied_model_contract_sha256") == sha256_file(applied_path)
        and bundle.get("applied_model_contract_identity_sha256") == applied_identity,
        "model bundle/applied-contract binding mismatch",
    )
    _require(
        bundle.get("model_contract_proposal_sha256") == sha256_file(proposal_path)
        and bundle.get("model_contract_proposal_identity_sha256") == proposal_identity,
        "model bundle/proposal binding mismatch",
    )
    _require(
        all(bundle.get(field) == value for field, value in expected_public.items())
        and isinstance(root_bundle, Mapping)
        and root_bundle.get("bundle_identity_sha256")
        == expected_public["root_bundle_identity_sha256"],
        "model bundle public expert/root identity mismatch",
    )
    _require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 5
        and [member.get("member_index") for member in checkpoints if isinstance(member, Mapping)]
        == list(range(5))
        and len({member.get("sha256") for member in checkpoints if isinstance(member, Mapping)}) == 5
        and all(
            isinstance(member, Mapping) and _is_sha(member.get("sha256"))
            for member in checkpoints
        ),
        "model bundle does not contain five distinct ordered train399 checkpoints",
    )
    _require(isinstance(members, list) and members, "model bundle member inventory is absent")
    normalized_members: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        _require(isinstance(member, Mapping), f"model bundle member {index} is invalid")
        relative = _safe_relative(member.get("path"), f"model bundle member {index}")
        _require(
            _is_sha(member.get("sha256"))
            and isinstance(member.get("bytes"), int)
            and member.get("bytes") >= 0,
            f"model bundle member {index} identity is invalid",
        )
        normalized_members.append({**dict(member), "path": relative})
    _require(
        normalized_members == sorted(normalized_members, key=lambda row: row["path"])
        and len({row["path"] for row in normalized_members}) == len(normalized_members)
        and bundle.get("member_count") == len(normalized_members)
        and bundle.get("bundle_sha256") == sha256_json(normalized_members)
        and bundle.get("bundle_size_bytes")
        == sum(int(row["bytes"]) for row in normalized_members),
        "model bundle member inventory identity/closure mismatch",
    )
    _require(
        bundle.get("blind_images_used") == 0
        and bundle.get("root_cap_region_statistics_included") is False
        and bundle.get("historical_or_provisional_backfill_used") is False,
        "model bundle violates formal release red lines",
    )
    manifest = _project_file(root, example_manifest, "example workflow manifest")
    example = _read_json(manifest, "example workflow manifest")
    manifest_identity = _sealed(example, "manifest_identity_sha256", "example workflow manifest")
    guards = example.get("guards")
    _require(example.get("schema_version") == WORKFLOW_SCHEMA and isinstance(guards, Mapping), "example workflow manifest schema/guards are invalid")
    _require(guards.get("blind_images_used") == 0 and guards.get("canonical_annotations_read") is False and guards.get("condition_metadata_used_for_routing") is False and guards.get("root_cap_region_output") is False, "example manifest violates inference red lines")
    capsule = _portable_capsule(
        project_root=root,
        portable_capsule_root=portable_capsule_root,
        example_manifest=manifest,
        proposal=proposal_path,
        applied=applied_path,
        model_bundle=bundle_path,
    )
    expected_path = _project_file(root, expected_example_identity, "expected example identity")
    expected = _read_json(expected_path, "expected example identity")
    expected_identity = _sealed(expected, "expected_identity_receipt_identity_sha256", "expected example identity")
    _require(expected.get("schema_version") == EXPECTED_SCHEMA and expected.get("status") == "locked_final_real_example_before_clean_install", "expected example identity is not a final prelock")
    _require(expected.get("input_kind") == "real_nonblind_release_example" and expected.get("release_authorized") is True and expected.get("development_or_synthetic_smoke") is False and expected.get("blind_images_used") == 0, "development/synthetic/non-final input cannot authorize a formal receipt")
    _require(
        isinstance(expected.get("tasks"), int) and expected.get("tasks") > 0,
        "final example task count is invalid",
    )
    expected_files = _canonical_expected_files(expected.get("canonical_output_files"))
    expected_output_identity = sha256_json(expected_files)
    _require(expected.get("expected_example_output_identity_sha256") == expected_output_identity, "prelocked expected output identity mismatch")
    expected_bindings = {
        "example_manifest_sha256": sha256_file(manifest),
        "example_manifest_identity_sha256": manifest_identity,
        "model_contract_proposal_sha256": sha256_file(proposal_path),
        "model_contract_proposal_identity_sha256": proposal_identity,
        "applied_model_contract_sha256": sha256_file(applied_path),
        "applied_model_contract_identity_sha256": applied_identity,
        "model_bundle_manifest_sha256": sha256_file(bundle_path),
        "model_bundle_manifest_identity_sha256": bundle_identity,
        "source_release_manifest_sha256": sha256_file(manifest_path),
        "source_release_tree_identity_sha256": source["tree_identity_sha256"],
        "formal_wheel_sha256": wheel_audit["sha256"],
        "portable_capsule_identity_sha256": capsule["identity"],
        "portable_capsule_tree_identity_sha256": capsule[
            "tree_identity_sha256"
        ],
        **expected_public,
    }
    _require(all(expected.get(field) == value for field, value in expected_bindings.items()), "expected example prelock cross-binding mismatch")
    lock_path = _project_file(root, dependency_lock, "dependency lock")
    wheelhouse_path = _project_directory(root, wheelhouse, "offline wheelhouse")
    dependency_lock_record = _dependency_lock(lock_path)
    locked_distributions = {
        row["name"] for row in dependency_lock_record["requirements"]
    }
    _require(
        REQUIRED_DEPLOYMENT_DISTRIBUTIONS.issubset(locked_distributions),
        "offline dependency lock omits audited deployment runtime distributions: "
        + ", ".join(
            sorted(REQUIRED_DEPLOYMENT_DISTRIBUTIONS - locked_distributions)
        ),
    )
    wheelhouse_record = _wheelhouse(wheelhouse_path)
    for requirement in dependency_lock_record["requirements"]:
        matches = [
            record
            for record in wheelhouse_record["files"]
            if record["distribution"] == requirement["name"]
            and record["version"] == requirement["version"]
            and record["sha256"] in requirement["allowed_sha256"]
        ]
        _require(
            matches,
            "offline wheelhouse does not contain a name/version/hash-authorized wheel "
            f"for dependency {requirement['name']}=={requirement['version']}",
        )
    return {
        "project_root": root,
        "source_release_root": source_root,
        "source_manifest_path": manifest_path,
        "source_manifest": source,
        "wheel_path": wheel_path,
        "wheel_audit": wheel_audit,
        "proposal_path": proposal_path,
        "proposal_identity": proposal_identity,
        "applied_path": applied_path,
        "applied_identity": applied_identity,
        "bundle_path": bundle_path,
        "bundle_identity": bundle_identity,
        "example_manifest_path": manifest,
        "example_manifest_identity": manifest_identity,
        "expected_path": expected_path,
        "expected_identity": expected_identity,
        "expected_files": expected_files,
        "expected_output_identity": expected_output_identity,
        "dependency_lock_path": lock_path,
        "dependency_lock": dependency_lock_record,
        "wheelhouse_path": wheelhouse_path,
        "wheelhouse": wheelhouse_record,
        "public": expected_public,
        "tasks": expected.get("tasks"),
        "capsule": capsule,
    }


def clean_install_plan(**kwargs: Any) -> dict[str, Any]:
    context = _authority_context(**kwargs)
    _require(isinstance(context["tasks"], int) and context["tasks"] > 0, "final example task count is invalid")
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "validated_not_executed",
        "default_plan_only": True,
        "execute_requires_explicit_flag": True,
        "environment_policy": "new_project_local_venv_no_system_site_packages",
        "dependency_strategy": "offline_wheelhouse_plus_sha256_requirements_lock",
        "formal_wheel_sha256": context["wheel_audit"]["sha256"],
        "source_release_manifest_sha256": sha256_file(context["source_manifest_path"]),
        "applied_model_contract_identity_sha256": context["applied_identity"],
        "model_contract_proposal_identity_sha256": context["proposal_identity"],
        "model_bundle_manifest_identity_sha256": context["bundle_identity"],
        "portable_capsule_identity_sha256": context["capsule"]["identity"],
        "portable_capsule_tree_identity_sha256": context["capsule"][
            "tree_identity_sha256"
        ],
        "expected_identity_receipt_identity_sha256": context["expected_identity"],
        "expected_example_output_identity_sha256": context["expected_output_identity"],
        "tasks": context["tasks"],
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    plan["plan_identity_sha256"] = sha256_json(plan)
    return plan


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(argv), cwd=cwd, env=dict(env), check=False, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise CleanInstallError(f"clean-install command failed ({completed.returncode}): {argv[0]} {argv[1] if len(argv) > 1 else ''}\n{completed.stderr[-2000:]}")
    return completed


def _command(runner: CommandRunner, actual: Sequence[str], audit: Sequence[str], *, cwd: Path, env: Mapping[str, str], records: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    completed = runner(list(actual), cwd=cwd, env=dict(env))
    if completed.returncode != 0:
        raise CleanInstallError(f"clean-install command failed ({completed.returncode}): {' '.join(audit)}")
    records.append({"argv": list(audit), "returncode": 0})
    return completed


def _output_records(output: Path, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary_paths = {
        "fusion/fusion_summary.json",
        "traits/summary.json",
        "distal_axis_profiles/summary.json",
    }
    actual_paths: set[str] = set()
    for directory in ("fusion", "traits", "distal_axis_profiles"):
        root = output / directory
        _require(root.is_dir() and not root.is_symlink(), f"canonical output directory is absent: {directory}")
        for path in root.rglob("*"):
            if path.is_file():
                _require(not path.is_symlink(), f"canonical output symlink is forbidden: {path}")
                relative = path.relative_to(output).as_posix()
                if relative not in summary_paths:
                    actual_paths.add(relative)
    expected_paths = {str(record["path"]) for record in expected}
    _require(
        actual_paths == expected_paths,
        "canonical example output file-set closure differs from the prelock",
    )
    observed = []
    for record in expected:
        relative = str(record["path"])
        path = output / relative
        _require(path.is_file() and not path.is_symlink(), f"canonical example output is absent: {relative}")
        observed.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return observed


def _verify_completed_output(context: Mapping[str, Any], output: Path, plan_path: Path) -> dict[str, Any]:
    _require(not (output.parent / "plan-only-analysis").exists(), "plan-only CLI unexpectedly created analysis output")
    plan = _read_json(plan_path, "installed CLI plan")
    _sealed(plan, "plan_identity_sha256", "installed CLI plan")
    _require(plan.get("schema_version") == "PHAxis-analysis-workflow-plan-1.0" and plan.get("status") == "planned_not_executed" and plan.get("tasks") == context["tasks"], "installed CLI plan-only verification failed")
    state = _read_json(output / "workflow_state.json", "clean-install workflow state")
    state_identity = _sealed(state, "state_identity_sha256", "clean-install workflow state")
    attempts = state.get("execution_attempts")
    latest_id = state.get("latest_execution_attempt_id")
    _require(
        state.get("schema_version") == "PHAxis-analysis-workflow-state-1.1"
        and state.get("status") == "completed"
        and isinstance(attempts, list)
        and len(attempts) == 1
        and latest_id == 1
        and state.get("latest_execution_fresh_direct_benchmark_eligible") is True,
        "clean-install workflow did not complete as one fresh execution",
    )
    attempt = attempts[0]
    stages = attempt.get("stages") if isinstance(attempt, Mapping) else None
    expected_stages = {
        "root_provider",
        "stageb_train399",
        "fusion",
        "traits",
        "distal_axis_profiles",
    }
    _require(
        isinstance(attempt, Mapping)
        and attempt.get("status") == "completed"
        and attempt.get("resume_requested") is False
        and attempt.get("resume_or_cache_used") is False
        and attempt.get("fresh_direct_benchmark_eligible") is True
        and attempt.get("review_overlays_excluded_from_benchmark_scope") is True
        and isinstance(stages, list)
        and {row.get("stage") for row in stages if isinstance(row, Mapping)}
        == expected_stages
        and all(
            isinstance(row, Mapping) and row.get("execution_status") == "executed_fresh"
            for row in stages
        ),
        "clean-install workflow used resume/cache or omitted a formal stage",
    )
    _require(
        state.get("condition_metadata_used_for_routing") is False
        and state.get("canonical_annotations_read") is False
        and state.get("blind_images_used") == 0
        and state.get("root_cap_region_output") is False,
        "clean-install workflow state violates inference red lines",
    )
    public = context["public"]
    summaries = (
        (output / "fusion/fusion_summary.json", "summary_identity_sha256", "fusion", "images", "PHAxis-fusion-run-1.1"),
        (output / "traits/summary.json", "export_identity_sha256", "traits", "tasks", "PHAxis-trait-export-1.0"),
        (output / "distal_axis_profiles/summary.json", "export_identity_sha256", "profiles", "tasks", "PHAxis-distal-axis-profile-export-1.0.0"),
    )
    summary_bindings: dict[str, dict[str, str]] = {}
    for path, identity_field, role, count_field, schema in summaries:
        payload = _read_json(path, f"{role} summary")
        identity = _sealed(payload, identity_field, f"{role} summary")
        _require(payload.get("schema_version") == schema and payload.get("status") == "completed" and payload.get(count_field) == context["tasks"], f"{role} summary is incomplete")
        _require(payload.get("model_contract_proposal_sha256") == sha256_file(context["proposal_path"]) and payload.get("model_contract_proposal_identity_sha256") == context["proposal_identity"], f"{role} summary proposal binding mismatch")
        if role == "fusion":
            _require(payload.get("model_bundle_id") == public["model_bundle_id"] and payload.get("root_expert") == public["root_expert_id"] and payload.get("hair_identity_count_expert") == public["hair_identity_count_expert"] and payload.get("blind_images_used") == 0 and payload.get("root_cap_region_output") is False and payload.get("condition_metadata_used_for_routing") is False and payload.get("canonical_annotations_read") is False, "fusion public identity/red lines changed")
        else:
            root_cap_field = "root_cap_region_statistics_included" if role == "traits" else "root_cap_region_output"
            _require(payload.get("model_bundle_id") == public["model_bundle_id"] and payload.get("root_expert_id") == public["root_expert_id"] and payload.get("blind_images_used") == 0 and payload.get("canonical_annotations_read") is False and payload.get("condition_metadata_used_for_model_routing") is False and payload.get(root_cap_field) is False, f"{role} public identity/red lines changed")
            if role == "traits":
                _require(payload.get("hair_identity_count_expert") == public["hair_identity_count_expert"], "traits hair expert identity changed")
        if role == "traits":
            for field, filename in (
                ("traits_sha256", "traits.csv"),
                ("image_traits_sha256", "image_traits.csv"),
                ("detailed_root_statistics_sha256", "detailed_root_statistics.csv"),
                ("hair_instances_sha256", "hair_instances.csv"),
                ("analysis_metadata_sha256", "analysis_metadata.csv"),
            ):
                _require(payload.get(field) == sha256_file(path.parent / filename), f"traits summary file binding mismatch: {filename}")
        if role == "profiles":
            _require(payload.get("profiles_csv_sha256") == sha256_file(path.parent / "distal_axis_profiles.csv"), "profile summary file binding mismatch")
            _require(
                payload.get("traits_csv_sha256")
                == sha256_file(output / "traits/traits.csv")
                and payload.get("hair_instances_csv_sha256")
                == sha256_file(output / "traits/hair_instances.csv"),
                "profile summary does not bind the verified trait/hair tables",
            )
        summary_bindings[role] = {"file_sha256": sha256_file(path), "identity_sha256": identity}
    observed = _output_records(output, context["expected_files"])
    identity = sha256_json(observed)
    _require(observed == context["expected_files"] and identity == context["expected_output_identity"], "clean-install example output differs from prelocked path-independent identity")
    return {"state_file_sha256": sha256_file(output / "workflow_state.json"), "state_identity_sha256": state_identity, "summaries": summary_bindings, "canonical_output_files": observed, "example_output_identity_sha256": identity}


def _relocated_plan_path_proof(plan: Mapping[str, Any], release_root: Path) -> dict[str, Any]:
    """Prove every absolute string in the executable plan stays in the copy.

    The plan contains each argv token separately, so this covers root-provider
    subprocess authorities as well as Stage-B, traits and profile paths.  A
    Windows authoring path is rejected on non-Windows too rather than being
    misinterpreted as a relative POSIX string.
    """

    absolute: list[dict[str, str]] = []

    def visit(value: Any, logical: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{logical}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{logical}[{index}]")
        elif isinstance(value, str):
            windows_absolute = PureWindowsPath(value).is_absolute()
            posix_absolute = PurePosixPath(value).is_absolute()
            if not windows_absolute and not posix_absolute:
                return
            _require(
                not windows_absolute or os.name == "nt",
                f"relocated plan contains an authoring-host Windows path: {logical}",
            )
            resolved = Path(value).resolve()
            _require(
                _inside(resolved, release_root),
                f"relocated plan absolute path escapes disposable release root: {logical}",
            )
            absolute.append(
                {
                    "field": logical,
                    "release_relative_path": resolved.relative_to(release_root).as_posix(),
                }
            )

    visit(plan, "plan")
    absolute.sort(key=lambda row: (row["field"], row["release_relative_path"]))
    _require(bool(absolute), "relocated workflow plan exposes no auditable absolute paths")
    return {
        "absolute_path_count": len(absolute),
        "absolute_paths": absolute,
        "absolute_path_identity_sha256": sha256_json(absolute),
        "all_absolute_runtime_paths_inside_disposable_release_root": True,
    }


def execute_clean_install_verification(
    *, project_root: Path, wheel: Path, source_release_root: Path,
    applied_model_contract: Path, model_contract_proposal: Path,
    model_bundle_manifest: Path, example_manifest: Path,
    portable_capsule_root: Path,
    expected_example_identity: Path, dependency_lock: Path, wheelhouse: Path,
    base_python: Path, work_root: Path, output: Path,
    cuda_visible_devices: str,
    runner: CommandRunner = _run,
) -> dict[str, Any]:
    context = _authority_context(
        project_root=project_root,
        wheel=wheel,
        source_release_root=source_release_root,
        applied_model_contract=applied_model_contract,
        model_contract_proposal=model_contract_proposal,
        model_bundle_manifest=model_bundle_manifest,
        example_manifest=example_manifest,
        portable_capsule_root=portable_capsule_root,
        expected_example_identity=expected_example_identity,
        dependency_lock=dependency_lock,
        wheelhouse=wheelhouse,
    )
    root = context["project_root"]
    python = _project_file(root, base_python, "base Python")
    _require(
        "envs" in {part.casefold() for part in python.relative_to(root).parts},
        "base Python must come from a project-local conda environment",
    )
    requested_work = _new_project_path(root, work_root, "clean-install work root")
    destination = _new_project_path(root, output, "clean-install receipt output")
    _require(
        not _inside(destination, requested_work),
        "formal receipt output must be outside the disposable clean-install work root",
    )
    _require(
        not requested_work.exists() and not destination.exists(),
        "work root and receipt output must both be absent",
    )
    _require(
        re.fullmatch(r"\d+(?:,\d+)*", cuda_visible_devices) is not None
        and len(set(cuda_visible_devices.split(",")))
        == len(cuda_visible_devices.split(",")),
        "CUDA_VISIBLE_DEVICES mapping is invalid or contains duplicates",
    )
    preflight_env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    preflight = runner(["nvidia-smi"], cwd=root, env=preflight_env)
    _require(
        preflight.returncode == 0 and bool(preflight.stdout.strip()),
        "nvidia-smi preflight failed before isolated clean-install workflow",
    )
    preflight_sha256 = hashlib.sha256(preflight.stdout.encode("utf-8")).hexdigest()

    requested_work.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_work.name}.attempt-",
            dir=requested_work.parent,
        )
    )
    receipt: dict[str, Any] | None = None
    try:
        release_root = attempt / "release"
        release_root.mkdir()
        copied_source = release_root / "source"
        copied_capsule = release_root / "model_capsule"
        copied_dist = release_root / "dist"
        copied_dependencies = release_root / "offline_dependencies"
        shutil.copytree(context["source_release_root"], copied_source)
        shutil.copytree(context["capsule"]["root"], copied_capsule)
        shutil.copytree(context["wheelhouse_path"], copied_dependencies / "wheelhouse")
        copied_dist.mkdir()
        copied_wheel = copied_dist / context["wheel_path"].name
        shutil.copyfile(context["wheel_path"], copied_wheel)
        copied_lock = copied_dependencies / "requirements.lock.txt"
        shutil.copyfile(context["dependency_lock_path"], copied_lock)
        copied_expected = release_root / "expected_identity.json"
        shutil.copyfile(context["expected_path"], copied_expected)
        copied_manifest = (
            copied_capsule
            / "model/examples/clean_install/release_example_manifest.json"
        )
        _require(
            sha256_file(copied_manifest) == sha256_file(context["example_manifest_path"])
            and sha256_file(copied_wheel) == context["wheel_audit"]["sha256"]
            and sha256_file(copied_lock) == context["dependency_lock"]["sha256"],
            "disposable release-root copy changed a formal authority",
        )

        env_root = release_root / "env"
        env_python = env_root / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        cli = env_root / ("Scripts/phaxis.exe" if os.name == "nt" else "bin/phaxis")
        run_root = release_root / "run"
        run_root.mkdir()
        plan_path = run_root / "plan.json"
        plan_analysis = run_root / "plan-only-analysis"
        analysis = run_root / "example-analysis"
        clean_env = dict(os.environ)
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "__PYVENV_LAUNCHER__",
        ):
            clean_env.pop(name, None)
        clean_env.update(
            {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1",
            }
        )
        commands: list[dict[str, Any]] = []
        _command(
            runner,
            [str(python), "-B", "-I", "-m", "venv", str(env_root)],
            ["<PROJECT_CONDA_PYTHON>", "-B", "-I", "-m", "venv", "<RELEASE_ROOT_ENV>"],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        _require(env_python.is_file(), "new isolated environment Python was not created")
        _command(
            runner,
            [
                str(env_python), "-B", "-I", "-m", "pip", "--isolated", "install",
                "--disable-pip-version-check", "--no-input", "--require-hashes",
                "--only-binary=:all:", "--no-index", "--find-links",
                str(copied_dependencies / "wheelhouse"), "-r", str(copied_lock),
            ],
            [
                "<ENV_PYTHON>", "-B", "-I", "-m", "pip", "--isolated", "install",
                "--disable-pip-version-check", "--no-input", "--require-hashes",
                "--only-binary=:all:", "--no-index", "--find-links",
                "<RELEASE_ROOT_WHEELHOUSE>", "-r", "<RELEASE_ROOT_SHA256_LOCK>",
            ],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        _command(
            runner,
            [
                str(env_python), "-B", "-I", "-m", "pip", "--isolated", "install",
                "--disable-pip-version-check", "--no-input", "--no-deps",
                "--no-index", str(copied_wheel),
            ],
            [
                "<ENV_PYTHON>", "-B", "-I", "-m", "pip", "--isolated", "install",
                "--disable-pip-version-check", "--no-input", "--no-deps",
                "--no-index", "<RELEASE_ROOT_FORMAL_PHAXIS_WHEEL>",
            ],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        _require(cli.is_file(), "formal wheel did not install the phaxis CLI entry point")
        _command(
            runner,
            [str(env_python), "-B", "-I", "-m", "pip", "--isolated", "check"],
            ["<ENV_PYTHON>", "-B", "-I", "-m", "pip", "--isolated", "check"],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        inventory_result = _command(
            runner,
            [str(env_python), "-B", "-I", "-m", "pip", "--isolated", "list", "--format=json"],
            ["<ENV_PYTHON>", "-B", "-I", "-m", "pip", "--isolated", "list", "--format=json"],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        try:
            inventory = sorted(
                json.loads(inventory_result.stdout),
                key=lambda row: str(row["name"]).casefold(),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise CleanInstallError("installed distribution inventory is invalid") from error
        installed_names = {
            re.sub(r"[-_.]+", "-", str(row.get("name", ""))).casefold()
            for row in inventory
        }
        _require(
            "phaxis" in installed_names
            and REQUIRED_DEPLOYMENT_DISTRIBUTIONS.issubset(installed_names),
            "isolated environment omits PHAxis or an audited deployment distribution",
        )
        modules_literal = json.dumps(sorted(DEPLOYMENT_IMPORT_MODULES))
        import_code = (
            "import importlib,json,phaxis,site,sys;"
            f"names={modules_literal};"
            "mods={n:getattr(importlib.import_module(n),'__file__',None) for n in names};"
            "mods['phaxis']=getattr(phaxis,'__file__',None);"
            "print(json.dumps({'version':phaxis.__version__,'user_site_enabled':bool(site.ENABLE_USER_SITE),"
            "'executable':sys.executable,'prefix':sys.prefix,'modules':mods}))"
        )
        import_result = _command(
            runner,
            [str(env_python), "-B", "-I", "-c", import_code],
            ["<ENV_PYTHON>", "-B", "-I", "-c", "<DEPLOYMENT_IMPORT_AND_ISOLATION_PROBE>"],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        try:
            import_probe = json.loads(import_result.stdout)
        except json.JSONDecodeError as error:
            raise CleanInstallError("PHAxis deployment import probe output is invalid") from error
        _require(
            import_probe.get("version") == VERSION
            and import_probe.get("user_site_enabled") is False
            and Path(str(import_probe.get("executable"))).resolve() == env_python.resolve()
            and Path(str(import_probe.get("prefix"))).resolve() == env_root.resolve(),
            "installed PHAxis executable/prefix/isolation changed",
        )
        module_files = import_probe.get("modules")
        _require(isinstance(module_files, Mapping), "deployment module path probe is absent")
        module_relpaths: dict[str, str] = {}
        for module in (*sorted(DEPLOYMENT_IMPORT_MODULES), "phaxis"):
            raw = module_files.get(module)
            _require(isinstance(raw, str) and raw, f"deployment module has no file: {module}")
            module_path = Path(raw).resolve()
            _require(
                _inside(module_path, env_root),
                f"deployment module was imported outside the clean environment: {module}",
            )
            module_relpaths[module] = module_path.relative_to(env_root).as_posix()

        version_result = _command(
            runner,
            [str(cli), "--version"],
            ["<PHAXIS_CLI>", "--version"],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        _require(version_result.stdout.strip() == "PHAxis 1.0.0", "installed CLI version changed")
        analyze_prefix = [str(env_python), "-B", "-I", "-m", "phaxis.cli", "analyze"]
        _command(
            runner,
            [
                *analyze_prefix, "--manifest", str(copied_manifest), "--output",
                str(plan_analysis), "--plan-output", str(plan_path),
            ],
            [
                "<ENV_PYTHON>", "-B", "-I", "-m", "phaxis.cli", "analyze",
                "--manifest", "<RELEASE_ROOT_EXAMPLE_MANIFEST>", "--output",
                "<PLAN_ONLY_OUTPUT_MUST_REMAIN_ABSENT>", "--plan-output", "<PLAN_JSON>",
            ],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        _require(
            plan_path.is_file() and not plan_analysis.exists(),
            "installed CLI plan-only semantics failed",
        )
        plan_payload = _read_json(plan_path, "installed relocated workflow plan")
        relocated_path_proof = _relocated_plan_path_proof(
            plan_payload, release_root.resolve()
        )
        root_stages = [
            stage
            for stage in plan_payload.get("stages", ())
            if isinstance(stage, Mapping) and stage.get("name") == "root_provider"
        ]
        _require(len(root_stages) == 1, "relocated workflow plan has no unique root stage")
        root_plan = root_stages[0].get("detail", {}).get("plan", {})
        _require(
            isinstance(root_plan, Mapping)
            and Path(str(root_plan.get("python_executable"))).resolve()
            == env_python.resolve(),
            "root-provider subprocess is not rebound to clean-environment Python",
        )
        for field in ("project", "bundle", "input_manifest", "output"):
            path = Path(str(root_plan.get(field) or "")).resolve()
            _require(
                _inside(path, release_root),
                f"root-provider plan {field} escapes disposable release root",
            )
        _command(
            runner,
            [
                *analyze_prefix, "--manifest", str(copied_manifest), "--output",
                str(analysis), "--execute",
            ],
            [
                "<ENV_PYTHON>", "-B", "-I", "-m", "phaxis.cli", "analyze",
                "--manifest", "<RELEASE_ROOT_EXAMPLE_MANIFEST>", "--output",
                "<NEW_EXAMPLE_ANALYSIS>", "--execute",
            ],
            cwd=release_root,
            env=clean_env,
            records=commands,
        )
        completed = _verify_completed_output(context, analysis, plan_path)
        public = context["public"]
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "completed_final_clean_install",
            "product": "PHAxis",
            "product_version": VERSION,
            "source_release_manifest_sha256": sha256_file(context["source_manifest_path"]),
            "source_release_tree_identity_sha256": context["source_manifest"]["tree_identity_sha256"],
            "formal_wheel": context["wheel_audit"],
            "applied_model_contract_sha256": sha256_file(context["applied_path"]),
            "applied_model_contract_identity_sha256": context["applied_identity"],
            "model_contract_proposal_sha256": sha256_file(context["proposal_path"]),
            "model_contract_proposal_identity_sha256": context["proposal_identity"],
            "model_bundle_manifest_sha256": sha256_file(context["bundle_path"]),
            "model_bundle_manifest_identity_sha256": context["bundle_identity"],
            "portable_capsule_identity_sha256": context["capsule"]["identity"],
            "portable_capsule_tree_identity_sha256": context["capsule"]["tree_identity_sha256"],
            **public,
            "expected_example_identity_receipt_sha256": sha256_file(context["expected_path"]),
            "expected_example_identity_receipt_identity_sha256": context["expected_identity"],
            "example_manifest_sha256": sha256_file(context["example_manifest_path"]),
            "example_manifest_identity_sha256": context["example_manifest_identity"],
            "example_output_identity_sha256": completed["example_output_identity_sha256"],
            "canonical_example_output_files": completed["canonical_output_files"],
            "workflow_execution": {
                key: value
                for key, value in completed.items()
                if key not in {"canonical_output_files", "example_output_identity_sha256"}
            },
            "installation": {
                "environment_backend": "venv_created_inside_disposable_release_root",
                "new_isolated_environment": True,
                "system_site_packages": False,
                "dependency_extra": "deployment",
                "dependency_strategy": "offline_deployment_wheelhouse_plus_sha256_lock_then_formal_wheel_no_deps",
                "dependency_lock": context["dependency_lock"],
                "offline_wheelhouse": context["wheelhouse"],
                "commands": commands,
                "installed_distributions": inventory,
                "installed_distribution_identity_sha256": sha256_json(inventory),
                "deployment_module_relpaths_inside_clean_env": module_relpaths,
                "deployment_modules_imported_from_clean_env": True,
                "root_subprocess_python_is_clean_env_python": True,
                "all_runtime_paths_inside_disposable_release_root": True,
                "relocated_plan_path_proof": relocated_path_proof,
                "authoring_workspace_runtime_dependency": False,
                "pip_check_passed": True,
                "cli_version": version_result.stdout.strip(),
                "import_version": import_probe["version"],
                "user_site_enabled": import_probe["user_site_enabled"],
                "plan_only_verified_before_execute": True,
                "explicit_execute_used": True,
                "cuda_visible_devices": cuda_visible_devices,
                "nvidia_smi_preflight_before_isolated_workflow": True,
                "nvidia_smi_preflight_stdout_sha256": preflight_sha256,
                "disposable_attempt_cleanup_policy": "always_remove_before_receipt_publication",
            },
            "real_final_example_input": True,
            "development_or_synthetic_smoke": False,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        }
    finally:
        shutil.rmtree(attempt, ignore_errors=True)

    _require(not attempt.exists() and not requested_work.exists(), "disposable clean-install attempt cleanup failed")
    _require(receipt is not None, "clean-install attempt did not produce a receipt")
    receipt["installation"]["disposable_attempt_removed_before_receipt_publication"] = True
    receipt["clean_install_receipt_identity_sha256"] = sha256_json(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise CleanInstallError(f"refusing to overwrite receipt: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-release-root", type=Path, required=True)
    parser.add_argument("--applied-model-contract", type=Path, required=True)
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--model-bundle-manifest", type=Path, required=True)
    parser.add_argument("--example-manifest", type=Path, required=True)
    parser.add_argument("--portable-capsule-root", type=Path, required=True)
    parser.add_argument("--expected-example-identity", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cuda-visible-devices",
        help="explicit physical-card visibility inherited by the isolated workflow",
    )
    parser.add_argument("--execute", action="store_true", help="create the clean environment, run inference/traits/profiles, and publish the receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    common = {
        "project_root": args.project_root,
        "wheel": args.wheel,
        "source_release_root": args.source_release_root,
        "applied_model_contract": args.applied_model_contract,
        "model_contract_proposal": args.model_contract_proposal,
        "model_bundle_manifest": args.model_bundle_manifest,
        "example_manifest": args.example_manifest,
        "portable_capsule_root": args.portable_capsule_root,
        "expected_example_identity": args.expected_example_identity,
        "dependency_lock": args.dependency_lock,
        "wheelhouse": args.wheelhouse,
    }
    try:
        if not args.execute:
            if any(
                value is not None
                for value in (
                    args.base_python,
                    args.work_root,
                    args.output,
                    args.cuda_visible_devices,
                )
            ):
                parser.error(
                    "--base-python/--work-root/--output/--cuda-visible-devices "
                    "are execution-only; add --execute"
                )
            result = clean_install_plan(**common)
        else:
            if (
                args.base_python is None
                or args.work_root is None
                or args.output is None
                or args.cuda_visible_devices is None
            ):
                parser.error(
                    "--execute requires --base-python, --work-root, --output, "
                    "and --cuda-visible-devices"
                )
            result = execute_clean_install_verification(
                **common,
                base_python=args.base_python,
                work_root=args.work_root,
                output=args.output,
                cuda_visible_devices=args.cuda_visible_devices,
            )
    except CleanInstallError as error:
        print(f"PHAxis clean-install verification failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
