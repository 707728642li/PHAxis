#!/usr/bin/env python3
"""Resolve, hash-lock, and materialize the formal offline wheelhouse.

The target interpreter/platform and index policy are explicit.  ``--check``
only inspects the formal wheel and prints a sealed plan.  ``--execute`` asks pip
to resolve binary wheels, rejects sdists/duplicates, writes a hash-required
lock file, and atomically publishes an offline wheelhouse plus receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
import uuid
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-offline-dependency-materialization-1.0"
STATUS = "completed_locked_cp312_win_amd64"
PLAN_SCHEMA = "PHAxis-offline-dependency-materialization-plan-1.0"
PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu130"
DEPLOYMENT_EXTRA = "deployment"
RESOLVED_SBOM_NAME = "SBOM.resolved.cdx.json"
RESOLVED_LICENSE_INVENTORY_NAME = "THIRD_PARTY_LICENSES.resolved.json"
REQUIRED_DEPLOYMENT_DISTRIBUTIONS = frozenset(
    {
        "imageio",
        "joblib",
        "matplotlib",
        "numpy",
        "opencv-python-headless",
        "packaging",
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


class DependencyMaterializationError(RuntimeError):
    """The formal dependency closure cannot be materialized safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DependencyMaterializationError(message)


def _normal_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _is_license_member(path: PurePosixPath) -> bool:
    name = path.name.upper()
    return any(
        name == stem
        or name.startswith(stem + ".")
        or name.startswith(stem + "-")
        or name.startswith(stem + "_")
        for stem in ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")
    )


def _wheel_metadata(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".whl", f"wheel is absent/invalid: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            _require(archive.testzip() is None, f"wheel archive is corrupt: {path.name}")
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            _require(len(names) == 1, f"wheel has invalid METADATA closure: {path.name}")
            metadata = BytesParser(policy=email_policy).parsebytes(archive.read(names[0]))
            license_files: list[dict[str, Any]] = []
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                _require(
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and relative.as_posix() == member.filename,
                    f"wheel has unsafe archive path: {path.name}:{member.filename}",
                )
                if not _is_license_member(relative):
                    continue
                mode = (member.external_attr >> 16) & 0o170000
                _require(
                    mode != 0o120000,
                    f"wheel license member is a symlink: {path.name}:{member.filename}",
                )
                content = archive.read(member)
                license_files.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise DependencyMaterializationError(f"cannot inspect wheel {path.name}: {error}") from error
    name = metadata.get("Name")
    version = metadata.get("Version")
    _require(isinstance(name, str) and name and isinstance(version, str) and version, f"wheel Name/Version is absent: {path.name}")
    license_expression = str(metadata.get("License-Expression") or "").strip()
    legacy_license = str(metadata.get("License") or "").strip()
    if legacy_license.casefold() in {"unknown", "n/a", "none"}:
        legacy_license = ""
    license_classifiers = sorted(
        str(value).strip()
        for value in metadata.get_all("Classifier") or ()
        if str(value).strip().startswith("License ::")
    )
    return {
        "filename": path.name,
        "distribution": _normal_name(name),
        "version": version,
        "requires_dist": list(metadata.get_all("Requires-Dist") or ()),
        "metadata_license_expression": license_expression,
        "metadata_legacy_license": legacy_license,
        "metadata_license_classifiers": license_classifiers,
        "metadata_license_files": sorted(
            str(value).strip()
            for value in metadata.get_all("License-File") or ()
            if str(value).strip()
        ),
        "license_files": license_files,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _target_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update(
        {
            "python_version": "3.12",
            "python_full_version": "3.12.0",
            "platform_system": "Windows",
            "platform_release": "11",
            "platform_version": "10.0.0",
            "sys_platform": "win32",
            "platform_machine": "AMD64",
            "os_name": "nt",
            "platform_python_implementation": "CPython",
            "implementation_name": "cpython",
            "implementation_version": "3.12.0",
            "extra": "",
        }
    )
    return environment


def _direct_requirements(
    metadata: Mapping[str, Any], *, selected_extra: str = DEPLOYMENT_EXTRA
) -> list[dict[str, str]]:
    environment = _target_environment()
    environment["extra"] = selected_extra
    records: list[dict[str, str]] = []
    for raw in metadata.get("requires_dist", ()):
        try:
            requirement = Requirement(str(raw))
        except Exception as error:
            raise DependencyMaterializationError(f"invalid Requires-Dist in formal wheel: {raw}") from error
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        records.append(
            {
                "name": _normal_name(requirement.name),
                "specifier": str(requirement.specifier),
                "marker": str(requirement.marker or ""),
            }
        )
    records.sort(key=lambda row: (row["name"], row["specifier"], row["marker"]))
    _require(len({row["name"] for row in records}) == len(records), "formal wheel repeats an active direct dependency")
    return records


def _has_license_evidence(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("metadata_license_expression")
        or metadata.get("metadata_legacy_license")
        or metadata.get("metadata_license_classifiers")
        or metadata.get("license_files")
    )


def _license_inventory(
    dependencies: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for _path, metadata in dependencies:
        evidence = {
            "metadata_license_expression": metadata[
                "metadata_license_expression"
            ],
            "metadata_legacy_license": metadata["metadata_legacy_license"],
            "metadata_license_classifiers": metadata[
                "metadata_license_classifiers"
            ],
            "metadata_license_files": metadata["metadata_license_files"],
            "license_files": metadata["license_files"],
        }
        artifacts.append(
            {
                "distribution": metadata["distribution"],
                "version": metadata["version"],
                "filename": metadata["filename"],
                "bytes": metadata["bytes"],
                "sha256": metadata["sha256"],
                **evidence,
                "license_evidence_identity_sha256": sha256_json(evidence),
                "machine_readable_spdx_expression_present": bool(
                    metadata["metadata_license_expression"]
                ),
                "license_evidence_present": _has_license_evidence(metadata),
            }
        )
    artifacts.sort(key=lambda row: row["distribution"])
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-resolved-third-party-license-inventory-1.0",
        "status": "resolved_artifact_evidence_inventory_requires_review",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "target": {
            "platform": "win_amd64",
            "python_version": "3.12",
            "implementation": "cp",
            "abi": "cp312",
            "extras": [DEPLOYMENT_EXTRA],
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "all_artifacts_have_license_evidence": all(
            row["license_evidence_present"] for row in artifacts
        ),
        "all_artifacts_have_machine_readable_spdx_expression": all(
            row["machine_readable_spdx_expression_present"]
            for row in artifacts
        ),
        "artifact_specific_license_review_required": True,
        "license_clearance_claimed": False,
        "resolved_transitive_dependency_claimed": True,
    }
    payload["resolved_license_inventory_identity_sha256"] = sha256_json(payload)
    return payload


def _purl(distribution: str, version: str) -> str:
    return (
        "pkg:pypi/"
        + quote(distribution, safe=".-_~")
        + "@"
        + quote(version, safe=".-_~")
    )


def _component_licenses(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    expression = str(metadata.get("metadata_license_expression") or "").strip()
    if expression:
        return [{"expression": expression}]
    legacy = str(metadata.get("metadata_legacy_license") or "").strip()
    classifiers = metadata.get("metadata_license_classifiers")
    if legacy:
        label = legacy
    elif isinstance(classifiers, list) and classifiers:
        label = "; ".join(map(str, classifiers))
    else:
        label = "SEE-LICENSE-FILES-IN-LOCKED-WHEEL"
    return [{"license": {"name": label}}]


def _active_transitive_names(
    metadata: Mapping[str, Any],
    *,
    available: frozenset[str],
) -> list[str]:
    environment = _target_environment()
    environment["extra"] = ""
    names: set[str] = set()
    for raw in metadata.get("requires_dist", ()):
        try:
            requirement = Requirement(str(raw))
        except Exception as error:
            raise DependencyMaterializationError(
                f"invalid Requires-Dist in resolved wheel {metadata['filename']}: {raw}"
            ) from error
        if requirement.marker is not None and not requirement.marker.evaluate(
            environment
        ):
            continue
        name = _normal_name(requirement.name)
        _require(
            name in available,
            f"resolved wheelhouse omits active dependency {name} required by "
            f"{metadata['distribution']}",
        )
        names.add(name)
    return sorted(names)


def _resolved_sbom(
    *,
    formal: Mapping[str, Any],
    dependencies: Sequence[tuple[Path, Mapping[str, Any]]],
    direct_names: frozenset[str],
) -> dict[str, Any]:
    by_name = {
        str(metadata["distribution"]): metadata
        for _path, metadata in dependencies
    }
    _require(
        len(by_name) == len(dependencies),
        "resolved SBOM cannot represent duplicate distributions",
    )
    available = frozenset(by_name)
    _require(
        direct_names.issubset(available),
        "resolved SBOM omits a formal direct dependency",
    )
    root_ref = _purl("phaxis", "1.0.0")
    refs = {
        name: _purl(name, str(metadata["version"]))
        for name, metadata in by_name.items()
    }
    components: list[dict[str, Any]] = []
    dependency_graph: list[dict[str, Any]] = [
        {
            "ref": root_ref,
            "dependsOn": [refs[name] for name in sorted(direct_names)],
        }
    ]
    for name in sorted(by_name):
        metadata = by_name[name]
        components.append(
            {
                "type": "library",
                "bom-ref": refs[name],
                "name": name,
                "version": metadata["version"],
                "purl": refs[name],
                "hashes": [
                    {"alg": "SHA-256", "content": metadata["sha256"]}
                ],
                "licenses": _component_licenses(metadata),
                "properties": [
                    {
                        "name": "phaxis:locked-wheel-filename",
                        "value": metadata["filename"],
                    },
                    {
                        "name": "phaxis:license-evidence-present",
                        "value": str(_has_license_evidence(metadata)).casefold(),
                    },
                ],
            }
        )
        dependency_graph.append(
            {
                "ref": refs[name],
                "dependsOn": [
                    refs[dependency]
                    for dependency in _active_transitive_names(
                        metadata,
                        available=available,
                    )
                ],
            }
        )
    closure_identity = sha256_json(
        [
            {
                "distribution": metadata["distribution"],
                "version": metadata["version"],
                "sha256": metadata["sha256"],
            }
            for _path, metadata in sorted(
                dependencies,
                key=lambda item: str(item[1]["distribution"]),
            )
        ]
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:"
        + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"https://phaxis.local/sbom/resolved/{closure_identity}",
            )
        ),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "phaxis",
                "version": "1.0.0",
                "purl": root_ref,
                "hashes": [
                    {"alg": "SHA-256", "content": formal["sha256"]}
                ],
                "licenses": [{"expression": "Apache-2.0"}],
            },
            "properties": [
                {
                    "name": "phaxis:resolution-target",
                    "value": "cp312-win_amd64",
                },
                {
                    "name": "phaxis:resolved-transitive-closure-claimed",
                    "value": "true",
                },
                {
                    "name": "phaxis:wheelhouse-identity-sha256",
                    "value": closure_identity,
                },
            ],
        },
        "components": components,
        "dependencies": dependency_graph,
    }


def _plan(*, formal_wheel: Path, python_executable: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    wheel = formal_wheel.resolve()
    python = python_executable.resolve()
    _require(python.is_file() and not python.is_symlink(), "resolver Python is absent or symlinked")
    metadata = _wheel_metadata(wheel)
    _require(metadata["distribution"] == "phaxis" and metadata["version"] == "1.0.0", "formal wheel is not PHAxis 1.0.0")
    requirements = _direct_requirements(metadata)
    _require(bool(requirements), "formal PHAxis wheel has no active runtime dependencies")
    direct_names = {row["name"] for row in requirements}
    missing = sorted(REQUIRED_DEPLOYMENT_DISTRIBUTIONS - direct_names)
    _require(
        not missing,
        "formal PHAxis wheel deployment extra omits audited runtime dependencies: "
        + ", ".join(missing),
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "validated_not_materialized",
        "default_check_only": True,
        "execute_requires_explicit_flag": True,
        "formal_wheel": {key: metadata[key] for key in ("filename", "bytes", "sha256", "distribution", "version")},
        "formal_wheel_requires_dist_identity_sha256": sha256_json(metadata["requires_dist"]),
        "active_direct_requirements": requirements,
        "audited_runtime_direct_distributions": sorted(
            REQUIRED_DEPLOYMENT_DISTRIBUTIONS
        ),
        "target": {
            "platform": "win_amd64",
            "python_version": "3.12",
            "implementation": "cp",
            "abi": "cp312",
            "extras": [DEPLOYMENT_EXTRA],
        },
        "resolver": {
            "binary_only": True,
            "primary_index": PYPI_INDEX,
            "pytorch_index": PYTORCH_INDEX,
            "pip_require_hashes_for_install": True,
            "network_used_only_during_materialization": True,
        },
        "blind_images_used": 0,
    }
    plan["plan_identity_sha256"] = sha256_json(plan)
    return plan, metadata


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def materialize_dependencies(
    *,
    formal_wheel: str | Path,
    python_executable: str | Path,
    output: str | Path,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    wheel = Path(formal_wheel).resolve()
    python = Path(python_executable).resolve()
    destination = Path(output).resolve()
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    plan, formal = _plan(formal_wheel=wheel, python_executable=python)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        download = staging / ".resolver-download"
        download.mkdir()
        argv = [
            str(python),
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--dest",
            str(download),
            "--only-binary=:all:",
            "--platform",
            "win_amd64",
            "--python-version",
            "312",
            "--implementation",
            "cp",
            "--abi",
            "cp312",
            "--index-url",
            PYPI_INDEX,
            "--extra-index-url",
            PYTORCH_INDEX,
            f"{wheel}[{DEPLOYMENT_EXTRA}]",
        ]
        completed = runner(
            argv,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        _require(completed.returncode == 0, "pip binary dependency resolution failed")
        candidates = sorted(download.iterdir(), key=lambda path: path.name.casefold())
        _require(bool(candidates) and all(path.is_file() and path.suffix.casefold() == ".whl" for path in candidates), "resolver produced an sdist, directory, or empty closure")
        inspected = [(path, _wheel_metadata(path)) for path in candidates]
        dependencies = [
            (path, record)
            for path, record in inspected
            if not (record["distribution"] == "phaxis" and record["version"] == "1.0.0")
        ]
        _require(bool(dependencies), "resolver produced no PHAxis dependencies")
        names = [record["distribution"] for _path, record in dependencies]
        _require(len(names) == len(set(names)), "resolver produced duplicate dependency distributions")
        direct_names = {row["name"] for row in plan["active_direct_requirements"]}
        _require(direct_names.issubset(set(names)), "resolved wheelhouse omits an active direct dependency")
        missing_license_evidence = sorted(
            str(record["distribution"])
            for _path, record in dependencies
            if not _has_license_evidence(record)
        )
        _require(
            not missing_license_evidence,
            "resolved wheels lack license metadata/files: "
            + ", ".join(missing_license_evidence),
        )

        wheelhouse = staging / "wheelhouse"
        wheelhouse.mkdir()
        records: list[dict[str, Any]] = []
        for source, metadata in sorted(dependencies, key=lambda item: item[1]["distribution"]):
            target = wheelhouse / source.name
            shutil.copyfile(source, target)
            _require(sha256_file(target) == metadata["sha256"], f"wheel copy hash changed: {source.name}")
            records.append(
                {
                    "distribution": metadata["distribution"],
                    "version": metadata["version"],
                    "filename": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": metadata["sha256"],
                }
            )
        records.sort(key=lambda row: row["distribution"])
        lock_path = staging / "requirements.lock.txt"
        lock_text = "".join(
            f"{row['distribution']}=={row['version']} --hash=sha256:{row['sha256']}\n"
            for row in records
        )
        lock_path.write_text(lock_text, encoding="utf-8", newline="\n")
        license_inventory = _license_inventory(dependencies)
        _require(
            license_inventory["all_artifacts_have_license_evidence"] is True,
            "resolved license inventory has an evidence gap",
        )
        license_inventory_path = staging / RESOLVED_LICENSE_INVENTORY_NAME
        atomic_write_json(license_inventory_path, license_inventory)
        resolved_sbom = _resolved_sbom(
            formal=formal,
            dependencies=dependencies,
            direct_names=frozenset(direct_names),
        )
        resolved_sbom_path = staging / RESOLVED_SBOM_NAME
        atomic_write_json(resolved_sbom_path, resolved_sbom)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "formal_wheel_sha256": formal["sha256"],
            "formal_wheel_requires_dist_identity_sha256": plan[
                "formal_wheel_requires_dist_identity_sha256"
            ],
            "target": plan["target"],
            "resolver": plan["resolver"],
            "active_direct_requirements": plan["active_direct_requirements"],
            "audited_runtime_direct_distributions": plan[
                "audited_runtime_direct_distributions"
            ],
            "requirements_lock_sha256": sha256_file(lock_path),
            "pip_require_hashes": True,
            "wheelhouse_files": records,
            "wheelhouse_file_count": len(records),
            "wheelhouse_identity_sha256": sha256_json(records),
            "resolved_requirement_set_identity_sha256": sha256_json(
                [
                    {
                        "distribution": row["distribution"],
                        "version": row["version"],
                        "sha256": row["sha256"],
                    }
                    for row in records
                ]
            ),
            "resolved_cyclonedx_sbom": {
                "filename": RESOLVED_SBOM_NAME,
                "sha256": sha256_file(resolved_sbom_path),
                "spec_version": "1.6",
                "serial_number": resolved_sbom["serialNumber"],
                "component_count_including_phaxis": len(records) + 1,
                "exact_versions_and_wheel_sha256_included": True,
                "dependency_graph_included": True,
            },
            "resolved_license_inventory": {
                "filename": RESOLVED_LICENSE_INVENTORY_NAME,
                "sha256": sha256_file(license_inventory_path),
                "identity_sha256": license_inventory[
                    "resolved_license_inventory_identity_sha256"
                ],
                "artifact_count": len(records),
                "all_artifacts_have_license_evidence": True,
                "artifact_specific_license_review_required": True,
            },
            "resolved_software_supply_chain_generated": True,
            "resolver_argv_audit": [
                "<PYTHON>",
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--platform=win_amd64",
                "--python-version=312",
                "--implementation=cp",
                "--abi=cp312",
                "--index-url=<PYPI_OFFICIAL>",
                "--extra-index-url=<PYTORCH_CU130_OFFICIAL>",
                "<FORMAL_PHAXIS_WHEEL>[deployment]",
            ],
            "sdists_used": False,
            "credentials_recorded": False,
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["dependency_materialization_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "receipt.json", receipt)
        shutil.rmtree(download)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return deepcopy(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.execute:
            result = materialize_dependencies(
                formal_wheel=args.formal_wheel,
                python_executable=args.python,
                output=args.output,
            )
            identity = result["dependency_materialization_identity_sha256"]
        else:
            result, _metadata = _plan(
                formal_wheel=args.formal_wheel.resolve(),
                python_executable=args.python.resolve(),
            )
            identity = result["plan_identity_sha256"]
    except DependencyMaterializationError as error:
        print(f"offline dependency materialization blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "identity_sha256": identity}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
