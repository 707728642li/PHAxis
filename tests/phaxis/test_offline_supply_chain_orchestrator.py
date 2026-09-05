from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.release_orchestrator import (
    ReleaseOrchestratorError,
    _Context,
    _validate_offline_dependencies,
)
from scripts.phaxis import materialize_offline_dependencies as materializer


def _wheel(
    path: Path,
    *,
    name: str,
    version: str,
    requires: list[str] | None = None,
    license_expression: str = "MIT",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"License-Expression: {license_expression}",
        "License-File: LICENSE.txt",
    ]
    metadata.extend(f"Requires-Dist: {item}" for item in (requires or ()))
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/licenses/LICENSE.txt",
            f"Synthetic {license_expression} license for {name}.\n",
        )
    return path


def _fixture(tmp_path: Path) -> tuple[_Context, dict, Path, dict]:
    formal = _wheel(
        tmp_path / "dist/phaxis-1.0.0-py3-none-any.whl",
        name="phaxis",
        version="1.0.0",
        requires=[
            f'{name}>=1; extra == "deployment"'
            for name in sorted(materializer.REQUIRED_DEPLOYMENT_DISTRIBUTIONS)
        ],
        license_expression="Apache-2.0",
    )
    dependencies = [
        _wheel(
            tmp_path
            / "resolver"
            / f"{name.replace('-', '_')}-1.0-py3-none-any.whl",
            name=name,
            version="1.0",
            requires=["torch>=1"] if name == "timm" else None,
        )
        for name in sorted(materializer.REQUIRED_DEPLOYMENT_DISTRIBUTIONS)
    ]

    def fake_runner(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        destination = Path(argv[argv.index("--dest") + 1])
        shutil.copyfile(formal, destination / formal.name)
        for wheel in dependencies:
            shutil.copyfile(wheel, destination / wheel.name)
        return subprocess.CompletedProcess(argv, 0, "resolved", "")

    output = tmp_path / "run/offline_dependencies/output"
    materializer.materialize_dependencies(
        formal_wheel=formal,
        python_executable=Path(__import__("sys").executable),
        output=output,
        runner=fake_runner,
    )
    artifacts = [
        {"name": "receipt", "path": str(output / "receipt.json"), "kind": "file"},
        {"name": "output", "path": str(output), "kind": "directory"},
        {
            "name": "dependency_lock",
            "path": str(output / "requirements.lock.txt"),
            "kind": "file",
        },
        {
            "name": "wheelhouse",
            "path": str(output / "wheelhouse"),
            "kind": "directory",
        },
        {
            "name": "resolved_sbom",
            "path": str(output / materializer.RESOLVED_SBOM_NAME),
            "kind": "file",
        },
        {
            "name": "resolved_license_inventory",
            "path": str(output / materializer.RESOLVED_LICENSE_INVENTORY_NAME),
            "kind": "file",
        },
    ]
    stage = {"name": "offline_dependencies", "artifacts": artifacts}
    artifact_paths = {
        ("offline_dependencies", str(row["name"])): Path(str(row["path"])).resolve()
        for row in artifacts
    }
    artifact_paths[("distributions", "wheel")] = formal.resolve()
    context = _Context(
        manifest_path=tmp_path / "manifest.json",
        manifest_file_sha256="0" * 64,
        manifest={},
        workspace=tmp_path,
        run_dir=tmp_path / "run",
        external_locks={},
        artifact_paths=artifact_paths,
        candidate_preview={},
    )
    receipt_path = output / "receipt.json"
    return context, stage, receipt_path, read_json(receipt_path)


def _rewrite_receipt_artifact_lock(
    receipt_path: Path, receipt: dict, role: str, artifact_path: Path
) -> dict:
    rewritten = deepcopy(receipt)
    rewritten[role]["sha256"] = sha256_file(artifact_path)
    rewritten.pop("dependency_materialization_identity_sha256")
    rewritten["dependency_materialization_identity_sha256"] = sha256_json(rewritten)
    atomic_write_json(receipt_path, rewritten)
    return rewritten


def test_orchestrator_accepts_semantically_closed_resolved_supply_chain(
    tmp_path: Path,
) -> None:
    context, stage, receipt_path, receipt = _fixture(tmp_path)
    _validate_offline_dependencies(context, stage, receipt_path, receipt)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("schema", "CycloneDX 1.6"),
        ("component", "component set"),
        ("dependency", "dependency relation"),
    ],
)
def test_orchestrator_rejects_semantic_sbom_tampering_even_when_rehashed(
    tmp_path: Path, mutation: str, match: str
) -> None:
    context, stage, receipt_path, receipt = _fixture(tmp_path)
    sbom_path = receipt_path.parent / materializer.RESOLVED_SBOM_NAME
    sbom = read_json(sbom_path)
    if mutation == "schema":
        sbom["specVersion"] = "1.5"
    elif mutation == "component":
        sbom["components"].pop()
    else:
        timm = next(row for row in sbom["components"] if row["name"] == "timm")
        graph = next(row for row in sbom["dependencies"] if row["ref"] == timm["bom-ref"])
        graph["dependsOn"] = []
    atomic_write_json(sbom_path, sbom)
    receipt = _rewrite_receipt_artifact_lock(
        receipt_path, receipt, "resolved_cyclonedx_sbom", sbom_path
    )
    with pytest.raises(ReleaseOrchestratorError, match=match):
        _validate_offline_dependencies(context, stage, receipt_path, receipt)


def test_orchestrator_rejects_license_member_relation_tampering_even_when_resealed(
    tmp_path: Path,
) -> None:
    context, stage, receipt_path, receipt = _fixture(tmp_path)
    inventory_path = (
        receipt_path.parent / materializer.RESOLVED_LICENSE_INVENTORY_NAME
    )
    inventory = read_json(inventory_path)
    inventory.pop("resolved_license_inventory_identity_sha256")
    inventory["artifacts"][0]["license_files"][0]["sha256"] = "f" * 64
    evidence = {
        key: inventory["artifacts"][0][key]
        for key in (
            "metadata_license_expression",
            "metadata_legacy_license",
            "metadata_license_classifiers",
            "metadata_license_files",
            "license_files",
        )
    }
    inventory["artifacts"][0]["license_evidence_identity_sha256"] = sha256_json(
        evidence
    )
    inventory["resolved_license_inventory_identity_sha256"] = sha256_json(inventory)
    atomic_write_json(inventory_path, inventory)
    receipt["resolved_license_inventory"]["identity_sha256"] = inventory[
        "resolved_license_inventory_identity_sha256"
    ]
    receipt = _rewrite_receipt_artifact_lock(
        receipt_path, receipt, "resolved_license_inventory", inventory_path
    )
    with pytest.raises(ReleaseOrchestratorError, match="license inventory"):
        _validate_offline_dependencies(context, stage, receipt_path, receipt)


def test_orchestrator_rejects_resolved_artifact_hash_tampering(tmp_path: Path) -> None:
    context, stage, receipt_path, receipt = _fixture(tmp_path)
    sbom_path = receipt_path.parent / materializer.RESOLVED_SBOM_NAME
    sbom_path.write_bytes(sbom_path.read_bytes() + b" ")
    with pytest.raises(ReleaseOrchestratorError, match="hash interlock"):
        _validate_offline_dependencies(context, stage, receipt_path, receipt)
