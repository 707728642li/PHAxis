from __future__ import annotations

import base64
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from phaxis.io import sha256_file, sha256_json
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)


SCRIPT = Path(__file__).parents[2] / "scripts/phaxis/build_clean_install_verification.py"
SPEC = importlib.util.spec_from_file_location("clean_install_verification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
clean = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean)


CANONICAL_BYTES = {
    "distal_axis_profiles/distal_axis_profiles.csv": b"task_id,bin,count\nexample,0,1\n",
    "fusion/predictions/example.json": b'{"prediction":"formal"}\n',
    "traits/analysis_metadata.csv": b"task_id,source\nexample,formal\n",
    "traits/detailed_root_statistics.csv": b"task_id,root\nexample,1\n",
    "traits/hair_instances.csv": b"task_id,hair\nexample,1\n",
    "traits/image_traits.csv": b"task_id,trait\nexample,1\n",
    "traits/traits.csv": b"task_id,name,value\nexample,length,1\n",
}
AUTHORITY_KEYS = {
    "project_root",
    "wheel",
    "source_release_root",
    "applied_model_contract",
    "model_contract_proposal",
    "model_bundle_manifest",
    "example_manifest",
    "portable_capsule_root",
    "expected_example_identity",
    "dependency_lock",
    "wheelhouse",
}


def _json(path: Path, payload: dict, identity_field: str | None = None) -> Path:
    data = deepcopy(payload)
    if identity_field is not None:
        data[identity_field] = sha256_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def _wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    package_files: dict[str, bytes],
    entry_points: str | None = None,
    license_files: dict[str, bytes] | None = None,
    omitted_members: tuple[str, ...] = (),
) -> Path:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    members = dict(package_files)
    license_file_values = list((license_files or {}).keys())
    license_headers = "".join(
        f"License-File: {relative}\n" for relative in license_file_values
    )
    metadata_version = "2.4" if license_file_values else "2.1"
    members[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: {metadata_version}\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n"
        f"{license_headers}\n"
    ).encode()
    for relative, data in (license_files or {}).items():
        members[f"{dist_info}/licenses/{relative}"] = data
    members[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: synthetic-test\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    if entry_points is not None:
        members[f"{dist_info}/entry_points.txt"] = entry_points.encode()
    for name in omitted_members:
        members.pop(name, None)
    record_name = f"{dist_info}/RECORD"
    rows = []
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([record_name, "", ""])
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    members[record_name] = buffer.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _canonical_records() -> list[dict]:
    return [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(CANONICAL_BYTES.items())
    ]


def _fixture(tmp_path: Path) -> dict:
    project = tmp_path / "project"
    project.mkdir(parents=True)

    source_release = project / "formal-source"
    source_files = {
        "LICENSE": b"Apache License 2.0 fixture\n",
        "src/phaxis/__init__.py": b'__version__ = "1.0.0"\n',
        "src/phaxis/cli.py": b"def main():\n    return 0\n",
        "src/phaxis/_vendor/tomli/LICENSE.txt": (
            b"Synthetic MIT license fixture for vendored Tomli.\n"
        ),
    }
    source_records = []
    for relative, data in sorted(source_files.items()):
        path = source_release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        source_records.append(
            {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    source_manifest = _json(
        source_release / "SOURCE_MANIFEST.json",
        {
            "schema_version": clean.SOURCE_SCHEMA,
            "distribution": "phaxis",
            "version": "1.0.0",
            "release_mode": "formal",
            "source_policy": "explicit_path_bounded_allowlist",
            "files": source_records,
            "tree_identity_sha256": sha256_json(source_records),
        },
    )
    wheel = _wheel(
        project / "dist/phaxis-1.0.0-py3-none-any.whl",
        distribution="phaxis",
        version="1.0.0",
        package_files={
            name.removeprefix("src/"): data
            for name, data in source_files.items()
            if name.startswith("src/phaxis/")
        },
        entry_points="[console_scripts]\nphaxis = phaxis.cli:main\n",
        license_files={
            relative: source_files[relative]
            for relative in clean.REQUIRED_PEP639_LICENSE_FILES
        },
    )

    capsule = project / "portable-capsule"
    checkpoint_files = []
    for index, seed in enumerate((3001, 3002, 3003, 3004, 3005)):
        checkpoint = capsule / f"model/assets/stageb/member-{index}-seed-{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"synthetic checkpoint {index} seed {seed}\n".encode())
        checkpoint_files.append(checkpoint)

    stageb_binding = {
        "expert_id": "PHAxis-StageB-train399-five-member",
        "candidate_bundle_identity_sha256": "1" * 64,
        "selection_receipt_identity_sha256": "2" * 64,
        "selected_model_metadata_identity_sha256": "3" * 64,
        "checkpoint_sha256": [sha256_file(path) for path in checkpoint_files],
        "selected_score_threshold": 0.5,
    }
    root_bundle_identity = "9" * 64
    derived = derive_public_identity(
        stageb_binding, root_bundle_identity_sha256=root_bundle_identity
    )
    proposal_payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "formal_release_status": "passed_proposal_not_official",
        "model_bundle_id": derived["model_bundle_id"],
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "stageb_binding": stageb_binding,
            "formal_gate_source_sha256": {
                "train399_candidate": "c" * 64,
                "train399_selection": "d" * 64,
                "train399_evaluation": "e" * 64,
                "root_exact283": "f" * 64,
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": stageb_binding[
                    "candidate_bundle_identity_sha256"
                ],
                "selection_receipt_identity_sha256": stageb_binding[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": stageb_binding[
                    "selected_model_metadata_identity_sha256"
                ],
                "root_exact283_audit_identity_sha256": "a" * 64
            },
        },
        "root_expert": {
            "expert_id": derived["root_expert_id"],
            "provider_role": derived["root_provider_role"],
            "pipeline_identity_sha256": "b" * 64,
            "bundle_identity_sha256": root_bundle_identity,
            "fresh_exact283_audit_identity_sha256": "a" * 64,
            "root_bundle_authority": {
                "pipeline_identity_sha256": "b" * 64,
                "bundle_identity_sha256": root_bundle_identity,
            },
        },
        "hair_identity_count_expert": {"expert_id": stageb_binding["expert_id"]},
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": derived["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": derived["root_expert_id"]
        },
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
        },
    }
    proposal = _json(
        capsule / "model/assets/runtime/model_contract_proposal.json",
        proposal_payload,
        "model_contract_identity_sha256",
    )
    proposal_value = json.loads(proposal.read_text(encoding="utf-8"))
    applied_payload = deepcopy(proposal_value)
    applied_payload.pop("model_contract_identity_sha256")
    applied_payload["formal_release_status"] = "passed"
    applied_payload["promotion"].update(
        {
            "status": "applied_formal_release",
            "official_apply_performed": True,
            "proposal_file_sha256": sha256_file(proposal),
            "proposal_identity_sha256": proposal_value[
                "model_contract_identity_sha256"
            ],
        }
    )
    applied = _json(
        capsule / "model/assets/runtime/applied_model_contract.json",
        applied_payload,
        "model_contract_identity_sha256",
    )
    applied_value = json.loads(applied.read_text(encoding="utf-8"))

    checkpoints = [
        {
            "member_index": index,
            "seed": seed,
            "sha256": sha256_file(checkpoint_files[index]),
            "bytes": checkpoint_files[index].stat().st_size,
            "package_path": checkpoint_files[index].relative_to(capsule).as_posix(),
        }
        for index, seed in enumerate((3001, 3002, 3003, 3004, 3005))
    ]
    public = {
        "model_bundle_id": derived["model_bundle_id"],
        "root_expert_id": derived["root_expert_id"],
        "root_bundle_identity_sha256": root_bundle_identity,
        "hair_identity_count_expert": stageb_binding["expert_id"],
    }
    runtime = capsule / "model/assets/runtime"
    root_runtime = runtime / "root"
    root_bundle_root = capsule / "model/assets/root_provider"
    root_bundle_registry = _json(
        root_bundle_root / "root_provider_bundle.json",
        {"schema_version": "synthetic-root-bundle-1.0", "files": []},
    )
    root_authorities = {
        "acquisition_gate": _json(root_runtime / "acquisition_gate.json", {"status": "passed"}),
        "deployment_metadata": _json(root_runtime / "deployment_metadata.json", {"tasks": 1}),
        "canonical_manifest": _json(root_runtime / "canonical_manifest.json", {"tasks": 1}),
        "deployment_manifest": _json(root_runtime / "deployment_manifest.json", {"tasks": 1}),
        "deployment_lock": _json(root_runtime / "deployment_lock.json", {"tasks": 1}),
    }
    # This authority is packaged for provenance but deliberately is not referenced
    # by the one-task portable workflow.
    _json(root_runtime / "reference_registry.json", {"tasks": 283})
    stageb_authorities = {
        "candidate_manifest": _json(runtime / "train399_candidate.json", {"status": "passed"}),
        "selected_model_metadata": _json(runtime / "selected_model_metadata.json", {"status": "passed"}),
        "selection_receipt": _json(runtime / "selection_receipt.json", {"status": "passed"}),
    }
    profile_contract = _json(
        runtime / "distal_axis_profile_contract.json", {"status": "passed"}
    )
    inputs = capsule / "model/examples/clean_install/inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    root_input = inputs / "root_input_manifest.csv"
    stageb_input = inputs / "stageb_input_manifest.csv"
    traits_input = inputs / "traits_metadata.csv"
    source_image = inputs / "sample_source_image.tif"
    root_input.write_text("task_id,image\nexample,sample_source_image.tif\n", encoding="utf-8")
    stageb_input.write_text("task_id,image\nexample,sample_source_image.tif\n", encoding="utf-8")
    traits_input.write_text("task_id,condition\nexample,release\n", encoding="utf-8")
    source_image.write_bytes(b"synthetic real nonblind image bytes")
    projection_receipt = _json(
        capsule / "model/examples/clean_install/projection_receipt.json",
        {"status": "completed", "tasks": 1, "blind_images_used": 0},
    )

    manifest_base = capsule / "model/examples/clean_install"

    def locked(path: Path) -> dict[str, str]:
        return {
            "path": Path(os.path.relpath(path, manifest_base)).as_posix(),
            "sha256": sha256_file(path),
        }

    example_manifest = _json(
        manifest_base / "release_example_manifest.json",
        {
            "schema_version": clean.WORKFLOW_SCHEMA,
            "model_contract_proposal": locked(proposal),
            "root_provider": {
                "project": ".",
                "bundle": {
                    "path": Path(os.path.relpath(root_bundle_root, manifest_base)).as_posix(),
                    "registry_sha256": sha256_file(root_bundle_registry),
                    "bundle_identity_sha256": root_bundle_identity,
                },
                "input_manifest": locked(root_input),
                "image_root": "inputs",
                **{name: locked(path) for name, path in root_authorities.items()},
                "cuda_visible_devices": "0",
                "physical_gpu": 0,
            },
            "stageb": {
                "input_manifest": locked(stageb_input),
                "image_root": "inputs",
                "checkpoints": [locked(path) for path in checkpoint_files],
                **{name: locked(path) for name, path in stageb_authorities.items()},
                "cuda_visible_devices": "0",
                "physical_gpu": 0,
                "internal_device": "cuda:0",
            },
            "traits": {"metadata_csv": locked(traits_input)},
            "distal_axis_profiles": {"contract_json": locked(profile_contract)},
            "release_example": {
                "input_kind": "real_nonblind_release_example",
                "release_authorized": True,
                "development_or_synthetic_smoke": False,
                "tasks": 1,
                "source_image_relpath": "inputs/sample_source_image.tif",
                "source_image_sha256": sha256_file(source_image),
                "portable_capsule_finalized": True,
                "authoring_workspace_paths_required": False,
                "runtime_python_inherited_from_clean_environment": True,
                "exact283_reference_registry_executed_for_one_task": False,
                "blind_images_used": 0,
            },
            "guards": {
                "blind_images_used": 0,
                "canonical_annotations_read": False,
                "condition_metadata_used_for_routing": False,
                "root_cap_region_output": False,
            },
        },
        "manifest_identity_sha256",
    )
    example_value = json.loads(example_manifest.read_text(encoding="utf-8"))

    capsule_receipt = _json(
        manifest_base / "receipt.json",
        {
            "schema_version": clean.CAPSULE_SCHEMA,
            "status": "completed_self_contained_raw_to_profiles_runtime",
            "workflow_manifest_sha256": sha256_file(example_manifest),
            "workflow_manifest_identity_sha256": example_value["manifest_identity_sha256"],
            "source_projection_receipt_sha256": sha256_file(projection_receipt),
            "root_reference_registry_packaged": True,
            "root_reference_registry_executed_for_one_task": False,
            "root_subprocess_python_rebound_to_active_interpreter": True,
            "authoring_workspace_paths_required": False,
            "canonical_annotations_read": False,
            "condition_metadata_used_for_routing": False,
            "blind_images_used": 0,
        },
        "portable_capsule_identity_sha256",
    )

    role_by_path = {
        record["package_path"]: "stageb_checkpoint" for record in checkpoints
    }
    members = []
    for path in sorted(capsule.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(capsule).as_posix()
        record = {
            "role": role_by_path.get(relative, "runtime_authority"),
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if relative in role_by_path:
            checkpoint = next(row for row in checkpoints if row["package_path"] == relative)
            record.update(
                {"member_index": checkpoint["member_index"], "seed": checkpoint["seed"]}
            )
        members.append(record)
    bundle = _json(
        capsule / "model/assets/MODEL_BUNDLE_MANIFEST.json",
        {
            "schema_version": clean.MODEL_BUNDLE_SCHEMA,
            "status": "completed_final_immutable_bundle",
            "product": "PHAxis",
            "product_version": "1.0.0",
            **public,
            "model_contract_proposal_sha256": sha256_file(proposal),
            "model_contract_proposal_identity_sha256": proposal_value[
                "model_contract_identity_sha256"
            ],
            "applied_model_contract_sha256": sha256_file(applied),
            "applied_model_contract_identity_sha256": applied_value[
                "model_contract_identity_sha256"
            ],
            "member_count": len(members),
            "members": members,
            "bundle_sha256": sha256_json(members),
            "bundle_size_bytes": sum(record["bytes"] for record in members),
            "stageb_checkpoints": checkpoints,
            "root_provider_bundle": {
                "bundle_identity_sha256": root_bundle_identity
            },
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
            "historical_or_provisional_backfill_used": False,
        },
        "model_bundle_manifest_identity_sha256",
    )
    bundle_value = json.loads(bundle.read_text(encoding="utf-8"))
    capsule_tree = sha256_json(
        [
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in members
        ]
    )

    wheelhouse = project / "wheelhouse"
    dependency_wheels = []
    for distribution in sorted(clean.REQUIRED_DEPLOYMENT_DISTRIBUTIONS):
        normalized = distribution.replace("-", "_")
        dependency_wheels.append(
            (
                distribution,
                _wheel(
                    wheelhouse / f"{normalized}-1.0-py3-none-any.whl",
                    distribution=distribution,
                    version="1.0",
                    package_files={f"{normalized}/__init__.py": b"VALUE = 1\n"},
                ),
            )
        )
    dependency_lock = project / "locks/requirements.txt"
    dependency_lock.parent.mkdir(parents=True)
    dependency_lock.write_text(
        "".join(
            f"{distribution}==1.0 --hash=sha256:{sha256_file(path)}\n"
            for distribution, path in dependency_wheels
        ),
        encoding="utf-8",
    )

    expected_records = _canonical_records()
    expected = _json(
        project / "example/expected.json",
        {
            "schema_version": clean.EXPECTED_SCHEMA,
            "status": "locked_final_real_example_before_clean_install",
            "input_kind": "real_nonblind_release_example",
            "release_authorized": True,
            "development_or_synthetic_smoke": False,
            "blind_images_used": 0,
            "tasks": 1,
            "canonical_output_files": expected_records,
            "expected_example_output_identity_sha256": sha256_json(expected_records),
            "example_manifest_sha256": sha256_file(example_manifest),
            "example_manifest_identity_sha256": example_value[
                "manifest_identity_sha256"
            ],
            "model_contract_proposal_sha256": sha256_file(proposal),
            "model_contract_proposal_identity_sha256": proposal_value[
                "model_contract_identity_sha256"
            ],
            "applied_model_contract_sha256": sha256_file(applied),
            "applied_model_contract_identity_sha256": applied_value[
                "model_contract_identity_sha256"
            ],
            "model_bundle_manifest_sha256": sha256_file(bundle),
            "model_bundle_manifest_identity_sha256": bundle_value[
                "model_bundle_manifest_identity_sha256"
            ],
            "source_release_manifest_sha256": sha256_file(source_manifest),
            "source_release_tree_identity_sha256": sha256_json(source_records),
            "formal_wheel_sha256": sha256_file(wheel),
            "portable_capsule_identity_sha256": json.loads(
                capsule_receipt.read_text(encoding="utf-8")
            )["portable_capsule_identity_sha256"],
            "portable_capsule_tree_identity_sha256": capsule_tree,
            **public,
        },
        "expected_identity_receipt_identity_sha256",
    )
    base_python = project / "envs/base/python.exe"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"synthetic project-conda python")
    return {
        "project_root": project,
        "wheel": wheel,
        "source_release_root": source_release,
        "applied_model_contract": applied,
        "model_contract_proposal": proposal,
        "model_bundle_manifest": bundle,
        "example_manifest": example_manifest,
        "portable_capsule_root": capsule,
        "expected_example_identity": expected,
        "dependency_lock": dependency_lock,
        "wheelhouse": wheelhouse,
        "base_python": base_python,
        "public": public,
        "proposal_identity": proposal_value["model_contract_identity_sha256"],
    }


class _FakeRunner:
    def __init__(
        self,
        fixture: dict,
        *,
        plan_writes_output: bool = False,
        tamper: bool = False,
        fail_execute_once: bool = False,
    ):
        self.fixture = fixture
        self.plan_writes_output = plan_writes_output
        self.tamper = tamper
        self.fail_execute_once = fail_execute_once
        self.env_root: Path | None = None

    def __call__(self, argv, *, cwd: Path, env: dict) -> subprocess.CompletedProcess[str]:
        assert "PYTHONPATH" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        stdout = ""
        if argv == ["nvidia-smi"]:
            stdout = "synthetic GPU0, 24576 MiB, 0 MiB used, 0% utilization\n"
        elif "venv" in argv:
            env_root = Path(argv[-1])
            self.env_root = env_root
            scripts = env_root / ("Scripts" if clean.os.name == "nt" else "bin")
            scripts.mkdir(parents=True)
            (scripts / ("python.exe" if clean.os.name == "nt" else "python")).write_bytes(b"python")
            (scripts / ("phaxis.exe" if clean.os.name == "nt" else "phaxis")).write_bytes(b"cli")
        elif "pip" in argv and "list" in argv:
            stdout = json.dumps(
                [{"name": "phaxis", "version": "1.0.0"}]
                + [
                    {"name": distribution, "version": "1.0"}
                    for distribution in sorted(clean.REQUIRED_DEPLOYMENT_DISTRIBUTIONS)
                ]
            )
        elif any(
            isinstance(value, str) and value.startswith("<DEPLOYMENT_IMPORT")
            for value in argv
        ):
            raise AssertionError("audit placeholder leaked into executed argv")
        elif "-c" in argv:
            assert self.env_root is not None
            modules = {}
            for module in (*sorted(clean.DEPLOYMENT_IMPORT_MODULES), "phaxis"):
                module_file = self.env_root / "Lib/site-packages" / module / "__init__.py"
                module_file.parent.mkdir(parents=True, exist_ok=True)
                module_file.write_bytes(b"# synthetic isolated module\n")
                modules[module] = str(module_file)
            stdout = json.dumps(
                {
                    "version": "1.0.0",
                    "user_site_enabled": False,
                    "executable": str(Path(argv[0]).resolve()),
                    "prefix": str(self.env_root.resolve()),
                    "modules": modules,
                }
            )
        elif argv[-1:] == ["--version"]:
            stdout = "PHAxis 1.0.0\n"
        elif "analyze" in argv and "--execute" not in argv:
            plan_path = Path(argv[argv.index("--plan-output") + 1])
            output_path = Path(argv[argv.index("--output") + 1])
            if self.plan_writes_output:
                output_path.mkdir(parents=True)
            _json(
                plan_path,
                {
                    "schema_version": "PHAxis-analysis-workflow-plan-1.0",
                    "status": "planned_not_executed",
                    "tasks": 1,
                    "stages": [
                        {
                            "name": "root_provider",
                            "detail": {
                                "plan": {
                                    "python_executable": str(Path(argv[0]).resolve()),
                                    "project": str(cwd.resolve()),
                                    "bundle": str((cwd / "model_capsule").resolve()),
                                    "input_manifest": str(
                                        Path(argv[argv.index("--manifest") + 1]).resolve()
                                    ),
                                    "output": str(output_path.resolve()),
                                }
                            },
                        }
                    ],
                },
                "plan_identity_sha256",
            )
        elif "analyze" in argv and "--execute" in argv:
            if self.fail_execute_once:
                self.fail_execute_once = False
                return subprocess.CompletedProcess(
                    argv, 17, stdout="", stderr="synthetic injected execute failure"
                )
            self._write_completed(Path(argv[argv.index("--output") + 1]))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def _write_completed(self, output: Path) -> None:
        payloads = dict(CANONICAL_BYTES)
        if self.tamper:
            payloads["traits/traits.csv"] += b"tamper\n"
        for relative, data in payloads.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        proposal = self.fixture["model_contract_proposal"]
        proposal_fields = {
            "model_contract_proposal_sha256": sha256_file(proposal),
            "model_contract_proposal_identity_sha256": self.fixture[
                "proposal_identity"
            ],
        }
        public = self.fixture["public"]
        _json(
            output / "fusion/fusion_summary.json",
            {
                "schema_version": "PHAxis-fusion-run-1.1",
                "status": "completed",
                "images": 1,
                "model_bundle_id": public["model_bundle_id"],
                "root_expert": public["root_expert_id"],
                "hair_identity_count_expert": public[
                    "hair_identity_count_expert"
                ],
                "condition_metadata_used_for_routing": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
                "root_cap_region_output": False,
                **proposal_fields,
            },
            "summary_identity_sha256",
        )
        traits = output / "traits"
        _json(
            traits / "summary.json",
            {
                "schema_version": "PHAxis-trait-export-1.0",
                "status": "completed",
                "tasks": 1,
                "model_bundle_id": public["model_bundle_id"],
                "root_expert_id": public["root_expert_id"],
                "hair_identity_count_expert": public[
                    "hair_identity_count_expert"
                ],
                "traits_sha256": sha256_file(traits / "traits.csv"),
                "image_traits_sha256": sha256_file(traits / "image_traits.csv"),
                "detailed_root_statistics_sha256": sha256_file(
                    traits / "detailed_root_statistics.csv"
                ),
                "hair_instances_sha256": sha256_file(traits / "hair_instances.csv"),
                "analysis_metadata_sha256": sha256_file(
                    traits / "analysis_metadata.csv"
                ),
                "condition_metadata_used_for_model_routing": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
                "root_cap_region_statistics_included": False,
                **proposal_fields,
            },
            "export_identity_sha256",
        )
        profiles = output / "distal_axis_profiles"
        _json(
            profiles / "summary.json",
            {
                "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
                "status": "completed",
                "tasks": 1,
                "model_bundle_id": public["model_bundle_id"],
                "root_expert_id": public["root_expert_id"],
                "profiles_csv_sha256": sha256_file(
                    profiles / "distal_axis_profiles.csv"
                ),
                "traits_csv_sha256": sha256_file(traits / "traits.csv"),
                "hair_instances_csv_sha256": sha256_file(
                    traits / "hair_instances.csv"
                ),
                "condition_metadata_used_for_model_routing": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
                "root_cap_region_output": False,
                **proposal_fields,
            },
            "export_identity_sha256",
        )
        stages = [
            {"stage": name, "execution_status": "executed_fresh"}
            for name in (
                "root_provider",
                "stageb_train399",
                "fusion",
                "traits",
                "distal_axis_profiles",
            )
        ]
        _json(
            output / "workflow_state.json",
            {
                "schema_version": "PHAxis-analysis-workflow-state-1.1",
                "status": "completed",
                "latest_execution_attempt_id": 1,
                "latest_execution_fresh_direct_benchmark_eligible": True,
                "execution_attempts": [
                    {
                        "status": "completed",
                        "resume_requested": False,
                        "resume_or_cache_used": False,
                        "fresh_direct_benchmark_eligible": True,
                        "review_overlays_excluded_from_benchmark_scope": True,
                        "stages": stages,
                    }
                ],
                "condition_metadata_used_for_routing": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
                "root_cap_region_output": False,
            },
            "state_identity_sha256",
        )


def _execute(fixture: dict, tmp_path: Path, name: str, runner: _FakeRunner) -> dict:
    return clean.execute_clean_install_verification(
        **{
            key: value
            for key, value in fixture.items()
            if key in AUTHORITY_KEYS | {"base_python"}
        },
        work_root=fixture["project_root"] / f"tmp/{name}",
        output=fixture["project_root"] / f"receipts/{name}.json",
        cuda_visible_devices="0",
        runner=runner,
    )


def test_default_plan_is_deterministic_check_only_and_never_runs_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("plan-only started a process/GPU probe")
        ),
    )
    common = {
        key: value for key, value in fixture.items() if key in AUTHORITY_KEYS
    }
    first = clean.clean_install_plan(**common)
    second = clean.clean_install_plan(**common)
    assert first == second
    assert first["status"] == "validated_not_executed"
    assert first["default_plan_only"] is True
    assert not (fixture["project_root"] / "tmp").exists()


def test_relocated_plan_path_proof_rejects_an_authoring_workspace_absolute_path(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "copied-release"
    release_root.mkdir()
    outside = (tmp_path / "authoring-workspace/model.pt").resolve()
    outside.parent.mkdir()
    outside.write_bytes(b"not portable")
    with pytest.raises(clean.CleanInstallError, match="escapes disposable release root"):
        clean._relocated_plan_path_proof(
            {"stages": [{"command": [str(outside)]}]}, release_root.resolve()
        )


def test_fake_offline_clean_install_is_sealed_path_independent_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("injected synthetic runner was bypassed")
        ),
    )
    first = _execute(fixture, tmp_path, "first", _FakeRunner(fixture))
    second = _execute(fixture, tmp_path, "second", _FakeRunner(fixture))
    assert first == second
    assert first["status"] == "completed_final_clean_install"
    assert first["formal_wheel"]["metadata_license_files"] == [
        "LICENSE",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
    ]
    assert first["formal_wheel"]["pep639_license_member_count"] == 2
    assert first["formal_wheel"]["license_file_hashes_verified"] is True
    assert first["example_output_identity_sha256"] == sha256_json(
        _canonical_records()
    )
    assert first["installation"]["dependency_strategy"] == (
        "offline_deployment_wheelhouse_plus_sha256_lock_then_formal_wheel_no_deps"
    )
    assert first["installation"][
        "nvidia_smi_preflight_before_isolated_workflow"
    ] is True
    dependency_command = first["installation"]["commands"][1]["argv"]
    assert "--require-hashes" in dependency_command
    assert "--only-binary=:all:" in dependency_command
    assert "--no-index" in dependency_command
    python_commands = [
        record["argv"]
        for record in first["installation"]["commands"]
        if record["argv"][0] in {"<PROJECT_CONDA_PYTHON>", "<ENV_PYTHON>"}
    ]
    assert python_commands
    assert all(command[1:3] == ["-B", "-I"] for command in python_commands)
    unsigned = deepcopy(first)
    assert unsigned.pop("clean_install_receipt_identity_sha256") == sha256_json(
        unsigned
    )


def test_clean_install_wheel_audit_rejects_resealed_missing_vendor_license(
    tmp_path: Path,
) -> None:
    source_files = {
        "LICENSE": b"Apache License 2.0 fixture\n",
        "src/phaxis/__init__.py": b'__version__ = "1.0.0"\n',
        "src/phaxis/_vendor/tomli/LICENSE.txt": b"Tomli MIT license fixture\n",
    }
    source_records = {
        relative: {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for relative, data in source_files.items()
    }
    wheel = _wheel(
        tmp_path / "phaxis-1.0.0-py3-none-any.whl",
        distribution="phaxis",
        version="1.0.0",
        package_files={
            relative.removeprefix("src/"): data
            for relative, data in source_files.items()
            if relative.startswith("src/phaxis/")
        },
        entry_points="[console_scripts]\nphaxis = phaxis.cli:main\n",
        license_files={
            relative: source_files[relative]
            for relative in clean.REQUIRED_PEP639_LICENSE_FILES
        },
        omitted_members=(
            "phaxis-1.0.0.dist-info/licenses/src/phaxis/_vendor/tomli/LICENSE.txt",
        ),
    )
    with pytest.raises(
        clean.CleanInstallError,
        match="wheel lacks required PEP 639 license member",
    ):
        clean._audit_wheel(wheel, source_records)


@pytest.mark.parametrize("failure", ["plan-output", "canonical-output"])
def test_execute_failure_never_publishes_formal_receipt(
    tmp_path: Path, failure: str
) -> None:
    fixture = _fixture(tmp_path)
    runner = _FakeRunner(
        fixture,
        plan_writes_output=failure == "plan-output",
        tamper=failure == "canonical-output",
    )
    receipt = fixture["project_root"] / "receipts/rejected.json"
    with pytest.raises(clean.CleanInstallError):
        _execute(fixture, tmp_path, "rejected", runner)
    assert not receipt.exists()
    assert not (fixture["project_root"] / "tmp/rejected").exists()
    assert not list((fixture["project_root"] / "tmp").glob(".rejected.attempt-*"))


def test_failed_attempt_is_cleaned_and_same_run_can_retry_without_manual_deletion(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runner = _FakeRunner(fixture, fail_execute_once=True)
    with pytest.raises(clean.CleanInstallError, match=r"clean-install command failed \(17\)"):
        _execute(fixture, tmp_path, "retryable", runner)
    receipt = fixture["project_root"] / "receipts/retryable.json"
    requested_work = fixture["project_root"] / "tmp/retryable"
    assert not receipt.exists()
    assert not requested_work.exists()
    assert not list(requested_work.parent.glob(".retryable.attempt-*"))

    completed = _execute(fixture, tmp_path, "retryable", runner)
    assert completed["status"] == "completed_final_clean_install"
    assert receipt.is_file()
    assert not requested_work.exists()
    assert not list(requested_work.parent.glob(".retryable.attempt-*"))


def test_dev_smoke_wheel_tamper_and_offline_lock_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "dev")
    expected = json.loads(
        fixture["expected_example_identity"].read_text(encoding="utf-8")
    )
    expected.pop("expected_identity_receipt_identity_sha256")
    expected["development_or_synthetic_smoke"] = True
    expected["expected_identity_receipt_identity_sha256"] = sha256_json(expected)
    fixture["expected_example_identity"].write_text(
        json.dumps(expected, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(clean.CleanInstallError, match="development/synthetic"):
        clean.clean_install_plan(
            **{key: value for key, value in fixture.items() if key in AUTHORITY_KEYS}
        )

    fixture = _fixture(tmp_path / "wheel")
    with fixture["wheel"].open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(clean.CleanInstallError, match="prelock cross-binding"):
        clean.clean_install_plan(
            **{key: value for key, value in fixture.items() if key in AUTHORITY_KEYS}
        )

    fixture = _fixture(tmp_path / "offline")
    rows = fixture["dependency_lock"].read_text(encoding="utf-8").splitlines()
    rows = [
        ("numpy==1.0 --hash=sha256:" + "f" * 64)
        if row.startswith("numpy==")
        else row
        for row in rows
    ]
    fixture["dependency_lock"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(clean.CleanInstallError, match="offline wheelhouse"):
        clean.clean_install_plan(
            **{key: value for key, value in fixture.items() if key in AUTHORITY_KEYS}
        )
