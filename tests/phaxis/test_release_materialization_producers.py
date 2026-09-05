from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest

from phaxis.io import atomic_write_json, sha256_file, sha256_json
from scripts.phaxis import build_clean_install_expected_identity as expected_builder
from scripts.phaxis import build_clean_install_sample_manifest as sample_builder
from scripts.phaxis import build_release_case_prelocks as case_builder
from scripts.phaxis import materialize_figure1_geometry as geometry_builder
from scripts.phaxis import materialize_offline_dependencies as dependency_builder


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_offline_target_marker_environment_is_hermetic_cp312_windows() -> None:
    environment = dependency_builder._target_environment()
    assert environment["python_version"] == "3.12"
    assert environment["os_name"] == "nt"
    assert environment["platform_python_implementation"] == "CPython"
    assert environment["platform_release"] == "11"
    assert environment["platform_version"] == "10.0.0"
    active = dependency_builder._direct_requirements(
        {
            "requires_dist": [
                "packaging>=24,<26",
                "legacy-only>=1; python_version < '3.11'",
                'colorama>=0.4; os_name == "nt"',
                'pypy-only>=1; platform_python_implementation == "PyPy"',
                'deployment-only>=1; extra == "deployment"',
                'publication-only>=1; extra == "publication"',
            ]
        }
    )
    assert {row["name"] for row in active} == {
        "colorama",
        "deployment-only",
        "packaging",
    }


def _application_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], list[str]]:
    case_ids = [task for _role, task in case_builder.CASE_TASKS]
    ids = case_ids + [f"RHSCU-synthetic-{index:03d}" for index in range(278)]
    images: dict[str, Path] = {}
    rows: list[dict[str, str]] = []
    fallback = tmp_path / "images/fallback.png"
    fallback.parent.mkdir(parents=True)
    Image.fromarray(np.full((10, 12), 128, dtype=np.uint8)).save(fallback)
    for index, task_id in enumerate(ids):
        if task_id in case_ids:
            image = tmp_path / "images" / f"{task_id}.png"
            Image.fromarray(
                np.full((10, 12), 40 + index * 10, dtype=np.uint8)
            ).save(image)
            images[task_id] = image
        else:
            image = fallback
        rows.append(
            {
                "task_id": task_id,
                "image_path": str(image.resolve()),
                "image_sha256": sha256_file(image),
                "um_per_px": "2.0",
                "source_megapixels": "0.00012",
            }
        )
    manifest = tmp_path / "application283.csv"
    _write_csv(manifest, list(rows[0]), rows)
    return manifest, images, ids


def test_release_case_prelocks_are_fixed_exact283_and_result_independent(
    tmp_path: Path,
) -> None:
    manifest, _images, _ids = _application_fixture(tmp_path)
    output = tmp_path / "case-prelocks"
    receipt = case_builder.build_case_prelocks(
        application_manifest=manifest,
        output=output,
    )
    unsigned = deepcopy(receipt)
    assert unsigned.pop("case_prelock_identity_sha256") == sha256_json(unsigned)
    assert receipt["status"] == case_builder.STATUS
    assert receipt["application_tasks"] == 283
    assert receipt["model_outputs_read"] is False
    assert receipt["condition_metadata_read"] is False
    with (output / "overlay_case_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["case_role"], row["task_id"]) for row in rows] == list(
        case_builder.CASE_TASKS
    )
    figure = json.loads(
        (output / "figure1_case_selection.json").read_text(encoding="utf-8")
    )
    identity = figure.pop("figure1_case_selection_identity_sha256")
    assert identity == sha256_json(figure)
    assert figure["task_id"] == case_builder.FIGURE1_TASK_ID
    assert figure["classic_challenge_panel_task"] is False
    with pytest.raises(case_builder.CasePrelockError, match="overwrite"):
        case_builder.build_case_prelocks(
            application_manifest=manifest,
            output=output,
        )


def _fusion_fixture(
    tmp_path: Path, manifest: Path, ids: list[str], source: Path
) -> Path:
    fusion = tmp_path / "fusion"
    (fusion / "predictions").mkdir(parents=True)
    (fusion / "masks").mkdir()
    (fusion / "axis_geometry").mkdir()
    task = case_builder.FIGURE1_TASK_ID
    source_sha = sha256_file(source)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[1:9, 5:8] = 255
    mask_path = fusion / "masks" / f"{task}.root.png"
    assert cv2.imwrite(str(mask_path), mask)
    axis_path = fusion / "axis_geometry" / f"{task}.npz"
    path_xy = np.asarray([[6.0, 8.0], [6.0, 5.0], [6.0, 1.0]], dtype=np.float32)
    np.savez_compressed(
        axis_path,
        path_xy=path_xy,
        distance_from_tip_px=np.asarray([0.0, 3.0, 7.0], dtype=np.float32),
        radius_px=np.asarray([1.0, 1.5, 1.0], dtype=np.float32),
        source_image_sha256=np.asarray(source_sha),
    )
    prediction = {
        "schema_version": "PHAxis-prediction-1.0",
        "task_id": task,
        "source_image_sha256": source_sha,
        "root_mask_relpath": f"masks/{task}.root.png",
        "root_mask_sha256": sha256_file(mask_path),
        "root_axis_geometry_relpath": f"axis_geometry/{task}.npz",
        "root_axis_geometry_sha256": sha256_file(axis_path),
        "root_cap_point_xy": [6.0, 8.0],
        "identity_hairs": [
            {
                "source_instance_id": "H-1",
                "points_xy": [[5.0, 5.0], [2.0, 4.0]],
            }
        ],
        "length_hairs": [
            {
                "identity_source_instance_id": "H-1",
                "points_xy": [[5.0, 5.0], [4.0, 4.5], [2.0, 4.0]],
            }
        ],
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read_during_inference": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }
    prediction_path = fusion / "predictions" / f"{task}.json"
    atomic_write_json(prediction_path, prediction)
    records = [
        {
            "task_id": task_id,
            "prediction_sha256": sha256_file(prediction_path)
            if task_id == task
            else sha256_json({"future": task_id}),
        }
        for task_id in ids
    ]
    summary = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "images": 283,
        "records": records,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(fusion / "fusion_summary.json", summary)
    return fusion


def test_figure1_geometry_is_derived_from_preselected_final_prediction(
    tmp_path: Path,
) -> None:
    manifest, images, ids = _application_fixture(tmp_path)
    prelocks = tmp_path / "prelocks"
    case_builder.build_case_prelocks(application_manifest=manifest, output=prelocks)
    source = images[case_builder.FIGURE1_TASK_ID]
    fusion = _fusion_fixture(tmp_path, manifest, ids, source)
    output = tmp_path / "figure1"
    receipt = geometry_builder.materialize_figure1_geometry(
        case_selection=prelocks / "figure1_case_selection.json",
        application_manifest=manifest,
        fusion_root=fusion,
        output=output,
    )
    unsigned = deepcopy(receipt)
    assert unsigned.pop("figure1_geometry_materialization_identity_sha256") == sha256_json(unsigned)
    assert receipt["status"] == geometry_builder.STATUS
    assert receipt["case_selection_changed_after_results"] is False
    geometry = json.loads(
        (output / "figure1_geometry.json").read_text(encoding="utf-8")
    )
    identity = geometry.pop("figure1_display_geometry_identity_sha256")
    assert identity == sha256_json(geometry)
    assert geometry["prediction_sha256"] == receipt["prediction_sha256"]
    assert len(geometry["root_polygon_xy"]) >= 3
    assert geometry["hair_identities"][0]["length_curve_xy"] is not None
    copied = output / receipt["source_image_file"]
    assert sha256_file(copied) == sha256_file(source)


def test_sample_manifest_projects_one_real_source_with_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    application, _images, ids = _application_fixture(tmp_path)
    prelocks = tmp_path / "prelocks"
    case_builder.build_case_prelocks(application_manifest=application, output=prelocks)
    case = json.loads(
        (prelocks / "figure1_case_selection.json").read_text(encoding="utf-8")
    )
    source_sha = case["source_image_sha256"]
    task = case["task_id"]
    rows = [
        {
            "task_id": task_id,
            "image_path": case["source_image_path"],
            "image_sha256": source_sha,
            "um_per_px": "2.0",
        }
        for task_id in ids
    ]
    root_rows = [
        {
            "image_id": row["task_id"],
            "input_path": row["image_path"],
            "source_image_sha256": row["image_sha256"],
            "source_um_per_px": row["um_per_px"],
        }
        for row in rows
    ]
    root_csv = tmp_path / "full/root.csv"
    stageb_csv = tmp_path / "full/stageb.csv"
    traits_csv = tmp_path / "full/traits.csv"
    deployment_metadata_csv = tmp_path / "full/deployment_metadata.csv"
    deployment_manifest_csv = tmp_path / "full/deployment_manifest.csv"
    canonical_csv = tmp_path / "full/canonical.csv"
    acquisition_gate = tmp_path / "full/acquisition_gate.json"
    deployment_lock = tmp_path / "full/deployment_manifest_lock.json"
    _write_csv(root_csv, list(root_rows[0]), root_rows)
    _write_csv(stageb_csv, list(rows[0]), rows)
    _write_csv(traits_csv, list(rows[0]), rows)
    deployment_rows = [
        {
            **row,
            "image_relpath": Path(row["image_path"]).name,
            "width": "12",
            "height": "10",
            "source_megapixels": "0.00012",
            "scale_provenance": "raw_image_classical_train399_locked",
            "analysis_scale_eligible": "true",
        }
        for row in rows
    ]
    _write_csv(
        deployment_metadata_csv, list(deployment_rows[0]), deployment_rows
    )
    _write_csv(
        deployment_manifest_csv, list(deployment_rows[0]), deployment_rows
    )
    _write_csv(
        canonical_csv,
        [
            "biological_unit_id",
            "canonical_view_selected",
            "acquisition_desirability_score",
            "focus_score",
            "robust_contrast",
            "historical_absolute_path",
        ],
        [
            {
                "biological_unit_id": row["task_id"],
                "canonical_view_selected": "true",
                "acquisition_desirability_score": "0.9",
                "focus_score": "0.8",
                "robust_contrast": "0.7",
                "historical_absolute_path": str(
                    (tmp_path / "authoring-only/source.tif").resolve()
                ),
            }
            for row in rows
        ],
    )
    atomic_write_json(
        acquisition_gate,
        {
            "schema_version": "RHPheno-acquisition-gate-1.0",
            "phenotype_model_independent": True,
            "manual_phenotype_truth_used": False,
            "thresholds": {},
        },
    )
    atomic_write_json(
        deployment_lock,
        {
            "schema_version": sample_builder.DEPLOYMENT_LOCK_SCHEMA,
            "status": "locked_before_phenotype_inference",
            "samples": len(deployment_rows),
            "manifest": "deployment_manifest.csv",
            "manifest_sha256": sha256_file(deployment_manifest_csv),
            "deployment_identity_sha256": "1" * 64,
            "source_qc_lock_identity_sha256": "2" * 64,
            "scale_policy": "raw_image_classical_train399_locked_fail_closed",
            "canonical_annotations_read": False,
            "phenotype_model_predictions_used": False,
            "blind_images_used": 0,
        },
    )
    full_path = tmp_path / "full/workflow.json"
    full_path.write_text("{}", encoding="utf-8")
    full = {
        "schema_version": "PHAxis-analysis-workflow-manifest-1.0",
        "root_provider": {
            "input_manifest": {"path": str(root_csv), "sha256": sha256_file(root_csv)},
            "acquisition_gate": {
                "path": str(acquisition_gate),
                "sha256": sha256_file(acquisition_gate),
            },
            "deployment_metadata": {
                "path": str(deployment_metadata_csv),
                "sha256": sha256_file(deployment_metadata_csv),
            },
            "canonical_manifest": {
                "path": str(canonical_csv),
                "sha256": sha256_file(canonical_csv),
            },
            "deployment_manifest": {
                "path": str(deployment_manifest_csv),
                "sha256": sha256_file(deployment_manifest_csv),
            },
            "deployment_lock": {
                "path": str(deployment_lock),
                "sha256": sha256_file(deployment_lock),
            },
            "image_root": str(tmp_path),
        },
        "stageb": {
            "input_manifest": {"path": str(stageb_csv), "sha256": sha256_file(stageb_csv)},
            "image_root": str(tmp_path),
        },
        "traits": {
            "metadata_csv": {"path": str(traits_csv), "sha256": sha256_file(traits_csv)}
        },
        "benchmark_contract": {},
        "guards": {
            "blind_images_used": 0,
            "canonical_annotations_read": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_output": False,
        },
        "manifest_identity_sha256": "f" * 64,
    }
    monkeypatch.setattr(sample_builder, "load_analysis_manifest", lambda _path: deepcopy(full))

    def fake_plan(path: Path, *, output: Path, review_overlays: bool) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for section, field in (
            ("root_provider", "input_manifest"),
            ("stageb", "input_manifest"),
            ("traits", "metadata_csv"),
        ):
            csv_path = path.parent / payload[section][field]["path"]
            with csv_path.open(encoding="utf-8", newline="") as handle:
                projected = list(csv.DictReader(handle))
            assert len(projected) == 1
        assert not output.exists() and review_overlays is False
        return {
            "tasks": 1,
            "manifest_identity_sha256": payload["manifest_identity_sha256"],
        }

    monkeypatch.setattr(sample_builder, "build_analysis_plan", fake_plan)
    output = tmp_path / "sample"
    receipt = sample_builder.build_sample_manifest(
        analysis_workflow_manifest=full_path,
        case_selection=prelocks / "figure1_case_selection.json",
        output=output,
    )
    assert receipt["tasks"] == 1
    assert receipt["portable_relative_input_paths"] is True
    manifest = json.loads(
        (output / "release_example_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["release_example"]["task_id"] == task
    assert manifest["root_provider"]["image_root"] == "inputs"
    assert "python_executable" not in manifest["root_provider"]
    assert manifest["root_provider"]["project"] == "."
    assert "historical_absolute_path" not in (
        output / "inputs/canonical_manifest.csv"
    ).read_text(encoding="utf-8")
    assert receipt["portable_data_and_root_authority_paths"] is True
    assert not Path(manifest["stageb"]["input_manifest"]["path"]).is_absolute()
    sample_image = next((output / "inputs").glob("sample_source_image*"))
    assert sha256_file(sample_image) == source_sha


def _wheel(
    path: Path,
    *,
    name: str,
    version: str,
    requires: list[str] | None = None,
    license_expression: str = "MIT",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    metadata.append(f"License-Expression: {license_expression}")
    metadata.append("License-File: LICENSE.txt")
    metadata.extend(f"Requires-Dist: {item}" for item in (requires or ()))
    distribution = name.replace("-", "_")
    with zipfile.ZipFile(path, "w") as archive:
        dist_info = f"{distribution}-{version}.dist-info"
        archive.writestr(
            f"{dist_info}/METADATA",
            "\n".join(metadata) + "\n",
        )
        archive.writestr(
            f"{dist_info}/licenses/LICENSE.txt",
            f"Synthetic {license_expression} license fixture for {name}.\n",
        )
    return path


def test_offline_dependency_materializer_resolves_binary_hash_closure(
    tmp_path: Path,
) -> None:
    formal = _wheel(
        tmp_path / "phaxis-1.0.0-py3-none-any.whl",
        name="phaxis",
        version="1.0.0",
        license_expression="Apache-2.0",
        requires=[
            "numpy>=2,<3",
            "packaging>=24,<26",
            "scipy>=1,<2",
            "legacy-only>=1; python_version < '3.11'",
            *[
                f'{name}>=1; extra == "deployment"'
                for name in sorted(
                    dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
                    - {"numpy", "packaging", "scipy"}
                )
            ],
            'not-for-deployment>=1; extra == "publication"',
        ],
    )
    dependency_wheels = [
        _wheel(
            tmp_path
            / "resolver"
            / f"{name.replace('-', '_')}-1.0-py3-none-any.whl",
            name=name,
            version="1.0",
            requires=["torch>=1"] if name == "timm" else None,
        )
        for name in sorted(dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS)
    ]

    def fake_runner(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(argv[argv.index("--dest") + 1])
        shutil.copyfile(formal, destination / formal.name)
        assert argv[-1].endswith("[deployment]")
        for wheel in dependency_wheels:
            shutil.copyfile(wheel, destination / wheel.name)
        return subprocess.CompletedProcess(argv, 0, "resolved", "")

    output = tmp_path / "offline"
    receipt = dependency_builder.materialize_dependencies(
        formal_wheel=formal,
        python_executable=Path(__import__("sys").executable),
        output=output,
        runner=fake_runner,
    )
    unsigned = deepcopy(receipt)
    assert unsigned.pop("dependency_materialization_identity_sha256") == sha256_json(unsigned)
    assert receipt["target"]["extras"] == ["deployment"]
    assert receipt["wheelhouse_file_count"] == len(
        dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
    )
    assert {row["distribution"] for row in receipt["wheelhouse_files"]} == {
        *dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS,
    }
    lock = (output / "requirements.lock.txt").read_text(encoding="utf-8")
    assert "numpy==1.0 --hash=sha256:" in lock
    assert "torch==1.0 --hash=sha256:" in lock
    assert "not-for-deployment" not in lock
    assert all(path.suffix == ".whl" for path in (output / "wheelhouse").iterdir())
    resolved_sbom_path = output / dependency_builder.RESOLVED_SBOM_NAME
    license_inventory_path = (
        output / dependency_builder.RESOLVED_LICENSE_INVENTORY_NAME
    )
    assert resolved_sbom_path.is_file()
    assert license_inventory_path.is_file()
    resolved_sbom = json.loads(resolved_sbom_path.read_text(encoding="utf-8"))
    assert resolved_sbom["bomFormat"] == "CycloneDX"
    assert resolved_sbom["specVersion"] == "1.6"
    assert resolved_sbom["metadata"]["component"]["name"] == "phaxis"
    assert len(resolved_sbom["components"]) == len(
        dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
    )
    component_by_name = {
        row["name"]: row for row in resolved_sbom["components"]
    }
    assert component_by_name["torch"]["hashes"][0]["alg"] == "SHA-256"
    timm_ref = component_by_name["timm"]["bom-ref"]
    torch_ref = component_by_name["torch"]["bom-ref"]
    assert next(
        row for row in resolved_sbom["dependencies"] if row["ref"] == timm_ref
    )["dependsOn"] == [torch_ref]
    license_inventory = json.loads(
        license_inventory_path.read_text(encoding="utf-8")
    )
    unsigned_license_inventory = deepcopy(license_inventory)
    assert unsigned_license_inventory.pop(
        "resolved_license_inventory_identity_sha256"
    ) == sha256_json(unsigned_license_inventory)
    assert license_inventory["artifact_count"] == len(
        dependency_builder.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
    )
    assert license_inventory["all_artifacts_have_license_evidence"] is True
    assert all(
        row["metadata_license_expression"] == "MIT"
        and row["license_files"]
        for row in license_inventory["artifacts"]
    )
    assert receipt["resolved_cyclonedx_sbom"]["sha256"] == sha256_file(
        resolved_sbom_path
    )
    assert receipt["resolved_license_inventory"]["sha256"] == sha256_file(
        license_inventory_path
    )


def test_expected_identity_inventory_is_path_sorted_and_clean_verifier_compatible(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    for relative in sorted(expected_builder.CANONICAL_REQUIRED_OUTPUTS):
        path = analysis / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    prediction = analysis / "fusion/predictions/BIO-001.json"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text("{}", encoding="utf-8")
    records = expected_builder._canonical_output_records(analysis)
    assert records == sorted(records, key=lambda row: row["path"])
    assert len(records) == len(expected_builder.CANONICAL_REQUIRED_OUTPUTS) + 1
    files = {}
    for name in ("example_manifest", "proposal", "applied", "model_bundle", "wheel"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = path
    source_manifest = tmp_path / "SOURCE_MANIFEST.json"
    source_manifest.write_text("source", encoding="utf-8")
    context = {
        "paths": files,
        "source_manifest_path": source_manifest,
        "source": {"tree_identity_sha256": "1" * 64},
        "example_identity": "2" * 64,
        "proposal_identity": "3" * 64,
        "applied_identity": "4" * 64,
        "bundle_identity": "5" * 64,
        "capsule": {
            "identity": "7" * 64,
            "tree_identity_sha256": "8" * 64,
        },
        "public": {
            "model_bundle_id": "PHAXIS-SYNTHETIC",
            "root_expert_id": "ROOT-SYNTHETIC",
            "root_bundle_identity_sha256": "6" * 64,
            "hair_identity_count_expert": "HAIR-SYNTHETIC",
        },
    }
    payload = expected_builder._expected_payload(context, records)
    identity = payload.pop("expected_identity_receipt_identity_sha256")
    assert identity == sha256_json(payload)
    assert payload["status"] == expected_builder.EXPECTED_STATUS
    assert payload["expected_example_output_identity_sha256"] == sha256_json(records)
