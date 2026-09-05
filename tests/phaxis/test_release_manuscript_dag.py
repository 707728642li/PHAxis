from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phaxis.io import sha256_file, sha256_json
from phaxis.manuscript_values import EVIDENCE_ARTIFACT_ROLES
from phaxis.release_orchestrator import (
    ReleaseOrchestratorError,
    _recoverable_root_bundle_materialization,
)
from phaxis.release_topology import MANDATORY_STAGE_ORDER, STAGE_DEPENDENCIES
from phaxis.root_provider.bundle import BUNDLE_ID, BUNDLE_SCHEMA
from scripts.phaxis import build_post_training_release_stage_contract as builder
from scripts.phaxis.materialize_verified_root_provider_bundle import (
    materialize_verified_bundle,
)
from scripts.phaxis.validate_manuscript_visual_qa import (
    ATTESTATION_SCHEMA,
    FINAL_STATUS,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    RENDER_SCHEMA,
    STRUCTURAL_SCHEMA,
    VISUAL_CHECKS,
    VisualQaError,
    seal_attestation,
    validate_visual_qa,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_VISUAL_CHECKS = (
    "no_text_or_figure_clipping",
    "no_object_overlap",
    "no_page_overflow_or_unintended_blank_page",
    "tables_legible_and_rows_not_split",
    "headings_legends_and_references_not_orphaned",
    "main_six_and_supplementary_nine_plates_legible",
    "headers_footers_page_numbers_and_line_numbers_correct",
    "editor_title_page_identity_and_declarations_complete",
    "reviewer_visible_pages_contain_no_identity_cues",
    "reviewer_visible_figure_pixels_contain_no_identity_cues",
    "figure4_prelocked_anchor_ids_and_deterministic_insets_verified",
    "figure4_audit_cards_match_axis_support_endpoint_and_eligibility_semantics",
)


def _stage_map() -> dict[str, dict]:
    contract = builder.build()
    assert contract["stage_count"] == 61
    assert tuple(stage["name"] for stage in contract["stages"]) == MANDATORY_STAGE_ORDER
    return {stage["name"]: stage for stage in contract["stages"]}


def _values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, token in enumerate(command[:-1])
        if token == option
    ]


def test_generated_manuscript_dag_is_real_and_fail_closed() -> None:
    checked_in = json.loads(
        (
            PROJECT_ROOT
            / "configs/phaxis/v1_0/post_training_release_stage_contract_1_8.json"
        ).read_text(encoding="utf-8")
    )
    generated = builder.build()
    assert checked_in == generated
    assert tuple(stage["name"] for stage in generated["stages"][53:58]) == (
        "submission_docx",
        "supplementary_docx",
        "manuscript_artifact_qa",
        "manuscript_render",
        "manuscript_visual_qa",
    )
    stages = _stage_map()
    manuscript = stages["manuscript"]
    assert _values(manuscript["command"], "--output") == [
        "{run_dir}/manuscript/manuscript.md"
    ]
    assert all(not item["path"].endswith(".docx") for item in manuscript["artifacts"])

    main_docx = stages["submission_docx"]
    supplement_docx = stages["supplementary_docx"]
    assert main_docx["command"][1].endswith("build_submission_docx.py")
    assert supplement_docx["command"][1].endswith("build_supplementary_docx.py")
    title_page = (
        "{run_dir}/submission_docx/PHAxis_Plant_Phenomics_title_page.docx"
    )
    anonymous_main = (
        "{run_dir}/submission_docx/PHAxis_Plant_Phenomics_anonymized_main.docx"
    )
    anonymous_supplement = (
        "{run_dir}/supplementary_docx/"
        "PHAxis_Plant_Phenomics_anonymized_supplement.docx"
    )
    upload_manifest = (
        "{run_dir}/manuscript_artifact_qa/upload-role-manifest.json"
    )
    assert _values(main_docx["command"], "--title-page-output") == [title_page]
    assert _values(main_docx["command"], "--anonymized-main-output") == [
        anonymous_main
    ]
    assert _values(main_docx["command"], "--submission-metadata") == [
        "{external:submission_title_metadata}"
    ]
    assert "--output" not in main_docx["command"]
    assert {artifact["path"] for artifact in main_docx["artifacts"]} >= {
        title_page,
        anonymous_main,
    }
    assert main_docx["receipt"]["status"] == (
        "completed_final_double_anonymous_submission_bundle"
    )

    assert _values(supplement_docx["command"], "--output") == [
        anonymous_supplement
    ]
    assert "--submission-metadata" not in supplement_docx["command"]
    assert "{external:submission_title_metadata}" not in supplement_docx["command"]
    assert not any(
        item.get("external") == "submission_title_metadata"
        for item in supplement_docx["inputs"]
    )
    assert {artifact["path"] for artifact in supplement_docx["artifacts"]} >= {
        anonymous_supplement
    }
    assert supplement_docx["receipt"]["status"] == (
        "completed_final_anonymized_supplementary_docx"
    )

    structural = stages["manuscript_artifact_qa"]
    assert structural["command"][1].endswith("verify_manuscript_artifacts.py")
    assert _values(structural["command"], "--title-page-docx") == [title_page]
    assert _values(structural["command"], "--anonymized-main-docx") == [
        anonymous_main
    ]
    assert _values(
        structural["command"], "--anonymized-supplement-docx"
    ) == [anonymous_supplement]
    assert _values(structural["command"], "--upload-manifest") == [
        upload_manifest
    ]
    assert not {"--main-docx", "--main-docx-receipt", "--supplement-docx"} & set(
        structural["command"]
    )
    assert upload_manifest in {
        artifact["path"] for artifact in structural["artifacts"]
    }
    assert structural["receipt"]["status"] == (
        "passed_double_anonymous_three_role_ooxml_closure"
    )

    render = stages["manuscript_render"]
    assert render["command"][1].endswith("render_manuscript_bundle.py")
    assert _values(render["command"], "--title-page-docx") == [title_page]
    assert _values(render["command"], "--anonymized-main-docx") == [
        anonymous_main
    ]
    assert _values(render["command"], "--anonymized-supplement-docx") == [
        anonymous_supplement
    ]
    assert _values(render["command"], "--upload-manifest") == [upload_manifest]
    assert {
        artifact["path"]
        for artifact in render["artifacts"]
        if artifact["path"].endswith(".pdf")
    } == {
        "{run_dir}/manuscript_render/output/title_page/title_page.pdf",
        "{run_dir}/manuscript_render/output/anonymized_main/anonymized_main.pdf",
        (
            "{run_dir}/manuscript_render/output/anonymized_supplement/"
            "anonymized_supplement.pdf"
        ),
    }
    assert render["receipt"]["status"] == (
        "completed_three_role_word_pdf_and_page_png_render"
    )

    visual = stages["manuscript_visual_qa"]
    assert visual["command"][1].endswith("validate_manuscript_visual_qa.py")
    assert visual["receipt"]["status"] == (
        "passed_author_verified_three_role_page_visual_qa"
    )
    assert STRUCTURAL_SCHEMA == "PHAxis-manuscript-artifact-structural-qa-2.0"
    assert RENDER_SCHEMA == "PHAxis-manuscript-pdf-page-render-2.0"
    assert ATTESTATION_SCHEMA == "PHAxis-manuscript-visual-qa-attestation-2.0"
    assert RECEIPT_SCHEMA == "PHAxis-manuscript-human-visual-qa-receipt-2.0"
    assert tuple(VISUAL_CHECKS) == EXPECTED_VISUAL_CHECKS
    # The attestation is an intentionally pre-existing human work item on a
    # resumed invocation; only the successful receipt may be an artifact.
    assert [artifact["name"] for artifact in visual["artifacts"]] == ["receipt"]
    assert set(STAGE_DEPENDENCIES["release_finalize"]) >= {
        "submission_docx",
        "supplementary_docx",
        "manuscript_artifact_qa",
        "manuscript_render",
        "manuscript_visual_qa",
    }


def test_generated_p0_wiring_uses_exact_producer_authorities() -> None:
    stages = _stage_map()

    assert stages["source_release"]["receipt"] == {
        **stages["source_release"]["receipt"],
        "status_field": "release_mode",
        "status": "formal",
    }
    assert stages["figures"]["receipt"]["status"] == (
        "final_sealed_strict_train399_only"
    )

    qc_root = stages["qcdev_root_provider"]
    assert qc_root["receipt"]["status"] == "completed_uncompared"
    assert "--reference-registry" not in qc_root["command"]

    profile_export = stages["profiles_exact283"]
    assert profile_export["command"][1].endswith(
        "/scripts/phaxis/export_cohort_distal_axis_profiles.py"
    )
    assert _values(profile_export["command"], "--cohorts-root") == [
        "{run_dir}/cohorts_exact283/output"
    ]
    assert profile_export["receipt"]["identity_field"] == (
        "cohort_profile_bundle_identity_sha256"
    )
    assert {artifact["name"]: artifact["path"] for artifact in profile_export["artifacts"]} == {
        "receipt": "{run_dir}/profiles_exact283/output/summary.json",
        "output": "{run_dir}/profiles_exact283/output",
        "primary_summary": (
            "{run_dir}/profiles_exact283/output/primary_clean261/summary.json"
        ),
        "primary_profiles": (
            "{run_dir}/profiles_exact283/output/primary_clean261/"
            "distal_axis_profiles.csv"
        ),
        "sensitivity_summary": (
            "{run_dir}/profiles_exact283/output/sensitivity_full283/summary.json"
        ),
        "sensitivity_profiles": (
            "{run_dir}/profiles_exact283/output/sensitivity_full283/"
            "distal_axis_profiles.csv"
        ),
    }

    profile_analysis = stages["profile_analysis"]
    assert profile_analysis["receipt"]["identity_field"] == (
        "analysis_identity_sha256"
    )
    profiles = profile_analysis["command"]
    primary_profile_root = (
        "{run_dir}/profiles_exact283/output/primary_clean261"
    )
    sensitivity_profile_root = (
        "{run_dir}/profiles_exact283/output/sensitivity_full283"
    )
    assert _values(profiles, "--primary-profiles") == [primary_profile_root]
    assert _values(profiles, "--sensitivity-profiles") == [
        sensitivity_profile_root
    ]
    assert primary_profile_root != sensitivity_profile_root

    measurement = stages["measurement_assurance"]["command"]
    assert _values(measurement, "--clean-traits") == [
        "{run_dir}/cohorts_exact283/output/primary_clean261/traits.csv"
    ]
    figure_inputs = stages["figure_inputs"]["command"]
    assert _values(figure_inputs, "--clean-traits") == [
        "{run_dir}/cohorts_exact283/output/primary_clean261/traits.csv"
    ]
    assert _values(figure_inputs, "--profiles") == [
        "{run_dir}/profiles_exact283/output/primary_clean261/summary.json"
    ]
    assert _values(figure_inputs, "--sensitivity-profiles-summary") == [
        "{run_dir}/profiles_exact283/output/sensitivity_full283/summary.json"
    ]

    for name in ("qcdev_fusion", "fusion_exact283"):
        assert stages[name]["command"][1] == "{workspace}/scripts/phaxis/run_cli.py"

    handover = stages["handover_model_asset_manifest"]
    assert _values(handover["command"], "--root-provider-verification-receipt") == [
        "{run_dir}/root_bundle_materialization/output/verification.json"
    ]
    assert "{run_dir}/root_provider_exact283/output/fresh_reference_audit.json" not in _values(
        handover["command"], "--root-provider-verification-receipt"
    )
    assert _values(handover["command"], "--portable-capsule-output") == [
        "{run_dir}/handover_model_asset_manifest/portable_capsule"
    ]
    assert next(
        artifact
        for artifact in handover["artifacts"]
        if artifact["name"] == "portable_capsule"
    )["kind"] == "directory"

    evidence = stages["evidence"]["command"]
    assert _values(evidence, "--figure-inputs") == [
        "{run_dir}/figure_inputs/output/figure_inputs.json"
    ]

    values = stages["values"]["command"]
    assert _values(values, "--figure-assembly-summary") == [
        "{run_dir}/figure_inputs/output/assembly_summary.json"
    ]
    role_paths = dict(
        entry.split("=", 1) for entry in _values(values, "--evidence-artifact")
    )
    assert tuple(role_paths) == EVIDENCE_ARTIFACT_ROLES
    assert role_paths["figure_inputs"].endswith("/figure_inputs.json")
    assert role_paths["figures"].endswith("/figure_assembly_summary.json")
    assert "evidence" not in role_paths

    expected_identity = stages["clean_install_expected_identity"]["command"]
    clean = stages["clean_install"]["command"]
    capsule = "{run_dir}/handover_model_asset_manifest/portable_capsule"
    example = capsule + "/model/examples/clean_install/release_example_manifest.json"
    assert _values(expected_identity, "--portable-capsule-root") == [capsule]
    assert _values(expected_identity, "--example-manifest") == [example]
    assert _values(clean, "--portable-capsule-root") == [capsule]
    assert _values(clean, "--example-manifest") == [example]


def test_benchmark_inventory_closes_exact_runtime_support_roles() -> None:
    inventory = _stage_map()["benchmark_artifact_inventory"]["command"]
    records = _values(inventory, "--artifact")
    parsed = [record.split("=", 2) for record in records]
    assert all(len(record) == 3 for record in parsed)
    counts: dict[str, int] = {}
    for role, _package_path, source_path in parsed:
        counts[role] = counts.get(role, 0) + 1
        assert source_path.startswith("{run_dir}/")
    assert counts == {
        "same_hardware_receipt": 1,
        "phaxis_production_summary": 1,
        "v1_production_summary": 1,
        "phaxis_sequential_summary": 1,
        "v1_sequential_summary": 1,
        "production_comparison_receipt": 1,
        "sequential_comparison_receipt": 1,
        "per_image_latency_csv": 2,
        "gpu_telemetry": 4,
        "hardware_preflight": 4,
    }


def test_src_layout_cli_wrapper_starts_without_callers_pythonpath(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/phaxis/run_cli.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "-1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PHAxis 1.0.0" in completed.stdout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_visual_qa_first_pass_materialises_then_sealed_review_passes(
    tmp_path: Path,
) -> None:
    roles = ("title_page", "anonymized_main", "anonymized_supplement")
    upload_manifest_sha256 = sha256_json(["upload-role-manifest", "bytes"])
    upload_manifest_identity_sha256 = sha256_json(
        ["upload-role-manifest", "identity"]
    )
    structural = {
        "schema_version": STRUCTURAL_SCHEMA,
        "status": "passed_double_anonymous_three_role_ooxml_closure",
        "reviewer_visible_identity_occurrence_count": 0,
        "deep_ooxml_anonymity_scan_passed": True,
        "submission_upload_role_manifest_sha256": upload_manifest_sha256,
        "submission_upload_role_manifest_identity_sha256": (
            upload_manifest_identity_sha256
        ),
        "blind_images_used": 0,
    }
    structural["qa_identity_sha256"] = sha256_json(structural)
    structural_path = tmp_path / "structural.json"
    _write_json(structural_path, structural)

    documents = {
        role: {
            "docx_sha256": sha256_json([role, "docx"]),
            "pdf": {"sha256": sha256_json([role, "pdf"])},
            "pages": 2 if role == "anonymized_main" else 1,
            "page_bundle_identity_sha256": sha256_json([role, "pages"]),
        }
        for role in roles
    }
    render = {
        "schema_version": RENDER_SCHEMA,
        "status": "completed_three_role_word_pdf_and_page_png_render",
        "visual_qa_completed": False,
        "documents": documents,
        "structural_qa_sha256": sha256_file(structural_path),
        "structural_qa_identity_sha256": structural["qa_identity_sha256"],
        "submission_upload_role_manifest_sha256": upload_manifest_sha256,
        "submission_upload_role_manifest_identity_sha256": (
            upload_manifest_identity_sha256
        ),
        "blind_images_used": 0,
    }
    render["render_identity_sha256"] = sha256_json(render)
    render_path = tmp_path / "render.json"
    _write_json(render_path, render)

    template = {
        "schema_version": ATTESTATION_SCHEMA,
        "status": "incomplete_human_page_review_not_for_submission",
        "render_receipt_sha256": sha256_file(render_path),
        "render_identity_sha256": render["render_identity_sha256"],
        "reviewer_full_name": None,
        "reviewed_at_utc": None,
        "reviewed_documents": {
            role: {
                "docx_sha256": record["docx_sha256"],
                "pdf_sha256": record["pdf"]["sha256"],
                "pages": record["pages"],
                "page_bundle_identity_sha256": record[
                    "page_bundle_identity_sha256"
                ],
                "all_pages_reviewed_at_original_resolution": False,
            }
            for role, record in documents.items()
        },
        "checks": {name: False for name in EXPECTED_VISUAL_CHECKS},
        "review_notes": None,
        "submission_visual_gate_passed": False,
    }
    template["attestation_identity_sha256"] = sha256_json(template)
    template_path = tmp_path / "template.json"
    _write_json(template_path, template)
    attestation_path = tmp_path / "work" / "VISUAL_QA_ATTESTATION.json"
    output_path = tmp_path / "work" / "receipt.json"

    with pytest.raises(VisualQaError, match="human visual QA is required"):
        validate_visual_qa(
            render_receipt=render_path,
            structural_qa=structural_path,
            template=template_path,
            attestation=attestation_path,
            output=output_path,
        )
    assert attestation_path.read_bytes() == template_path.read_bytes()
    assert not output_path.exists()

    reviewed = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert reviewed["schema_version"] == ATTESTATION_SCHEMA
    assert set(reviewed["reviewed_documents"]) == set(roles)
    assert set(reviewed["checks"]) == set(EXPECTED_VISUAL_CHECKS)
    assert len(reviewed["checks"]) == 12
    assert not any(reviewed["checks"].values())
    reviewed.update(
        {
            "status": FINAL_STATUS,
            "reviewer_full_name": "Human Reviewer",
            "reviewed_at_utc": "2026-08-29T12:00:00+00:00",
            "review_notes": "Every rendered page inspected at original resolution.",
            "submission_visual_gate_passed": True,
        }
    )
    reviewed["checks"] = {key: True for key in reviewed["checks"]}
    for record in reviewed["reviewed_documents"].values():
        record["all_pages_reviewed_at_original_resolution"] = True
    _write_json(attestation_path, reviewed)
    seal_attestation(attestation_path)

    receipt = validate_visual_qa(
        render_receipt=render_path,
        structural_qa=structural_path,
        template=template_path,
        attestation=attestation_path,
        output=output_path,
    )
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["status"] == RECEIPT_STATUS
    assert receipt["documents_reviewed"] == 3
    assert receipt["editor_only_documents_reviewed"] == 1
    assert receipt["reviewer_visible_documents_reviewed"] == 2
    assert receipt["reviewer_visible_identity_occurrence_count"] == 0
    assert receipt["pages_reviewed"] == 4
    assert receipt["submission_upload_role_manifest_sha256"] == (
        upload_manifest_sha256
    )
    assert receipt["submission_upload_role_manifest_identity_sha256"] == (
        upload_manifest_identity_sha256
    )
    assert receipt["submission_use_allowed"] is True
    assert receipt["visual_qa_identity_sha256"] == sha256_json(
        {key: value for key, value in receipt.items() if key != "visual_qa_identity_sha256"}
    )

    incomplete_checks = dict(reviewed)
    incomplete_checks.pop("attestation_identity_sha256")
    incomplete_checks["checks"] = dict(incomplete_checks["checks"])
    incomplete_checks["checks"].pop("reviewer_visible_figure_pixels_contain_no_identity_cues")
    incomplete_path = tmp_path / "incomplete-checks.json"
    _write_json(incomplete_path, incomplete_checks)
    seal_attestation(incomplete_path)
    with pytest.raises(
        VisualQaError,
        match="all twelve page-layout, identity and Fig.4 audit checks",
    ):
        validate_visual_qa(
            render_receipt=render_path,
            structural_qa=structural_path,
            template=template_path,
            attestation=incomplete_path,
            output=tmp_path / "incomplete-checks-receipt.json",
        )


def test_root_bundle_container_is_atomic_exact_and_recoverable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "asset.bin"
    asset.write_bytes(b"registered model byte")
    files = [
        {
            "path": asset.name,
            "bytes": asset.stat().st_size,
            "sha256": sha256_file(asset),
        }
    ]
    contracts = {"synthetic": "strict-test-only"}
    registry = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": BUNDLE_ID,
        "status": "materialized",
        "files": files,
        "contracts": contracts,
        "root_effect_slice_files": 1,
        "bundle_identity_sha256": sha256_json(
            {
                "schema_version": BUNDLE_SCHEMA,
                "bundle_id": BUNDLE_ID,
                "files": files,
                "contracts": contracts,
            }
        ),
    }
    _write_json(source / "root_provider_bundle.json", registry)
    # Simulate runtime cache debris in the frozen source.  Exact
    # materialisation copies only registry members and never mutates/deletes it.
    cache = source / "__pycache__" / "debris.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"not a registered model asset")

    output = tmp_path / "run" / "root_bundle_materialization" / "output"
    receipt = materialize_verified_bundle(source_bundle=source, output=output)
    assert receipt["exact_file_closure_passed"] is True
    assert cache.is_file()
    assert set(path.name for path in output.iterdir()) == {
        "bundle",
        "verification.json",
    }
    assert not (output / "bundle" / "__pycache__").exists()

    stage = {
        "name": "root_bundle_materialization",
        "artifacts": [
            {
                "name": "receipt",
                "path": str(output / "verification.json"),
            },
            {"name": "bundle", "path": str(output / "bundle")},
            {
                "name": "bundle_manifest",
                "path": str(output / "bundle" / "root_provider_bundle.json"),
            },
        ],
    }
    assert _recoverable_root_bundle_materialization(stage) is True

    (output / "bundle" / "unlisted.bin").write_bytes(b"tamper")
    with pytest.raises(
        ReleaseOrchestratorError,
        match="recovery verification failed",
    ):
        _recoverable_root_bundle_materialization(stage)
