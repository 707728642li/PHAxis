from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visual = _load(
    "phaxis_double_anonymous_visual",
    "scripts/phaxis/validate_manuscript_visual_qa.py",
)
render = _load(
    "phaxis_double_anonymous_render",
    "scripts/phaxis/render_manuscript_bundle.py",
)


def test_word_wrapper_forced_cleanup_wait_is_bounded_and_task_owned() -> None:
    wrapper = PROJECT_ROOT / "scripts/phaxis/render_docx_with_word_com_windows.ps1"
    source = wrapper.read_text(encoding="utf-8")
    assert "$preExistingWordIds" in source
    assert "$wordProcessId -notin $preExistingWordIds" in source
    assert "Stop-Process -Id $wordProcessId -Force" in source
    assert "Stop-Process -Name" not in source
    assert sum(
        line.strip().startswith("Stop-Process ") for line in source.splitlines()
    ) == 1
    assert (
        "$forcedCleanupDeadline = [DateTime]::UtcNow.AddSeconds(5)" in source
    )
    assert "Start-Sleep -Milliseconds 100" in source
    forced_wait = source.index("$forcedCleanupDeadline")
    fail_closed = source.index(
        'throw "Task-owned WINWORD process $wordProcessId survived cleanup."'
    )
    assert forced_wait < fail_closed


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _sealed(payload: dict, field: str) -> dict:
    result = dict(payload)
    result[field] = visual.sha256_json(result)
    return result


def test_render_producer_materializes_exact_three_roles_and_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    title = tmp_path / "title.docx"
    main = tmp_path / "main.docx"
    supplement = tmp_path / "supplement.docx"
    for path, content in (
        (title, b"editor-only title page"),
        (main, b"anonymous main"),
        (supplement, b"anonymous supplement"),
    ):
        path.write_bytes(content)

    upload = {
        "schema_version": render.UPLOAD_MANIFEST_SCHEMA,
        "status": "sealed_editor_and_reviewer_upload_roles",
        "editor_only_document_count": 1,
        "reviewer_visible_document_count": 2,
        "reviewer_visible_identity_occurrence_count": 0,
    }
    upload["upload_manifest_identity_sha256"] = render.sha256_json(upload)
    upload_path = _write_json(tmp_path / "upload.json", upload)
    qa = {
        "schema_version": render.STRUCTURAL_SCHEMA,
        "status": "passed_double_anonymous_three_role_ooxml_closure",
        "ooxml_zip_magic_and_required_structure_passed": True,
        "title_page_ooxml": {"sha256": render.sha256_file(title)},
        "main_ooxml": {"sha256": render.sha256_file(main)},
        "supplement_ooxml": {"sha256": render.sha256_file(supplement)},
        "submission_upload_role_manifest_sha256": render.sha256_file(upload_path),
        "submission_upload_role_manifest_identity_sha256": upload[
            "upload_manifest_identity_sha256"
        ],
    }
    qa["qa_identity_sha256"] = render.sha256_json(qa)
    qa_path = _write_json(tmp_path / "qa.json", qa)

    monkeypatch.setattr(render, "_executable", lambda _command, _role: Path(sys.executable))

    def fake_run(command: list[str], *, role: str) -> subprocess.CompletedProcess[str]:
        del role
        if "-OutputPdf" in command:
            pdf = Path(command[command.index("-OutputPdf") + 1])
            status = Path(command[command.index("-StatusJson") + 1])
            pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            _write_json(
                status,
                {
                    "schema_version": render.WORD_STATUS_SCHEMA,
                    "status": "complete",
                    "word_visible": False,
                    "read_only": True,
                    "macros_enabled": False,
                    "pages": 1,
                    "renderer_version": "mock-word",
                    "forced_task_owned_process_cleanup": False,
                },
            )
        else:
            prefix = Path(command[-1])
            Image.new("RGB", (1200, 1600), "white").save(
                prefix.parent / "page-1.png"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(render, "_run_checked", fake_run)
    destination = tmp_path / "rendered"
    result = render.render_manuscript_bundle(
        title_page_docx=title,
        anonymized_main_docx=main,
        anonymized_supplement_docx=supplement,
        structural_qa=qa_path,
        upload_manifest=upload_path,
        output=destination,
    )

    assert tuple(result["documents"]) == (
        "title_page",
        "anonymized_main",
        "anonymized_supplement",
    )
    assert all(record["pages"] == 1 for record in result["documents"].values())
    template = json.loads(
        (destination / "VISUAL_QA_ATTESTATION_TEMPLATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(template["reviewed_documents"]) == set(result["documents"])
    assert len(template["checks"]) == 12
    assert template["checks"][
        "figure4_prelocked_anchor_ids_and_deterministic_insets_verified"
    ] is False
    assert template["checks"][
        "figure4_audit_cards_match_axis_support_endpoint_and_eligibility_semantics"
    ] is False
    assert not any(template["checks"].values())
    assert template["submission_visual_gate_passed"] is False
    with pytest.raises(render.ManuscriptRenderError, match="refusing to overwrite"):
        render.render_manuscript_bundle(
            title_page_docx=title,
            anonymized_main_docx=main,
            anonymized_supplement_docx=supplement,
            structural_qa=qa_path,
            upload_manifest=upload_path,
            output=destination,
        )


def test_visual_gate_requires_and_closes_exact_three_document_roles(
    tmp_path: Path,
) -> None:
    roles = ("title_page", "anonymized_main", "anonymized_supplement")
    documents = {
        role: {
            "docx_sha256": visual.sha256_json([role, "docx"]),
            "pdf": {"sha256": visual.sha256_json([role, "pdf"])},
            "pages": index,
            "page_bundle_identity_sha256": visual.sha256_json([role, "pages"]),
        }
        for index, role in enumerate(roles, start=1)
    }
    upload_sha = visual.sha256_json("upload bytes")
    upload_identity = visual.sha256_json("upload identity")
    structural = _sealed(
        {
            "schema_version": visual.STRUCTURAL_SCHEMA,
            "status": "passed_double_anonymous_three_role_ooxml_closure",
            "reviewer_visible_identity_occurrence_count": 0,
            "deep_ooxml_anonymity_scan_passed": True,
            "submission_upload_role_manifest_sha256": upload_sha,
            "submission_upload_role_manifest_identity_sha256": upload_identity,
            "blind_images_used": 0,
        },
        "qa_identity_sha256",
    )
    structural_path = _write_json(tmp_path / "structural.json", structural)
    render = _sealed(
        {
            "schema_version": visual.RENDER_SCHEMA,
            "status": "completed_three_role_word_pdf_and_page_png_render",
            "visual_qa_completed": False,
            "documents": documents,
            "structural_qa_sha256": visual.sha256_file(structural_path),
            "structural_qa_identity_sha256": structural["qa_identity_sha256"],
            "submission_upload_role_manifest_sha256": upload_sha,
            "submission_upload_role_manifest_identity_sha256": upload_identity,
            "blind_images_used": 0,
        },
        "render_identity_sha256",
    )
    render_path = _write_json(tmp_path / "render.json", render)
    reviewed = {
        role: {
            "docx_sha256": record["docx_sha256"],
            "pdf_sha256": record["pdf"]["sha256"],
            "pages": record["pages"],
            "page_bundle_identity_sha256": record["page_bundle_identity_sha256"],
            "all_pages_reviewed_at_original_resolution": True,
        }
        for role, record in documents.items()
    }
    checks = {key: True for key in visual.VISUAL_CHECKS}
    attestation = _sealed(
        {
            "schema_version": visual.ATTESTATION_SCHEMA,
            "status": visual.FINAL_STATUS,
            "render_receipt_sha256": visual.sha256_file(render_path),
            "render_identity_sha256": render["render_identity_sha256"],
            "reviewer_full_name": "Visual QA Reviewer",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reviewed_documents": reviewed,
            "checks": checks,
            "review_notes": "Every page was reviewed at original resolution.",
            "submission_visual_gate_passed": True,
        },
        "attestation_identity_sha256",
    )
    template_base = dict(attestation)
    template_base.pop("attestation_identity_sha256")
    template_base.update(
        {
            "status": "incomplete_human_page_review_not_for_submission",
            "reviewer_full_name": None,
            "reviewed_at_utc": None,
            "checks": {key: False for key in checks},
            "review_notes": None,
            "submission_visual_gate_passed": False,
            "reviewed_documents": {
                role: {
                    **record,
                    "all_pages_reviewed_at_original_resolution": False,
                }
                for role, record in reviewed.items()
            },
        }
    )
    template = _sealed(template_base, "attestation_identity_sha256")
    template_path = _write_json(tmp_path / "template.json", template)
    attestation_path = _write_json(tmp_path / "attestation.json", attestation)

    result = visual.validate_visual_qa(
        render_receipt=render_path,
        structural_qa=structural_path,
        template=template_path,
        attestation=attestation_path,
        output=tmp_path / "visual-receipt.json",
    )
    assert result["documents_reviewed"] == 3
    assert result["editor_only_documents_reviewed"] == 1
    assert result["reviewer_visible_documents_reviewed"] == 2
    assert result["reviewer_visible_identity_occurrence_count"] == 0
    assert result["pages_reviewed"] == 6
    assert result["submission_visual_gate_passed"] is True
    with pytest.raises(visual.VisualQaError, match="refusing to overwrite"):
        visual.validate_visual_qa(
            render_receipt=render_path,
            structural_qa=structural_path,
            template=template_path,
            attestation=attestation_path,
            output=tmp_path / "visual-receipt.json",
        )

    wrong_checks = dict(attestation)
    wrong_checks.pop("attestation_identity_sha256")
    wrong_checks["checks"] = dict(wrong_checks["checks"])
    wrong_checks["checks"].pop("no_object_overlap")
    wrong_checks["checks"]["unrecognized_visual_claim"] = True
    wrong_checks = _sealed(wrong_checks, "attestation_identity_sha256")
    wrong_checks_path = _write_json(tmp_path / "attestation-wrong-checks.json", wrong_checks)
    with pytest.raises(visual.VisualQaError, match="twelve page-layout"):
        visual.validate_visual_qa(
            render_receipt=render_path,
            structural_qa=structural_path,
            template=template_path,
            attestation=wrong_checks_path,
            output=tmp_path / "wrong-checks-receipt.json",
        )

    structural_drift = dict(structural)
    structural_drift.pop("qa_identity_sha256")
    structural_drift["unexpected_drift"] = True
    structural_drift = _sealed(structural_drift, "qa_identity_sha256")
    structural_drift_path = _write_json(
        tmp_path / "structural-drift.json", structural_drift
    )
    with pytest.raises(visual.VisualQaError, match="structural QA authority"):
        visual.validate_visual_qa(
            render_receipt=render_path,
            structural_qa=structural_drift_path,
            template=template_path,
            attestation=attestation_path,
            output=tmp_path / "structural-drift-receipt.json",
        )
