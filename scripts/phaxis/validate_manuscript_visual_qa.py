"""Validate the human, page-by-page visual-QA attestation for final documents.

If ``--attestation`` is absent, the renderer's incomplete template is copied
there create-only and the command exits blocked.  The release can then be
resumed only after a named reviewer has inspected every main and supplementary
page, changed every explicit check to true, set a review timestamp and notes,
and resealed the edited JSON with ``--seal-attestation``.  No machine path can
assert that a page looked correct.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


ATTESTATION_SCHEMA = "PHAxis-manuscript-visual-qa-attestation-2.0"
RENDER_SCHEMA = "PHAxis-manuscript-pdf-page-render-2.0"
STRUCTURAL_SCHEMA = "PHAxis-manuscript-artifact-structural-qa-2.0"
RECEIPT_SCHEMA = "PHAxis-manuscript-human-visual-qa-receipt-2.0"
FINAL_STATUS = "complete_author_verified_page_visual_qa"
RECEIPT_STATUS = "passed_author_verified_three_role_page_visual_qa"
VISUAL_CHECKS = (
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


class VisualQaError(RuntimeError):
    """The required human visual review is absent, incomplete, or unbound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualQaError(message)


def _verify_seal(payload: Mapping[str, Any], field: str, role: str) -> None:
    identity = payload.get(field)
    unsigned = deepcopy(dict(payload))
    unsigned.pop(field, None)
    _require(
        isinstance(identity, str)
        and re.fullmatch(r"[0-9a-f]{64}", identity) is not None
        and sha256_json(unsigned) == identity,
        f"{role} identity seal mismatch",
    )


def seal_attestation(path: str | Path) -> dict[str, Any]:
    attestation_path = Path(path).resolve()
    payload = read_json(attestation_path)
    _require(payload.get("schema_version") == ATTESTATION_SCHEMA, "attestation schema changed")
    payload.pop("attestation_identity_sha256", None)
    payload["attestation_identity_sha256"] = sha256_json(payload)
    temporary = attestation_path.with_name(f".{attestation_path.name}.sealed.tmp")
    _require(not temporary.exists(), f"temporary seal path exists: {temporary}")
    atomic_write_json(temporary, payload)
    os.replace(temporary, attestation_path)
    return payload


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one JSON receipt without a replace-capable race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        atomic_write_json(temporary, payload)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise VisualQaError(f"refusing to overwrite: {path}") from error
        except OSError as error:
            raise VisualQaError(
                f"create-only visual-QA receipt publication failed: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def validate_visual_qa(
    *,
    render_receipt: str | Path,
    structural_qa: str | Path,
    template: str | Path,
    attestation: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    render_path = Path(render_receipt).resolve()
    structural_path = Path(structural_qa).resolve()
    template_path = Path(template).resolve()
    attestation_path = Path(attestation).resolve()
    output_path = Path(output).resolve()
    _require(not output_path.exists(), f"refusing to overwrite: {output_path}")
    _require(template_path.is_file() and not template_path.is_symlink(), "visual-QA template is absent")
    if not attestation_path.exists():
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                attestation_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(template_path.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
        raise VisualQaError(
            "human visual QA is required: an incomplete attestation template "
            f"was materialized at {attestation_path}; review every rendered page, "
            "complete it, reseal it with --seal-attestation, then resume the same release"
        )
    _require(attestation_path.is_file() and not attestation_path.is_symlink(), "visual-QA attestation is not a regular file")
    render = read_json(render_path)
    structural = read_json(structural_path)
    template_payload = read_json(template_path)
    payload = read_json(attestation_path)
    _verify_seal(render, "render_identity_sha256", "render receipt")
    _verify_seal(structural, "qa_identity_sha256", "structural QA")
    _verify_seal(template_payload, "attestation_identity_sha256", "visual-QA template")
    _verify_seal(payload, "attestation_identity_sha256", "visual-QA attestation")
    _require(
        render.get("schema_version") == RENDER_SCHEMA
        and render.get("status") == "completed_three_role_word_pdf_and_page_png_render"
        and render.get("visual_qa_completed") is False,
        "render receipt is not the pre-review final render",
    )
    _require(
        structural.get("schema_version") == STRUCTURAL_SCHEMA
        and structural.get("status") == "passed_double_anonymous_three_role_ooxml_closure"
        and structural.get("reviewer_visible_identity_occurrence_count") == 0
        and structural.get("deep_ooxml_anonymity_scan_passed") is True,
        "structural manuscript QA is not a pass",
    )
    _require(
        render.get("structural_qa_sha256") == sha256_file(structural_path)
        and render.get("structural_qa_identity_sha256")
        == structural.get("qa_identity_sha256"),
        "render receipt does not bind this structural QA authority",
    )
    _require(
        render.get("submission_upload_role_manifest_sha256")
        == structural.get("submission_upload_role_manifest_sha256")
        and render.get("submission_upload_role_manifest_identity_sha256")
        == structural.get("submission_upload_role_manifest_identity_sha256"),
        "render and structural QA disagree on the upload-role manifest",
    )
    _require(
        render.get("blind_images_used") == 0
        and structural.get("blind_images_used") == 0,
        "manuscript visual chain is blind-tainted",
    )
    expected_roles = {"title_page", "anonymized_main", "anonymized_supplement"}
    template_documents = template_payload.get("reviewed_documents")
    render_documents = render.get("documents")
    _require(
        template_payload.get("schema_version") == ATTESTATION_SCHEMA
        and template_payload.get("status")
        == "incomplete_human_page_review_not_for_submission"
        and template_payload.get("render_receipt_sha256") == sha256_file(render_path)
        and template_payload.get("render_identity_sha256")
        == render.get("render_identity_sha256")
        and template_payload.get("submission_visual_gate_passed") is False,
        "visual-QA template does not bind this incomplete render review",
    )
    _require(
        isinstance(template_documents, Mapping)
        and isinstance(render_documents, Mapping)
        and set(template_documents) == set(render_documents) == expected_roles,
        "visual-QA template document set is incomplete",
    )
    for role in expected_roles:
        template_record = template_documents[role]
        rendered_record = render_documents[role]
        _require(
            isinstance(template_record, Mapping)
            and isinstance(rendered_record, Mapping)
            and template_record.get("docx_sha256")
            == rendered_record.get("docx_sha256")
            and template_record.get("pdf_sha256")
            == rendered_record.get("pdf", {}).get("sha256")
            and template_record.get("pages") == rendered_record.get("pages")
            and template_record.get("page_bundle_identity_sha256")
            == rendered_record.get("page_bundle_identity_sha256")
            and template_record.get("all_pages_reviewed_at_original_resolution")
            is False,
            f"{role} visual-QA template record differs from the render",
        )
    _require(
        isinstance(template_payload.get("checks"), Mapping)
        and set(template_payload["checks"]) == set(VISUAL_CHECKS)
        and not any(template_payload["checks"].values()),
        "visual-QA template checks are not the exact incomplete twelve-check set",
    )
    _require(payload.get("schema_version") == ATTESTATION_SCHEMA, "visual-QA attestation schema changed")
    _require(payload.get("status") == FINAL_STATUS, "visual-QA attestation is not author-verified final")
    _require(
        payload.get("render_receipt_sha256") == sha256_file(render_path)
        and payload.get("render_identity_sha256") == render.get("render_identity_sha256"),
        "visual-QA attestation does not bind this render receipt",
    )
    reviewer = payload.get("reviewer_full_name")
    reviewed_at = payload.get("reviewed_at_utc")
    notes = payload.get("review_notes")
    _require(isinstance(reviewer, str) and reviewer.strip(), "visual-QA reviewer name is absent")
    _require(isinstance(notes, str) and notes.strip(), "visual-QA review notes are absent")
    _require(isinstance(reviewed_at, str), "visual-QA review timestamp is absent")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise VisualQaError("visual-QA review timestamp is invalid") from error
    _require(timestamp.tzinfo is not None, "visual-QA review timestamp must include timezone")
    checks = payload.get("checks")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == set(VISUAL_CHECKS)
        and all(value is True for value in checks.values()),
        "all twelve page-layout, identity and Fig.4 audit checks must be explicitly true",
    )
    reviewed_documents = payload.get("reviewed_documents")
    _require(
        isinstance(reviewed_documents, Mapping)
        and isinstance(render_documents, Mapping)
        and set(reviewed_documents)
        == set(render_documents)
        == expected_roles,
        "visual-QA document set is incomplete",
    )
    total_pages = 0
    for role in ("title_page", "anonymized_main", "anonymized_supplement"):
        reviewed = reviewed_documents[role]
        rendered = render_documents[role]
        _require(isinstance(reviewed, Mapping) and isinstance(rendered, Mapping), f"{role} visual-QA record is invalid")
        expected = {
            "docx_sha256": rendered.get("docx_sha256"),
            "pdf_sha256": rendered.get("pdf", {}).get("sha256"),
            "pages": rendered.get("pages"),
            "page_bundle_identity_sha256": rendered.get("page_bundle_identity_sha256"),
        }
        _require(
            all(reviewed.get(field) == value for field, value in expected.items()),
            f"{role} visual-QA record differs from the rendered artifact",
        )
        _require(
            reviewed.get("all_pages_reviewed_at_original_resolution") is True,
            f"{role} pages were not all reviewed at original resolution",
        )
        total_pages += int(rendered["pages"])
    _require(payload.get("submission_visual_gate_passed") is True, "submission visual gate is not explicitly passed")
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "render_receipt_sha256": sha256_file(render_path),
        "render_identity_sha256": render["render_identity_sha256"],
        "structural_qa_sha256": sha256_file(structural_path),
        "structural_qa_identity_sha256": structural["qa_identity_sha256"],
        "submission_upload_role_manifest_sha256": render[
            "submission_upload_role_manifest_sha256"
        ],
        "submission_upload_role_manifest_identity_sha256": render[
            "submission_upload_role_manifest_identity_sha256"
        ],
        "attestation_sha256": sha256_file(attestation_path),
        "attestation_identity_sha256": payload["attestation_identity_sha256"],
        "reviewer_full_name": reviewer.strip(),
        "reviewed_at_utc": reviewed_at,
        "documents_reviewed": 3,
        "editor_only_documents_reviewed": 1,
        "reviewer_visible_documents_reviewed": 2,
        "reviewer_visible_identity_occurrence_count": 0,
        "pages_reviewed": total_pages,
        "all_pages_reviewed_at_original_resolution": True,
        "submission_visual_gate_passed": True,
        "submission_use_allowed": True,
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
    }
    result["visual_qa_identity_sha256"] = sha256_json(result)
    _write_create_only_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-receipt", type=Path)
    parser.add_argument("--structural-qa", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seal-attestation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seal_attestation:
        _require(
            args.render_receipt is None
            and args.structural_qa is None
            and args.template is None
            and args.output is None,
            "--seal-attestation accepts only --attestation",
        )
        payload = seal_attestation(args.attestation)
        print(payload["attestation_identity_sha256"])
        return 0
    _require(
        all(value is not None for value in (args.render_receipt, args.structural_qa, args.template, args.output)),
        "validation requires --render-receipt, --structural-qa, --template and --output",
    )
    result = validate_visual_qa(
        render_receipt=args.render_receipt,
        structural_qa=args.structural_qa,
        template=args.template,
        attestation=args.attestation,
        output=args.output,
    )
    print(result["visual_qa_identity_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VisualQaError, OSError, ValueError, TypeError) as error:
        print(f"blocked: {error}", file=sys.stderr)
        raise SystemExit(3)
