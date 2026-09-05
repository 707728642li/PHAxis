"""Render the three-role PHAxis submission bundle to PDF and page PNGs.

The producer uses the checked-in, task-owned Microsoft Word COM wrapper and a
resolved Poppler ``pdftoppm`` executable.  It records renderer identities,
PDF/page hashes and page counts, then writes an incomplete human visual-QA
template.  Rendering alone never marks visual QA as passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-manuscript-pdf-page-render-2.0"
STATUS = "completed_three_role_word_pdf_and_page_png_render"
STRUCTURAL_SCHEMA = "PHAxis-manuscript-artifact-structural-qa-2.0"
UPLOAD_MANIFEST_SCHEMA = "PHAxis-submission-upload-role-manifest-1.0"
WORD_STATUS_SCHEMA = "PHAxis-word-com-docx-render-1.0"
ATTESTATION_SCHEMA = "PHAxis-manuscript-visual-qa-attestation-2.0"
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


class ManuscriptRenderError(RuntimeError):
    """The real document renderer or page rasterizer failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptRenderError(message)


def _read_json_utf8_sig(path: Path, role: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{role} is absent or a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ManuscriptRenderError(f"{role} is not JSON") from error
    _require(isinstance(payload, dict), f"{role} must contain one object")
    return payload


def _executable(command: str | Path, role: str) -> Path:
    value = str(command)
    resolved = shutil.which(value)
    path = Path(resolved if resolved else value).resolve()
    _require(path.is_file(), f"{role} executable is absent: {command}")
    return path


def _pdf_record(path: Path, *, role: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{role} PDF is absent")
    raw = path.read_bytes()
    _require(raw.startswith(b"%PDF-"), f"{role} PDF magic is invalid")
    _require(b"%%EOF" in raw[-4096:], f"{role} PDF has no terminal EOF marker")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "pdf_magic": "%PDF-"}


def _page_records(directory: Path, *, expected_pages: int, role: str) -> list[dict[str, Any]]:
    def page_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError) as error:
            raise ManuscriptRenderError(f"{role} page filename is invalid: {path.name}") from error

    pages = sorted(directory.glob("page-*.png"), key=page_number)
    _require(len(pages) == expected_pages, f"{role} page raster count is {len(pages)}, expected {expected_pages}")
    records: list[dict[str, Any]] = []
    for index, path in enumerate(pages, start=1):
        _require(page_number(path) == index, f"{role} page numbering is not contiguous")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        _require(width >= 1000 and height >= 1000, f"{role} page {index} is too small")
        _require(mode in {"RGB", "RGBA"}, f"{role} page {index} has unexpected mode {mode}")
        records.append(
            {
                "page": index,
                "filename": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "width_px": width,
                "height_px": height,
                "mode": mode,
            }
        )
    return records


def _run_checked(command: list[str], *, role: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    _require(
        completed.returncode == 0,
        f"{role} failed ({completed.returncode}): {(completed.stderr or completed.stdout)[-1000:]}",
    )
    return completed


def render_manuscript_bundle(
    *,
    title_page_docx: str | Path,
    anonymized_main_docx: str | Path,
    anonymized_supplement_docx: str | Path,
    structural_qa: str | Path,
    upload_manifest: str | Path,
    output: str | Path,
    powershell: str | Path = "powershell.exe",
    pdftoppm: str | Path = "pdftoppm",
) -> dict[str, Any]:
    title_path = Path(title_page_docx).resolve()
    main_path = Path(anonymized_main_docx).resolve()
    supplement_path = Path(anonymized_supplement_docx).resolve()
    qa_path = Path(structural_qa).resolve()
    upload_manifest_path = Path(upload_manifest).resolve()
    destination = Path(output).resolve()
    _require(not destination.exists(), f"refusing to overwrite: {destination}")
    _require(
        title_path.is_file() and main_path.is_file() and supplement_path.is_file(),
        "one or more three-role DOCX files are absent",
    )
    qa = read_json(qa_path)
    upload_roles = read_json(upload_manifest_path)
    qa_unsigned = dict(qa)
    qa_identity = qa_unsigned.pop("qa_identity_sha256", None)
    _require(
        isinstance(qa_identity, str) and sha256_json(qa_unsigned) == qa_identity,
        "structural QA identity seal mismatch",
    )
    upload_unsigned = dict(upload_roles)
    upload_identity = upload_unsigned.pop("upload_manifest_identity_sha256", None)
    _require(
        isinstance(upload_identity, str)
        and sha256_json(upload_unsigned) == upload_identity,
        "upload-role manifest identity seal mismatch",
    )
    _require(
        qa.get("schema_version") == STRUCTURAL_SCHEMA
        and qa.get("status") == "passed_double_anonymous_three_role_ooxml_closure"
        and qa.get("ooxml_zip_magic_and_required_structure_passed") is True,
        "structural OOXML QA is not a pass",
    )
    _require(
        qa.get("title_page_ooxml", {}).get("sha256") == sha256_file(title_path)
        and qa.get("main_ooxml", {}).get("sha256") == sha256_file(main_path)
        and qa.get("supplement_ooxml", {}).get("sha256") == sha256_file(supplement_path),
        "DOCX files differ from structural QA authority",
    )
    _require(
        upload_roles.get("schema_version") == UPLOAD_MANIFEST_SCHEMA
        and upload_roles.get("status") == "sealed_editor_and_reviewer_upload_roles"
        and upload_roles.get("editor_only_document_count") == 1
        and upload_roles.get("reviewer_visible_document_count") == 2
        and upload_roles.get("reviewer_visible_identity_occurrence_count") == 0
        and qa.get("submission_upload_role_manifest_sha256")
        == sha256_file(upload_manifest_path)
        and qa.get("submission_upload_role_manifest_identity_sha256")
        == upload_roles.get("upload_manifest_identity_sha256"),
        "submission upload-role manifest is not the sealed three-role authority",
    )
    powershell_path = _executable(powershell, "PowerShell")
    pdftoppm_path = _executable(pdftoppm, "pdftoppm")
    wrapper = Path(__file__).with_name("render_docx_with_word_com_windows.ps1").resolve()
    _require(wrapper.is_file() and not wrapper.is_symlink(), "Word COM wrapper is absent")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".manuscript-render-", dir=destination.parent)).resolve()
    try:
        documents: dict[str, Any] = {}
        for role, docx in (
            ("title_page", title_path),
            ("anonymized_main", main_path),
            ("anonymized_supplement", supplement_path),
        ):
            role_dir = staging / role
            pages_dir = role_dir / "pages_150dpi"
            pages_dir.mkdir(parents=True)
            pdf = role_dir / f"{role}.pdf"
            word_status = role_dir / "word_render_status.json"
            _run_checked(
                [
                    str(powershell_path),
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(wrapper),
                    "-InputDocx",
                    str(docx),
                    "-OutputPdf",
                    str(pdf),
                    "-StatusJson",
                    str(word_status),
                ],
                role=f"{role} Word COM render",
            )
            status = _read_json_utf8_sig(word_status, f"{role} Word render status")
            pages = status.get("pages")
            _require(
                status.get("schema_version") == WORD_STATUS_SCHEMA
                and status.get("status") == "complete"
                and status.get("word_visible") is False
                and status.get("read_only") is True
                and status.get("macros_enabled") is False
                and isinstance(pages, int)
                and pages > 0,
                f"{role} Word renderer did not complete safely",
            )
            pdf_record = _pdf_record(pdf, role=role)
            prefix = pages_dir / "page"
            _run_checked(
                [
                    str(pdftoppm_path),
                    "-png",
                    "-r",
                    "150",
                    str(pdf),
                    str(prefix),
                ],
                role=f"{role} Poppler page rasterization",
            )
            page_records = _page_records(pages_dir, expected_pages=pages, role=role)
            documents[role] = {
                "docx_sha256": sha256_file(docx),
                "pdf": pdf_record,
                "pages": pages,
                "page_raster_dpi": 150,
                "page_png_records": page_records,
                "page_bundle_identity_sha256": sha256_json(page_records),
                "word_render_status_sha256": sha256_file(word_status),
                "word_renderer_version": status.get("renderer_version"),
                "forced_task_owned_process_cleanup": status.get(
                    "forced_task_owned_process_cleanup"
                ),
            }
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "structural_qa_sha256": sha256_file(qa_path),
            "structural_qa_identity_sha256": qa.get("qa_identity_sha256"),
            "submission_upload_role_manifest_sha256": sha256_file(
                upload_manifest_path
            ),
            "submission_upload_role_manifest_identity_sha256": upload_roles.get(
                "upload_manifest_identity_sha256"
            ),
            "word_com_wrapper_sha256": sha256_file(wrapper),
            "powershell_executable": str(powershell_path),
            "powershell_executable_sha256": sha256_file(powershell_path),
            "pdftoppm_executable": str(pdftoppm_path),
            "pdftoppm_executable_sha256": sha256_file(pdftoppm_path),
            "documents": documents,
            "pdf_magic_passed": True,
            "page_rasterization_completed": True,
            "visual_qa_completed": False,
            "submission_use_allowed": False,
            "blind_images_used": 0,
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
        }
        result["render_identity_sha256"] = sha256_json(result)
        atomic_write_json(staging / "receipt.json", result)
        attestation: dict[str, Any] = {
            "schema_version": ATTESTATION_SCHEMA,
            "status": "incomplete_human_page_review_not_for_submission",
            "render_receipt_sha256": sha256_file(staging / "receipt.json"),
            "render_identity_sha256": result["render_identity_sha256"],
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
            "checks": {name: False for name in VISUAL_CHECKS},
            "review_notes": None,
            "submission_visual_gate_passed": False,
        }
        attestation["attestation_identity_sha256"] = sha256_json(attestation)
        atomic_write_json(staging / "VISUAL_QA_ATTESTATION_TEMPLATE.json", attestation)
        os.replace(staging, destination)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title-page-docx", type=Path, required=True)
    parser.add_argument("--anonymized-main-docx", type=Path, required=True)
    parser.add_argument("--anonymized-supplement-docx", type=Path, required=True)
    parser.add_argument("--structural-qa", type=Path, required=True)
    parser.add_argument("--upload-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--powershell", default="powershell.exe")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = render_manuscript_bundle(**vars(args))
    print(result["render_identity_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManuscriptRenderError, OSError, ValueError, TypeError) as error:
        print(f"blocked: {error}", file=sys.stderr)
        raise SystemExit(2)
