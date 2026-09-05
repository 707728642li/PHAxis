from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from tests.phaxis import test_submission_docx_builder as main_fixture
from tests.phaxis import test_supplementary_docx_builder as supplement_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load(
    "phaxis_double_anonymous_full_chain_gate",
    "scripts/phaxis/verify_manuscript_artifacts.py",
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_stage53_to_stage55_three_role_chain_is_hash_closed(tmp_path: Path) -> None:
    main_master = main_fixture._manuscript(
        tmp_path / "main-master.md", placeholders=False
    )
    main_manuscript = tmp_path / "main-compiled.md"
    main_manuscript.write_bytes(main_master.read_bytes())
    metadata = main_fixture._metadata(tmp_path / "metadata-v2.json", final=True)
    main_compile_path = tmp_path / "main-compile.json"
    main_compile = main_fixture._compile_receipt(
        main_compile_path, main_manuscript
    )
    main_compile["master_sha256"] = hashlib.sha256(
        main_master.read_bytes()
    ).hexdigest()
    main_compile.pop("receipt_identity_sha256")
    main_compile["receipt_identity_sha256"] = main_fixture.builder._canonical_hash(
        main_compile
    )
    _write_json(main_compile_path, main_compile)

    figure_summary = supplement_fixture._figure_suite(
        tmp_path / "figure-suite", main_compile_path
    )
    title_page = tmp_path / "submission" / "PHAxis_title_page.docx"
    anonymous_main = tmp_path / "submission" / "PHAxis_anonymized_main.docx"
    main_docx_receipt = tmp_path / "submission" / "receipt.json"
    main_fixture.builder.build_submission_docx(
        mode="final",
        manuscript=main_manuscript,
        submission_metadata=metadata,
        compile_receipt=main_compile_path,
        figure_summary=figure_summary,
        title_page_output=title_page,
        anonymized_main_output=anonymous_main,
        receipt=main_docx_receipt,
    )

    supplement_master = supplement_fixture._supplement(
        tmp_path / "supplement-master.md", final=True
    )
    supplement_manuscript = tmp_path / "supplement-compiled.md"
    supplement_manuscript.write_bytes(supplement_master.read_bytes())
    anonymous_supplement = (
        tmp_path / "supplement" / "PHAxis_anonymized_supplement.docx"
    )
    supplement_docx_receipt = tmp_path / "supplement" / "receipt.json"
    supplement_fixture.builder.build_supplementary_docx(
        mode="final",
        supplement=supplement_manuscript,
        output=anonymous_supplement,
        main_manuscript=main_manuscript,
        main_compile_receipt=main_compile_path,
        figure_summary=figure_summary,
        receipt=supplement_docx_receipt,
    )

    figures = json.loads(figure_summary.read_text(encoding="utf-8"))
    table_identity = {
        stem: record["item_identity_sha256"]
        for stem, record in figures["supplementary_tables"].items()
    }
    supplement_compile = {
        "schema_version": gate.SUPPLEMENT_COMPILE_SCHEMA,
        "status": "completed_strict_final_supplementary_compilation",
        "master_sha256": hashlib.sha256(supplement_master.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(
            supplement_manuscript.read_bytes()
        ).hexdigest(),
        "main_compile_receipt_sha256": hashlib.sha256(
            main_compile_path.read_bytes()
        ).hexdigest(),
        "main_compile_receipt_identity_sha256": main_compile[
            "receipt_identity_sha256"
        ],
        "supplementary_table_data_resource_count": 10,
        "supplementary_table_data_materialized": True,
        "supplementary_table_bundle_receipt_sha256": figures[
            "supplementary_table_bundle_receipt_sha256"
        ],
        "supplementary_table_bundle_identity_sha256": figures[
            "supplementary_table_bundle_identity_sha256"
        ],
        "supplementary_table_item_identity_sha256": table_identity,
        "blind_images_used": 0,
    }
    supplement_compile["receipt_identity_sha256"] = gate._canonical_hash(
        supplement_compile
    )
    supplement_compile_path = _write_json(
        tmp_path / "supplement-compile.json", supplement_compile
    )

    qa_path = tmp_path / "artifact-qa" / "receipt.json"
    upload_path = tmp_path / "artifact-qa" / "upload-role-manifest.json"
    result = gate.verify_manuscript_artifacts(
        main_master=main_master,
        supplement_master=supplement_master,
        main_manuscript=main_manuscript,
        main_compile_receipt=main_compile_path,
        supplement_manuscript=supplement_manuscript,
        supplement_compile_receipt=supplement_compile_path,
        submission_metadata=metadata,
        figure_summary=figure_summary,
        title_page_docx=title_page,
        anonymized_main_docx=anonymous_main,
        submission_docx_receipt=main_docx_receipt,
        anonymized_supplement_docx=anonymous_supplement,
        supplement_docx_receipt=supplement_docx_receipt,
        output=qa_path,
        upload_manifest=upload_path,
    )

    assert result["reviewer_visible_identity_occurrence_count"] == 0
    assert result["title_page_ooxml"]["reviewer_visible"] is False
    assert result["main_ooxml"]["identity_denylist_hit_count"] == 0
    assert result["supplement_ooxml"]["identity_denylist_hit_count"] == 0
    assert result["main_ooxml"]["image_placement_count"] == 6
    assert result["supplement_ooxml"]["image_placement_count"] == 9
    # OOXML may safely deduplicate byte-identical plates into one media part;
    # logical placements, relationships and the physical part set remain closed.
    assert result["main_ooxml"]["embedded_media_count"] >= 1
    assert result["supplement_ooxml"]["embedded_media_count"] >= 1
    upload = json.loads(upload_path.read_text(encoding="utf-8"))
    assert set(upload["roles"]["editor_only"]) == {"title_page"}
    assert set(upload["roles"]["reviewer_visible"]) == {
        "anonymized_main",
        "anonymized_supplement",
    }
    assert upload["reviewer_visible_identity_occurrence_count"] == 0
    unsigned = dict(upload)
    identity = unsigned.pop("upload_manifest_identity_sha256")
    assert gate._canonical_hash(unsigned) == identity
