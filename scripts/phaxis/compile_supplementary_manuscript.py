"""Compile the final PHAxis supplementary Markdown from its static master.

The supplementary source currently has no machine-number placeholders.  It is
nevertheless a development master and must not be passed directly to the final
DOCX builder.  This producer binds that master to the completed main-manuscript
compile receipt and final 6+9 figure suite, changes only the controlled status
frontmatter, and emits a create-only UTF-8 Markdown document plus a sealed
receipt.  It never invents result values or author metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.supplementary_tables import (  # noqa: E402
    SupplementaryTableError,
    validate_supplementary_table_data_bundle,
)


SCHEMA_VERSION = "PHAxis-supplementary-manuscript-compile-receipt-1.0"
STATUS = "completed_strict_final_supplementary_compilation"
MAIN_COMPILE_SCHEMA = "PHAxis-manuscript-compile-receipt-1.2"
MAIN_COMPILE_STATUS = "completed_strict_final_manuscript_compilation"
FIGURE_SCHEMA = "PHAxis-publication-figure-suite-1.0"
FIGURE_STATUS = "final_sealed_strict_train399_only"
MASTER_STATUS = (
    "**Status:** Evidence-bound development master; final machine values "
    "pending; not for submission  "
)
FINAL_STATUS = (
    "**Status:** Final evidence-bound supplementary source; submission is "
    "allowed only after final document rendering and author visual QA  "
)
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
SUPPLEMENT_FIRST_LINE = re.compile(
    r'^# Supplementary Materials for [“"](?P<title>.+?)[”"]\s*$'
)
COMPANION_TITLE_LINE = re.compile(
    r'^\*\*Companion main manuscript:\*\*\s+[“"](?P<title>.+?)[”"]\s*$',
    re.MULTILINE,
)


class SupplementaryCompileError(RuntimeError):
    """The supplementary source or one of its final authorities is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SupplementaryCompileError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SupplementaryCompileError("payload is not finite canonical JSON") from error


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_bytes(path: Path, role: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{role} is absent or a symlink")
    return path.read_bytes()


def _read_text(path: Path, role: str) -> tuple[bytes, str]:
    raw = _read_bytes(path, role)
    try:
        return raw, raw.decode("utf-8")
    except UnicodeError as error:
        raise SupplementaryCompileError(f"{role} must be UTF-8") from error


def _read_json(path: Path, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes(path, role)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SupplementaryCompileError(f"{role} is not strict UTF-8 JSON") from error
    _require(isinstance(payload, dict), f"{role} must contain one JSON object")
    return raw, payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_seal(payload: Mapping[str, Any], field: str, role: str) -> None:
    _require(_is_sha256(payload.get(field)), f"{role} has no valid {field}")
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field)
    _require(_canonical_hash(unsigned) == observed, f"{role} identity seal mismatch")


def _publish_new(path: Path, raw: bytes) -> None:
    _require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SupplementaryCompileError(f"refusing to overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_companion_titles(*, main_text: str, master_text: str) -> str:
    """Bind both supplementary title declarations to the compiled main title."""

    main_title_match = re.match(r"^#\s+(.+?)\s*$", main_text, re.MULTILINE)
    _require(main_title_match is not None, "compiled main manuscript has no title")
    main_title = main_title_match.group(1)

    master_lines = master_text.splitlines()
    _require(bool(master_lines), "supplementary master is empty")
    first_line_match = SUPPLEMENT_FIRST_LINE.fullmatch(master_lines[0])
    _require(
        first_line_match is not None
        and first_line_match.group("title") == main_title,
        "supplementary first-line title differs from the compiled main manuscript",
    )

    companion_matches = list(COMPANION_TITLE_LINE.finditer(master_text))
    _require(
        len(companion_matches) == 1,
        "supplementary master must define exactly one Companion main manuscript title",
    )
    _require(
        companion_matches[0].group("title") == main_title,
        "supplementary Companion main manuscript title differs from the compiled main manuscript",
    )
    return main_title


def compile_supplementary_manuscript(
    *,
    master: str | Path,
    main_manuscript: str | Path,
    main_compile_receipt: str | Path,
    figure_summary: str | Path,
    output: str | Path,
    receipt: str | Path | None = None,
) -> dict[str, Any]:
    master_path = Path(master).resolve()
    main_path = Path(main_manuscript).resolve()
    main_receipt_path = Path(main_compile_receipt).resolve()
    figure_path = Path(figure_summary).resolve()
    output_path = Path(output).resolve()
    receipt_path = (
        Path(receipt).resolve()
        if receipt is not None
        else output_path.with_name(f"{output_path.name}.receipt.json")
    )
    _require(output_path.suffix.casefold() == ".md", "output must use .md")
    _require(output_path != receipt_path, "output and receipt paths must differ")
    _require(not output_path.exists() and not receipt_path.exists(), "refusing to overwrite output or receipt")

    master_raw, master_text = _read_text(master_path, "supplementary master")
    main_raw, main_text = _read_text(main_path, "compiled main manuscript")
    main_receipt_raw, main_receipt = _read_json(main_receipt_path, "main compile receipt")
    figure_raw, figures = _read_json(figure_path, "final figure summary")

    _require(
        main_receipt.get("schema_version") == MAIN_COMPILE_SCHEMA
        and main_receipt.get("status") == MAIN_COMPILE_STATUS,
        "main compile receipt is not final",
    )
    _verify_seal(main_receipt, "receipt_identity_sha256", "main compile receipt")
    _require(main_receipt.get("output_sha256") == _sha256(main_raw), "main manuscript/receipt hash mismatch")
    _require(PLACEHOLDER.search(main_text) is None, "compiled main manuscript retains tokens")
    _require(
        figures.get("schema_version") == FIGURE_SCHEMA
        and figures.get("status") == FIGURE_STATUS
        and figures.get("submission_use_allowed") is True,
        "figure suite is not final and submission-eligible",
    )
    claim = figures.get("claim_contract")
    _require(
        isinstance(claim, Mapping)
        and claim.get("main_figure_count") == 6
        and claim.get("supplementary_figure_count") == 9
        and claim.get("supplementary_table_data_resource_count") == 10,
        "figure suite does not contain the required 6+9 plates and S1--S10 data",
    )
    table_receipt_relative = figures.get("supplementary_table_bundle_receipt")
    _require(
        isinstance(table_receipt_relative, str) and bool(table_receipt_relative),
        "figure suite omits the supplementary table/data receipt path",
    )
    table_receipt_path = (figure_path.parent / table_receipt_relative).resolve()
    _require(
        table_receipt_path.is_relative_to(figure_path.parent),
        "supplementary table/data receipt escapes the figure suite",
    )
    try:
        table_bundle = validate_supplementary_table_data_bundle(
            table_receipt_path, require_final=True
        )
    except SupplementaryTableError as error:
        raise SupplementaryCompileError(
            f"supplementary Table/Data S1--S10 validation failed: {error}"
        ) from error
    _require(
        figures.get("supplementary_tables") == table_bundle["items"]
        and figures.get("supplementary_table_bundle_receipt_sha256")
        == table_bundle["receipt_sha256"]
        and figures.get("supplementary_table_bundle_identity_sha256")
        == table_bundle["bundle_identity_sha256"]
        and figures.get("supplementary_table_bundle_sha256")
        == table_bundle["bundle_file_sha256"],
        "figure summary and supplementary table/data bundle differ",
    )
    _require(
        figures.get("model_contract_proposal_identity_sha256")
        == main_receipt.get("model_contract_proposal_identity_sha256"),
        "main manuscript and figure suite use different model authority",
    )
    _require(PLACEHOLDER.search(master_text) is None, "supplementary master retains machine tokens")
    _require(master_text.count(MASTER_STATUS) == 1, "supplementary master status contract changed")
    _require(FINAL_STATUS not in master_text, "supplementary master is already marked final")
    _require(
        len(re.findall(r"^### Figure S[1-9]\.\s", master_text, re.MULTILINE)) == 9,
        "supplementary master does not define exactly Figure S1--S9",
    )
    _require(
        len(re.findall(r"^### (?:Table|Data) S(?:[1-9]|10)\.\s", master_text, re.MULTILINE)) == 10,
        "supplementary master does not define exactly Table/Data S1--S10",
    )
    _validate_companion_titles(main_text=main_text, master_text=master_text)

    compiled = master_text.replace(MASTER_STATUS, FINAL_STATUS, 1)
    output_raw = compiled.encode("utf-8")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "master_sha256": _sha256(master_raw),
        "main_manuscript_sha256": _sha256(main_raw),
        "main_compile_receipt_sha256": _sha256(main_receipt_raw),
        "main_compile_receipt_identity_sha256": main_receipt["receipt_identity_sha256"],
        "figure_summary_sha256": _sha256(figure_raw),
        "figure_suite_identity_sha256": figures.get("figure_suite_identity_sha256"),
        "model_contract_proposal_identity_sha256": main_receipt[
            "model_contract_proposal_identity_sha256"
        ],
        "output_sha256": _sha256(output_raw),
        "status_frontmatter_replacements": 1,
        "numeric_or_author_values_inserted": 0,
        "unresolved_token_count": 0,
        "main_figure_count": 6,
        "supplementary_figure_count": 9,
        "supplementary_table_data_resource_count": 10,
        "supplementary_table_data_materialized": True,
        "supplementary_table_bundle_receipt_sha256": table_bundle[
            "receipt_sha256"
        ],
        "supplementary_table_bundle_identity_sha256": table_bundle[
            "bundle_identity_sha256"
        ],
        "supplementary_table_item_identity_sha256": {
            stem: record["item_identity_sha256"]
            for stem, record in table_bundle["items"].items()
        },
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
    }
    result["receipt_identity_sha256"] = _canonical_hash(result)
    receipt_raw = (
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _publish_new(output_path, output_raw)
    try:
        _publish_new(receipt_path, receipt_raw)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--main-manuscript", type=Path, required=True)
    parser.add_argument("--main-compile-receipt", type=Path, required=True)
    parser.add_argument("--figure-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compile_supplementary_manuscript(
        master=args.master,
        main_manuscript=args.main_manuscript,
        main_compile_receipt=args.main_compile_receipt,
        figure_summary=args.figure_summary,
        output=args.output,
        receipt=args.receipt,
    )
    print(result["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupplementaryCompileError as error:
        print(f"blocked: {error}", file=os.sys.stderr)
        raise SystemExit(2)
