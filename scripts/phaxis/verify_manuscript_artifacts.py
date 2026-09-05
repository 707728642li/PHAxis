"""Verify the three-role PHAxis double-anonymous submission bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata
from xml.etree import ElementTree as ET
import zipfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import sha256_file  # noqa: E402
from phaxis.supplementary_tables import (  # noqa: E402
    SupplementaryTableError,
    validate_supplementary_table_data_bundle,
)


SCHEMA_VERSION = "PHAxis-manuscript-artifact-structural-qa-2.0"
STATUS = "passed_double_anonymous_three_role_ooxml_closure"
UPLOAD_MANIFEST_SCHEMA = "PHAxis-submission-upload-role-manifest-1.0"
UPLOAD_MANIFEST_STATUS = "sealed_editor_and_reviewer_upload_roles"
MAIN_COMPILE_SCHEMA = "PHAxis-manuscript-compile-receipt-1.2"
SUPPLEMENT_COMPILE_SCHEMA = "PHAxis-supplementary-manuscript-compile-receipt-1.0"
MAIN_DOCX_SCHEMA = "PHAxis-submission-docx-build-2.0"
SUPPLEMENT_DOCX_SCHEMA = "PHAxis-supplementary-docx-build-2.0"
FIGURE_SCHEMA = "PHAxis-publication-figure-suite-1.0"
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
EDITOR_ONLY_DECLARATIONS = (
    "Acknowledgments",
    "Funding",
    "Author Contributions",
    "Competing Interests",
)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DC_NS = "http://purl.org/dc/elements/1.1/"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
REQUIRED_PARTS = frozenset(
    {
        "[Content_Types].xml",
        "_rels/.rels",
        "docProps/core.xml",
        "word/document.xml",
        "word/_rels/document.xml.rels",
        "word/styles.xml",
    }
)


class ManuscriptArtifactError(RuntimeError):
    """The final manuscript artifact graph is not structurally closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptArtifactError(message)


def _canonical_hash(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManuscriptArtifactError("payload is not finite canonical JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bytes(path: Path, role: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{role} is absent or a symlink")
    return path.read_bytes()


def _read_json(path: Path, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes(path, role)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ManuscriptArtifactError(f"{role} is not strict UTF-8 JSON") from error
    _require(isinstance(payload, dict), f"{role} must contain one JSON object")
    return raw, payload


def _read_text(path: Path, role: str) -> tuple[bytes, str]:
    raw = _read_bytes(path, role)
    try:
        return raw, raw.decode("utf-8")
    except UnicodeError as error:
        raise ManuscriptArtifactError(f"{role} must be UTF-8") from error


def _verify_seal(payload: Mapping[str, Any], field: str, role: str) -> None:
    identity = payload.get(field)
    _require(
        isinstance(identity, str)
        and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity),
        f"{role} has no valid {field}",
    )
    unsigned = deepcopy(dict(payload))
    unsigned.pop(field, None)
    _require(_canonical_hash(unsigned) == identity, f"{role} identity seal mismatch")


def _normalized_search_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _markdown_visible_text(value: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    text = re.sub(r"[*_`]", "", text)
    return _normalized_search_text(text)


def _compact_identity_text(value: str) -> str:
    return "".join(character for character in value if character.isalnum())


def _editor_only_declarations(main_text: str) -> dict[str, str]:
    labels = {
        "acknowledgments": "Acknowledgments",
        "funding": "Funding",
        "author contributions": "Author Contributions",
        "competing interests": "Competing Interests",
    }
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in main_text.splitlines():
        heading = re.match(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", line)
        if heading:
            current = labels.get(heading.group(1).strip().casefold())
            if current is not None:
                _require(current not in result, f"duplicate editor-only declaration: {current}")
                result[current] = []
            continue
        if current is not None:
            result[current].append(line)
    closed = {label: "\n".join(result.get(label, [])).strip() for label in labels.values()}
    for label, value in closed.items():
        _require(bool(value), f"compiled main manuscript lacks declaration: {label}")
    return closed


def _h2_section(main_text: str, title: str) -> str:
    values: list[str] = []
    collecting = False
    found = 0
    for line in main_text.splitlines():
        heading = re.match(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", line)
        if heading:
            semantic = heading.group(1).strip().casefold()
            collecting = semantic == title.casefold()
            if collecting:
                found += 1
            continue
        if collecting:
            values.append(line)
    _require(found == 1, f"compiled main manuscript requires exactly one {title} section")
    value = "\n".join(values).strip()
    _require(bool(value), f"compiled main manuscript has an empty {title} section")
    return value


def _availability_contract(main_text: str) -> dict[str, Any]:
    data = _h2_section(main_text, "Data Availability")
    code = _h2_section(main_text, "Code Availability")
    combined = _markdown_visible_text(f"{data}\n{code}")
    _require(
        re.search(r"available\s+(?:only\s+)?upon\s+(?:reasonable\s+)?request", combined)
        is None,
        "Data Availability is only an available-upon-request statement",
    )
    data_urls = re.findall(r"https?://[^\s)`]+", data, flags=re.I)
    code_urls = re.findall(r"https?://[^\s)`]+", code, flags=re.I)
    _require(bool(data_urls), "Data Availability has no executable public URL")
    _require(bool(code_urls), "Code Availability has no executable public URL")
    _require("license" in combined, "availability statements omit a license")
    _require(
        any(
            marker in combined
            for marker in (
                "dataset card",
                "annotation schema",
                "documentation",
                "readme",
                "environment lock",
            )
        ),
        "availability statements omit basic documentation assets",
    )
    _require(
        "example input" in combined and "expected output" in combined,
        "availability statements omit reviewer-runnable example assets",
    )
    return {
        "data_availability_sha256": hashlib.sha256(
            _markdown_visible_text(data).encode("utf-8")
        ).hexdigest(),
        "code_availability_sha256": hashlib.sha256(
            _markdown_visible_text(code).encode("utf-8")
        ).hexdigest(),
        "data_url_count": len(data_urls),
        "code_url_count": len(code_urls),
        "available_upon_request_only": False,
        "license_declared": True,
        "basic_documentation_declared": True,
        "reviewer_runnable_example_declared": True,
    }


def _identity_denylist(
    metadata: Mapping[str, Any], declarations: Mapping[str, str]
) -> list[dict[str, str]]:
    values: list[tuple[str, str]] = []
    for index, author in enumerate(metadata.get("authors", []), start=1):
        if not isinstance(author, Mapping):
            continue
        for field in ("full_name", "email", "orcid", "postal_address"):
            value = author.get(field)
            if isinstance(value, str) and value.strip():
                values.append((f"author_{index}_{field}", value.strip()))
    for affiliation in metadata.get("affiliations", []):
        if isinstance(affiliation, Mapping):
            value = affiliation.get("text")
            if isinstance(value, str) and value.strip():
                values.append((f"affiliation_{affiliation.get('id')}", value.strip()))
    for label, value in declarations.items():
        visible = _markdown_visible_text(value)
        if visible:
            values.append((f"declaration_body_{label}", visible))
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, value in values:
        normalized = _normalized_search_text(value)
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(
            {
                "label": label,
                "normalized": normalized,
                "compact": _compact_identity_text(normalized),
                "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            }
        )
    _require(bool(result), "submission metadata produced an empty identity denylist")
    return result


def _denylist_hits(text: str, denylist: Sequence[Mapping[str, str]]) -> list[str]:
    normalized = _normalized_search_text(text)
    compact = _compact_identity_text(normalized)
    return [
        str(row["label"])
        for row in denylist
        if str(row["normalized"]) in normalized
        or (
            len(str(row.get("compact", ""))) >= 6
            and str(row["compact"]) in compact
        )
    ]


def _raw_denylist_hits(raw: bytes, denylist: Sequence[Mapping[str, str]]) -> list[str]:
    hits: list[str] = []
    lowered = raw.lower()
    for row in denylist:
        value = str(row["normalized"])
        encodings = (
            value.encode("utf-8", errors="ignore"),
            value.encode("utf-16le", errors="ignore"),
        )
        if any(candidate and candidate.lower() in lowered for candidate in encodings):
            hits.append(str(row["label"]))
    return hits


def _embedded_image_metadata(raw: bytes, *, role: str, member: str) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            entries: dict[str, str] = {}
            for key, value in image.info.items():
                if isinstance(value, bytes):
                    rendered = value.decode("utf-8", errors="replace")
                elif isinstance(value, (str, int, float, bool)):
                    rendered = str(value)
                else:
                    continue
                entries[f"info:{key}"] = rendered
            try:
                for key, value in image.getexif().items():
                    entries[f"exif:{key}"] = str(value)
            except (AttributeError, TypeError, ValueError):
                pass
            return {
                "member": member,
                "format": image.format,
                "metadata_entry_count": len(entries),
                "metadata_sha256": _canonical_hash(entries),
                "_metadata_text": "\n".join(f"{key}={value}" for key, value in entries.items()),
            }
    except Exception as error:
        raise ManuscriptArtifactError(f"{role} embedded image is invalid: {member}") from error


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _publish_create_only(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise ManuscriptArtifactError(f"refusing to overwrite: {destination}") from error
    except OSError as error:
        raise ManuscriptArtifactError(f"create-only publication failed: {destination}") from error


def _publish_transaction(members: Sequence[tuple[Path, Path]]) -> None:
    _require(bool(members), "QA publication transaction is empty")
    destinations = [destination for _, destination in members]
    _require(len(destinations) == len(set(destinations)), "QA outputs are duplicated")
    _require(
        len({path.parent.resolve() for path in destinations}) == 1,
        "QA receipt and upload manifest must share one directory",
    )
    _require(not any(path.exists() for path in destinations), "QA output already exists")
    published: list[Path] = []
    try:
        for source, destination in members:
            _publish_create_only(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise


def _figure_png_closure(summary_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == FIGURE_SCHEMA
        and payload.get("status") == "final_sealed_strict_train399_only"
        and payload.get("submission_use_allowed") is True,
        "publication figure summary is not final",
    )
    base = summary_path.parent.resolve()
    roles = (("figures", "figure_bundle_sha256", 6), ("supplementary_figures", "supplementary_figure_bundle_sha256", 9))
    records: dict[str, Any] = {}
    for record_field, hash_field, expected_count in roles:
        figures = payload.get(record_field)
        hashes = payload.get(hash_field)
        _require(
            isinstance(figures, Mapping)
            and isinstance(hashes, Mapping)
            and len(figures) == len(hashes) == expected_count
            and list(figures) == list(hashes),
            f"{record_field} is not an ordered {expected_count}-plate closure",
        )
        role_records = []
        for stem, record in figures.items():
            _require(isinstance(record, Mapping), f"{record_field}/{stem} is invalid")
            bundle = record.get("bundle")
            files = bundle.get("files") if isinstance(bundle, Mapping) else None
            raw_path = files.get("png") if isinstance(files, Mapping) else None
            _require(isinstance(raw_path, str) and raw_path, f"{record_field}/{stem} PNG is absent")
            path = Path(raw_path)
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            _require(path.is_file() and not path.is_symlink(), f"{record_field}/{stem} PNG is absent")
            expected_sha = hashes[stem].get("png") if isinstance(hashes[stem], Mapping) else None
            observed_sha = sha256_file(path)
            _require(observed_sha == expected_sha, f"{record_field}/{stem} PNG hash mismatch")
            role_records.append({"stem": stem, "sha256": observed_sha, "bytes": path.stat().st_size})
        records[record_field] = role_records
    return records


def _inspect_docx(
    path: Path,
    *,
    expected_media: int,
    role: str,
    reviewer_visible: bool,
    denylist: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    raw = _read_bytes(path, role)
    _require(raw.startswith(b"PK\x03\x04"), f"{role} lacks OOXML ZIP magic")
    _require(zipfile.is_zipfile(path), f"{role} is not a valid ZIP archive")
    try:
        with zipfile.ZipFile(path) as archive:
            _require(archive.testzip() is None, f"{role} contains a corrupt ZIP member")
            member_names = archive.namelist()
            names = set(member_names)
            _require(len(member_names) == len(names), f"{role} contains duplicate ZIP members")
            for name in member_names:
                _require(
                    name == name.replace("\\", "/")
                    and not name.startswith("/")
                    and ".." not in Path(name).parts,
                    f"{role} contains a non-canonical ZIP member: {name}",
                )
            missing = sorted(REQUIRED_PARTS - names)
            _require(not missing, f"{role} lacks required OOXML parts: {missing}")
            document_raw = archive.read("word/document.xml")
            styles_raw = archive.read("word/styles.xml")
            relationships_raw = archive.read("word/_rels/document.xml.rels")
            content_types_raw = archive.read("[Content_Types].xml")
            parsed_xml: dict[str, ET.Element] = {}
            search_fragments: list[str] = [path.name, "\n".join(member_names)]
            for part_name in sorted(
                name
                for name in names
                if name.endswith((".xml", ".rels"))
                or name == "[Content_Types].xml"
            ):
                part_raw = archive.read(part_name)
                try:
                    parsed = ET.fromstring(part_raw)
                except ET.ParseError as error:
                    raise ManuscriptArtifactError(f"{role} {part_name} XML is invalid") from error
                parsed_xml[part_name] = parsed
                search_fragments.extend(value for value in parsed.itertext() if value)
                search_fragments.extend(
                    value
                    for element in parsed.iter()
                    for value in element.attrib.values()
                    if value
                )
            root = parsed_xml["word/document.xml"]
            _require(root.tag == f"{{{W_NS}}}document", f"{role} has the wrong document root")
            text = "\n".join(node.text or "" for node in root.findall(f".//{{{W_NS}}}t"))
            paragraph_texts = [
                _normalized_search_text(
                    "".join(
                        node.text or ""
                        for node in paragraph.findall(f".//{{{W_NS}}}t")
                    )
                )
                for paragraph in root.findall(f".//{{{W_NS}}}p")
            ]
            declaration_heading_hits = sorted(
                label
                for label in EDITOR_ONLY_DECLARATIONS
                if _normalized_search_text(label)
                in {
                    re.sub(r"^\d+\.\s*", "", value)
                    for value in paragraph_texts
                }
            )
            _require(PLACEHOLDER.search(text) is None, f"{role} contains unresolved machine tokens")
            _require("PROVISIONAL" not in text.upper(), f"{role} contains a provisional marker")
            media = sorted(name for name in names if name.startswith("word/media/") and not name.endswith("/"))
            image_placements = root.findall(f".//{{{A_NS}}}blip")
            _require(
                len(image_placements) == expected_media,
                f"{role} places {len(image_placements)} images, expected {expected_media}",
            )
            relationships = parsed_xml["word/_rels/document.xml.rels"]
            image_relationships = [
                row
                for row in list(relationships)
                if str(row.attrib.get("Type", "")).endswith("/image")
            ]
            _require(
                len(image_relationships) == len(media),
                f"{role} image relationship/media-part closure changed",
            )
            image_relationship_ids = {
                str(row.attrib.get("Id", "")) for row in image_relationships
            }
            placement_relationship_ids = [
                str(blip.attrib.get(f"{{{R_NS}}}embed", ""))
                for blip in image_placements
            ]
            _require(
                all(placement_relationship_ids)
                and set(placement_relationship_ids).issubset(image_relationship_ids),
                f"{role} image placements are not internally relationship-bound",
            )
            relationship_media = {
                posixpath.normpath(
                    posixpath.join("word", str(row.attrib.get("Target", "")))
                )
                for row in image_relationships
                if row.attrib.get("TargetMode") != "External"
            }
            _require(
                relationship_media == set(media),
                f"{role} image relationships do not close over embedded media",
            )
            _require(root.find(f".//{{{W_NS}}}sectPr") is not None, f"{role} has no section geometry")
            external_targets: list[str] = []
            for name, parsed in parsed_xml.items():
                if not name.endswith(".rels"):
                    continue
                for relationship in list(parsed):
                    if relationship.attrib.get("TargetMode") == "External":
                        target = str(relationship.attrib.get("Target", ""))
                        _require(bool(target), f"{role} has an empty external relationship")
                        _require(
                            not target.casefold().startswith(("file:", "mailto:"))
                            and not target.startswith(("\\\\", "//")),
                            f"{role} has an identity- or host-bearing external relationship",
                        )
                        external_targets.append(target)

            embedded_images = sorted(
                name
                for name in names
                if not name.endswith("/")
                and Path(name).suffix.casefold()
                in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"}
            )
            image_metadata: list[dict[str, Any]] = []
            for member in embedded_images:
                member_raw = archive.read(member)
                record = _embedded_image_metadata(member_raw, role=role, member=member)
                metadata_text = str(record.pop("_metadata_text"))
                metadata_hits = sorted(
                    set(
                        _denylist_hits(metadata_text, denylist)
                        + _raw_denylist_hits(member_raw, denylist)
                    )
                )
                record["identity_denylist_hits"] = metadata_hits
                image_metadata.append(record)

            package_text = "\n".join(search_fragments + external_targets)
            identity_hits = sorted(set(_denylist_hits(package_text, denylist)))
            media_identity_hits = sorted(
                {
                    hit
                    for record in image_metadata
                    for hit in record["identity_denylist_hits"]
                }
            )
            tracked_tags = {
                f"{{{W_NS}}}ins",
                f"{{{W_NS}}}del",
                f"{{{W_NS}}}moveFrom",
                f"{{{W_NS}}}moveTo",
            }
            tracked_change_count = sum(
                element.tag in tracked_tags
                for parsed in parsed_xml.values()
                for element in parsed.iter()
            )
            hidden_text_count = sum(
                element.tag in {f"{{{W_NS}}}vanish", f"{{{W_NS}}}webHidden"}
                for parsed in parsed_xml.values()
                for element in parsed.iter()
            )
            identity_parts = sorted(
                name
                for name in names
                if name.startswith(
                    (
                        "word/comments",
                        "word/people",
                        "word/person",
                        "word/embeddings/",
                        "docProps/custom",
                    )
                )
            )
            core = parsed_xml["docProps/core.xml"]
            creators = [
                _normalized_search_text(element.text or "")
                for element in core.findall(f".//{{{DC_NS}}}creator")
            ]
            modified_by = [
                _normalized_search_text(element.text or "")
                for element in core.findall(f".//{{{CP_NS}}}lastModifiedBy")
            ]
            self_reference_hits = sorted(
                phrase
                for phrase in (
                    "at our institution",
                    "our institution",
                    "our laboratory at",
                    "the authors' institution",
                    "we previously published",
                )
                if phrase in _normalized_search_text(package_text)
            )
            if reviewer_visible:
                _require(not identity_hits, f"{role} exposes identity terms: {identity_hits}")
                _require(not media_identity_hits, f"{role} embedded image metadata exposes identity: {media_identity_hits}")
                _require(
                    not declaration_heading_hits,
                    f"{role} retains editor-only declaration headings: {declaration_heading_hits}",
                )
                _require(not identity_parts, f"{role} retains identity-bearing OOXML parts: {identity_parts}")
                _require(tracked_change_count == 0, f"{role} retains tracked changes")
                _require(hidden_text_count == 0, f"{role} retains hidden text")
                _require(not any(creators) and not any(modified_by), f"{role} core creator metadata is not empty")
                _require(not self_reference_hits, f"{role} retains self-identifying prose: {self_reference_hits}")
            return {
                "sha256": _sha256(raw),
                "bytes": len(raw),
                "zip_magic": "PK0304",
                "zip_crc_passed": True,
                "required_ooxml_parts_present": True,
                "embedded_media_count": len(media),
                "image_placement_count": len(image_placements),
                "embedded_image_part_count": len(embedded_images),
                "image_relationship_count": len(image_relationships),
                "section_count": len(root.findall(f".//{{{W_NS}}}sectPr")),
                "table_count": len(root.findall(f".//{{{W_NS}}}tbl")),
                "reviewer_visible": reviewer_visible,
                "xml_and_relationship_part_count_scanned": len(parsed_xml),
                "external_relationship_targets": external_targets,
                "embedded_image_metadata": image_metadata,
                "identity_denylist_hit_count": len(identity_hits) + len(media_identity_hits),
                "identity_denylist_hits": sorted(set(identity_hits + media_identity_hits)),
                "editor_only_declaration_heading_hits": declaration_heading_hits,
                "identity_bearing_part_count": len(identity_parts),
                "tracked_change_count": tracked_change_count,
                "hidden_text_count": hidden_text_count,
                "core_creator_values": creators,
                "core_last_modified_by_values": modified_by,
                "self_reference_hits": self_reference_hits,
                "_normalized_visible_text": _normalized_search_text(text),
                "_normalized_visible_paragraphs": paragraph_texts,
                "_normalized_package_text": _normalized_search_text(package_text),
            }
    except zipfile.BadZipFile as error:
        raise ManuscriptArtifactError(f"{role} is not valid OOXML") from error


def _validate_title_page_completeness(
    record: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    declarations: Mapping[str, str],
) -> None:
    paragraphs = record.get("_normalized_visible_paragraphs")
    _require(isinstance(paragraphs, list), "title-page visible paragraph inventory is absent")
    declaration_indices: dict[str, int] = {}
    for label in EDITOR_ONLY_DECLARATIONS:
        normalized_label = _normalized_search_text(label)
        matches = [
            index
            for index, paragraph in enumerate(paragraphs)
            if re.sub(r"^\d+\.\s*", "", str(paragraph)) == normalized_label
        ]
        _require(
            len(matches) == 1,
            f"editor-only title page declaration heading is absent or duplicated: {label}",
        )
        declaration_indices[label] = matches[0]
    ordered_indices = [declaration_indices[label] for label in EDITOR_ONLY_DECLARATIONS]
    _require(
        ordered_indices == sorted(ordered_indices),
        "editor-only title page declaration ordering changed",
    )
    front_text = _normalized_search_text(
        "\n".join(str(value) for value in paragraphs[: min(ordered_indices)])
    )
    required_front: list[tuple[str, str]] = []
    for index, author in enumerate(metadata.get("authors", []), start=1):
        _require(isinstance(author, Mapping), f"title metadata author {index} is invalid")
        for field in ("full_name", "email", "orcid"):
            value = author.get(field)
            if isinstance(value, str) and value.strip():
                required_front.append((f"author_{index}_{field}", _normalized_search_text(value)))
        if author.get("corresponding_author") is True:
            value = author.get("postal_address")
            _require(isinstance(value, str) and value.strip(), "corresponding postal address is absent")
            _require(
                author.get("postal_address_author_verified") is True,
                "corresponding postal address is not author-verified",
            )
            required_front.append((f"author_{index}_postal_address", _normalized_search_text(value)))
    for affiliation in metadata.get("affiliations", []):
        _require(isinstance(affiliation, Mapping), "title metadata affiliation is invalid")
        required_front.append(
            (
                f"affiliation_{affiliation.get('id')}",
                _normalized_search_text(affiliation.get("text", "")),
            )
        )
    missing = [
        label for label, value in required_front if value and value not in front_text
    ]
    for index, label in enumerate(EDITOR_ONLY_DECLARATIONS):
        start = declaration_indices[label] + 1
        end = (
            declaration_indices[EDITOR_ONLY_DECLARATIONS[index + 1]]
            if index + 1 < len(EDITOR_ONLY_DECLARATIONS)
            else len(paragraphs)
        )
        declaration_text = _normalized_search_text(
            "\n".join(str(value) for value in paragraphs[start:end])
        )
        expected = _markdown_visible_text(declarations[label])
        if expected and expected not in declaration_text:
            missing.append(f"declaration_body_{label}")
    _require(not missing, f"editor-only title page is incomplete: {missing}")


def _validate_supplement_ooxml_table_count(
    receipt: Mapping[str, Any],
    ooxml: Mapping[str, Any],
) -> None:
    _require(
        receipt.get("embedded_markdown_table_count") == ooxml.get("table_count"),
        "supplement DOCX declared table count differs from OOXML",
    )


def verify_manuscript_artifacts(
    *,
    main_master: str | Path,
    supplement_master: str | Path,
    main_manuscript: str | Path,
    main_compile_receipt: str | Path,
    supplement_manuscript: str | Path,
    supplement_compile_receipt: str | Path,
    submission_metadata: str | Path,
    figure_summary: str | Path,
    title_page_docx: str | Path,
    anonymized_main_docx: str | Path,
    submission_docx_receipt: str | Path,
    anonymized_supplement_docx: str | Path,
    supplement_docx_receipt: str | Path,
    output: str | Path,
    upload_manifest: str | Path,
) -> dict[str, Any]:
    paths = {
        name: Path(value).resolve()
        for name, value in locals().items()
        if name not in {"output", "upload_manifest"}
    }
    output_path = Path(output).resolve()
    upload_manifest_path = Path(upload_manifest).resolve()
    _require(output_path != upload_manifest_path, "QA receipt and upload manifest paths collide")
    _require(
        output_path.parent == upload_manifest_path.parent,
        "QA receipt and upload manifest must share one directory",
    )
    _require(
        not output_path.exists() and not upload_manifest_path.exists(),
        "refusing to overwrite QA receipt or upload manifest",
    )

    main_master_raw, _ = _read_text(paths["main_master"], "main master")
    supplement_master_raw, _ = _read_text(paths["supplement_master"], "supplement master")
    main_raw, main_text = _read_text(paths["main_manuscript"], "compiled main manuscript")
    supplement_raw, supplement_text = _read_text(paths["supplement_manuscript"], "compiled supplement")
    main_compile_raw, main_compile = _read_json(paths["main_compile_receipt"], "main compile receipt")
    supplement_compile_raw, supplement_compile = _read_json(paths["supplement_compile_receipt"], "supplement compile receipt")
    metadata_raw, metadata = _read_json(paths["submission_metadata"], "submission metadata")
    figure_raw, figures = _read_json(paths["figure_summary"], "figure summary")
    main_docx_receipt_raw, main_docx_receipt_payload = _read_json(paths["submission_docx_receipt"], "submission DOCX receipt")
    supplement_docx_receipt_raw, supplement_docx_receipt_payload = _read_json(paths["supplement_docx_receipt"], "supplement DOCX receipt")

    for payload, field, role in (
        (main_compile, "receipt_identity_sha256", "main compile receipt"),
        (supplement_compile, "receipt_identity_sha256", "supplement compile receipt"),
        (metadata, "metadata_identity_sha256", "submission metadata"),
        (main_docx_receipt_payload, "receipt_identity_sha256", "main DOCX receipt"),
        (supplement_docx_receipt_payload, "receipt_identity_sha256", "supplement DOCX receipt"),
    ):
        _verify_seal(payload, field, role)
    _require(
        metadata.get("schema_version") == "PHAxis-submission-title-metadata-2.0"
        and metadata.get("status") == "complete_author_verified_submission_metadata",
        "submission metadata is not author-verified v2",
    )
    _require(
        main_compile.get("schema_version") == MAIN_COMPILE_SCHEMA
        and main_compile.get("status") == "completed_strict_final_manuscript_compilation",
        "main compile receipt is not final",
    )
    _require(
        supplement_compile.get("schema_version") == SUPPLEMENT_COMPILE_SCHEMA
        and supplement_compile.get("status") == "completed_strict_final_supplementary_compilation",
        "supplement compile receipt is not final",
    )
    _require(main_compile.get("master_sha256") == _sha256(main_master_raw), "main master authority mismatch")
    _require(main_compile.get("output_sha256") == _sha256(main_raw), "main Markdown authority mismatch")
    _require(supplement_compile.get("master_sha256") == _sha256(supplement_master_raw), "supplement master authority mismatch")
    _require(supplement_compile.get("output_sha256") == _sha256(supplement_raw), "supplement Markdown authority mismatch")
    _require(supplement_compile.get("main_compile_receipt_sha256") == _sha256(main_compile_raw), "supplement does not bind the main compile receipt")
    _require(PLACEHOLDER.search(main_text) is None and PLACEHOLDER.search(supplement_text) is None, "compiled Markdown retains machine tokens")
    declarations = _editor_only_declarations(main_text)
    availability = _availability_contract(main_text)
    denylist = _identity_denylist(metadata, declarations)
    figure_closure = _figure_png_closure(paths["figure_summary"], figures)
    table_receipt_relative = figures.get("supplementary_table_bundle_receipt")
    _require(
        isinstance(table_receipt_relative, str) and bool(table_receipt_relative),
        "figure summary omits supplementary Table/Data receipt",
    )
    table_receipt_path = (
        paths["figure_summary"].parent / table_receipt_relative
    ).resolve()
    _require(
        table_receipt_path.is_relative_to(paths["figure_summary"].parent),
        "supplementary Table/Data receipt escapes figure suite",
    )
    try:
        table_bundle = validate_supplementary_table_data_bundle(
            table_receipt_path, require_final=True
        )
    except SupplementaryTableError as error:
        raise ManuscriptArtifactError(
            f"supplementary Table/Data S1--S10 validation failed: {error}"
        ) from error
    table_identity_map = {
        stem: record["item_identity_sha256"]
        for stem, record in table_bundle["items"].items()
    }
    _require(
        figures.get("supplementary_tables") == table_bundle["items"]
        and figures.get("supplementary_table_bundle_receipt_sha256")
        == table_bundle["receipt_sha256"]
        and figures.get("supplementary_table_bundle_identity_sha256")
        == table_bundle["bundle_identity_sha256"]
        and figures.get("supplementary_table_bundle_sha256")
        == table_bundle["bundle_file_sha256"]
        and supplement_compile.get("supplementary_table_data_resource_count")
        == 10
        and supplement_compile.get("supplementary_table_data_materialized")
        is True
        and supplement_compile.get("supplementary_table_bundle_receipt_sha256")
        == table_bundle["receipt_sha256"]
        and supplement_compile.get("supplementary_table_bundle_identity_sha256")
        == table_bundle["bundle_identity_sha256"]
        and supplement_compile.get("supplementary_table_item_identity_sha256")
        == table_identity_map,
        "supplement compiler does not bind exact S1--S10 table/data bundle",
    )

    expected_receipts = (
        (
            main_docx_receipt_payload,
            MAIN_DOCX_SCHEMA,
            "completed_final_double_anonymous_submission_bundle",
            _sha256(main_raw),
            "manuscript_sha256",
        ),
        (
            supplement_docx_receipt_payload,
            SUPPLEMENT_DOCX_SCHEMA,
            "completed_final_anonymized_supplementary_docx",
            _sha256(supplement_raw),
            "supplement_sha256",
        ),
    )
    for payload, schema, status, source_sha, source_field in expected_receipts:
        _require(
            payload.get("schema_version") == schema
            and payload.get("status") == status
            and payload.get("mode") == "final"
            and payload.get("submission_use_allowed") is True,
            f"{schema} is not final",
        )
        _require(payload.get(source_field) == source_sha, f"{schema} source hash mismatch")
        _require(payload.get("figure_summary_sha256") == _sha256(figure_raw), f"{schema} figure-summary hash mismatch")
        _require(payload.get("blind_images_used") == 0, f"{schema} used blind images")
    _require(
        main_docx_receipt_payload.get("submission_metadata_sha256")
        == _sha256(metadata_raw),
        "submission DOCX metadata hash mismatch",
    )
    _require(
        supplement_docx_receipt_payload.get("submission_metadata_consumed") is False
        and "submission_metadata_sha256" not in supplement_docx_receipt_payload
        and "submission_metadata_identity_sha256" not in supplement_docx_receipt_payload,
        "anonymous supplement consumed or retained submission metadata",
    )
    _require(
        main_docx_receipt_payload.get("compile_receipt_sha256") == _sha256(main_compile_raw)
        and supplement_docx_receipt_payload.get("main_compile_receipt_sha256") == _sha256(main_compile_raw),
        "DOCX receipts do not bind the main compile receipt",
    )
    _require(
        supplement_docx_receipt_payload.get("main_manuscript_sha256") == _sha256(main_raw),
        "supplement DOCX does not bind the compiled main manuscript",
    )
    _require(
        supplement_docx_receipt_payload.get(
            "supplementary_table_data_resource_count"
        )
        == 10
        and supplement_docx_receipt_payload.get(
            "supplementary_table_data_materialized"
        )
        is True
        and supplement_docx_receipt_payload.get(
            "supplementary_table_bundle_receipt_sha256"
        )
        == table_bundle["receipt_sha256"]
        and supplement_docx_receipt_payload.get(
            "supplementary_table_bundle_identity_sha256"
        )
        == table_bundle["bundle_identity_sha256"]
        and supplement_docx_receipt_payload.get(
            "supplementary_table_item_identity_sha256"
        )
        == table_identity_map,
        "supplement DOCX does not bind exact S1--S10 table/data bundle",
    )

    title_ooxml = _inspect_docx(
        paths["title_page_docx"],
        expected_media=0,
        role="editor-only title-page DOCX",
        reviewer_visible=False,
        denylist=denylist,
    )
    main_ooxml = _inspect_docx(
        paths["anonymized_main_docx"],
        expected_media=6,
        role="anonymized main DOCX",
        reviewer_visible=True,
        denylist=denylist,
    )
    supplement_ooxml = _inspect_docx(
        paths["anonymized_supplement_docx"],
        expected_media=9,
        role="anonymized supplement DOCX",
        reviewer_visible=True,
        denylist=denylist,
    )
    _validate_title_page_completeness(
        title_ooxml,
        metadata=metadata,
        declarations=declarations,
    )
    for title in ("Data Availability", "Code Availability"):
        expected = _markdown_visible_text(_h2_section(main_text, title))
        _require(
            expected in str(main_ooxml.get("_normalized_visible_text", "")),
            f"anonymized main DOCX does not preserve {title}",
        )
    _validate_supplement_ooxml_table_count(
        supplement_docx_receipt_payload, supplement_ooxml
    )
    _require(
        main_docx_receipt_payload.get("title_page_docx_sha256")
        == title_ooxml["sha256"]
        and main_docx_receipt_payload.get("anonymized_main_docx_sha256")
        == main_ooxml["sha256"]
        and main_docx_receipt_payload.get("title_page_separate") is True
        and main_docx_receipt_payload.get("anonymized_main_separate") is True,
        "submission DOCX receipt does not bind both role-separated documents",
    )
    _require(supplement_docx_receipt_payload.get("docx_sha256") == supplement_ooxml["sha256"], "supplement DOCX receipt hash mismatch")

    title_ooxml.pop("_normalized_package_text", None)
    main_ooxml.pop("_normalized_package_text", None)
    supplement_ooxml.pop("_normalized_package_text", None)
    title_ooxml.pop("_normalized_visible_text", None)
    main_ooxml.pop("_normalized_visible_text", None)
    supplement_ooxml.pop("_normalized_visible_text", None)
    title_ooxml.pop("_normalized_visible_paragraphs", None)
    main_ooxml.pop("_normalized_visible_paragraphs", None)
    supplement_ooxml.pop("_normalized_visible_paragraphs", None)

    upload_payload: dict[str, Any] = {
        "schema_version": UPLOAD_MANIFEST_SCHEMA,
        "status": UPLOAD_MANIFEST_STATUS,
        "target_journal": "Plant Phenomics",
        "submission_model": "double_anonymous",
        "roles": {
            "editor_only": {
                "title_page": {
                    "filename": paths["title_page_docx"].name,
                    "sha256": title_ooxml["sha256"],
                }
            },
            "reviewer_visible": {
                "anonymized_main": {
                    "filename": paths["anonymized_main_docx"].name,
                    "sha256": main_ooxml["sha256"],
                },
                "anonymized_supplement": {
                    "filename": paths["anonymized_supplement_docx"].name,
                    "sha256": supplement_ooxml["sha256"],
                },
            },
        },
        "excluded_from_upload_roles": {
            "compiled_markdown": [
                {"role": "main", "sha256": _sha256(main_raw)},
                {"role": "supplement", "sha256": _sha256(supplement_raw)},
            ],
            "human_authority_json": [
                {"role": "submission_metadata", "sha256": _sha256(metadata_raw)}
            ],
            "receipts": [
                {"role": "submission_docx", "sha256": _sha256(main_docx_receipt_raw)},
                {"role": "supplement_docx", "sha256": _sha256(supplement_docx_receipt_raw)},
            ],
        },
        "editor_only_document_count": 1,
        "reviewer_visible_document_count": 2,
        "reviewer_visible_identity_occurrence_count": 0,
        "reviewer_visible_ooxml_deep_scan_passed": True,
        "blind_images_used": 0,
    }
    upload_payload["upload_manifest_identity_sha256"] = _canonical_hash(upload_payload)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "main_master_sha256": _sha256(main_master_raw),
        "supplement_master_sha256": _sha256(supplement_master_raw),
        "main_compile_receipt_sha256": _sha256(main_compile_raw),
        "main_compile_receipt_identity_sha256": main_compile["receipt_identity_sha256"],
        "supplement_compile_receipt_sha256": _sha256(supplement_compile_raw),
        "supplement_compile_receipt_identity_sha256": supplement_compile["receipt_identity_sha256"],
        "submission_metadata_sha256": _sha256(metadata_raw),
        "submission_metadata_identity_sha256": metadata["metadata_identity_sha256"],
        "figure_summary_sha256": _sha256(figure_raw),
        "figure_suite_identity_sha256": figures.get("figure_suite_identity_sha256"),
        "figure_png_closure": figure_closure,
        "supplementary_table_data_closure": {
            "ordered_item_count": 10,
            "bundle_receipt_sha256": table_bundle["receipt_sha256"],
            "bundle_identity_sha256": table_bundle["bundle_identity_sha256"],
            "source_authority_sha256": table_bundle[
                "source_authority_sha256"
            ],
            "ordered_item_identity_sha256": table_identity_map,
        },
        "availability_statement_closure": availability,
        "data_and_code_availability_present": True,
        "submission_docx_build_receipt_sha256": _sha256(main_docx_receipt_raw),
        "submission_docx_build_receipt_identity_sha256": main_docx_receipt_payload["receipt_identity_sha256"],
        "supplement_docx_build_receipt_sha256": _sha256(supplement_docx_receipt_raw),
        "supplement_docx_build_receipt_identity_sha256": supplement_docx_receipt_payload["receipt_identity_sha256"],
        "title_page_ooxml": title_ooxml,
        "main_ooxml": main_ooxml,
        "supplement_ooxml": supplement_ooxml,
        "document_roles": {
            "editor_only": ["title_page"],
            "reviewer_visible": ["anonymized_main", "anonymized_supplement"],
        },
        "reviewer_visible_identity_occurrence_count": 0,
        "reviewer_visible_core_identity_occurrence_count": 0,
        "reviewer_visible_tracked_change_count": 0,
        "reviewer_visible_hidden_text_count": 0,
        "reviewer_visible_embedded_image_identity_occurrence_count": 0,
        "deep_ooxml_anonymity_scan_passed": True,
        "editor_only_title_page_completeness_passed": True,
        "submission_upload_role_manifest_sha256": None,
        "submission_upload_role_manifest_identity_sha256": upload_payload[
            "upload_manifest_identity_sha256"
        ],
        "ooxml_zip_magic_and_required_structure_passed": True,
        "master_authority_closure_passed": True,
        "figure_input_closure_passed": True,
        "supplementary_table_data_closure_passed": True,
        "submission_use_allowed_before_visual_qa": False,
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".manuscript-qa-", dir=output_path.parent))
    try:
        temporary_upload = staging / "upload_manifest.json"
        _write_json(temporary_upload, upload_payload)
        result["submission_upload_role_manifest_sha256"] = sha256_file(
            temporary_upload
        )
        result["qa_identity_sha256"] = _canonical_hash(result)
        temporary_receipt = staging / "receipt.json"
        _write_json(temporary_receipt, result)
        _publish_transaction(
            (
                (temporary_upload, upload_manifest_path),
                (temporary_receipt, output_path),
            )
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-master", type=Path, required=True)
    parser.add_argument("--supplement-master", type=Path, required=True)
    parser.add_argument("--main-manuscript", type=Path, required=True)
    parser.add_argument("--main-compile-receipt", type=Path, required=True)
    parser.add_argument("--supplement-manuscript", type=Path, required=True)
    parser.add_argument("--supplement-compile-receipt", type=Path, required=True)
    parser.add_argument("--submission-metadata", type=Path, required=True)
    parser.add_argument("--figure-summary", type=Path, required=True)
    parser.add_argument("--title-page-docx", type=Path, required=True)
    parser.add_argument("--anonymized-main-docx", type=Path, required=True)
    parser.add_argument("--submission-docx-receipt", type=Path, required=True)
    parser.add_argument("--anonymized-supplement-docx", type=Path, required=True)
    parser.add_argument("--supplement-docx-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_manuscript_artifacts(**vars(args))
    print(result["qa_identity_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManuscriptArtifactError, OSError, ValueError, TypeError) as error:
        print(f"blocked: {error}", file=sys.stderr)
        raise SystemExit(2)
