#!/usr/bin/env python3
"""Build and verify the PHAxis 1.0.0 wheel and sdist from a sealed source tree."""

from __future__ import annotations

import argparse
import base64
import csv
from copy import deepcopy
from email.policy import default as email_policy
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence
import zipfile


# Distribution construction is allowed to create only its explicit build
# outputs.  Importing release helpers must not mutate the sealed source tree.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-release-distributions-1.0"
SOURCE_SCHEMA = "PHAxis-source-release-manifest-2.0"
PRODUCT = "phaxis"
VERSION = "1.0.0"
SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"
SOURCE_SUPPLY_CHAIN_FILES = (
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES.json",
    "SBOM.cdx.json",
)
REQUIRED_PEP639_LICENSE_FILES = (
    "LICENSE",
    "src/phaxis/_vendor/tomli/LICENSE.txt",
)
RELEASE_ASSET_INVENTORY_NAME = "release_asset_inventory.json"
RELEASE_CHECKSUMS_NAME = "SHA256SUMS"
PEP517_SDIST_GENERATED_MEMBERS = (
    "PKG-INFO",
    "setup.cfg",
    "src/phaxis.egg-info/PKG-INFO",
    "src/phaxis.egg-info/SOURCES.txt",
    "src/phaxis.egg-info/dependency_links.txt",
    "src/phaxis.egg-info/entry_points.txt",
    "src/phaxis.egg-info/requires.txt",
    "src/phaxis.egg-info/top_level.txt",
)
WHEEL_PROHIBITED_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "blind",
    "data",
    "legacy_project",
    "models",
    "predictions",
    "weights",
}
WHEEL_PROHIBITED_SUFFIXES = {
    ".bmp",
    ".ckpt",
    ".engine",
    ".gif",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".webp",
}


class DistributionBuildError(RuntimeError):
    """A source authority, build command, or distribution closure failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionBuildError(message)


def _sealed(payload: Mapping[str, Any], field: str, *, role: str) -> None:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(
        isinstance(observed, str) and observed == sha256_json(unsigned),
        f"{role}: {field} does not seal the complete object",
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    audit_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    record = {
        "argv": list(audit_argv if audit_argv is not None else command),
        "returncode": completed.returncode,
        "stdout_sha256": sha256_json(completed.stdout.splitlines()),
        "stderr_sha256": sha256_json(completed.stderr.splitlines()),
    }
    _require(
        completed.returncode == 0,
        f"distribution command failed ({completed.returncode}): {completed.stderr[-1000:]}",
    )
    return record


def _directory_lock(root: Path) -> dict[str, Any]:
    """Hash every file under ``root`` without publishing host paths."""

    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        _require(not path.is_symlink(), "source release contains a symbolic link")
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "identity_sha256": sha256_json(records),
    }


def _copy_manifest_exact_source(
    source_root: Path,
    source_manifest: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Create the private PEP 517 input from manifest-authorized bytes only."""

    raw_records = source_manifest.get("files")
    _require(isinstance(raw_records, list), "source release manifest files is not a list")
    destination.mkdir(parents=True)
    copied: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in raw_records:
        _require(isinstance(raw_record, Mapping), "source manifest record is invalid")
        relative = str(raw_record.get("path", ""))
        pure = PurePosixPath(relative)
        _require(
            bool(relative)
            and pure.as_posix() == relative
            and not pure.is_absolute()
            and ".." not in pure.parts
            and relative != SOURCE_MANIFEST_NAME
            and relative not in seen,
            f"source manifest path is invalid or duplicated: {relative}",
        )
        seen.add(relative)
        source = source_root.joinpath(*pure.parts)
        target = destination.joinpath(*pure.parts)
        _require(
            source.is_file() and not source.is_symlink(),
            f"manifest-authorized source file is absent/invalid: {relative}",
        )
        expected_bytes = raw_record.get("bytes")
        expected_sha256 = raw_record.get("sha256")
        _require(
            source.stat().st_size == expected_bytes
            and sha256_file(source) == expected_sha256,
            f"manifest-authorized source identity mismatch: {relative}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        _require(
            target.stat().st_size == expected_bytes
            and sha256_file(target) == expected_sha256,
            f"private source-copy identity mismatch: {relative}",
        )
        copied.append(
            {"path": relative, "bytes": expected_bytes, "sha256": expected_sha256}
        )

    manifest_source = source_root / SOURCE_MANIFEST_NAME
    manifest_target = destination / SOURCE_MANIFEST_NAME
    shutil.copy2(manifest_source, manifest_target)
    manifest_record = {
        "path": SOURCE_MANIFEST_NAME,
        "bytes": manifest_target.stat().st_size,
        "sha256": sha256_file(manifest_target),
    }
    copied.append(manifest_record)
    observed = _directory_lock(destination)
    expected_records = sorted(copied, key=lambda row: row["path"])
    _require(
        observed["file_count"] == len(expected_records)
        and observed["identity_sha256"] == sha256_json(expected_records),
        "private build input is not the exact SOURCE_MANIFEST-authorized copy",
    )
    return {
        "role": "private_manifest_exact_source_copy",
        "source_manifest_sha256": manifest_record["sha256"],
        "file_count_including_manifest": len(expected_records),
        "tree_identity_sha256": sha256_json(expected_records),
        "manifest_exact_copy_verified": True,
    }


def _source_supply_chain_records(
    source_root: Path,
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the PHAxis-authored notice/license/SBOM source closure."""

    records = source_manifest.get("files")
    _require(isinstance(records, list), "source release manifest files is not a list")
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, Mapping)
    }
    _require(
        all(relative in by_path for relative in SOURCE_SUPPLY_CHAIN_FILES),
        "source release omits PHAxis supply-chain metadata",
    )
    validated: list[dict[str, Any]] = []
    for relative in SOURCE_SUPPLY_CHAIN_FILES:
        path = source_root / relative
        record = by_path[relative]
        _require(path.is_file() and not path.is_symlink(), f"source supply-chain file is absent/invalid: {relative}")
        _require(
            path.stat().st_size == record.get("bytes")
            and sha256_file(path) == record.get("sha256"),
            f"source supply-chain identity mismatch: {relative}",
        )
        validated.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    try:
        sbom = json.loads((source_root / "SBOM.cdx.json").read_text(encoding="utf-8"))
        inventory = json.loads(
            (source_root / "THIRD_PARTY_LICENSES.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DistributionBuildError(f"source supply-chain JSON is invalid: {error}") from error
    unsigned_inventory = dict(inventory) if isinstance(inventory, Mapping) else {}
    inventory_identity = unsigned_inventory.pop("inventory_identity_sha256", None)
    sbom_metadata = sbom.get("metadata") if isinstance(sbom, Mapping) else None
    sbom_component = (
        sbom_metadata.get("component")
        if isinstance(sbom_metadata, Mapping)
        else None
    )
    _require(
        isinstance(sbom, Mapping)
        and sbom.get("bomFormat") == "CycloneDX"
        and sbom.get("specVersion") == "1.6"
        and isinstance(sbom_component, Mapping)
        and sbom_component.get("name") == PRODUCT
        and sbom_component.get("version") == VERSION
        and isinstance(sbom.get("components"), list)
        and bool(sbom["components"]),
        "source SBOM is not the PHAxis 1.0.0 CycloneDX authority",
    )
    _require(
        isinstance(inventory, Mapping)
        and inventory.get("schema_version")
        == "PHAxis-third-party-license-inventory-1.0"
        and inventory.get("product") == "PHAxis"
        and inventory.get("product_version") == VERSION
        and isinstance(inventory_identity, str)
        and inventory_identity == sha256_json(unsigned_inventory),
        "source third-party license inventory is not sealed",
    )
    notice_text = (source_root / "NOTICE").read_text(encoding="utf-8")
    third_party_text = (source_root / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    _require(
        "PHAxis 1.0.0" in notice_text
        and "PHAxis 1.0.0 third-party notices" in third_party_text
        and "RHPheno" not in notice_text
        and "RHPheno" not in third_party_text,
        "source notice metadata is legacy or does not identify PHAxis 1.0.0",
    )
    return validated


def _safe_wheel_member(value: str) -> PurePosixPath:
    _require(
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not value.startswith("/"),
        f"unsafe wheel member path: {value!r}",
    )
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == value,
        f"unsafe wheel member path: {value!r}",
    )
    return relative


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(
        "ascii"
    )


def _audit_wheel(
    wheel: Path,
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact package code, metadata, entry point, and RECORD closure."""

    expected_filename = f"{PRODUCT}-{VERSION}-py3-none-any.whl"
    _require(
        wheel.name == expected_filename,
        f"wheel filename is not the canonical pure-Python artifact: {wheel.name}",
    )
    records = source_manifest.get("files")
    _require(isinstance(records, list), "source release manifest files is not a list")
    expected_code: dict[str, tuple[str, int]] = {}
    source_file_identities: dict[str, tuple[str, int]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "source release manifest record is invalid")
        relative = record.get("path")
        if isinstance(relative, str):
            identity = (str(record.get("sha256")), int(record.get("bytes")))
            source_file_identities[relative] = identity
            if relative.startswith("src/phaxis/"):
                expected_code[relative.removeprefix("src/")] = identity
    _require(bool(expected_code), "source release manifest contains no PHAxis package code")
    _require(
        all(relative in source_file_identities for relative in REQUIRED_PEP639_LICENSE_FILES),
        "source release manifest omits a required PEP 639 license file",
    )

    try:
        with zipfile.ZipFile(wheel, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _require(bool(names), "wheel archive is empty")
            _require(
                len(names) == len(set(names)),
                "wheel archive contains duplicate member paths",
            )
            _require(
                all(not info.is_dir() for info in infos),
                "wheel archive contains explicit directory members",
            )
            for info in infos:
                relative = _safe_wheel_member(info.filename)
                unix_mode = (info.external_attr >> 16) & 0o170000
                _require(
                    unix_mode != stat.S_IFLNK,
                    f"wheel archive contains a symlink: {info.filename}",
                )
                folded_parts = {part.casefold() for part in relative.parts}
                _require(
                    not WHEEL_PROHIBITED_PATH_PARTS.intersection(folded_parts),
                    f"wheel archive contains a prohibited payload path: {info.filename}",
                )
                _require(
                    relative.suffix.casefold() not in WHEEL_PROHIBITED_SUFFIXES,
                    f"wheel archive contains a prohibited payload suffix: {info.filename}",
                )

            dist_info = f"{PRODUCT}-{VERSION}.dist-info/"
            metadata_name = dist_info + "METADATA"
            wheel_metadata_name = dist_info + "WHEEL"
            entry_points_name = dist_info + "entry_points.txt"
            record_name = dist_info + "RECORD"
            for required in (
                metadata_name,
                wheel_metadata_name,
                entry_points_name,
                record_name,
            ):
                _require(required in names, f"wheel archive lacks required member: {required}")
            other_metadata = [
                name
                for name in names
                if name.endswith(".dist-info/METADATA") and name != metadata_name
            ]
            _require(not other_metadata, "wheel archive contains a second dist-info authority")

            metadata_bytes = archive.read(metadata_name)
            metadata = BytesParser(policy=email_policy).parsebytes(metadata_bytes)
            _require(
                metadata.get("Metadata-Version") == "2.4"
                and str(metadata.get("Name", "")).casefold() == PRODUCT
                and metadata.get("Version") == VERSION
                and metadata.get("Summary")
                == "Reproducible Arabidopsis primary-root and root-hair phenotyping"
                and metadata.get("Requires-Python") == ">=3.10"
                and metadata.get("License-Expression") == "Apache-2.0",
                "wheel METADATA is not canonical PHAxis 1.0.0 metadata",
            )
            license_file_headers = metadata.get_all("License-File", [])
            _require(
                license_file_headers == list(REQUIRED_PEP639_LICENSE_FILES),
                "wheel METADATA License-File headers are not the exact PHAxis and vendored Tomli set",
            )
            expected_dist_info_licenses = {
                dist_info + "licenses/" + relative: source_file_identities[relative]
                for relative in REQUIRED_PEP639_LICENSE_FILES
            }
            for name, (expected_sha256, expected_bytes) in expected_dist_info_licenses.items():
                _require(
                    name in names,
                    f"wheel archive lacks required PEP 639 license member: {name}",
                )
                content = archive.read(name)
                _require(
                    hashlib.sha256(content).hexdigest() == expected_sha256
                    and len(content) == expected_bytes,
                    f"wheel PEP 639 license member differs from source authority: {name}",
                )
            observed_dist_info_licenses = {
                name for name in names if name.startswith(dist_info + "licenses/")
            }
            _require(
                observed_dist_info_licenses == set(expected_dist_info_licenses),
                "wheel PEP 639 license member closure is not the exact PHAxis and vendored Tomli set",
            )
            wheel_metadata_bytes = archive.read(wheel_metadata_name)
            wheel_metadata = BytesParser(policy=email_policy).parsebytes(
                wheel_metadata_bytes
            )
            _require(
                wheel_metadata.get("Wheel-Version") == "1.0"
                and str(wheel_metadata.get("Root-Is-Purelib", "")).casefold()
                == "true"
                and "py3-none-any" in wheel_metadata.get_all("Tag", []),
                "wheel WHEEL metadata is not the canonical pure-Python tag",
            )
            entry_points_bytes = archive.read(entry_points_name)
            entry_points = entry_points_bytes.decode("utf-8")
            compact_entry_points = " ".join(entry_points.split())
            _require(
                "[console_scripts]" in entry_points
                and "phaxis = phaxis.cli:main" in compact_entry_points,
                "wheel entry_points.txt lacks the canonical phaxis CLI",
            )

            record_bytes = archive.read(record_name)
            rows = list(csv.reader(record_bytes.decode("utf-8").splitlines()))
            _require(
                all(len(row) == 3 for row in rows),
                "wheel RECORD contains a malformed row",
            )
            record_map = {row[0]: (row[1], row[2]) for row in rows}
            _require(
                len(record_map) == len(rows),
                "wheel RECORD contains duplicate member rows",
            )
            _require(
                set(record_map) == set(names),
                "wheel RECORD member closure differs from the archive",
            )
            for name in names:
                digest, size = record_map[name]
                if name == record_name:
                    _require(
                        digest == "" and size == "",
                        "wheel RECORD self-row must be unhashed",
                    )
                    continue
                data = archive.read(name)
                _require(
                    digest == "sha256=" + _record_digest(data)
                    and size == str(len(data)),
                    f"wheel RECORD digest/size mismatch: {name}",
                )

            observed_code = {
                name: (hashlib.sha256(archive.read(name)).hexdigest(), len(archive.read(name)))
                for name in names
                if name.startswith("phaxis/")
            }
            _require(
                observed_code == expected_code,
                "wheel PHAxis code differs from the exact source-manifest package tree",
            )
            unexpected_payload = sorted(
                name
                for name in names
                if not name.startswith(("phaxis/", dist_info))
            )
            _require(
                not unexpected_payload,
                "wheel archive contains payload outside phaxis and dist-info: "
                + ", ".join(unexpected_payload),
            )
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile) as error:
        raise DistributionBuildError(f"cannot audit wheel archive: {error}") from error

    code_identity_rows = [
        {"path": path, "sha256": digest, "bytes": size}
        for path, (digest, size) in sorted(expected_code.items())
    ]
    return {
        "archive_filename": wheel.name,
        "archive_sha256": sha256_file(wheel),
        "distribution": PRODUCT,
        "version": VERSION,
        "wheel_tag": "py3-none-any",
        "metadata_member": metadata_name,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "entry_points_member": entry_points_name,
        "entry_points_sha256": hashlib.sha256(entry_points_bytes).hexdigest(),
        "entry_point": "phaxis = phaxis.cli:main",
        "record_member": record_name,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "record_member_count": len(record_map),
        "record_verified": True,
        "metadata_license_files": list(REQUIRED_PEP639_LICENSE_FILES),
        "pep639_license_member_count": len(expected_dist_info_licenses),
        "license_file_hashes_verified": True,
        "source_package_file_count": len(expected_code),
        "source_package_identity_sha256": sha256_json(code_identity_rows),
        "source_package_hashes_verified": True,
        "unexpected_payload_members": 0,
        "prohibited_payload_members": 0,
    }


def _build_toolchain_record(python: Path, *, cwd: Path) -> dict[str, Any]:
    """Capture the exact interpreter and packaging-tool versions used to build."""

    probe = (
        "import importlib.metadata as m, json, platform, sys; "
        "print(json.dumps({'implementation': platform.python_implementation(), "
        "'python_version': platform.python_version(), 'python_cache_tag': "
        "sys.implementation.cache_tag, 'packages': {name: m.version(name) for name "
        "in ('build','setuptools','wheel','twine')}}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-B", "-I", "-c", probe],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    _require(
        completed.returncode == 0,
        "cannot inspect the distribution build toolchain: "
        + completed.stderr[-1000:],
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise DistributionBuildError(
            f"distribution build-toolchain probe returned invalid JSON: {error}"
        ) from error
    packages = payload.get("packages") if isinstance(payload, Mapping) else None
    _require(
        isinstance(packages, Mapping)
        and set(packages) == {"build", "setuptools", "wheel", "twine"}
        and all(isinstance(value, str) and bool(value) for value in packages.values()),
        "distribution build-toolchain versions are incomplete",
    )
    record = {
        "implementation": payload.get("implementation"),
        "python_version": payload.get("python_version"),
        "python_cache_tag": payload.get("python_cache_tag"),
        "python_executable_filename": python.name,
        "python_executable_sha256": sha256_file(python),
        "packages": dict(sorted(packages.items())),
        "probe_isolated": True,
        "cuda_visible_devices": "-1",
        "exact_versions_recorded": True,
        "build_isolation_used": False,
    }
    record["build_toolchain_identity_sha256"] = sha256_json(record)
    return record


def _audit_sdist(
    sdist: Path,
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove authored closure plus the exact PEP 517 generated metadata set."""

    records = source_manifest.get("files")
    _require(isinstance(records, list), "source release manifest files is not a list")
    authored_records: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "source release manifest record is invalid")
        relative = record.get("path")
        _require(
            isinstance(relative, str)
            and relative
            and relative != SOURCE_MANIFEST_NAME
            and PurePosixPath(relative).as_posix() == relative
            and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts,
            "source release manifest contains an invalid path",
        )
        _require(relative not in authored_records, f"duplicate source manifest path: {relative}")
        authored_records[relative] = record

    archive_root = f"{PRODUCT}-{VERSION}"
    try:
        with tarfile.open(sdist, mode="r:gz") as archive:
            members = archive.getmembers()
            member_names = [PurePosixPath(member.name).as_posix() for member in members]
            _require(
                len(member_names) == len(set(member_names)),
                "sdist contains duplicate archive member paths",
            )
            file_members: dict[str, tarfile.TarInfo] = {}
            for member, normalized in zip(members, member_names, strict=True):
                path = PurePosixPath(normalized)
                _require(
                    not path.is_absolute()
                    and ".." not in path.parts
                    and len(path.parts) >= 1
                    and path.parts[0] == archive_root,
                    f"sdist member escapes canonical archive root: {normalized}",
                )
                _require(
                    member.isfile() or member.isdir(),
                    f"sdist contains a non-file/non-directory member: {normalized}",
                )
                if not member.isfile():
                    continue
                _require(
                    len(path.parts) > 1,
                    f"sdist file has no relative member path: {normalized}",
                )
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                _require(relative not in file_members, f"duplicate sdist file member: {relative}")
                file_members[relative] = member

            authored_paths = set(authored_records) | {SOURCE_MANIFEST_NAME}
            observed_paths = set(file_members)
            missing_authored = sorted(authored_paths - observed_paths)
            _require(
                not missing_authored,
                "sdist is missing authored source-release members: "
                + ", ".join(missing_authored),
            )
            generated_paths = observed_paths - authored_paths
            allowed_generated = set(PEP517_SDIST_GENERATED_MEMBERS)
            unexpected_generated = sorted(generated_paths - allowed_generated)
            missing_generated = sorted(allowed_generated - generated_paths)
            _require(
                not unexpected_generated,
                "sdist contains unexpected generated members: "
                + ", ".join(unexpected_generated),
            )
            _require(
                not missing_generated,
                "sdist is missing required PEP 517 generated members: "
                + ", ".join(missing_generated),
            )

            source_manifest_bytes = source_manifest_path.read_bytes()
            expected_hashes = {
                relative: str(record.get("sha256"))
                for relative, record in authored_records.items()
            }
            expected_hashes[SOURCE_MANIFEST_NAME] = hashlib.sha256(
                source_manifest_bytes
            ).hexdigest()
            for relative in sorted(authored_paths):
                extracted = archive.extractfile(file_members[relative])
                _require(extracted is not None, f"cannot read sdist member: {relative}")
                observed_sha256 = hashlib.sha256(extracted.read()).hexdigest()
                _require(
                    observed_sha256 == expected_hashes[relative],
                    f"sdist authored member SHA-256 mismatch: {relative}",
                )

            generated_records: list[dict[str, Any]] = []
            for relative in PEP517_SDIST_GENERATED_MEMBERS:
                member = file_members[relative]
                extracted = archive.extractfile(member)
                _require(extracted is not None, f"cannot read generated sdist member: {relative}")
                content = extracted.read()
                generated_records.append(
                    {
                        "path": relative,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    except (OSError, tarfile.TarError) as error:
        raise DistributionBuildError(f"cannot audit sdist archive: {error}") from error

    return {
        "archive_filename": sdist.name,
        "archive_sha256": sha256_file(sdist),
        "canonical_archive_root": archive_root,
        "source_manifest_self_covered": True,
        "source_manifest_member": SOURCE_MANIFEST_NAME,
        "source_manifest_member_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "authored_members_including_source_manifest": len(authored_records) + 1,
        "authored_member_hashes_verified": True,
        "allowed_pep517_generated_members": list(PEP517_SDIST_GENERATED_MEMBERS),
        "observed_pep517_generated_members": generated_records,
        "unexpected_generated_members": 0,
        "unexpected_generated_member_paths": [],
        "missing_allowed_generated_members": 0,
        "missing_allowed_generated_member_paths": [],
        "total_sdist_file_members": len(file_members),
    }


def build_release_distributions(
    *,
    source_release_root: str | Path,
    source_release_manifest: str | Path,
    output: str | Path,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Build both formats, run metadata checks, and atomically publish them."""

    source_root = Path(source_release_root).resolve()
    source_manifest_path = Path(source_release_manifest).resolve()
    destination = Path(output).resolve()
    python = Path(python_executable).resolve()
    _require(source_root.is_dir(), f"source release root is absent: {source_root}")
    _require(
        source_manifest_path == source_root / "SOURCE_MANIFEST.json"
        and source_manifest_path.is_file(),
        "source release manifest must be SOURCE_MANIFEST.json in the source root",
    )
    source_manifest_file_sha256 = sha256_file(source_manifest_path)
    source_manifest = read_json(source_manifest_path)
    _require(
        sha256_file(source_manifest_path) == source_manifest_file_sha256,
        "source release manifest changed while it was being loaded",
    )
    _require(
        source_manifest.get("schema_version") == SOURCE_SCHEMA
        and source_manifest.get("distribution") == PRODUCT
        and source_manifest.get("version") == VERSION
        and source_manifest.get("release_mode") == "formal",
        "source release is not the formal PHAxis 1.0.0 authority",
    )
    records = source_manifest.get("files")
    _require(
        isinstance(records, list)
        and source_manifest.get("tree_identity_sha256") == sha256_json(records),
        "source release manifest tree identity is invalid",
    )
    source_supply_chain = _source_supply_chain_records(source_root, source_manifest)
    _require(python.is_file(), f"build Python is absent: {python}")
    _require(not destination.exists(), f"refusing to overwrite: {destination}")
    _require(
        destination != source_root and source_root not in destination.parents,
        "distribution output must be outside the sealed source release",
    )
    source_before_lock = _directory_lock(source_root)

    attempt_root = destination.parent / f".{destination.name}.attempt-{os.getpid()}"
    _require(
        not attempt_root.exists(),
        f"distribution attempt already exists: {attempt_root}",
    )
    private_source = attempt_root / "private-manifest-exact-source-copy"
    attempt = attempt_root / "publish"
    attempt.mkdir(parents=True)
    try:
        private_copy = _copy_manifest_exact_source(
            source_root,
            source_manifest,
            private_source,
        )
        _require(
            private_copy["source_manifest_sha256"]
            == source_manifest_file_sha256,
            "private source copy contains a different SOURCE_MANIFEST",
        )
        build_toolchain = _build_toolchain_record(python, cwd=private_source)
    except BaseException:
        shutil.rmtree(attempt_root, ignore_errors=True)
        raise
    commands: list[dict[str, Any]] = []
    try:
        commands.append(
            _run(
                [
                    str(python),
                    "-B",
                    "-I",
                    "-m",
                    "build",
                    "--no-isolation",
                    "--wheel",
                    "--sdist",
                    "--outdir",
                    str(attempt),
                    str(private_source),
                ],
                cwd=private_source,
                audit_argv=[
                    "<BUILD_PYTHON>",
                    "-B",
                    "-I",
                    "-m",
                    "build",
                    "--no-isolation",
                    "--wheel",
                    "--sdist",
                    "--outdir",
                    "<PRIVATE_DISTRIBUTION_OUTPUT>",
                    "<PRIVATE_MANIFEST_EXACT_SOURCE_COPY>",
                ],
            )
        )
        built = sorted(
            (path for path in attempt.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
        wheels = [path for path in built if path.suffix == ".whl"]
        sdists = [path for path in built if path.name.endswith(".tar.gz")]
        _require(
            len(wheels) == 1 and len(sdists) == 1 and len(built) == 2,
            "build did not produce exactly one wheel and one sdist",
        )
        expected_prefix = f"{PRODUCT}-{VERSION}"
        _require(
            wheels[0].name == expected_prefix + "-py3-none-any.whl"
            and sdists[0].name == expected_prefix + ".tar.gz",
            "distribution filenames do not identify phaxis 1.0.0",
        )
        wheel_archive_audit = _audit_wheel(
            wheels[0],
            source_manifest=source_manifest,
        )
        sdist_archive_audit = _audit_sdist(
            sdists[0],
            source_manifest_path=private_source / SOURCE_MANIFEST_NAME,
            source_manifest=source_manifest,
        )
        commands.append(
            _run(
                [
                    str(python),
                    "-B",
                    "-I",
                    "-m",
                    "twine",
                    "check",
                    str(wheels[0]),
                    str(sdists[0]),
                ],
                cwd=private_source,
                audit_argv=[
                    "<BUILD_PYTHON>",
                    "-B",
                    "-I",
                    "-m",
                    "twine",
                    "check",
                    "<PRIVATE_WHEEL>",
                    "<PRIVATE_SDIST>",
                ],
            )
        )
        artifacts = [
            {
                "filename": path.name,
                "kind": "wheel" if path.suffix == ".whl" else "sdist",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (wheels[0], sdists[0])
        ]
        release_sbom = attempt / f"{PRODUCT}-{VERSION}.cdx.json"
        release_notices = attempt / f"{PRODUCT}-{VERSION}-THIRD_PARTY_NOTICES.md"
        release_license_inventory = (
            attempt / f"{PRODUCT}-{VERSION}-THIRD_PARTY_LICENSES.json"
        )
        shutil.copyfile(private_source / "SBOM.cdx.json", release_sbom)
        shutil.copyfile(private_source / "THIRD_PARTY_NOTICES.md", release_notices)
        shutil.copyfile(
            private_source / "THIRD_PARTY_LICENSES.json",
            release_license_inventory,
        )
        _require(
            sha256_file(release_sbom)
            == next(
                row["sha256"]
                for row in source_supply_chain
                if row["path"] == "SBOM.cdx.json"
            )
            and sha256_file(release_notices)
            == next(
                row["sha256"]
                for row in source_supply_chain
                if row["path"] == "THIRD_PARTY_NOTICES.md"
            ),
            "copied release supply-chain asset hash changed",
        )
        _require(
            sha256_file(release_license_inventory)
            == next(
                row["sha256"]
                for row in source_supply_chain
                if row["path"] == "THIRD_PARTY_LICENSES.json"
            ),
            "copied release license inventory hash changed",
        )
        release_assets = sorted(
            [
                *artifacts,
                {
                    "filename": release_sbom.name,
                    "kind": "cyclonedx_sbom",
                    "bytes": release_sbom.stat().st_size,
                    "sha256": sha256_file(release_sbom),
                },
                {
                    "filename": release_notices.name,
                    "kind": "third_party_notices",
                    "bytes": release_notices.stat().st_size,
                    "sha256": sha256_file(release_notices),
                },
                {
                    "filename": release_license_inventory.name,
                    "kind": "third_party_license_inventory",
                    "bytes": release_license_inventory.stat().st_size,
                    "sha256": sha256_file(release_license_inventory),
                },
            ],
            key=lambda row: row["filename"],
        )
        checksum_path = attempt / RELEASE_CHECKSUMS_NAME
        checksum_text = "".join(
            f"{row['sha256']}  {row['filename']}\n" for row in release_assets
        )
        checksum_path.write_text(checksum_text, encoding="utf-8", newline="\n")
        release_asset_inventory: dict[str, Any] = {
            "schema_version": "PHAxis-release-asset-inventory-1.0",
            "status": "sealed_release_assets",
            "distribution": PRODUCT,
            "version": VERSION,
            "source_release_manifest_sha256": source_manifest_file_sha256,
            "source_tree_identity_sha256": source_manifest["tree_identity_sha256"],
            "assets": release_assets,
            "asset_count": len(release_assets),
            "checksum_manifest": {
                "filename": RELEASE_CHECKSUMS_NAME,
                "algorithm": "SHA-256",
                "entries": len(release_assets),
                "sha256": sha256_file(checksum_path),
            },
            "source_supply_chain": source_supply_chain,
            "blind_images_used": 0,
        }
        release_asset_inventory["release_asset_inventory_identity_sha256"] = (
            sha256_json(release_asset_inventory)
        )
        inventory_path = attempt / RELEASE_ASSET_INVENTORY_NAME
        atomic_write_json(inventory_path, release_asset_inventory)
        source_after_lock = _directory_lock(source_root)
        _require(
            source_after_lock == source_before_lock,
            "sealed source release changed during distribution construction",
        )
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_wheel_sdist_verified",
            "distribution": PRODUCT,
            "version": VERSION,
            "source_release_manifest_sha256": source_manifest_file_sha256,
            "source_tree_identity_sha256": source_manifest["tree_identity_sha256"],
            "artifacts": artifacts,
            "release_assets": release_assets,
            "release_asset_inventory": {
                "filename": RELEASE_ASSET_INVENTORY_NAME,
                "sha256": sha256_file(inventory_path),
                "identity_sha256": release_asset_inventory[
                    "release_asset_inventory_identity_sha256"
                ],
            },
            "release_checksums": {
                "filename": RELEASE_CHECKSUMS_NAME,
                "algorithm": "SHA-256",
                "entries": len(release_assets),
                "sha256": sha256_file(checksum_path),
            },
            "source_supply_chain": source_supply_chain,
            "build_toolchain": build_toolchain,
            "wheel_archive_audit": wheel_archive_audit,
            "sdist_archive_audit": sdist_archive_audit,
            "commands": commands,
            "private_build_input": private_copy,
            "source_release_input_immutable": True,
            "source_release_before_lock": source_before_lock,
            "source_release_after_lock": source_after_lock,
            "build_isolation_used": False,
            "twine_check_passed": True,
            "canonical_annotations_read": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["distribution_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(attempt / "distribution_receipt.json", receipt)
        os.replace(attempt, destination)
        shutil.rmtree(attempt_root, ignore_errors=True)
        published = read_json(destination / "distribution_receipt.json")
        _sealed(published, "distribution_identity_sha256", role="distribution receipt")
        return published
    except BaseException:
        shutil.rmtree(attempt_root, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release-root", type=Path, required=True)
    parser.add_argument("--source-release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    try:
        receipt = build_release_distributions(
            source_release_root=args.source_release_root,
            source_release_manifest=args.source_release_manifest,
            output=args.output,
            python_executable=args.python,
        )
    except (DistributionBuildError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
