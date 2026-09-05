from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.release_orchestrator import (
    PEP517_SDIST_GENERATED_MEMBERS as ORCHESTRATOR_PEP517_SDIST_GENERATED_MEMBERS,
    _directory_lock as _stage_directory_lock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _script_module(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_sdist(
    path: Path,
    *,
    source_root: Path,
    generated_members: tuple[str, ...],
    tampered_authored: dict[str, bytes] | None = None,
    extra_generated_members: tuple[str, ...] = (),
) -> None:
    tampered = tampered_authored or {}
    prefix = "phaxis-1.0.0"
    with tarfile.open(path, mode="w:gz") as archive:
        for source in sorted(item for item in source_root.rglob("*") if item.is_file()):
            relative = source.relative_to(source_root).as_posix()
            content = tampered.get(relative, source.read_bytes())
            member = tarfile.TarInfo(f"{prefix}/{relative}")
            member.size = len(content)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(content))
        for relative in (*generated_members, *extra_generated_members):
            content = f"generated metadata: {relative}\n".encode("utf-8")
            member = tarfile.TarInfo(f"{prefix}/{relative}")
            member.size = len(content)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(content))


def _wheel_record_digest(content: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(
        b"="
    ).decode("ascii")


def _write_synthetic_wheel(
    path: Path,
    *,
    source_root: Path,
    version: str = "1.0.0",
    entry_point: str = "phaxis = phaxis.cli:main",
    corrupt_record_member: str | None = None,
    extra_members: dict[str, bytes] | None = None,
    omitted_members: tuple[str, ...] = (),
    license_files: tuple[str, ...] = (
        "LICENSE",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
    ),
) -> None:
    dist_info = "phaxis-1.0.0.dist-info"
    members = {
        source.relative_to(source_root / "src").as_posix(): source.read_bytes()
        for source in sorted((source_root / "src/phaxis").rglob("*"))
        if source.is_file()
    }
    license_headers = "".join(f"License-File: {item}\n" for item in license_files)
    members.update(
        {
            f"{dist_info}/METADATA": (
                "Metadata-Version: 2.4\n"
                "Name: phaxis\n"
                f"Version: {version}\n"
                "Summary: Reproducible Arabidopsis primary-root and root-hair phenotyping\n"
                "License-Expression: Apache-2.0\n"
                f"{license_headers}"
                "Requires-Python: >=3.10\n\n"
            ).encode("utf-8"),
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: PHAxis synthetic test\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n\n"
            ).encode("utf-8"),
            f"{dist_info}/entry_points.txt": (
                f"[console_scripts]\n{entry_point}\n"
            ).encode("utf-8"),
            f"{dist_info}/licenses/LICENSE": (source_root / "LICENSE").read_bytes(),
            f"{dist_info}/licenses/src/phaxis/_vendor/tomli/LICENSE.txt": (
                source_root / "src/phaxis/_vendor/tomli/LICENSE.txt"
            ).read_bytes(),
        }
    )
    members.update(extra_members or {})
    for name in omitted_members:
        members.pop(name, None)
    record_name = f"{dist_info}/RECORD"
    rows = []
    for name, content in sorted(members.items()):
        digest = _wheel_record_digest(content)
        if name == corrupt_record_member:
            digest = "0" * 43
        rows.append([name, f"sha256={digest}", str(len(content))])
    rows.append([record_name, "", ""])
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    members[record_name] = record_buffer.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)


def _write_source_supply_chain(source: Path) -> list[dict]:
    payloads = {
        "LICENSE": "Apache License 2.0 fixture\n",
        "NOTICE": "PHAxis 1.0.0\nApache-2.0\n",
        "THIRD_PARTY_NOTICES.md": (
            "# PHAxis 1.0.0 third-party notices\n\nSynthetic test authority.\n"
        ),
    }
    inventory = {
        "schema_version": "PHAxis-third-party-license-inventory-1.0",
        "status": "complete_declared_direct_dependency_inventory",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "dependencies": [{"name": "numpy", "license_expression": "BSD-3-Clause"}],
    }
    inventory["inventory_identity_sha256"] = sha256_json(inventory)
    payloads["THIRD_PARTY_LICENSES.json"] = (
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    payloads["SBOM.cdx.json"] = json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "version": 1,
            "metadata": {
                "component": {
                    "type": "application",
                    "name": "phaxis",
                    "version": "1.0.0",
                }
            },
            "components": [{"type": "library", "name": "numpy"}],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    records = []
    package_init = source / "src/phaxis/__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    records.append(
        {
            "path": "src/phaxis/__init__.py",
            "bytes": package_init.stat().st_size,
            "sha256": sha256_file(package_init),
            "origin": "project:src/phaxis/__init__.py",
        }
    )
    vendor_license = source / "src/phaxis/_vendor/tomli/LICENSE.txt"
    vendor_license.parent.mkdir(parents=True)
    vendor_license.write_text(
        "Synthetic MIT license fixture for vendored Tomli.\n",
        encoding="utf-8",
        newline="\n",
    )
    records.append(
        {
            "path": "src/phaxis/_vendor/tomli/LICENSE.txt",
            "bytes": vendor_license.stat().st_size,
            "sha256": sha256_file(vendor_license),
            "origin": "project:src/phaxis/_vendor/tomli/LICENSE.txt",
        }
    )
    for relative, content in sorted(payloads.items()):
        path = source / relative
        path.write_text(content, encoding="utf-8", newline="\n")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "origin": "generated:test",
            }
        )
    return sorted(records, key=lambda row: row["path"])


def test_distribution_stage_builds_wheel_and_sdist_as_one_atomic_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module(
        "phaxis_release_distribution_test",
        "scripts/phaxis/build_release_distributions.py",
    )
    assert module.PEP517_SDIST_GENERATED_MEMBERS == (
        ORCHESTRATOR_PEP517_SDIST_GENERATED_MEMBERS
    )
    source = tmp_path / "source"
    source.mkdir()
    records = _write_source_supply_chain(source)
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    atomic_write_json(source / "SOURCE_MANIFEST.json", manifest)
    stage40_sentinel_lock = _stage_directory_lock(source)

    commands: list[list[str]] = []

    def fake_run(command, *, cwd, audit_argv=None):
        command = list(command)
        commands.append(command)
        if "build" in command:
            out = Path(command[command.index("--outdir") + 1])
            private_source = Path(command[-1])
            assert cwd == private_source
            _write_synthetic_wheel(
                out / "phaxis-1.0.0-py3-none-any.whl",
                source_root=private_source,
            )
            _write_synthetic_sdist(
                out / "phaxis-1.0.0.tar.gz",
                source_root=private_source,
                generated_members=module.PEP517_SDIST_GENERATED_MEMBERS,
            )
            (private_source / "src/phaxis.egg-info").mkdir(parents=True)
            (private_source / "src/phaxis.egg-info/PKG-INFO").write_text(
                "generated only in private build input\n", encoding="utf-8"
            )
        return {
            "argv": list(audit_argv or command),
            "returncode": 0,
            "stdout_sha256": sha256_json([]),
            "stderr_sha256": sha256_json([]),
        }

    monkeypatch.setattr(module, "_run", fake_run)
    toolchain = {
        "implementation": "CPython",
        "python_version": "3.12.0",
        "python_cache_tag": "cpython-312",
        "python_executable_filename": "python.exe" if os.name == "nt" else "python",
        "python_executable_sha256": "a" * 64,
        "packages": {
            "build": "1.2.2",
            "setuptools": "80.0.0",
            "twine": "6.1.0",
            "wheel": "0.45.1",
        },
        "probe_isolated": True,
        "cuda_visible_devices": "-1",
        "exact_versions_recorded": True,
        "build_isolation_used": False,
    }
    toolchain["build_toolchain_identity_sha256"] = sha256_json(toolchain)
    monkeypatch.setattr(module, "_build_toolchain_record", lambda python, cwd: toolchain)
    destination = tmp_path / "dist"
    receipt = module.build_release_distributions(
        source_release_root=source,
        source_release_manifest=source / "SOURCE_MANIFEST.json",
        output=destination,
        python_executable=sys.executable,
    )
    assert receipt["status"] == "completed_wheel_sdist_verified"
    assert receipt["source_release_input_immutable"] is True
    assert receipt["source_release_before_lock"] == receipt["source_release_after_lock"]
    assert _stage_directory_lock(source) == stage40_sentinel_lock
    assert receipt["private_build_input"]["manifest_exact_copy_verified"] is True
    assert not (source / "src/phaxis.egg-info").exists()
    assert {row["kind"] for row in receipt["artifacts"]} == {"wheel", "sdist"}
    assert receipt["distribution_identity_sha256"] == sha256_json(
        {key: value for key, value in receipt.items() if key != "distribution_identity_sha256"}
    )
    assert (destination / "distribution_receipt.json").is_file()
    assert (destination / "phaxis-1.0.0.cdx.json").is_file()
    assert (destination / "phaxis-1.0.0-THIRD_PARTY_NOTICES.md").is_file()
    assert (destination / "phaxis-1.0.0-THIRD_PARTY_LICENSES.json").is_file()
    assert sha256_file(
        destination / "phaxis-1.0.0-THIRD_PARTY_LICENSES.json"
    ) == sha256_file(source / "THIRD_PARTY_LICENSES.json")
    assert (destination / "release_asset_inventory.json").is_file()
    assert (destination / "SHA256SUMS").is_file()
    assert {row["kind"] for row in receipt["release_assets"]} == {
        "wheel",
        "sdist",
        "cyclonedx_sbom",
        "third_party_notices",
        "third_party_license_inventory",
    }
    inventory = read_json(destination / "release_asset_inventory.json")
    unsigned_inventory = dict(inventory)
    inventory_identity = unsigned_inventory.pop(
        "release_asset_inventory_identity_sha256"
    )
    assert inventory_identity == sha256_json(unsigned_inventory)
    assert inventory["assets"] == receipt["release_assets"]
    assert receipt["release_asset_inventory"]["sha256"] == sha256_file(
        destination / "release_asset_inventory.json"
    )
    assert receipt["release_checksums"]["sha256"] == sha256_file(
        destination / "SHA256SUMS"
    )
    assert (destination / "SHA256SUMS").read_text(encoding="utf-8") == "".join(
        f"{row['sha256']}  {row['filename']}\n"
        for row in receipt["release_assets"]
    )
    assert any("twine" in command for command in commands)
    assert all(command[1:3] == ["-B", "-I"] for command in commands)
    assert receipt["build_toolchain"] == toolchain
    public_commands = json.dumps(receipt["commands"], sort_keys=True)
    assert str(tmp_path) not in public_commands
    assert "<PRIVATE_MANIFEST_EXACT_SOURCE_COPY>" in public_commands
    wheel_audit = receipt["wheel_archive_audit"]
    assert wheel_audit["archive_filename"] == "phaxis-1.0.0-py3-none-any.whl"
    assert wheel_audit["wheel_tag"] == "py3-none-any"
    assert wheel_audit["entry_point"] == "phaxis = phaxis.cli:main"
    assert wheel_audit["record_verified"] is True
    assert wheel_audit["source_package_file_count"] == 2
    assert wheel_audit["source_package_hashes_verified"] is True
    assert wheel_audit["metadata_license_files"] == [
        "LICENSE",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
    ]
    assert wheel_audit["pep639_license_member_count"] == 2
    assert wheel_audit["license_file_hashes_verified"] is True
    assert wheel_audit["unexpected_payload_members"] == 0
    assert wheel_audit["prohibited_payload_members"] == 0
    sdist_audit = receipt["sdist_archive_audit"]
    assert sdist_audit["source_manifest_self_covered"] is True
    assert sdist_audit["source_manifest_member"] == "SOURCE_MANIFEST.json"
    assert sdist_audit["allowed_pep517_generated_members"] == list(
        module.PEP517_SDIST_GENERATED_MEMBERS
    )
    assert len(sdist_audit["observed_pep517_generated_members"]) == 8
    assert sdist_audit["unexpected_generated_members"] == 0
    assert sdist_audit["unexpected_generated_member_paths"] == []
    assert sdist_audit["missing_allowed_generated_members"] == 0
    sdist_artifact = next(row for row in receipt["artifacts"] if row["kind"] == "sdist")
    assert sdist_audit["archive_sha256"] == sdist_artifact["sha256"]
    with pytest.raises(module.DistributionBuildError, match="overwrite"):
        module.build_release_distributions(
            source_release_root=source,
            source_release_manifest=source / "SOURCE_MANIFEST.json",
            output=destination,
            python_executable=sys.executable,
        )

    changed_source = tmp_path / "source-mutated-during-stage41"
    shutil.copytree(source, changed_source)

    def mutate_original_source(command, *, cwd, audit_argv=None):
        record = fake_run(command, cwd=cwd, audit_argv=audit_argv)
        if "build" in command:
            (changed_source / "NOTICE").write_text(
                "PHAxis 1.0.0\nconcurrent mutation\n",
                encoding="utf-8",
            )
        return record

    monkeypatch.setattr(module, "_run", mutate_original_source)
    changed_output = tmp_path / "dist-mutated-source"
    with pytest.raises(module.DistributionBuildError, match="changed during"):
        module.build_release_distributions(
            source_release_root=changed_source,
            source_release_manifest=changed_source / "SOURCE_MANIFEST.json",
            output=changed_output,
            python_executable=sys.executable,
        )
    assert not changed_output.exists()


def test_distribution_real_pep517_build_uses_only_private_source_copy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("build")
    pytest.importorskip("twine")
    module = _script_module(
        "phaxis_release_distribution_real_pep517_test",
        "scripts/phaxis/build_release_distributions.py",
    )
    source = tmp_path / "sealed-source"
    source.mkdir()
    records = _write_source_supply_chain(source)
    authored = {
        "README.md": "# PHAxis\n",
        "MANIFEST.in": (
            "include SOURCE_MANIFEST.json\n"
            "include README.md LICENSE MANIFEST.in pyproject.toml\n"
            "include NOTICE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.json SBOM.cdx.json\n"
            "recursive-include src *.py\n"
            "recursive-include src/phaxis/_vendor/tomli LICENSE.txt py.typed\n"
        ),
        "pyproject.toml": '''[build-system]
requires = ["setuptools>=77", "wheel>=0.45"]
build-backend = "setuptools.build_meta"

[project]
name = "phaxis"
version = "1.0.0"
description = "Reproducible Arabidopsis primary-root and root-hair phenotyping"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"]
dependencies = ["numpy>=1.26,<3"]

[project.scripts]
phaxis = "phaxis.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = false

[tool.setuptools.package-data]
"phaxis._vendor.tomli" = ["LICENSE.txt", "py.typed"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["phaxis", "phaxis.*"]
''',
        "src/phaxis/cli.py": "def main():\n    return 0\n",
    }
    for relative, content in authored.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "origin": f"generated:test:{relative}",
            }
        )
    for project_vendor in sorted(
        path
        for path in (PROJECT_ROOT / "src/phaxis/_vendor").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".txt", ".typed"}
    ):
        relative = project_vendor.relative_to(PROJECT_ROOT).as_posix()
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_vendor, target)
        existing = next((row for row in records if row["path"] == relative), None)
        record = {
            "path": relative,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "origin": f"project:test:{relative}",
        }
        if existing is None:
            records.append(record)
        else:
            existing.update(record)
    records.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    atomic_write_json(source / "SOURCE_MANIFEST.json", manifest)
    stage40_sentinel_lock = _stage_directory_lock(source)

    receipt = module.build_release_distributions(
        source_release_root=source,
        source_release_manifest=source / "SOURCE_MANIFEST.json",
        output=tmp_path / "distributions",
        python_executable=sys.executable,
    )
    assert receipt["source_release_input_immutable"] is True
    assert receipt["source_release_before_lock"] == receipt["source_release_after_lock"]
    assert _stage_directory_lock(source) == stage40_sentinel_lock
    assert not (source / "src/phaxis.egg-info").exists()
    assert not (source / "build").exists()
    assert all(
        not argument.startswith(str(tmp_path))
        for command in receipt["commands"]
        for argument in command["argv"]
    )
    distribution_root = tmp_path / "distributions"
    with zipfile.ZipFile(distribution_root / "phaxis-1.0.0-py3-none-any.whl") as wheel:
        wheel_members = set(wheel.namelist())
    assert "phaxis/_vendor/tomli/_parser.py" in wheel_members
    assert "phaxis/_vendor/tomli/LICENSE.txt" in wheel_members
    assert any(
        member.endswith(".dist-info/licenses/src/phaxis/_vendor/tomli/LICENSE.txt")
        for member in wheel_members
    )
    with tarfile.open(distribution_root / "phaxis-1.0.0.tar.gz", mode="r:gz") as sdist:
        sdist_members = set(sdist.getnames())
    assert "phaxis-1.0.0/src/phaxis/_vendor/tomli/_parser.py" in sdist_members
    assert "phaxis-1.0.0/src/phaxis/_vendor/tomli/LICENSE.txt" in sdist_members


def test_sdist_audit_rejects_tampered_authored_member(tmp_path: Path) -> None:
    module = _script_module(
        "phaxis_release_distribution_tamper_test",
        "scripts/phaxis/build_release_distributions.py",
    )
    source = tmp_path / "source"
    source.mkdir()
    readme = source / "README.md"
    readme.write_text("authored source\n", encoding="utf-8")
    records = [
        {
            "path": "README.md",
            "bytes": readme.stat().st_size,
            "sha256": sha256_file(readme),
            "origin": "generated:test",
        }
    ]
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    manifest_path = source / "SOURCE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    archive = tmp_path / "phaxis-1.0.0.tar.gz"
    _write_synthetic_sdist(
        archive,
        source_root=source,
        generated_members=module.PEP517_SDIST_GENERATED_MEMBERS,
        tampered_authored={"README.md": b"tampered after authorship\n"},
    )
    with pytest.raises(
        module.DistributionBuildError,
        match="sdist authored member SHA-256 mismatch: README.md",
    ):
        module._audit_sdist(
            archive,
            source_manifest_path=manifest_path,
            source_manifest=manifest,
        )


def test_sdist_audit_rejects_extra_generated_member(tmp_path: Path) -> None:
    module = _script_module(
        "phaxis_release_distribution_extra_test",
        "scripts/phaxis/build_release_distributions.py",
    )
    source = tmp_path / "source"
    source.mkdir()
    records: list[dict] = []
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    manifest_path = source / "SOURCE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    archive = tmp_path / "phaxis-1.0.0.tar.gz"
    _write_synthetic_sdist(
        archive,
        source_root=source,
        generated_members=module.PEP517_SDIST_GENERATED_MEMBERS,
        extra_generated_members=("src/phaxis.egg-info/rogue.txt",),
    )
    with pytest.raises(
        module.DistributionBuildError,
        match="sdist contains unexpected generated members: .*rogue.txt",
    ):
        module._audit_sdist(
            archive,
            source_manifest_path=manifest_path,
            source_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("writer_kwargs", "message"),
    (
        ({"version": "1.0.1"}, "wheel METADATA is not canonical"),
        (
            {"entry_point": "phaxis = phaxis.not_cli:main"},
            "entry_points.txt lacks the canonical phaxis CLI",
        ),
        (
            {"corrupt_record_member": "phaxis/__init__.py"},
            "wheel RECORD digest/size mismatch: phaxis/__init__.py",
        ),
        (
            {"license_files": ("LICENSE",)},
            "wheel METADATA License-File headers are not the exact",
        ),
        (
            {
                "omitted_members": (
                    "phaxis-1.0.0.dist-info/licenses/src/phaxis/_vendor/tomli/LICENSE.txt",
                )
            },
            "wheel archive lacks required PEP 639 license member",
        ),
        (
            {
                "extra_members": {
                    "phaxis-1.0.0.dist-info/licenses/src/phaxis/_vendor/tomli/LICENSE.txt": b"re-sealed replacement\n"
                }
            },
            "wheel PEP 639 license member differs from source authority",
        ),
        (
            {
                "extra_members": {
                    "phaxis-1.0.0.dist-info/licenses/UNREVIEWED.txt": b"unreviewed\n"
                }
            },
            "wheel PEP 639 license member closure is not the exact",
        ),
        (
            {"extra_members": {"phaxis/weights/model.pt": b"not a model"}},
            "wheel archive contains a prohibited payload path",
        ),
        (
            {"extra_members": {"phaxis/rogue.py": b"ROGUE = True\n"}},
            "wheel PHAxis code differs from the exact source-manifest package tree",
        ),
    ),
)
def test_wheel_audit_fails_closed_on_metadata_record_or_payload_drift(
    tmp_path: Path,
    writer_kwargs: dict,
    message: str,
) -> None:
    module = _script_module(
        "phaxis_release_wheel_negative_test_" + hashlib.sha256(
            message.encode("utf-8")
        ).hexdigest()[:8],
        "scripts/phaxis/build_release_distributions.py",
    )
    source = tmp_path / "source"
    source.mkdir()
    records = _write_source_supply_chain(source)
    manifest = {
        "schema_version": "PHAxis-source-release-manifest-2.0",
        "distribution": "phaxis",
        "version": "1.0.0",
        "release_mode": "formal",
        "files": records,
        "tree_identity_sha256": sha256_json(records),
    }
    wheel = tmp_path / "phaxis-1.0.0-py3-none-any.whl"
    _write_synthetic_wheel(wheel, source_root=source, **writer_kwargs)
    with pytest.raises(module.DistributionBuildError, match=message):
        module._audit_wheel(wheel, source_manifest=manifest)


def test_benchmark_inventory_requires_every_direct_run_and_support_authority(
    tmp_path: Path,
) -> None:
    module = _script_module(
        "phaxis_benchmark_inventory_test",
        "scripts/phaxis/build_benchmark_artifact_inventory.py",
    )
    project = tmp_path / "project"
    inputs = project / "inputs"
    inputs.mkdir(parents=True)
    exact_roles = sorted(module.EXACT_ROLES)
    specs: list[str] = []
    for index, role in enumerate(exact_roles):
        source = inputs / f"{role}.json"
        source.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        specs.append(
            f"{role}=model/benchmark/{role}.json={source.relative_to(project).as_posix()}"
        )
    for index, role in enumerate(
        ("per_image_latency_csv", "per_image_latency_csv", "gpu_telemetry", "hardware_preflight")
    ):
        source = inputs / f"support-{index}.txt"
        source.write_text(f"{role}\n", encoding="utf-8")
        specs.append(
            f"{role}=model/benchmark/support-{index}.txt={source.relative_to(project).as_posix()}"
        )
    receipt = module.build_inventory(
        project_root=project,
        artifacts=specs,
        output="out/inventory.csv",
        receipt="out/inventory_receipt.json",
    )
    assert receipt["status"] == "completed_explicit_benchmark_inventory"
    assert receipt["role_counts"]["per_image_latency_csv"] == 2
    with (project / "out/inventory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(specs)
    assert all(row["release_authorized"] == "true" for row in rows)

    incomplete = [value for value in specs if not value.startswith("v1_sequential_summary=")]
    with pytest.raises(module.InventoryError, match="v1_sequential_summary"):
        module.build_inventory(
            project_root=project,
            artifacts=incomplete,
            output="out/incomplete.csv",
            receipt="out/incomplete_receipt.json",
        )


def test_production_manifest_cross_checks_root_raw_task_path_and_scale(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.tif"
    image.write_bytes(b"raw-image")
    digest = sha256_file(image)
    metadata = tmp_path / "metadata.csv"
    review = tmp_path / "review.csv"
    root = tmp_path / "root.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "image_sha256", "um_per_px", "source_megapixels"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "S1",
                "image_sha256": digest,
                "um_per_px": "2.5",
                "source_megapixels": "1.0",
            }
        )
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "image_path", "image_sha256"))
        writer.writeheader()
        writer.writerow({"task_id": "S1", "image_path": str(image), "image_sha256": digest})
    with root.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_id", "input_path", "source_um_per_px"),
        )
        writer.writeheader()
        writer.writerow({"image_id": "S1", "input_path": str(image), "source_um_per_px": "2.5"})
    output = tmp_path / "production"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/phaxis/build_production_manifest.py"),
        "--analysis-metadata",
        str(metadata),
        "--review-manifest",
        str(review),
        "--root-input-manifest",
        str(root),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = read_json(output / "summary.json")
    assert summary["root_source_alignment"] == "passed_exact_task_path_scale_identity"
    assert summary["root_input_manifest_sha256"] == sha256_file(root)

    bad_root = tmp_path / "bad-root.csv"
    bad_root.write_text(root.read_text(encoding="utf-8").replace("2.5", "2.6"), encoding="utf-8")
    refused = subprocess.run(
        [*command[:-3], str(bad_root), *command[-2:-1], str(tmp_path / "bad-output")],
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "physical scales differ" in refused.stderr


def test_handover_report_publication_rolls_back_unsealed_payload(tmp_path: Path) -> None:
    module = _script_module(
        "phaxis_handover_report_test",
        "scripts/phaxis/handover_manifest_producers.py",
    )
    project = tmp_path / "project"
    project.mkdir()
    payload = project / "payload.csv"
    payload.write_text("a\n", encoding="utf-8")
    report = {
        "schema_version": "PHAxis-handover-materialisation-plan-1.0",
        "status": "created",
        "plan_identity_sha256": "f" * 64,
    }
    receipt = project / "receipt.json"
    module.publish_report_no_overwrite(
        project_root=project,
        report=report,
        receipt=receipt,
    )
    assert read_json(receipt) == report

    orphan = project / "orphan.csv"
    orphan.write_text("b\n", encoding="utf-8")
    with pytest.raises(module.ProducerError, match="overwrite"):
        module.publish_report_no_overwrite(
            project_root=project,
            report=report,
            receipt=receipt,
            rollback_outputs=(orphan,),
        )
    assert not orphan.exists()
