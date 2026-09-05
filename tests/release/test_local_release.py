from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from phaxis.cli import main
from phaxis.local_demo import run_demo
from phaxis.offline_report import build_report


@pytest.mark.parametrize("zero,expected", [(False, 2), (True, 0)])
def test_demo_numerical_truth(tmp_path, zero, expected):
    root = tmp_path / "demo"
    receipt = run_demo(root, zero_hairs=zero)
    assert receipt["observed_hair_identities"] == expected
    with (root / "traits/image_traits.csv").open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert len(reader.fieldnames) == 82
        assert len(rows) == 1
    with (root / "traits/hair_instances.csv").open(encoding="utf-8-sig") as handle:
        hairs = list(csv.DictReader(handle))
        assert len(hairs) == expected
        if not zero:
            # The geometric curve measures 14 um; the unmatched identity remains missing.
            assert sorted(row["length_um"] for row in hairs) == ["", "14.0"]
    model = json.loads((root / "report/report_data.json").read_text())
    assert model["synthetic"]
    for name, digest in model["input_sha256"].items():
        assert hashlib.sha256((root / "report/tables" / name).read_bytes()).hexdigest() == digest


def test_reproducible_tables(tmp_path):
    for label in ("a", "b"):
        run_demo(tmp_path / label)
    for name in ("traits.csv", "image_traits.csv", "hair_instances.csv"):

        def rows(label):
            with (tmp_path / label / "traits" / name).open(encoding="utf-8-sig") as handle:
                return [
                    {k: v for k, v in row.items() if k != "prediction_sha256"}
                    for row in csv.DictReader(handle)
                ]

        # Fusion prediction hashes include the execution timestamp, not a changed phenotype.
        assert rows("a") == rows("b")


def test_existing_output_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)


def test_report_requires_tables(tmp_path):
    with pytest.raises(ValueError, match="image_traits"):
        build_report(tmp_path, tmp_path / "new")


def test_report_escapes_untrusted_text(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "image_traits.csv").write_text("task_id,note\nT1,<script>alert(1)</script>\n")
    dest = tmp_path / "report"
    build_report(source, dest)
    content = (dest / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "https://" not in content


def test_private_path_not_exported(tmp_path):
    (tmp_path / "image_traits.csv").write_text("task_id,note\nT1,C:/Users/private/input.tif\n")
    with pytest.raises(ValueError, match="Sensitive"):
        build_report(tmp_path, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_cli_error_exit(capsys):
    assert main(["report", "--traits", "missing-release-fixture", "--output", "unused"]) == 2
    assert "PHAxis error" in capsys.readouterr().err
