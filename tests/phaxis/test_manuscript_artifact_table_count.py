from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/phaxis/verify_manuscript_artifacts.py"


def _module():
    specification = importlib.util.spec_from_file_location(
        "verify_manuscript_artifacts_table_count", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_supplement_docx_declared_table_count_must_match_physical_ooxml() -> None:
    module = _module()
    module._validate_supplement_ooxml_table_count(
        {"embedded_markdown_table_count": 3}, {"table_count": 3}
    )
    with pytest.raises(
        module.ManuscriptArtifactError,
        match="declared table count differs from OOXML",
    ):
        module._validate_supplement_ooxml_table_count(
            {"embedded_markdown_table_count": 3}, {"table_count": 2}
        )
