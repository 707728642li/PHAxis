"""Portable, asset-free CPU suite; excludes release-control/private-asset tests."""

from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
names = [
    "cli_contract",
    "contract_validation",
    "fusion_contract",
    "traits_export",
    "trait_export_user_journeys",
    "workflow",
    "biological_presence_metric",
    "phenotype_contract",
    "phenotype_catalog",
    "evaluation_metrics",
    "io_contract",
]
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="-1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
paths = ["tests/release"] + [f"tests/phaxis/test_{name}.py" for name in names]
raise SystemExit(
    subprocess.call(
        [sys.executable, "-B", "-m", "pytest", *paths, "-q", *sys.argv[1:]], cwd=root, env=env
    )
)
