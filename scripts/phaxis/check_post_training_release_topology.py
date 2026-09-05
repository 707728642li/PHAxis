#!/usr/bin/env python
"""Inspect the real PHAxis formal-release producer DAG without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Topology inspection imports the packaged DAG from the source checkout.  It
# must never add bytecode outside SOURCE_MANIFEST.json.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.release_topology import (  # noqa: E402
    ReleaseTopologyError,
    validate_release_topology,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_release_topology(project_root=args.project_root)
    except (OSError, ReleaseTopologyError, ValueError) as error:
        print(f"PHAxis release producer topology blocked: {error}", file=sys.stderr)
        return 2
    # This is intentionally stdout-only.  A topology diagnostic is not a
    # release receipt and this entry point has no --output publication mode.
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 3 if report.get("declared_capability_gaps") else 0


if __name__ == "__main__":
    raise SystemExit(main())
