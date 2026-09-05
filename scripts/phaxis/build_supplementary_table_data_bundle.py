#!/usr/bin/env python
"""Build a create-only PHAxis S1--S10 reviewer data bundle from a source map.

The formal 61-stage release invokes the same producer in-process from the
existing ``figures`` stage.  This CLI is the reproducible CPU-only standalone
route for audits and fixtures; it neither discovers sources nor executes a
model.  The source-map JSON must name the exact roles required by
``phaxis.supplementary_tables.TABLE_SPECS``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.supplementary_tables import (  # noqa: E402
    FINAL_STATUS,
    PROVISIONAL_STATUS,
    SOURCE_MAP_SCHEMA,
    materialize_supplementary_table_data_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final", "provisional"), required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_map_path = args.source_map.resolve()
    payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCE_MAP_SCHEMA:
        raise ValueError("supplementary source-map schema changed")
    raw_sources = payload.get("source_paths")
    if not isinstance(raw_sources, dict):
        raise ValueError("supplementary source-map source_paths must be an object")
    sources = {}
    for role, raw in raw_sources.items():
        path = Path(str(raw))
        if not path.is_absolute():
            path = source_map_path.parent / path
        sources[str(role)] = path.resolve()
    result = materialize_supplementary_table_data_bundle(
        output=args.output,
        status=FINAL_STATUS if args.mode == "final" else PROVISIONAL_STATUS,
        source_paths=sources,
        source_identities=payload.get("source_identities", {}),
        figure_input_manifest_sha256=payload["figure_input_manifest_sha256"],
        figure_input_assembly_identity_sha256=payload[
            "figure_input_assembly_identity_sha256"
        ],
        model_contract_proposal_identity_sha256=payload[
            "model_contract_proposal_identity_sha256"
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
