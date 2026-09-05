"""Portable, hash-locked provider for the PHAxis Hybrid-Max root branch.

The public PHAxis wheel stays small.  This package verifies and orchestrates
the separately distributed frozen V1/V20.12/Q8/Hybrid-Max model bundle.
"""

from .bundle import (
    BundleError,
    build_bundle,
    collect_bundle_artifacts,
    verify_bundle,
)
from .reference import (
    audit_fresh_reference,
    build_reference_registry,
    verify_reference_registry,
)
from .runtime import PipelineConfig, build_execution_plan, run_pipeline

__all__ = [
    "BundleError",
    "build_bundle",
    "build_reference_registry",
    "audit_fresh_reference",
    "build_execution_plan",
    "collect_bundle_artifacts",
    "PipelineConfig",
    "run_pipeline",
    "verify_bundle",
    "verify_reference_registry",
]
