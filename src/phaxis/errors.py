"""Dependency-free public exceptions shared by PHAxis contract modules."""

from __future__ import annotations


class ContractError(RuntimeError):
    """A PHAxis identity, provenance, or expert-boundary contract was violated."""

