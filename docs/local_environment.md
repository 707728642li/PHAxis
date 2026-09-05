# Reproducible local environments

The delivery's pypi/requirements-cpu-win-py312.lock records exact base CPU and build dependencies plus SHA-256; the sibling wheelhouse enables offline installation. This lock is platform-specific, not a GPU deployment or cross-platform lock.

The local packaging environment is a dedicated Conda prefix. Clean wheel and sdist tests each use a fresh venv created from that Conda interpreter with no inherited site-packages. Neither uses the research checkout or PYTHONPATH for import.

The declaration SBOM inventories base and optional requirements. The resolved CPU SBOM inventories the tested wheelhouse. Full `[deployment]` resolution, model capsule and associated licenses require a separate deployment release check.
