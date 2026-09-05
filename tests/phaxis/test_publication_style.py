from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
import pytest  # noqa: E402

from phaxis.publication_style import (
    check_text_budget,
    configure_publication_style,
    save_figure_bundle,
)


def test_save_figure_bundle_has_locked_formats_and_dimensions(tmp_path) -> None:
    configure_publication_style()
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.set_xlabel("Distance (mm)")
    axis.set_ylabel("Hair density")

    manifest = save_figure_bundle(
        figure,
        tmp_path / "figure",
        width_mm=90.0,
        height_mm=60.0,
        check_edge_ink=False,
    )

    assert set(manifest["files"]) == {"pdf", "png", "tiff"}
    assert all(len(value) == 64 for value in manifest["sha256"].values())
    with Image.open(tmp_path / "figure.png") as image:
        assert image.width == pytest.approx(90.0 / 25.4 * 600.0, abs=2)
        assert image.height == pytest.approx(60.0 / 25.4 * 600.0, abs=2)
    with Image.open(tmp_path / "figure.tiff") as image:
        assert image.mode == "RGB"
        assert image.width == pytest.approx(90.0 / 25.4 * 300.0, abs=2)


def test_text_budget_rejects_overlong_annotation() -> None:
    configure_publication_style()
    figure, axis = plt.subplots()
    axis.text(0.5, 0.5, "x" * 43)
    with pytest.raises(RuntimeError, match="text budget exceeded"):
        check_text_budget(figure)
    plt.close(figure)


def test_canvas_wider_than_submission_contract_is_rejected(tmp_path) -> None:
    figure, _ = plt.subplots()
    with pytest.raises(ValueError, match="180-mm"):
        save_figure_bundle(
            figure,
            tmp_path / "too_wide",
            width_mm=180.1,
            height_mm=80.0,
        )
    plt.close(figure)
