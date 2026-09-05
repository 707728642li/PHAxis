"""Publication-quality plotting primitives owned by the PHAxis package.

The functions in this module define a small, deterministic figure contract:
fixed physical dimensions, editable PDF text, lossless high-resolution raster
exports, atomic writes, and inexpensive checks for clipped content.  Keeping
the implementation in :mod:`phaxis` makes the manuscript builders portable;
they do not need the historical RHAxis NextGen source tree.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
from PIL import Image

from .io import sha256_file


MM_PER_INCH = 25.4
PALETTE: Mapping[str, str] = {
    "navy": "#14213D",
    "plum": "#87396F",
    "teal": "#239B95",
    "orange": "#D45B2A",
    "gold": "#D99A2B",
    "ink": "#14213D",
    "grey": "#707887",
    "light_grey": "#DCE2E8",
    "pale_teal": "#E5F3F1",
    "pale_plum": "#F2E7EE",
}


def mm_to_inches(value: float) -> float:
    return float(value) / MM_PER_INCH


def _font_family() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in available else "Liberation Sans"


def configure_publication_style() -> None:
    """Apply the fixed Plant Phenomics-oriented Matplotlib style contract."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": _font_family(),
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.05) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def check_text_budget(
    figure: plt.Figure,
    *,
    tick_characters: int = 26,
    axis_title_characters: int = 46,
    legend_title_characters: int = 20,
    legend_item_characters: int = 28,
    annotation_characters: int = 42,
    maximum_lines: int = 3,
    explicit_exemptions: Sequence[str] = (),
) -> None:
    """Fail when manuscript text exceeds the locked character budget."""
    exemptions = set(explicit_exemptions)

    def inspect(text: str, limit: int, context: str, lines: int = maximum_lines) -> None:
        if not text or text in exemptions:
            return
        split = text.splitlines() or [text]
        if len(split) > lines or max(len(item) for item in split) > limit:
            raise RuntimeError(
                f"figure text budget exceeded ({context}): {text!r}; "
                f"limit={limit} chars/{lines} lines"
            )

    for axis in figure.axes:
        inspect(axis.get_xlabel(), axis_title_characters, "x axis title", 2)
        inspect(axis.get_ylabel(), axis_title_characters, "y axis title", 2)
        inspect(axis.get_title(), annotation_characters, "axis title", 2)
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            inspect(tick.get_text(), tick_characters, "tick", 2)
        legend = axis.get_legend()
        if legend is not None:
            inspect(legend.get_title().get_text(), legend_title_characters, "legend title", 2)
            for item in legend.get_texts():
                inspect(item.get_text(), legend_item_characters, "legend item", 2)
        known = {
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
        }
        for item in axis.texts:
            if item not in known:
                inspect(item.get_text(), annotation_characters, "annotation")


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".partial{path.suffix}", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _edge_ink_summary(path: Path, width: int = 2) -> dict[str, int]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        pixels = image.load()
        counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for x in range(image.width):
            for y in range(width):
                if min(pixels[x, y]) < 245:
                    counts["top"] += 1
            for y in range(max(0, image.height - width), image.height):
                if min(pixels[x, y]) < 245:
                    counts["bottom"] += 1
        for y in range(width, max(width, image.height - width)):
            for x in range(width):
                if min(pixels[x, y]) < 245:
                    counts["left"] += 1
            for x in range(max(0, image.width - width), image.width):
                if min(pixels[x, y]) < 245:
                    counts["right"] += 1
        return counts


def _edge_ink_pixels(path: Path, width: int = 2) -> int:
    return sum(_edge_ink_summary(path, width=width).values())


def save_figure_bundle(
    figure: plt.Figure,
    base_path: Path,
    *,
    width_mm: float,
    height_mm: float,
    text_exemptions: Sequence[str] = (),
    check_edge_ink: bool = True,
) -> dict[str, Any]:
    """Atomically write editable PDF, 600-dpi PNG, and 300-dpi RGB TIFF."""
    if base_path.suffix:
        raise ValueError("figure bundle base path must not have a suffix")
    if width_mm > 180.0 + 1e-9:
        raise ValueError("publication canvas exceeds the 180-mm double-column limit")
    configure_publication_style()
    figure.set_size_inches(mm_to_inches(width_mm), mm_to_inches(height_mm), forward=True)
    check_text_budget(figure, explicit_exemptions=text_exemptions)
    figure.canvas.draw()
    outputs = {
        "pdf": base_path.with_suffix(".pdf"),
        "png": base_path.with_suffix(".png"),
        "tiff": base_path.with_suffix(".tiff"),
    }
    temporary = {name: _temporary(path) for name, path in outputs.items()}
    try:
        figure.savefig(temporary["pdf"], format="pdf", bbox_inches=None, pad_inches=0, facecolor="white")
        figure.savefig(
            temporary["png"], format="png", dpi=600, bbox_inches=None, pad_inches=0, facecolor="white"
        )
        raster = _temporary(base_path.with_suffix(".render300.png"))
        try:
            figure.savefig(raster, format="png", dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
            with Image.open(raster) as opened:
                opened.convert("RGB").save(
                    temporary["tiff"],
                    format="TIFF",
                    dpi=(300.0, 300.0),
                    compression="tiff_lzw",
                )
        finally:
            raster.unlink(missing_ok=True)
        edge_ink = _edge_ink_summary(temporary["png"])
        if check_edge_ink and sum(edge_ink.values()) > 0:
            raise RuntimeError(
                f"figure has ink in the outer two pixels: {base_path.name}; {edge_ink}"
            )
        for name, destination in outputs.items():
            os.replace(temporary[name], destination)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)

    with Image.open(outputs["png"]) as png:
        png_size = list(png.size)
        png_dpi = [float(value) for value in png.info.get("dpi", ())]
    with Image.open(outputs["tiff"]) as tiff:
        tiff_size = list(tiff.size)
        tiff_mode = tiff.mode
        tiff_dpi = [float(value) for value in tiff.info.get("dpi", ())]
    return {
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "k": 1,
        "files": {name: str(path.resolve()) for name, path in outputs.items()},
        "sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "png_pixels": png_size,
        "png_dpi": png_dpi,
        "tiff_pixels": tiff_size,
        "tiff_dpi": tiff_dpi,
        "tiff_mode": tiff_mode,
        "edge_ink_pixels_outer_2px": _edge_ink_pixels(outputs["png"]),
        "edge_ink_by_side_outer_2px": _edge_ink_summary(outputs["png"]),
    }
