"""Generate HistoX Slide Processing documentation figures.

The figures are computed from a public TCGA whole-slide image with the current
HistoX implementation.  The script never writes to the source SVS file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import histox as hx
from histox.slide import qc


ACCENT = "#ef6548"
GREEN = "#15803d"
INK = "#172033"
MUTED = "#667085"
WHITE = "#ffffff"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
TCGA_SAMPLE = (
    "TCGA-3L-AA1B-01Z-00-DX1."
    "8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs"
)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def mono_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_MONO, size)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def fill_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fill a fixed panel while preserving aspect ratio, as in the legacy figures."""
    return ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    width: int = 5,
    head: int = 12,
) -> None:
    """Draw a simple black arrow matching the legacy workflow figures."""
    draw.line((start, end), fill="black", width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base_x, base_y = x2 - head * ux, y2 - head * uy
    draw.polygon(
        [
            (x2, y2),
            (base_x + 0.55 * head * px, base_y + 0.55 * head * py),
            (base_x - 0.55 * head * px, base_y - 0.55 * head * py),
        ],
        fill="black",
    )


def tissue_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Crop a slide thumbnail to the tissue extent reported by a QC mask."""
    bbox = bbox_from_kept_mask(mask, image)
    return image.crop(bbox)


def labeled_panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    *,
    size: tuple[int, int] = (880, 660),
) -> Image.Image:
    """Put a restrained label above a large, borderless scientific image."""
    label_height = 78
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, fill=INK, font=font(27, bold=True))
    draw.text((18, 46), subtitle, fill=MUTED, font=font(17))
    image_box = fit_image(image, (size[0], size[1] - label_height))
    canvas.paste(image_box, (0, label_height))
    return canvas


def compose_grid(
    panels: Sequence[Image.Image],
    *,
    columns: int,
    output: Path,
    gap: int = 12,
) -> None:
    """Compose panels without card borders, shadows, or a duplicate title."""
    if not panels:
        raise ValueError("At least one panel is required")
    rows = (len(panels) + columns - 1) // columns
    panel_width = max(item.width for item in panels)
    panel_height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB",
        (columns * panel_width + (columns - 1) * gap, rows * panel_height + (rows - 1) * gap),
        WHITE,
    )
    for index, item in enumerate(panels):
        row, column = divmod(index, columns)
        canvas.paste(item, (column * (panel_width + gap), row * (panel_height + gap)))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def binary_mask_image(mask: np.ndarray, *, black_is_true: bool) -> Image.Image:
    """Render a Boolean mask as a direct black-and-white diagnostic."""
    values = np.asarray(mask, dtype=bool)
    if black_is_true:
        values = ~values
    return Image.fromarray(values.astype(np.uint8) * 255).convert("RGB")


def crop_like(image: Image.Image, bbox: tuple[int, int, int, int], reference: Image.Image) -> Image.Image:
    """Apply a crop derived from another image with the same field of view."""
    if image.size != reference.size:
        image = image.resize(reference.size, Image.Resampling.NEAREST)
    return image.crop(bbox)


def bbox_from_kept_mask(mask: np.ndarray, image: Image.Image) -> tuple[int, int, int, int]:
    """Map the non-excluded mask extent to thumbnail pixel coordinates."""
    kept = ~np.asarray(mask, dtype=bool)
    y, x = np.where(kept)
    if not len(x):
        return (0, 0, image.width, image.height)
    mask_height, mask_width = kept.shape
    left = int(x.min() / mask_width * image.width)
    top = int(y.min() / mask_height * image.height)
    right = int((x.max() + 1) / mask_width * image.width)
    bottom = int((y.max() + 1) / mask_height * image.height)
    padding = max(int(max(right - left, bottom - top) * 0.08), 12)
    return (
        max(left - padding, 0),
        max(top - padding, 0),
        min(right + padding, image.width),
        min(bottom + padding, image.height),
    )


def selected_centers(wsi: hx.WSI) -> list[tuple[int, int]]:
    half = wsi.full_extract_px // 2
    return [
        (int(x + half), int(y + half))
        for x, y, grid_x, grid_y in wsi.coord
        if wsi.grid[int(grid_x), int(grid_y)]
    ]


def make_wsi(path: Path, *, tile_px=256, tile_um=1024, **kwargs) -> hx.WSI:
    return hx.WSI(
        str(path),
        tile_px=tile_px,
        tile_um=tile_um,
        verbose=False,
        use_edge_tiles=True,
        **kwargs,
    )


def build_workflow_figure(slide_path: Path, output: Path) -> dict:
    """Recreate the compact legacy extraction workflow with new HistoX tiles."""
    wsi = make_wsi(slide_path, tile_px=256, tile_um=512, roi_method="ignore")
    otsu_mask = qc.Otsu()(wsi)
    wsi.apply_qc_mask(otsu_mask)
    thumb = wsi.thumb(width=900, low_res=True).convert("RGB")
    centers = selected_centers(wsi)
    if len(centers) < 3:
        raise RuntimeError("Not enough tissue tiles to build the workflow figure")
    sample_indices = [len(centers) // 5, len(centers) // 2, 4 * len(centers) // 5]
    slide_width, slide_height = wsi.dimensions
    tiles: list[Image.Image] = []
    for index in sample_indices:
        center_x, center_y = centers[index]
        x = int(center_x / slide_width * thumb.width)
        y = int(center_y / slide_height * thumb.height)
        radius = max(min(thumb.width, thumb.height) // 12, 28)
        crop = thumb.crop(
            (
                max(x - radius, 0),
                max(y - radius, 0),
                min(x + radius, thumb.width),
                min(y + radius, thumb.height),
            )
        )
        tiles.append(fill_image(crop, (72, 72)))

    normalizer = hx.norm.StainNormalizer("reinhard_mask")
    normalizer.fit("v3")
    normalized_tiles = [
        Image.fromarray(
            np.asarray(normalizer.transform(np.asarray(tile)), dtype=np.uint8)
        )
        for tile in tiles
    ]

    canvas = Image.new("RGB", (1200, 349), WHITE)
    draw = ImageDraw.Draw(canvas)
    label_font = font(23, bold=True)
    small_font = font(20)
    optional_font = font(17)

    # Input slide stack.
    for offset in (0, 18, 36):
        draw.rounded_rectangle(
            (15 + offset, 120 + offset, 155 + offset, 180 + offset),
            radius=3,
            fill="#e9e9e9",
            outline="black",
            width=3,
        )
    for number, y in enumerate((160, 181, 202), start=1):
        draw.text((0, y), str(number), fill="black", font=small_font)

    # Optional ROI and QC branches use the same slide and mask as the main path.
    draw.rounded_rectangle((255, 4, 420, 73), radius=4, fill="#eeeeee", outline="black", width=3)
    roi_preview = fill_image(thumb, (128, 58))
    canvas.paste(roi_preview, (273, 10))
    draw.rounded_rectangle((255, 277, 420, 346), radius=4, fill="#eeeeee", outline="black", width=3)
    mask_preview = fill_image(binary_mask_image(otsu_mask, black_is_true=False), (128, 58))
    canvas.paste(mask_preview, (273, 283))
    draw.text((148, 54), "ROI", fill="black", font=label_font)
    draw.text((148, 277), "QC", fill="black", font=label_font)
    draw.text((216, 59), "*", fill="#1248ff", font=font(27, bold=True))
    draw.text((216, 281), "*", fill="#1248ff", font=font(27, bold=True))
    draw_arrow(draw, (187, 91), (229, 58), width=4, head=11)
    draw_arrow(draw, (187, 255), (229, 291), width=4, head=11)

    draw_arrow(draw, (208, 173), (497, 173), width=4, head=12)
    draw.text((275, 139), "Extract tiles", fill="black", font=label_font)
    draw.text((207, 183), "at μm or magnification", fill="black", font=label_font)
    draw_arrow(draw, (451, 52), (495, 77), width=4, head=11)
    draw_arrow(draw, (451, 299), (495, 274), width=4, head=11)
    draw.text((465, 27), "Filter", fill="black", font=font(20))
    draw.text((465, 306), "Filter", fill="black", font=font(20))

    def paste_tile_stack(tile: Image.Image, x: int, y: int, size: int) -> None:
        for offset in (0, 4, 8):
            scaled = tile.resize((size, size), Image.Resampling.LANCZOS)
            canvas.paste(scaled, (x + offset, y + offset))
            draw.rectangle((x + offset, y + offset, x + offset + size, y + offset + size), outline="black")

    for row, tile in enumerate(tiles):
        paste_tile_stack(tile, 520, 48 + row * 92, 72)
        draw.text((507, 83 + row * 92), str(row + 1), fill="black", font=small_font)
        paste_tile_stack(tile, 710, 79 + row * 72, 55)
        paste_tile_stack(normalized_tiles[row], 920, 79 + row * 72, 55)

    draw_arrow(draw, (608, 173), (698, 173), width=4, head=12)
    draw.text((610, 186), "Resize", fill="black", font=label_font)
    draw_arrow(draw, (781, 173), (908, 173), width=4, head=12)
    draw.text((772, 186), "Normalize", fill="black", font=label_font)
    draw.text((896, 149), "*", fill="#1248ff", font=font(27, bold=True))

    draw_arrow(draw, (989, 173), (1145, 173), width=4, head=12)
    draw.text((995, 139), "TFRecords", fill="black", font=label_font)
    for y in (115, 161, 207):
        draw.rounded_rectangle((1160, y, 1195, y + 34), radius=3, fill="black")
        draw.rectangle((1166, y + 7, 1189, y + 11), fill="white")
    draw_arrow(draw, (1044, 178), (1067, 218), width=4, head=10)
    draw.text((1072, 198), "*", fill="#1248ff", font=font(27, bold=True))
    draw.rounded_rectangle((1030, 224, 1135, 296), radius=17, fill=WHITE, outline="black", width=3)
    draw.text((1057, 231), "PDF", fill="black", font=label_font)
    draw.text((1040, 259), "Report", fill="black", font=label_font)
    draw.text((1088, 323), "* optional", fill="black", font=optional_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return {
        "retained_tiles_after_otsu": int(wsi.estimated_num_tiles),
        "displayed_tile_samples": 3,
        "normalizer": "reinhard_mask",
        "normalizer_preset": "v3",
    }


def build_roi_figure(slide_path: Path, output: Path) -> dict:
    probe = make_wsi(slide_path, roi_method="ignore")
    width, height = probe.dimensions
    roi = np.array(
        [
            (0.06 * width, 0.46 * height),
            (0.22 * width, 0.33 * height),
            (0.36 * width, 0.38 * height),
            (0.44 * width, 0.56 * height),
            (0.37 * width, 0.84 * height),
            (0.12 * width, 0.91 * height),
            (0.045 * width, 0.70 * height),
        ],
        dtype=np.int64,
    )
    methods: list[tuple[str, float | str]] = [
        ("0.01", 0.01),
        ('"center"', "center"),
        ("1", 1.0),
    ]
    panels: list[Image.Image] = []
    counts: dict[str, int] = {}
    for label, method in methods:
        slide = make_wsi(slide_path, roi_method="ignore", roi_filter_method=method)
        slide.load_roi_array(roi, label="demo-region", name="ROI_demo")
        slide.roi_method = "inside"
        slide.process_rois()
        rendered = slide.thumb(
            width=1000,
            coords=selected_centers(slide),
            rois=True,
            rect_color="black",
            rect_linewidth=3,
            color="black",
            linewidth=4,
            low_res=True,
        )
        left = int(roi[:, 0].min() / width * rendered.width)
        top = int(roi[:, 1].min() / height * rendered.height)
        right = int(roi[:, 0].max() / width * rendered.width)
        bottom = int(roi[:, 1].max() / height * rendered.height)
        padding = max(int(max(right - left, bottom - top) * 0.07), 10)
        cropped = rendered.crop(
            (
                max(left - padding, 0),
                max(top - padding, 0),
                min(right + padding, rendered.width),
                min(bottom + padding, rendered.height),
            )
        )
        panels.append(fill_image(cropped, (324, 323)))
        counts[str(method)] = int(slide.estimated_num_tiles)

    canvas = Image.new("RGB", (971, 423), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = "roi_filter_method"
    title_font = mono_font(25)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((971 - (title_box[2] - title_box[0])) // 2, 15), title, fill="black", font=title_font)
    panel_x = (0, 324, 647)
    for x, (label, _), panel in zip(panel_x, methods, panels):
        header_font = mono_font(23)
        label_box = draw.textbbox((0, 0), label, font=header_font)
        draw.text((x + (324 - (label_box[2] - label_box[0])) // 2, 65), label, fill="black", font=header_font)
        canvas.paste(panel, (x, 100))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, dpi=(300, 300), optimize=True)
    return {
        "coverage_0.01_tiles": counts["0.01"],
        "center_rule_tiles": counts["center"],
        "coverage_1.0_tiles": counts["1.0"],
    }


def build_qc_figure(slide_path: Path, output: Path, method: str) -> dict:
    wsi = make_wsi(slide_path, roi_method="ignore")
    original = wsi.thumb(width=1100, low_res=True)
    before = wsi.estimated_num_tiles

    if method == "otsu":
        algorithm = qc.Otsu()
        mask = algorithm(wsi)
    else:
        raise ValueError(f"Unknown QC method: {method}")

    if mask is None:
        raise RuntimeError(f"{method} returned no mask")
    mask = np.asarray(mask, dtype=bool)
    tissue_mask = binary_mask_image(mask, black_is_true=False)

    candidate_slide = make_wsi(slide_path, roi_method="ignore")
    candidate_grid = candidate_slide.thumb(
        width=1100,
        coords=selected_centers(candidate_slide),
        rect_color="black",
        rect_linewidth=2,
        low_res=True,
    )
    wsi.apply_qc_mask(mask)
    retained = wsi.thumb(
        width=1100,
        coords=selected_centers(wsi),
        rect_color="black",
        rect_linewidth=2,
        low_res=True,
    )
    after = wsi.estimated_num_tiles
    crop = bbox_from_kept_mask(mask, original)
    raw_panel = fill_image(original.crop(crop), (500, 456))
    candidate_panel = fill_image(crop_like(candidate_grid, crop, original), (500, 456))
    mask_panel = fill_image(crop_like(tissue_mask, crop, original), (500, 456))
    retained_panel = fill_image(crop_like(retained, crop, original), (500, 456))

    canvas = Image.new("RGB", (1000, 912), WHITE)
    canvas.paste(raw_panel, (0, 0))
    canvas.paste(candidate_panel, (500, 0))
    canvas.paste(mask_panel, (0, 456))
    canvas.paste(retained_panel, (500, 456))
    draw = ImageDraw.Draw(canvas)
    draw.line((499, 0, 499, 912), fill="black", width=2)
    draw.line((0, 455, 1000, 455), fill="black", width=2)
    draw.rectangle((0, 0, 999, 911), outline="black", width=2)
    draw.text((12, 416), "Raw", fill="black", font=font(28, bold=True))
    draw.text((12, 872), "Otsu", fill="black", font=font(28, bold=True))
    draw_arrow(draw, (487, 228), (515, 228), width=4, head=11)
    draw_arrow(draw, (487, 684), (515, 684), width=4, head=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return {
        "candidate_tiles": before,
        "retained_tiles": after,
        "excluded_fraction": float(mask.mean()),
    }


def build_blur_figure(blur_slide_path: Path, output: Path) -> dict:
    wsi = make_wsi(blur_slide_path, tile_px=512, tile_um=512, roi_method="ignore")
    candidate_tiles = wsi.estimated_num_tiles
    otsu_mask = qc.Otsu()(wsi)
    wsi.apply_qc_mask(otsu_mask)
    grid_after_otsu = wsi.grid.copy()

    algorithm = qc.GaussianV2(
        mpp=8,
        sigma=3,
        threshold=0.12,
        tile_px=256,
        verbose=False,
        filter_threads=4,
        persistent_threads=False,
    )
    blur_mask = algorithm(wsi)
    if blur_mask is None:
        raise RuntimeError("GaussianV2 returned no mask")
    wsi.apply_qc_mask(blur_mask)
    grid_after_both = wsi.grid.copy()
    rejected = np.argwhere(np.logical_and(grid_after_otsu, ~grid_after_both))

    input_slide = wsi.thumb(width=1100, low_res=True)
    final_grid_image = wsi.thumb(
        width=1100,
        coords=selected_centers(wsi),
        rect_color="black",
        rect_linewidth=2,
        low_res=True,
    )

    crop = bbox_from_kept_mask(otsu_mask, input_slide)
    otsu_display = binary_mask_image(otsu_mask, black_is_true=False).resize(
        input_slide.size, Image.Resampling.NEAREST
    )
    blur_display = binary_mask_image(blur_mask, black_is_true=False).resize(
        input_slide.size, Image.Resampling.NEAREST
    )
    combined_mask = Image.fromarray(
        np.maximum(np.asarray(otsu_display), np.asarray(blur_display)).astype(np.uint8)
    )
    mask_panel = fill_image(combined_mask.crop(crop), (500, 457))
    grid_panel = fill_image(crop_like(final_grid_image, crop, input_slide), (500, 457))

    canvas = Image.new("RGB", (1000, 457), WHITE)
    canvas.paste(mask_panel, (0, 0))
    canvas.paste(grid_panel, (500, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((499, 0, 499, 457), fill="black", width=2)
    draw.rectangle((0, 0, 999, 456), outline="black", width=2)
    draw.text((12, 417), "Blur", fill="black", font=font(28, bold=True))
    draw_arrow(draw, (487, 228), (515, 228), width=4, head=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return {
        "candidate_tiles": candidate_tiles,
        "tissue_tiles_after_otsu": int(grid_after_otsu.sum()),
        "retained_tiles_after_combined_qc": int(grid_after_both.sum()),
        "rejected_tiles_after_gaussian": int(len(rejected)),
        "rejected_grid_locations": rejected.tolist(),
        "final_qc": "otsu + gaussian_v2",
        "gaussian_parameters": {"mpp": 8, "sigma": 3, "threshold": 0.12},
    }


def build_normalization_figure(slide_path: Path, output: Path) -> dict:
    wsi = make_wsi(slide_path, roi_method="ignore")
    original_thumb = wsi.thumb(width=1200, low_res=True).convert("RGB")
    otsu_mask = qc.Otsu()(wsi)
    original = fill_image(tissue_crop(original_thumb, otsu_mask), (620, 520))
    methods: list[tuple[str, str | None]] = [
        ("Original", None),
        ("Macenko", "macenko"),
        ("Reinhard", "reinhard"),
        ("Reinhard Mask", "reinhard_mask"),
        ("Reinhard Fast", "reinhard_fast"),
        ("Reinhard Fast Mask", "reinhard_fast_mask"),
        ("Vahadane (sklearn)", "vahadane_sklearn"),
        ("Vahadane (spams)", "vahadane_spams"),
    ]
    panels: list[tuple[str, Image.Image]] = [("Original", original)]
    presets: dict[str, str] = {}
    for label, method in methods[1:]:
        try:
            normalizer = hx.norm.StainNormalizer(method)
            preset = "v2" if method.startswith("vahadane") else "v3"
            normalizer.fit(preset)
            presets[method] = preset
            normalized = np.asarray(
                normalizer.transform(np.asarray(original)), dtype=np.uint8
            )
        except ImportError as error:
            raise RuntimeError(
                f"{method} requires its optional runtime dependency; "
                "install it before regenerating the eight-panel comparison"
            ) from error
        panels.append((label, Image.fromarray(normalized)))

    canvas = Image.new("RGB", (1440, 720), WHITE)
    draw = ImageDraw.Draw(canvas)
    panel_width, panel_height = 344, 292
    x_positions = (10, 368, 726, 1084)
    image_y = (44, 386)
    label_y = (20, 362)
    label_font = font(13)
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, 4)
        x = x_positions[column]
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (x + (panel_width - (label_box[2] - label_box[0])) // 2, label_y[row]),
            label,
            fill="black",
            font=label_font,
        )
        canvas.paste(fill_image(image, (panel_width, panel_height)), (x, image_y[row]))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, dpi=(72, 72), optimize=True)
    return {
        "methods": [method for _, method in methods if method is not None],
        "presets": presets,
        "input": "whole-slide thumbnail",
        "layout": "legacy 4x2 comparison",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--blur-wsi", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.wsi.name != TCGA_SAMPLE:
        print(f"Warning: figures are being generated from {args.wsi.name}, not the documented TCGA sample")
    blur_wsi = args.blur_wsi or args.wsi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "source": {
            "filename": args.wsi.name,
            "program": "TCGA",
            "data_portal": "NCI Genomic Data Commons",
            "access": "open",
        },
        "blur_source": {
            "filename": blur_wsi.name,
            "program": "TCGA",
            "data_portal": "NCI Genomic Data Commons",
            "access": "open",
        },
        "histox_version": hx.__version__,
        "workflow": build_workflow_figure(
            args.wsi, args.output_dir / "histox_wsi_pipeline.png"
        ),
        "roi": build_roi_figure(args.wsi, args.output_dir / "histox_roi_filter.png"),
        "otsu": build_qc_figure(args.wsi, args.output_dir / "histox_otsu.png", "otsu"),
        "blur": build_blur_figure(blur_wsi, args.output_dir / "histox_blur.png"),
        "normalization": build_normalization_figure(
            args.wsi, args.output_dir / "histox_stain_normalization.png"
        ),
    }
    (args.output_dir / "generation_metadata.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
