"""Run HistoX slide processing from one open-access TCGA whole slide.

The first run downloads the 1.23 GB diagnostic slide used by the documentation
figures into the HistoX data cache. Later runs reuse the verified local copy.
Pass ``--slide`` to use an existing copy of that exact slide.
"""

# BEGIN SETUP
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import histox as hx
from histox import data
from histox.slide import qc
from histox.slide.report import ExtractionReport


DATASET_NAME = "tcga-coad-slide-processing"
SLIDE_FILENAME = (
    "TCGA-3L-AA1B-01Z-00-DX1."
    "8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs"
)
# END SETUP


# BEGIN SELECTED_CENTERS_HELPER
def selected_centers(slide: hx.WSI) -> list[tuple[int, int]]:
    """Return retained tile centers in base-level pixel coordinates."""
    half = slide.full_extract_px // 2
    return [
        (int(x + half), int(y + half))
        for x, y, grid_x, grid_y in slide.coord
        if slide.grid[int(grid_x), int(grid_y)]
    ]
# END SELECTED_CENTERS_HELPER


# BEGIN COUNT_FILTERED_TILES_HELPER
def count_filtered_tiles(slide: hx.WSI, **kwargs: Any) -> int:
    """Count a dry-run tile generator without writing image files."""
    generator = slide.build_generator(
        img_format="numpy",
        dry_run=True,
        deterministic=True,
        show_progress=False,
        num_threads=1,
        **kwargs,
    )
    if generator is None:
        return 0
    return sum(1 for _ in generator())
# END COUNT_FILTERED_TILES_HELPER


# BEGIN SAVE_COMPARISON_HELPER
def save_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    output: Path,
) -> None:
    """Save a side-by-side RGB comparison."""
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(processed)
    axes[1].set_title("Reinhard mask · preset v3")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output, dpi=160)
    plt.close(figure)
# END SAVE_COMPARISON_HELPER


def resolve_slide(
    local_slide: Optional[Path],
    data_dir: Optional[Path],
) -> tuple[Path, Dict[str, Any]]:
    """Download the registered slide, or validate a user-supplied copy."""
    record = data.get_dataset(DATASET_NAME)
    source = {
        "dataset": record.name,
        "version": record.version,
        "provider": record.provider.name,
        "provider_file_id": record.provider.dataset_id,
        "access": record.access,
        "terms_url": record.terms_url,
    }
    if local_slide is not None:
        slide_path = local_slide.expanduser().resolve()
        if not slide_path.is_file():
            raise FileNotFoundError("Slide does not exist: {}".format(slide_path))
        if slide_path.name != SLIDE_FILENAME:
            raise ValueError(
                "--slide must point to the documented TCGA file: {}".format(
                    SLIDE_FILENAME
                )
            )
        source["transfer"] = "local override"
        return slide_path, source

    # BEGIN DOWNLOAD
    # HistoX streams into a .part file, resumes byte ranges, and verifies MD5.
    download = data.download_dataset(
        DATASET_NAME,
        path=data_dir,
        timeout=300,
    )
    slide_path = download.destination / SLIDE_FILENAME
    print("Slide:", slide_path)
    print("Provider:", download.dataset.provider.name)
    print("Access:", download.dataset.access)
    print("Terms:", download.dataset.terms_url)
    # END DOWNLOAD
    source["transfer"] = "downloaded or reused"
    source["provenance"] = str(download.provenance_path)
    return slide_path, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "HistoX dataset cache root. If omitted, HISTOX_CACHE_DIR, "
            "XDG_CACHE_HOME/histox, or ~/.cache/histox is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("histox-slide-processing-output"),
        help="Directory for generated figures, tiles, reports, and results.",
    )
    parser.add_argument(
        "--slide",
        type=Path,
        help="Existing copy of the documented TCGA slide; skips the download.",
    )
    args = parser.parse_args()
    data_dir = (
        args.data_dir.expanduser().resolve() if args.data_dir is not None else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_path, source = resolve_slide(args.slide, data_dir)

    # BEGIN OPEN
    # tile_px is the model input size; tile_um is its physical field of view.
    slide = hx.WSI(
        str(slide_path),
        tile_px=256,
        tile_um=1024,
        roi_method="ignore",
        use_edge_tiles=True,
        verbose=False,
    )
    metadata = {
        "dimensions": list(slide.dimensions),
        "mpp": float(slide.mpp),
        "levels": [list(level) for level in slide.level_dimensions],
        "grid_shape": list(slide.grid.shape),
        "candidate_tiles": int(slide.estimated_num_tiles),
    }
    print("Metadata:", json.dumps(metadata, indent=2))
    slide.thumb(width=800, low_res=True).save(output_dir / "01_thumbnail.png")
    # END OPEN

    # BEGIN TILE
    # Grid (4, 10) is a verified tissue position in this exact TCGA slide.
    grid_xy = (4, 10)
    tile = slide[grid_xy]
    if tile is None:
        raise RuntimeError("The selected tissue tile was unavailable")
    Image.fromarray(tile).save(output_dir / "02_tissue_tile.png")
    print("Tile:", grid_xy, slide.grid_to_coord(*grid_xy), tile.shape, tile.dtype)
    # END TILE

    # BEGIN ROI
    roi_slide = hx.WSI(
        str(slide_path),
        tile_px=256,
        tile_um=1024,
        roi_method="ignore",
        roi_filter_method="center",
        use_edge_tiles=True,
        verbose=False,
    )
    width, height = roi_slide.dimensions
    # Define a reproducible polygon in base-level slide coordinates.
    roi_points = np.asarray(
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
    roi_slide.load_roi_array(
        roi_points,
        name="ROI_tutorial",
        label="demo-region",
    )
    roi_slide.roi_method = "inside"
    roi_slide.process_rois()
    roi_tiles = int(roi_slide.estimated_num_tiles)
    roi_slide.thumb(
        width=800,
        rois=True,
        coords=selected_centers(roi_slide),
        rect_color="black",
        low_res=True,
    ).save(output_dir / "03_roi_tiles.png")
    print("ROI tiles:", roi_tiles)
    # END ROI

    # BEGIN OTSU
    otsu_slide = hx.WSI(
        str(slide_path),
        tile_px=256,
        tile_um=1024,
        roi_method="ignore",
        use_edge_tiles=True,
        verbose=False,
    )
    before_otsu = int(otsu_slide.estimated_num_tiles)
    # True pixels in HistoX QC masks are excluded from extraction.
    otsu_mask = np.asarray(qc.Otsu()(otsu_slide), dtype=bool)
    otsu_slide.apply_qc_mask(otsu_mask)
    after_otsu = int(otsu_slide.estimated_num_tiles)
    Image.fromarray((otsu_mask.astype(np.uint8) * 255)).save(
        output_dir / "04_otsu_mask.png"
    )
    otsu_slide.thumb(
        width=800,
        coords=selected_centers(otsu_slide),
        rect_color="black",
        low_res=True,
    ).save(output_dir / "05_otsu_tiles.png")
    print("Otsu tiles:", before_otsu, "->", after_otsu)
    # END OTSU

    # BEGIN GAUSSIAN
    # Use a denser grid here so local blur rejection is visible.
    gaussian_slide = hx.WSI(
        str(slide_path),
        tile_px=512,
        tile_um=512,
        roi_method="ignore",
        use_edge_tiles=True,
        verbose=False,
    )
    gaussian_otsu = np.asarray(qc.Otsu()(gaussian_slide), dtype=bool)
    gaussian_slide.apply_qc_mask(gaussian_otsu)
    after_gaussian_otsu = int(gaussian_slide.estimated_num_tiles)
    gaussian_mask = qc.GaussianV2(
        mpp=8,
        sigma=3,
        threshold=0.12,
        tile_px=256,
        verbose=False,
        filter_threads=4,
        persistent_threads=False,
    )(gaussian_slide)
    gaussian_slide.apply_qc_mask(gaussian_mask)
    after_gaussian = int(gaussian_slide.estimated_num_tiles)
    gaussian_slide.thumb(
        width=800,
        coords=selected_centers(gaussian_slide),
        rect_color="black",
        low_res=True,
    ).save(output_dir / "06_otsu_gaussian_tiles.png")
    print("Otsu + Gaussian tiles:", after_gaussian_otsu, "->", after_gaussian)
    # END GAUSSIAN

    # BEGIN PIXEL_FILTERS
    # A dry run decodes tiles and reports counts without writing image files.
    no_pixel_filter = count_filtered_tiles(
        otsu_slide,
        grayspace_fraction=1.0,
        whitespace_fraction=1.0,
    )
    default_grayspace = count_filtered_tiles(
        otsu_slide,
        grayspace_fraction=0.6,
        grayspace_threshold=0.05,
        whitespace_fraction=1.0,
    )
    whitespace_filtered = count_filtered_tiles(
        otsu_slide,
        grayspace_fraction=1.0,
        whitespace_fraction=0.5,
        whitespace_threshold=230,
    )
    print(
        "Pixel filters:",
        no_pixel_filter,
        default_grayspace,
        whitespace_filtered,
    )
    # END PIXEL_FILTERS

    # BEGIN NORMALIZATION
    # Fit a bundled target preset, then transform the same verified RGB tile.
    normalizer = hx.norm.StainNormalizer("reinhard_mask")
    normalizer.fit("v3")
    normalized_tile = np.asarray(normalizer.transform(tile), dtype=np.uint8)
    save_comparison(
        tile,
        normalized_tile,
        output_dir / "07_stain_normalization.png",
    )
    # END NORMALIZATION

    # BEGIN EXPORT
    # Reuse the Otsu-filtered WSI; only retained positions are exported.
    report = otsu_slide.extract_tiles(
        tiles_dir=str(output_dir / "tiles"),
        img_format="png",
        report=True,
        grayspace_fraction=1.0,
        whitespace_fraction=1.0,
        num_threads=1,
        show_progress=False,
    )
    if report is None:
        raise RuntimeError("Tile extraction returned no report")
    report_path = output_dir / "08_extraction_report.pdf"
    ExtractionReport([report], title="HistoX TCGA tutorial report").save(
        str(report_path)
    )
    tile_dir = output_dir / "tiles" / otsu_slide.name
    exported_tiles = len(list(tile_dir.glob("*.png")))
    print("Exported tiles:", exported_tiles)
    print("Report:", report_path)
    # END EXPORT

    results = {
        "source": source,
        "slide": slide_path.name,
        "metadata": metadata,
        "roi_tiles": roi_tiles,
        "otsu_tiles": {
            "candidate": before_otsu,
            "retained": after_otsu,
        },
        "otsu_gaussian_tiles": {
            "after_otsu": after_gaussian_otsu,
            "after_gaussian": after_gaussian,
        },
        "pixel_filter_tiles": {
            "disabled": no_pixel_filter,
            "grayspace_default": default_grayspace,
            "whitespace_0.5": whitespace_filtered,
        },
        "exported_tiles": exported_tiles,
        "report": report_path.name,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Results:", output_dir / "results.json")


if __name__ == "__main__":
    main()
