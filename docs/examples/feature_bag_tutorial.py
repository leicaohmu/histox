"""Generate and validate feature bags for an existing HistoX project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import histox as hx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ResNet-50 feature bags from existing HistoX TFRecords. "
            "This script does not extract tiles automatically."
        )
    )
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--tile-um", default="20x")
    parser.add_argument("--batch-size", default=32, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = hx.load_project(str(args.project))
    dataset = project.dataset(tile_px=224, tile_um=args.tile_um)

    slides = dataset.slides()
    tfrecords = dataset.tfrecords()
    print(f"Slides selected: {len(slides)}")
    print(f"TFRecords found: {len(tfrecords)}")
    if not slides:
        raise SystemExit("No slides selected; inspect project sources and annotations.")
    if not tfrecords:
        raise SystemExit(
            "No TFRecords found. Extract tiles for tile_px=224 and "
            f"tile_um={args.tile_um!r}, then rerun this script."
        )

    encoder = hx.build_feature_extractor(
        "resnet50_imagenet",
        tile_px=224,
        backend="torch",
    )
    if encoder.num_features != 2048:
        raise RuntimeError(
            f"Unexpected ResNet-50 feature width: {encoder.num_features}"
        )

    bag_dir = Path(
        project.generate_feature_bags(
            encoder,
            dataset=dataset,
            outdir=str(args.outdir),
            batch_size=args.batch_size,
            min_tiles=1,
        )
    )
    bag_paths = sorted(bag_dir.glob("*.pt"))
    if not bag_paths:
        raise RuntimeError("No .pt feature bags were created; inspect the HistoX log.")

    bag_path = bag_paths[0]
    index_path = bag_path.with_suffix(".index.npz")
    config_path = bag_dir / "bags_config.json"
    if not index_path.exists() or not config_path.exists():
        raise RuntimeError("The feature bag is missing its index or configuration.")

    bag = torch.load(bag_path, map_location="cpu")
    coordinates = np.load(index_path)["arr_0"]
    config = json.loads(config_path.read_text())

    if bag.ndim != 2 or bag.shape[1] != 2048:
        raise RuntimeError(f"Unexpected bag shape: {tuple(bag.shape)}")
    if bag.shape[0] != coordinates.shape[0]:
        raise RuntimeError(
            "Feature rows and coordinate rows differ: "
            f"{bag.shape[0]} != {coordinates.shape[0]}"
        )

    print(f"Output directory: {bag_dir}")
    print(f"Validated bag: {bag_path.name}")
    print(f"Feature shape: {tuple(bag.shape)}")
    print(f"Coordinate shape: {coordinates.shape}")
    print(f"Extractor configuration: {config}")


if __name__ == "__main__":
    main()
