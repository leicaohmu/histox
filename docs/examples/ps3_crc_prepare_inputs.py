#!/usr/bin/env python
"""Prepare HistoX PS3 CRC inputs from user-owned slides and report text.

The manifest must contain exactly one selected diagnostic slide and one
machine-readable, de-identified report per patient:

``patient_id,slide_path,report_path``

Paths may be absolute or relative to the manifest. RNA is supplied separately
as a patient-by-gene CSV whose first column is the patient index. This script
creates PLIP patch bags, aligned WSI coordinates, and PLIP report-segment
embeddings. Gene scaling and PANTHER fitting deliberately remain fold-specific
and are performed later.

PDF parsing, OCR, de-identification, raw RNA quantification, and TCGA access
approval are outside this script's boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

import histox as hx
from histox.mil import PS3HistologyPrototypeEncoder, PS3ReportEncoder


REQUIRED_COLUMNS = ("patient_id", "slide_path", "report_path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest(path: Path) -> pd.DataFrame:
    manifest_path = path.expanduser().resolve()
    table = pd.read_csv(manifest_path)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in table]
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    table = table.loc[:, REQUIRED_COLUMNS].copy()
    table["patient_id"] = table["patient_id"].astype(str)
    if table["patient_id"].duplicated().any():
        duplicate = table.loc[
            table["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(
            "Select one diagnostic slide per patient before preparation; "
            f"duplicate patient: {duplicate}"
        )
    base = manifest_path.parent
    for column in ("slide_path", "report_path"):
        resolved = []
        for raw in table[column].astype(str):
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing {column}: {candidate}")
            resolved.append(candidate)
        table[column] = resolved
    return table.sort_values("patient_id").reset_index(drop=True)


def iter_split_files(split_root: Path, fold_pattern: str) -> Iterable[Path]:
    for fold in range(5):
        fold_dir = split_root / fold_pattern.format(fold=fold)
        yield fold_dir / "train.csv"
        yield fold_dir / "test.csv"


def check_patient_contract(
    manifest: pd.DataFrame,
    rna_path: Path,
    split_root: Path,
    fold_pattern: str,
    patient_column: str,
) -> Dict[str, object]:
    rna = pd.read_csv(rna_path, index_col=0)
    rna.index = rna.index.astype(str)
    if rna.index.duplicated().any():
        raise ValueError("RNA table contains duplicate patient rows")
    manifest_ids = set(manifest["patient_id"])
    fold_rows: List[Dict[str, object]] = []
    all_split_ids = set()
    for split_path in iter_split_files(split_root, fold_pattern):
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        split = pd.read_csv(split_path)
        if patient_column not in split:
            raise ValueError(
                f"{split_path} lacks patient column {patient_column!r}"
            )
        ids = set(split[patient_column].astype(str))
        all_split_ids.update(ids)
        fold_rows.append({"path": str(split_path), "patients": len(ids)})
    missing_manifest = sorted(all_split_ids - manifest_ids)
    missing_rna = sorted(all_split_ids - set(rna.index))
    if missing_manifest or missing_rna:
        raise ValueError(
            "Split coverage failed: "
            f"missing manifest={missing_manifest[:5]}, missing RNA={missing_rna[:5]}"
        )
    expression = rna.loc[sorted(all_split_ids)].to_numpy(dtype=np.float64)
    if not np.isfinite(expression).all():
        raise ValueError("RNA contains non-finite values for split patients")
    return {
        "manifest_patients": len(manifest_ids),
        "split_patients": len(all_split_ids),
        "rna_patients": len(rna),
        "rna_genes": len(rna.columns),
        "split_files": fold_rows,
    }


def output_path(root: Path, modality: str, patient_id: str) -> Path:
    path = root / modality / f"{patient_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_histology(
    manifest: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    extractor = hx.build_feature_extractor(
        "plip",
        backend="torch",
        model_id=args.plip_model,
        revision=args.plip_revision,
        device=args.device,
        mixed_precision=False,
    )
    records = []
    for row in manifest.itertuples(index=False):
        destination = output_path(output_root, "histology_bags", row.patient_id)
        coordinate_destination = output_path(
            output_root, "histology_coordinates", row.patient_id
        )
        if destination.exists() and not args.overwrite:
            if not coordinate_destination.exists():
                raise FileNotFoundError(
                    f"{coordinate_destination} is missing for an existing bag. "
                    "Run again with --overwrite to recreate aligned features "
                    "and coordinates."
                )
            bag = torch.load(destination, map_location="cpu")
            coordinates = torch.load(
                coordinate_destination, map_location="cpu"
            )
            transfer = "reused"
        else:
            slide = hx.WSI(
                str(row.slide_path),
                tile_px=256,
                tile_um=128,
                roi_method="ignore",
                verbose=False,
            )
            slide.qc("otsu")
            feature_grid = extractor(
                slide,
                batch_size=args.batch_size,
                dtype=np.float32,
            )
            bag, grid_coordinates = (
                PS3HistologyPrototypeEncoder
                .flatten_feature_grid_with_coordinates(feature_grid)
            )
            slide_coordinates = slide.get_tile_coord(anchor="center")
            coordinate_lookup = {
                (int(grid_x), int(grid_y)): (
                    int(x), int(y), int(grid_x), int(grid_y)
                )
                for x, y, grid_x, grid_y in slide_coordinates
            }
            try:
                coordinates = torch.tensor([
                    coordinate_lookup[(int(grid_x), int(grid_y))]
                    for grid_x, grid_y in grid_coordinates.tolist()
                ], dtype=torch.int64)
            except KeyError as exc:
                raise RuntimeError(
                    "Extracted feature grid is not aligned with WSI tile coordinates"
                ) from exc
            torch.save(bag.cpu(), destination)
            torch.save(coordinates.cpu(), coordinate_destination)
            transfer = "created"
        bag = PS3HistologyPrototypeEncoder.flatten_feature_grid(bag)
        coordinates = torch.as_tensor(coordinates, dtype=torch.int64)
        if bag.shape[1] != 512:
            raise RuntimeError(
                f"{row.patient_id}: expected 512 PLIP features, got {bag.shape[1]}"
            )
        if coordinates.ndim != 2 or coordinates.shape != (len(bag), 4):
            raise RuntimeError(
                f"{row.patient_id}: expected coordinates shaped "
                f"({len(bag)}, 4), got {tuple(coordinates.shape)}"
            )
        records.append({
            "patient_id": row.patient_id,
            "modality": "histology",
            "source": str(row.slide_path),
            "output": str(destination),
            "coordinate_output": str(coordinate_destination),
            "items": int(bag.shape[0]),
            "features": int(bag.shape[1]),
            "status": transfer,
            "sha256": sha256(destination),
            "coordinate_sha256": sha256(coordinate_destination),
        })
        print(
            f"histology patient={row.patient_id} patches={bag.shape[0]} "
            f"status={transfer}",
            flush=True,
        )
    return records


def prepare_reports(
    manifest: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    encoder = PS3ReportEncoder(
        model_id=args.plip_model,
        revision=args.plip_revision,
        device=args.device,
    )
    records = []
    for row in manifest.itertuples(index=False):
        destination = output_path(
            output_root, "report_embeddings", row.patient_id
        )
        if destination.exists() and not args.overwrite:
            embedding = torch.load(destination, map_location="cpu")
            transfer = "reused"
        else:
            report_text = row.report_path.read_text(encoding="utf-8")
            embedding = encoder.encode(report_text)
            torch.save(embedding.cpu(), destination)
            transfer = "created"
        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        if embedding.ndim != 2 or embedding.shape[1] != 512:
            raise RuntimeError(
                f"{row.patient_id}: unexpected report shape {tuple(embedding.shape)}"
            )
        records.append({
            "patient_id": row.patient_id,
            "modality": "report",
            "source": str(row.report_path),
            "output": str(destination),
            "items": int(embedding.shape[0]),
            "features": int(embedding.shape[1]),
            "status": transfer,
            "sha256": sha256(destination),
        })
        print(
            f"report patient={row.patient_id} segments={embedding.shape[0]} "
            f"status={transfer}",
            flush=True,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--fold-pattern",
        default="TCGA_COADREAD_dss_survival_k={fold}",
    )
    parser.add_argument("--patient-column", default="case_id")
    parser.add_argument("--output", type=Path, default=Path("histox-ps3-inputs"))
    parser.add_argument(
        "--stage",
        choices=("check", "histology", "reports", "all"),
        default="all",
    )
    parser.add_argument("--plip-model", default="vinid/plip")
    parser.add_argument("--plip-revision")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if "{fold}" not in args.fold_pattern:
        parser.error("--fold-pattern must contain {fold}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    manifest = resolve_manifest(args.manifest)
    rna_path = args.rna.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not rna_path.is_file():
        parser.error(f"--rna is not a file: {rna_path}")
    if not split_root.is_dir():
        parser.error(f"--split-root is not a directory: {split_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    checks = check_patient_contract(
        manifest,
        rna_path,
        split_root,
        args.fold_pattern,
        args.patient_column,
    )
    print(json.dumps(checks, indent=2), flush=True)
    records: List[Dict[str, object]] = []
    if args.stage in ("histology", "all"):
        records.extend(prepare_histology(manifest, output_root, args))
    if args.stage in ("reports", "all"):
        records.extend(prepare_reports(manifest, output_root, args))

    pd.DataFrame(records).to_csv(
        output_root / "prepared_artifacts.csv", index=False
    )
    provenance = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "rna": str(rna_path),
        "split_root": str(split_root),
        "plip_model": args.plip_model,
        "plip_revision": args.plip_revision,
        "tile_px": 256,
        "tile_um": 128,
        "mpp": 0.5,
        "qc": "otsu",
        "input_checks": checks,
        "artifacts": len(records),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
