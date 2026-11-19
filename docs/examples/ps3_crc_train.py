#!/usr/bin/env python
"""Train HistoX PS3 from prepared bags or retained fold features.

The script consumes five patient-level split directories, prepared Xena RNA,
Hallmark signatures, and PLIP report embeddings. Histology can be supplied as
patient-level PLIP patch bags created by ``ps3_crc_prepare_inputs.py`` or as
the retained fold-specific PANTHER tensors used by the validated HistoX run.
It trains one or more seeds and writes checkpoints, predictions, histories,
and C-index summaries.

Use ``--histology-bag-root`` for the from-source track. Use the default
``--embedding-glob`` contract for the exactly validated retained-feature
track. The defaults reproduce the COADREAD experiment. For another cancer
type, set ``--report-selection-mode paper`` and ``--reference-profile none``
alongside that cohort's split and outcome-column options.

Security: this workflow reads pickle and PyTorch files. Only use artifacts you
created or independently trust, and acknowledge this with
``--trust-input-files``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from histox import DataSpec, Experiment
from histox.mil import (
    PS3HistologyPrototypeEncoder,
    PS3PathwayPreprocessor,
    PS3ReportEncoder,
)
from histox.workflows.engines import survival_concordance_index


REFERENCE_CINDEX = {
    0: 0.9181818181818182,
    1: 0.9072164948453608,
    2: 0.7188755020080321,
    3: 0.7,
    4: 0.842391304347826,
}


@dataclass
class SplitData:
    patient_ids: List[str]
    histology: torch.Tensor
    genes: torch.Tensor
    reports: List[torch.Tensor]
    duration: torch.Tensor
    event: torch.Tensor


class MultimodalDataset(Dataset):
    def __init__(self, data: SplitData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "patient_id": self.data.patient_ids[index],
            "histology": self.data.histology[index],
            "genes": self.data.genes[index],
            "report": self.data.reports[index],
            "duration": self.data.duration[index],
            "event": self.data.event[index],
        }


class EventSafeBatchSampler(Sampler[List[int]]):
    """Random batches with a minimal swap that guarantees one event per batch."""

    def __init__(self, events: torch.Tensor, batch_size: int) -> None:
        self.events = torch.as_tensor(events).reshape(-1).bool()
        self.batch_size = int(batch_size)
        self.n_batches = int(math.ceil(len(self.events) / self.batch_size))
        if int(self.events.sum()) < self.n_batches:
            raise ValueError("Not enough events to place one in every Cox batch")

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        order = torch.randperm(len(self.events)).tolist()
        batches = [
            order[start:start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]
        event_counts = [sum(bool(self.events[index]) for index in batch) for batch in batches]
        for empty_index, count in enumerate(event_counts):
            if count:
                continue
            donor_index = next(index for index, donor_count in enumerate(event_counts) if donor_count > 1)
            donor_position = next(
                position for position, sample_index in enumerate(batches[donor_index])
                if bool(self.events[sample_index])
            )
            empty_position = next(
                position for position, sample_index in enumerate(batches[empty_index])
                if not bool(self.events[sample_index])
            )
            batches[donor_index][donor_position], batches[empty_index][empty_position] = (
                batches[empty_index][empty_position],
                batches[donor_index][donor_position],
            )
            event_counts[donor_index] -= 1
            event_counts[empty_index] += 1
        yield from batches


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def patient_table(
    split_dir: Path,
    split: str,
    rna: pd.DataFrame,
    patient_column: str,
) -> pd.DataFrame:
    table = pd.read_csv(split_dir / f"{split}.csv")
    if patient_column not in table.columns:
        raise ValueError(
            f"{split}.csv does not contain patient column {patient_column!r}"
        )
    table[patient_column] = table[patient_column].astype(str)
    table = table.drop_duplicates(patient_column).set_index(patient_column)
    patient_ids = sorted(table.index)
    missing = [patient_id for patient_id in patient_ids if patient_id not in rna.index]
    if missing:
        raise ValueError(
            f"{split}.csv has {len(missing)} patients missing from RNA; "
            f"first: {missing[:5]}"
        )
    return table.loc[patient_ids]


def prepare_fold(args: argparse.Namespace, fold: int):
    split_dir = args.split_root / args.fold_pattern.format(fold=fold)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Fold directory does not exist: {split_dir}")
    rna = pd.read_csv(args.rna, index_col=0)
    rna.index = rna.index.astype(str)
    signatures = pd.read_csv(args.signatures)
    train_table = patient_table(
        split_dir, "train", rna, args.patient_column
    )
    test_table = patient_table(
        split_dir, "test", rna, args.patient_column
    )
    train_ids = list(train_table.index)
    test_ids = list(test_table.index)
    overlap = sorted(set(train_ids).intersection(test_ids))
    if overlap:
        raise ValueError(
            f"Fold {fold} leaks {len(overlap)} patients; first: {overlap[:5]}"
        )

    pathway = PS3PathwayPreprocessor().fit(rna.loc[train_ids], signatures)
    train_genes = pathway.transform(rna.loc[train_ids])
    test_genes = pathway.transform(rna.loc[test_ids])

    def labels(table: pd.DataFrame):
        duration = torch.as_tensor(
            table[args.duration_column].to_numpy(), dtype=torch.float32
        )
        if args.event_column:
            event_values = table[args.event_column].to_numpy()
        else:
            event_values = 1 - table[args.censorship_column].to_numpy()
        event = torch.as_tensor(event_values, dtype=torch.float32)
        return duration, event

    train_duration, train_event = labels(train_table)
    test_duration, test_event = labels(test_table)

    if args.histology_bag_root is None:
        embedding_pattern = Path(args.embedding_glob).expanduser()
        if embedding_pattern.is_absolute():
            embedding_paths = (
                [embedding_pattern.resolve()]
                if embedding_pattern.is_file()
                else []
            )
        else:
            embedding_paths = sorted(split_dir.glob(args.embedding_glob))
        if len(embedding_paths) != 1:
            raise ValueError(
                f"Fold {fold} expected one embedding pickle matching "
                f"{args.embedding_glob!r}, found {len(embedding_paths)}"
            )
        embedding_path = embedding_paths[0]
        embedded = load_pickle(embedding_path)
        train_histology = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
            embedded["train"]["X"], args.n_histology_prototypes
        )
        test_histology = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
            embedded["test"]["X"], args.n_histology_prototypes
        )
        if train_histology.shape[-1] != args.histology_dim:
            raise ValueError(
                f"PANTHER token dimension is {train_histology.shape[-1]}, "
                f"but --histology-dim is {args.histology_dim}"
            )

        def verify_embedding_labels(
            split: str,
            duration: torch.Tensor,
            event: torch.Tensor,
        ) -> None:
            retained = embedded[split]["y"]
            retained_event = np.asarray(
                [row[0] for row in retained], dtype=np.float32
            )
            retained_duration = np.asarray(
                [row[1] for row in retained], dtype=np.float32
            )
            if (
                not np.array_equal(retained_event, event.numpy())
                or not np.allclose(retained_duration, duration.numpy())
            ):
                raise RuntimeError(
                    f"Fold {fold} {split}: retained histology rows do not "
                    "match patient labels"
                )

        verify_embedding_labels("train", train_duration, train_event)
        verify_embedding_labels("test", test_duration, test_event)
        histology_source = str(embedding_path)
        histology_sha256 = sha256(embedding_path)
    else:
        cache_path = args.output / "prepared_histology" / f"fold_{fold}.pt"
        if cache_path.exists() and not args.rebuild_histology:
            cached = torch.load(cache_path, map_location="cpu")
            if cached["train_ids"] != train_ids or cached["test_ids"] != test_ids:
                raise RuntimeError(
                    f"Fold {fold}: cached PANTHER patient order does not "
                    "match the current split"
                )
            train_histology = cached["train_histology"].float()
            test_histology = cached["test_histology"].float()
        else:
            def load_bags(patient_ids: Sequence[str]) -> List[torch.Tensor]:
                bags = []
                for patient_id in patient_ids:
                    path = args.histology_bag_root / f"{patient_id}.pt"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing histology bag: {path}")
                    bags.append(torch.load(path, map_location="cpu").float())
                return bags

            train_bags = load_bags(train_ids)
            test_bags = load_bags(test_ids)
            histology_encoder = PS3HistologyPrototypeEncoder(
                n_prototypes=args.n_histology_prototypes,
                patches_per_prototype=args.patches_per_prototype,
                max_iter=args.prototype_max_iter,
                n_init=args.prototype_n_init,
                em_iterations=1,
                tau=1.0,
                covariance_prior=1.0,
                random_state=args.prototype_seed,
            ).fit(train_bags)
            if histology_encoder.output_dim != args.histology_dim:
                raise ValueError(
                    f"PANTHER output dimension is {histology_encoder.output_dim}, "
                    f"but --histology-dim is {args.histology_dim}"
                )
            train_histology = histology_encoder.transform_many(train_bags)
            test_histology = histology_encoder.transform_many(test_bags)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "train_ids": train_ids,
                "test_ids": test_ids,
                "train_histology": train_histology,
                "test_histology": test_histology,
                "centers": histology_encoder.centers.cpu(),
            }, cache_path)
        histology_source = str(cache_path)
        histology_sha256 = sha256(cache_path)

    train_reports = [
        torch.load(args.report_root / f"{patient_id}.pt", map_location="cpu").float()
        for patient_id in train_ids
    ]
    test_reports = [
        torch.load(args.report_root / f"{patient_id}.pt", map_location="cpu").float()
        for patient_id in test_ids
    ]
    train_max_segments = max(int(report.shape[0]) for report in train_reports)
    test_max_segments = max(int(report.shape[0]) for report in test_reports)
    train = SplitData(train_ids, train_histology, train_genes, train_reports, train_duration, train_event)
    test = SplitData(test_ids, test_histology, test_genes, test_reports, test_duration, test_event)
    metadata = {
        "fold": fold,
        "histology_source": histology_source,
        "histology_sha256": histology_sha256,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_events": int(train_event.sum().item()),
        "test_events": int(test_event.sum().item()),
        "train_max_report_segments": train_max_segments,
        "test_max_report_segments": test_max_segments,
        "report_padding": (
            "train batches pad to at least the training-fold maximum; "
            "evaluation batches extend to their longest report without truncation"
        ),
        "n_genes": len(pathway.gene_names),
        "n_pathways": len(pathway.pathway_gene_indices),
        "reference_c_index": (
            REFERENCE_CINDEX[fold]
            if args.reference_profile == "coadread"
            else None
        ),
    }
    return train, test, pathway, metadata


def make_collate(pad_to: int):
    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        reports, report_mask = PS3ReportEncoder.collate(
            [item["report"] for item in batch],
            pad_to=max(
                pad_to,
                max(int(item["report"].shape[0]) for item in batch),
            ),
        )
        return {
            "patient_id": [str(item["patient_id"]) for item in batch],
            "histology": torch.stack([item["histology"] for item in batch]),
            "rna": torch.stack([item["genes"] for item in batch]),
            "report": reports,
            "report_mask": report_mask,
            "duration": torch.stack([item["duration"] for item in batch]),
            "event": torch.stack([item["event"] for item in batch]),
        }

    return collate


def model_config(
    pathway: PS3PathwayPreprocessor,
    dropout: float,
    n_histology_prototypes: int,
    histology_dim: int,
    report_selection_mode: str,
    report_representation_mode: str,
    report_residual_scale: float,
    prototype_embedding_mode: str,
    fusion_residual: bool,
) -> Dict[str, object]:
    return {
        "n_genes": len(pathway.gene_names),
        "pathway_gene_indices": pathway.pathway_gene_indices,
        "histology_dim": histology_dim,
        "report_dim": 512,
        "n_histology_prototypes": n_histology_prototypes,
        "n_report_prototypes": 10,
        "embedding_dim": 256,
        "modality_embedding_dim": 32,
        "n_fusion_layers": 1,
        "feedforward_dim": 512,
        "dropout": dropout,
        "report_selection_mode": report_selection_mode,
        "report_representation_mode": report_representation_mode,
        "report_residual_scale": report_residual_scale,
        "prototype_embedding_mode": prototype_embedding_mode,
        "fusion_residual": fusion_residual,
    }


def make_experiment(
    pathway: PS3PathwayPreprocessor,
    args: argparse.Namespace,
    output: Path,
    dropout: float,
) -> Experiment:
    """Bind the prepared cohort schema to the registered PS3 method."""
    return Experiment(
        task="survival",
        method="ps3",
        data_spec=DataSpec(
            modalities=("histology", "report", "rna"),
            fields={
                "patient_id": args.patient_column,
                "duration": args.duration_column,
            "event": (
                args.event_column
                or "1 - {}".format(args.censorship_column)
            ),
            },
        ),
        method_config=model_config(
            pathway,
            dropout=dropout,
            n_histology_prototypes=args.n_histology_prototypes,
            histology_dim=args.histology_dim,
            report_selection_mode=args.report_selection_mode,
            report_representation_mode=args.report_representation_mode,
            report_residual_scale=args.report_residual_scale,
            prototype_embedding_mode=args.prototype_embedding_mode,
            fusion_residual=args.fusion_residual,
        ),
        output=output,
        metadata={"cohort": args.fold_pattern},
    )


def validate_reference_metrics(reference_results: Path) -> Dict[int, float]:
    """Recompute every retained reference score before using the local metric."""
    recomputed = {}
    for fold, expected in REFERENCE_CINDEX.items():
        matches = sorted(
            (reference_results / f"k={fold}").glob("*coattn_text*")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Fold {fold} expected one retained result directory, "
                f"found {len(matches)}"
            )
        result_dir = matches[0]
        prediction = load_pickle(result_dir / "test_results.pkl")
        observed = survival_concordance_index(
            1 - np.asarray(prediction["all_censorships"]).astype(int),
            np.asarray(prediction["all_event_times"]),
            np.asarray(prediction["all_risk_scores"]),
        )
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"C-index parity failed for fold {fold}: {observed} != {expected}"
            )
        recomputed[fold] = observed
    return recomputed


def run_overfit(
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
    steps: int,
) -> None:
    train, _, pathway, metadata = prepare_fold(args, fold)
    event_indices = torch.where(train.event > 0)[0][:8]
    censored_indices = torch.where(train.event == 0)[0][:24]
    selected = torch.cat((event_indices, censored_indices))
    subset = SplitData(
        [train.patient_ids[index] for index in selected.tolist()],
        train.histology[selected],
        train.genes[selected],
        [train.reports[index] for index in selected.tolist()],
        train.duration[selected],
        train.event[selected],
    )
    loader = DataLoader(
        MultimodalDataset(subset),
        batch_size=len(subset.patient_ids),
        shuffle=False,
        collate_fn=make_collate(metadata["train_max_report_segments"]),
    )
    experiment = make_experiment(
        pathway,
        args,
        output=args.output / "overfit" / f"fold_{fold}",
        dropout=0.0,
    )
    model = experiment.build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def report_step(row: Dict[str, object]) -> None:
        step = int(row["epoch"])
        if step == 1 or step % 10 == 0 or step == steps:
            score = experiment.evaluate(loader=loader, device=device)
            print(
                f"overfit step={step}/{steps} "
                f"loss={row['mean_batch_loss']:.6f} "
                f"c_index={score.metrics['c_index']:.6f}",
                flush=True,
            )

    fitted = experiment.fit(
        train_loader=loader,
        epochs=steps,
        evaluation_loaders={"train": loader},
        device=device,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer=optimizer,
        scheduler=False,
        progress=report_step,
        save_outputs=False,
    )
    history = fitted.metadata["history"]
    print(json.dumps({
        "mode": "overfit",
        "fold": fold,
        "n": len(subset.patient_ids),
        "events": int(subset.event.sum().item()),
        "steps": steps,
        "initial_loss": history[0]["mean_batch_loss"],
        "final_loss": fitted.metrics["final_loss"],
        "final_c_index": fitted.metrics["train_c_index"],
    }, indent=2))


def train_fold(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    seed_everything(seed)
    fold_start = time.time()
    train, test, pathway, metadata = prepare_fold(args, fold)
    fold_dir = args.output / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        MultimodalDataset(train),
        batch_sampler=EventSafeBatchSampler(train.event, args.batch_size),
        num_workers=0,
        collate_fn=make_collate(metadata["train_max_report_segments"]),
    )
    train_eval_loader = DataLoader(
        MultimodalDataset(train),
        batch_size=len(train.patient_ids),
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(metadata["train_max_report_segments"]),
    )
    test_loader = DataLoader(
        MultimodalDataset(test),
        batch_size=len(test.patient_ids),
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(metadata["train_max_report_segments"]),
    )

    experiment = make_experiment(
        pathway,
        args,
        output=fold_dir,
        dropout=args.dropout,
    )
    model = experiment.build_model()
    print(
        f"fold={fold} start train={len(train.patient_ids)} test={len(test.patient_ids)} "
        f"events={metadata['train_events']}/{metadata['test_events']} params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    def report_epoch(row: Dict[str, object]) -> None:
        epoch = int(row["epoch"])
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"fold={fold} epoch={epoch}/{args.epochs} "
                f"loss={row['mean_batch_loss']:.6f} "
                f"lr={row['learning_rate']:.8f}",
                flush=True,
            )

    fitted = experiment.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        evaluation_loaders={
            "train": train_eval_loader,
            "test": test_loader,
        },
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        warmup_steps=len(train_loader),
        progress=report_epoch,
    )
    checkpoint_path = Path(fitted.artifacts["checkpoint"].path)

    metrics = {
        **metadata,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "optimizer": "AdamW with no decay on bias/norm/1D parameters",
        "scheduler": "cosine with one warmup epoch",
        "report_selection_mode": args.report_selection_mode,
        "report_representation_mode": args.report_representation_mode,
        "report_residual_scale": args.report_residual_scale,
        "prototype_embedding_mode": args.prototype_embedding_mode,
        "fusion_residual": args.fusion_residual,
        "train_c_index": fitted.metrics["train_c_index"],
        "test_c_index": fitted.metrics["test_c_index"],
        "difference_from_reference": (
            fitted.metrics["test_c_index"] - REFERENCE_CINDEX[fold]
            if args.reference_profile == "coadread"
            else None
        ),
        "skipped_zero_event_batches": 0,
        "batching": "random batches with minimal event-preserving swaps",
        "elapsed_seconds": time.time() - fold_start,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_folds(value: str) -> List[int]:
    folds = [int(item) for item in value.split(",")]
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be comma-separated values from 0 to 4")
    return folds


def parse_seeds(value: str) -> List[int]:
    seeds = [int(item) for item in value.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique comma-separated integers")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--fold-pattern",
        default="TCGA_COADREAD_dss_survival_k={fold}",
        help="Directory pattern below --split-root; must contain {fold}.",
    )
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--embedding-glob",
        default="embeddings/*.pkl",
        help=(
            "Fold-relative glob resolving to one retained PANTHER pickle; "
            "ignored when --histology-bag-root is supplied."
        ),
    )
    parser.add_argument(
        "--histology-bag-root",
        type=Path,
        help=(
            "Directory containing patient_id.pt PLIP patch bags created by "
            "ps3_crc_prepare_inputs.py. Enables fold-specific PANTHER fitting."
        ),
    )
    parser.add_argument("--patient-column", default="case_id")
    parser.add_argument("--duration-column", default="dss_survival_days")
    parser.add_argument(
        "--event-column",
        help="Observed-event column. If omitted, use 1 - --censorship-column.",
    )
    parser.add_argument("--censorship-column", default="dss_censorship")
    parser.add_argument(
        "--report-selection-mode",
        choices=("paper", "crc_reproduction", "traceable"),
        default="crc_reproduction",
        help=(
            "Use paper for interpolation, crc_reproduction for the validated "
            "COADREAD behavior, or traceable to select only real segments."
        ),
    )
    parser.add_argument(
        "--prototype-embedding-mode",
        choices=("modality", "prototype"),
        default="modality",
        help=(
            "Use one learned embedding per prototype position (reference PS3 "
            "behavior) or one shared embedding per modality."
        ),
    )
    parser.add_argument(
        "--report-representation-mode",
        choices=("attended", "residual"),
        default="attended",
        help=(
            "Use post-attention report prototypes, or add each selected "
            "source-segment embedding through a residual connection."
        ),
    )
    parser.add_argument(
        "--report-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the source-segment residual in residual report "
            "representation mode."
        ),
    )
    residual_group = parser.add_mutually_exclusive_group()
    residual_group.add_argument(
        "--fusion-residual",
        action="store_true",
        dest="fusion_residual",
        help="Add each fusion layer input to its attention output.",
    )
    residual_group.add_argument(
        "--no-fusion-residual",
        action="store_false",
        dest="fusion_residual",
        help="Disable the attention residual, matching the retained CRC setup.",
    )
    parser.set_defaults(fusion_residual=True)
    parser.add_argument(
        "--reference-profile",
        choices=("none", "coadread"),
        default="coadread",
        help=(
            "Attach the retained COADREAD fold scores to outputs, or none "
            "for a cohort without that reference."
        ),
    )
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument(
        "--trust-input-files",
        action="store_true",
        help="Acknowledge that pickle and .pt inputs can execute code when loaded.",
    )
    parser.add_argument("--mode", choices=("overfit", "train"), default="train")
    parser.add_argument("--folds", type=parse_folds, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3])
    parser.add_argument("--output", type=Path, default=Path("histox-ps3-crc-results"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-histology-prototypes", type=int, default=16)
    parser.add_argument("--histology-dim", type=int, default=1025)
    parser.add_argument("--patches-per-prototype", type=int, default=100000)
    parser.add_argument("--prototype-max-iter", type=int, default=50)
    parser.add_argument("--prototype-n-init", type=int, default=3)
    parser.add_argument("--prototype-seed", type=int, default=1)
    parser.add_argument("--rebuild-histology", action="store_true")
    parser.add_argument("--device", help="Torch device, for example cuda, cuda:1, or cpu.")
    parser.add_argument("--overfit-steps", type=int, default=60)
    args = parser.parse_args()
    if not args.trust_input_files:
        parser.error(
            "retained pickle/.pt artifacts are unsafe when untrusted; inspect "
            "their provenance, then pass --trust-input-files"
        )
    if "{fold}" not in args.fold_pattern:
        parser.error("--fold-pattern must contain {fold}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.overfit_steps <= 0:
        parser.error("epoch, batch-size, and overfit-step values must be positive")
    if args.n_histology_prototypes <= 0 or args.histology_dim <= 0:
        parser.error("histology dimensions must be positive")
    if not math.isfinite(args.report_residual_scale) or args.report_residual_scale <= 0:
        parser.error("--report-residual-scale must be a positive finite number")
    if (
        args.patches_per_prototype <= 0
        or args.prototype_max_iter <= 0
        or args.prototype_n_init <= 0
    ):
        parser.error("prototype preparation values must be positive")

    for name in ("split_root", "report_root"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_dir():
            parser.error(f"--{name.replace('_', '-')} is not a directory: {path}")
        setattr(args, name, path)
    if args.histology_bag_root is not None:
        args.histology_bag_root = args.histology_bag_root.expanduser().resolve()
        if not args.histology_bag_root.is_dir():
            parser.error(
                "--histology-bag-root is not a directory: "
                f"{args.histology_bag_root}"
            )
    for name in ("rna", "signatures"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{name} is not a file: {path}")
        setattr(args, name, path)
    if args.reference_results is not None:
        if args.reference_profile != "coadread":
            parser.error(
                "--reference-results requires --reference-profile coadread"
            )
        args.reference_results = args.reference_results.expanduser().resolve()
        if not args.reference_results.is_dir():
            parser.error(
                f"--reference-results is not a directory: {args.reference_results}"
            )

    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seeds[0])
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"device={device} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if args.reference_results is not None:
        print(
            "reference_metric_parity="
            + json.dumps(
                validate_reference_metrics(args.reference_results), sort_keys=True
            ),
            flush=True,
        )

    if args.mode == "overfit":
        run_overfit(args, args.folds[0], device, args.overfit_steps)
        return

    seed_summaries = []
    for seed in args.seeds:
        metrics = [
            train_fold(args, fold, seed, device)
            for fold in args.folds
        ]
        fold_scores = [float(row["test_c_index"]) for row in metrics]
        summary = {
            "seed": seed,
            "folds": args.folds,
            "test_c_index": fold_scores,
            "mean_test_c_index": float(np.mean(fold_scores)),
            "population_sd_test_c_index": float(np.std(fold_scores, ddof=0)),
            "reference_test_c_index": (
                [REFERENCE_CINDEX[fold] for fold in args.folds]
                if args.reference_profile == "coadread"
                else None
            ),
            "reference_mean_test_c_index": (
                float(np.mean([REFERENCE_CINDEX[fold] for fold in args.folds]))
                if args.reference_profile == "coadread"
                else None
            ),
        }
        seed_dir = args.output / f"seed_{seed}"
        (seed_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        seed_summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    seed_means = [row["mean_test_c_index"] for row in seed_summaries]
    aggregate = {
        "seeds": args.seeds,
        "folds": args.folds,
        "seed_summaries": seed_summaries,
        "mean_across_seed_means": float(np.mean(seed_means)),
        "sample_sd_across_seed_means": (
            float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
