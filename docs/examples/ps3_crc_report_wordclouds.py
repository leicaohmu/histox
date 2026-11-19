"""Create PS3 risk-stratified pathology-report word clouds.

The script interprets every COADREAD patient with the checkpoint from that
patient's held-out fold, maps the most histology-attended report prototype
back to its source PLIP text segment, and compares phrases below and above the
out-of-fold median predicted risk.

Example
-------
python docs/examples/ps3_crc_report_wordclouds.py \
    --validation-root runs/histox-ps3-validation \
    --split-root data/ps3_crc/splits \
    --rna data/ps3_crc/rna_clean.csv \
    --signatures data/ps3_crc/hallmarks_signatures.csv \
    --reports-csv data/ps3_crc/TCGA_Reports.csv \
    --report-embeddings data/ps3_crc/report_embeddings \
    --output runs/ps3_report_words \
    --trust-inputs
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from transformers import CLIPProcessor
from wordcloud import WordCloud

from histox.mil import (
    PS3HistologyPrototypeEncoder,
    PS3PathwayPreprocessor,
    PS3ReportEncoder,
)
from histox.mil.models import PS3SurvivalModel


CASE_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
FOLD_PATTERN = "TCGA_COADREAD_dss_survival_k={fold}"
CLINICAL_STOP_WORDS = {
    "additional", "also", "black", "block", "case", "clinical", "cm", "comments",
    "copy", "date", "description", "diagnosis", "examined", "examination", "gross",
    "history", "material", "microscopic", "mm", "page", "part", "received", "red",
    "report", "representative", "section", "sectioned", "sections", "specified",
    "specimen", "submitted", "summary", "tan", "white", "yellow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--reports-csv", type=Path, required=True)
    parser.add_argument("--report-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="vinid/plip")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-phrases", type=int, default=70)
    parser.add_argument("--min-document-frequency", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument(
        "--trust-inputs",
        action="store_true",
        help="Allow loading the retained PANTHER pickle files.",
    )
    return parser.parse_args()


def read_split(path: Path, rna: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["case_id"] = table["case_id"].astype(str)
    if table["case_id"].duplicated().any():
        label_columns = ["dss_survival_days", "dss_censorship"]
        conflicting = (
            table.groupby("case_id")[label_columns].nunique(dropna=False) > 1
        ).any(axis=1)
        if conflicting.any():
            raise ValueError(
                "Conflicting patient labels in {}; first: {}".format(
                    path, conflicting[conflicting].index[:5].tolist()
                )
            )
        table = table.drop_duplicates("case_id")
    table = table.set_index("case_id")
    missing = sorted(set(table.index) - set(rna.index))
    if missing:
        raise ValueError(
            "{} patients from {} are missing from RNA; first: {}".format(
                len(missing), path, missing[:5]
            )
        )
    return table.loc[sorted(table.index)]


def load_raw_reports(path: Path) -> pd.DataFrame:
    reports = pd.read_csv(path)
    required = {"patient_filename", "text"}
    if not required.issubset(reports.columns):
        raise ValueError(
            "reports CSV must contain patient_filename and text columns"
        )
    reports["case_id"] = reports["patient_filename"].astype(str).str.extract(
        CASE_PATTERN, expand=False
    )
    reports = reports.dropna(subset=["case_id", "text"])
    if reports["case_id"].duplicated().any():
        duplicated = reports.loc[
            reports["case_id"].duplicated(keep=False), "case_id"
        ].unique()
        raise ValueError(
            "reports CSV contains duplicate case IDs; first: {}".format(
                duplicated[:5].tolist()
            )
        )
    return reports.set_index("case_id")


def build_model(pathway: PS3PathwayPreprocessor) -> PS3SurvivalModel:
    return PS3SurvivalModel(
        n_genes=len(pathway.gene_names),
        pathway_gene_indices=pathway.pathway_gene_indices,
        histology_dim=1025,
        report_dim=512,
        n_histology_prototypes=16,
        n_report_prototypes=10,
        embedding_dim=256,
        modality_embedding_dim=32,
        n_fusion_layers=1,
        feedforward_dim=512,
        dropout=0.1,
        report_selection_mode="crc_reproduction",
    )


def verify_retained_labels(
    retained_labels: Sequence[Sequence[float]],
    table: pd.DataFrame,
) -> None:
    expected_duration = table["dss_survival_days"].to_numpy(dtype=np.float32)
    expected_event = (
        1 - table["dss_censorship"].to_numpy(dtype=np.float32)
    )
    retained_event = np.asarray(
        [row[0] for row in retained_labels], dtype=np.float32
    )
    retained_duration = np.asarray(
        [row[1] for row in retained_labels], dtype=np.float32
    )
    if not np.array_equal(retained_event, expected_event):
        raise RuntimeError("Retained event labels do not match the test split")
    if not np.allclose(retained_duration, expected_duration):
        raise RuntimeError("Retained durations do not match the test split")


def fold_evidence(
    fold: int,
    args: argparse.Namespace,
    rna: pd.DataFrame,
    signatures: pd.DataFrame,
    raw_reports: pd.DataFrame,
    tokenizer,
) -> Tuple[List[Dict[str, object]], float]:
    fold_dir = args.split_root / FOLD_PATTERN.format(fold=fold)
    result_dir = args.validation_root / "fold_{}".format(fold)
    metrics = json.loads((result_dir / "metrics.json").read_text())
    train = read_split(fold_dir / "train.csv", rna)
    test = read_split(fold_dir / "test.csv", rna)
    pathway = PS3PathwayPreprocessor().fit(rna.loc[train.index], signatures)
    test_genes = pathway.transform(rna.loc[test.index])

    embedding_path = Path(metrics["embedding_path"])
    if not embedding_path.is_file():
        candidates = sorted(fold_dir.glob("embeddings/*.pkl"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected one retained PANTHER file for fold {}".format(fold)
            )
        embedding_path = candidates[0]
    with embedding_path.open("rb") as handle:
        retained = pickle.load(handle)
    verify_retained_labels(retained["test"]["y"], test)
    # These checkpoints were trained from the retained ``allcat`` vectors by
    # reshaping each serialized vector to 16 x 1025. Preserve that exact input
    # layout here so the reloaded model reproduces its saved held-out risks.
    histology = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
        retained["test"]["X"],
        16,
        layout="crc_reproduction",
    )

    report_tensors = [
        torch.load(
            args.report_embeddings / "{}.pt".format(patient_id),
            map_location="cpu",
            weights_only=True,
        ).float()
        for patient_id in test.index
    ]
    pad_to = max(
        int(metrics["train_max_report_segments"]),
        max(int(report.shape[0]) for report in report_tensors),
    )
    padded_reports, report_mask = PS3ReportEncoder.collate(
        report_tensors, pad_to=pad_to
    )

    device = torch.device(args.device)
    model = build_model(pathway).to(device)
    state = torch.load(
        result_dir / "model_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        evidence = model.report_segment_evidence(
            histology_prototypes=histology.to(device),
            gene_expression=test_genes.to(device),
            report_embeddings=padded_reports.to(device),
            report_mask=report_mask.to(device),
        )

    predictions = pd.read_csv(result_dir / "test_predictions.csv")
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions = predictions.set_index("patient_id")
    if set(predictions.index) != set(test.index):
        raise RuntimeError("Saved predictions do not match fold {}".format(fold))
    computed = evidence["risk"].detach().cpu().reshape(-1).numpy()
    saved = predictions.loc[test.index, "log_risk"].to_numpy(dtype=np.float64)
    maximum_risk_difference = float(np.max(np.abs(computed - saved)))
    if not np.allclose(computed, saved, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "Fold {} risks differ from saved predictions (max {:.3g})".format(
                fold, maximum_risk_difference
            )
        )

    source_indices = evidence["selected_source_segment"].cpu().tolist()
    prototype_indices = evidence["selected_report_prototype"].cpu().tolist()
    attention_scores = evidence["selected_score"].cpu().tolist()
    valid_prototype_counts = evidence[
        "report_prototype_source_valid"
    ].sum(dim=1).cpu().tolist()

    rows: List[Dict[str, object]] = []
    for row_index, patient_id in enumerate(test.index):
        if patient_id not in raw_reports.index:
            raise KeyError("Raw report missing for {}".format(patient_id))
        report_text = str(raw_reports.at[patient_id, "text"])
        segments = PS3ReportEncoder.segment_text(
            report_text,
            tokenizer=tokenizer,
            max_tokens=77,
        )
        expected_segments = int(report_tensors[row_index].shape[0])
        if len(segments) != expected_segments:
            raise RuntimeError(
                "{} has {} source segments but {} saved embeddings".format(
                    patient_id, len(segments), expected_segments
                )
            )
        source_index = int(source_indices[row_index])
        rows.append({
            "patient_id": patient_id,
            "fold": fold,
            "duration": float(predictions.at[patient_id, "duration"]),
            "event": int(predictions.at[patient_id, "event"]),
            "log_risk": float(saved[row_index]),
            "report_segment_count": len(segments),
            "traceable_report_prototypes": int(
                valid_prototype_counts[row_index]
            ),
            "selected_report_prototype": int(prototype_indices[row_index]),
            "selected_report_prototype_label": "R{}".format(
                int(prototype_indices[row_index])
            ),
            "source_segment_index": source_index,
            "attention_score": float(attention_scores[row_index]),
            "selected_text": segments[source_index],
        })
    return rows, maximum_risk_difference


def phrase_table(
    evidence: pd.DataFrame,
    min_document_frequency: int,
) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=None,
        ngram_range=(2, 2),
        min_df=min_document_frequency,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    matrix = vectorizer.fit_transform(evidence["selected_text"].fillna(""))
    phrases = vectorizer.get_feature_names_out()
    # Filter after constructing contiguous bigrams. Removing stop words before
    # n-gram construction can create phrases that never occurred in the
    # report, and dropping ``no``/``not`` would erase clinical negation.
    removable = set(ENGLISH_STOP_WORDS) - {"no", "not", "none", "without"}
    negation = {"no", "not", "none", "without"}
    keep = np.asarray([
        not any(
            token in removable or token in CLINICAL_STOP_WORDS
            for token in phrase.split()
        )
        and phrase.split()[-1] not in negation
        for phrase in phrases
    ])
    matrix = matrix[:, keep]
    phrases = phrases[keep]
    records: List[Dict[str, object]] = []
    for group in ("low", "high"):
        group_mask = evidence["risk_group"].eq(group).to_numpy()
        group_matrix = matrix[group_mask]
        counts = np.asarray(group_matrix.sum(axis=0)).reshape(-1)
        document_counts = np.asarray((group_matrix > 0).sum(axis=0)).reshape(-1)
        patients = int(group_mask.sum())
        for index in np.flatnonzero(counts):
            records.append({
                "risk_group": group,
                "phrase": str(phrases[index]),
                "count": int(counts[index]),
                "document_frequency": int(document_counts[index]),
                "patients": patients,
                "count_per_patient": float(counts[index] / patients),
                "document_fraction": float(document_counts[index] / patients),
                "n_words": 2,
            })
    return pd.DataFrame.from_records(records).sort_values(
        ["risk_group", "count", "document_frequency", "phrase"],
        ascending=[True, False, False, True],
    )


def save_wordclouds(
    frequencies: pd.DataFrame,
    destination: Path,
    max_phrases: int,
    random_state: int,
) -> None:
    colors = {"low": "#1d4ed8", "high": "#dc2626"}
    titles = {"low": "Low predicted risk", "high": "High predicted risk"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    for axis, group in zip(axes, ("low", "high")):
        current = frequencies[frequencies["risk_group"].eq(group)].head(
            max_phrases
        )
        values = dict(zip(current["phrase"], current["count_per_patient"]))
        cloud = WordCloud(
            width=1500,
            height=650,
            background_color="white",
            max_words=max_phrases,
            prefer_horizontal=1.0,
            relative_scaling=0.5,
            random_state=random_state,
            collocations=False,
            color_func=lambda *unused, color=colors[group], **kwargs: color,
        ).generate_from_frequencies(values)
        axis.imshow(cloud, interpolation="bilinear")
        patients = int(current["patients"].iloc[0])
        axis.set_title("{} (n={})".format(titles[group], patients), fontsize=16)
        axis.axis("off")
    figure.suptitle(
        "Pathology-report phrases selected by PS3 histology-to-report attention",
        fontsize=17,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.trust_inputs:
        raise PermissionError(
            "Retained histology inputs are pickle files. Re-run with "
            "--trust-inputs only after verifying their source."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rna = pd.read_csv(args.rna, index_col=0)
    rna.index = rna.index.astype(str)
    signatures = pd.read_csv(args.signatures)
    raw_reports = load_raw_reports(args.reports_csv)
    processor = CLIPProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )

    rows: List[Dict[str, object]] = []
    risk_differences: Dict[str, float] = {}
    for fold in range(args.folds):
        current, difference = fold_evidence(
            fold, args, rna, signatures, raw_reports, processor.tokenizer
        )
        rows.extend(current)
        risk_differences[str(fold)] = difference
        print(
            "fold={} patients={} max_risk_difference={:.3g}".format(
                fold, len(current), difference
            )
        )

    evidence = pd.DataFrame.from_records(rows).sort_values("patient_id")
    if evidence["patient_id"].duplicated().any():
        raise RuntimeError("A patient appears in more than one held-out fold")
    median_risk = float(evidence["log_risk"].median())
    evidence["risk_group"] = np.where(
        evidence["log_risk"] <= median_risk, "low", "high"
    )
    evidence_path = args.output / "ps3_report_segment_evidence.csv"
    evidence.to_csv(evidence_path, index=False)

    frequencies = phrase_table(
        evidence,
        min_document_frequency=args.min_document_frequency,
    )
    frequency_path = args.output / "ps3_report_phrase_frequencies.csv"
    frequencies.to_csv(frequency_path, index=False)
    figure_path = args.output / "ps3_report_wordclouds.png"
    save_wordclouds(
        frequencies,
        destination=figure_path,
        max_phrases=args.max_phrases,
        random_state=args.random_state,
    )

    summary = {
        "patients": int(len(evidence)),
        "folds": int(args.folds),
        "median_log_risk": median_risk,
        "risk_group_rule": "low <= median; high > median",
        "risk_group_counts": {
            name: int(value)
            for name, value in evidence["risk_group"].value_counts().items()
        },
        "maximum_saved_risk_difference_by_fold": risk_differences,
        "reports_with_fewer_than_10_segments": int(
            evidence["report_segment_count"].lt(10).sum()
        ),
        "phrase_min_document_frequency": int(
            args.min_document_frequency
        ),
        "phrase_ngram_range": [2, 2],
        "histology_layout": "crc_reproduction",
        "wordcloud_random_state": int(args.random_state),
        "outputs": {
            "evidence": str(evidence_path),
            "phrase_frequencies": str(frequency_path),
            "figure": str(figure_path),
        },
    }
    summary_path = args.output / "ps3_report_wordcloud_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("patients={}".format(len(evidence)))
    print("median_log_risk={:.9f}".format(median_risk))
    print(evidence["risk_group"].value_counts().sort_index().to_string())
    for group in ("low", "high"):
        top = frequencies[frequencies["risk_group"].eq(group)].head(12)
        print("\n{}-risk top phrases".format(group))
        print(top[["phrase", "count", "document_frequency"]].to_string(index=False))
    print("\n{}".format(summary_path))


if __name__ == "__main__":
    main()
