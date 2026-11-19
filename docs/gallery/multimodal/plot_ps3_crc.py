"""
Train PS3 with the HistoX workflow
==================================

This tutorial follows one patient-level TCGA COADREAD fold from prepared
multimodal features to training, held-out evaluation, saved predictions, and
joint prototype attention. It uses the public HistoX ``Experiment`` lifecycle
throughout; there is no second training loop hidden behind the notebook.

Start here when PANTHER histology tensors, PLIP report embeddings, an RNA
table, pathway signatures, and patient-level split files are ready. To create
those inputs from WSI files and report text, first read
:doc:`Prepare multimodal inputs for PS3
</tutorials/ps3_data_preparation>`.

By the end you will have:

* checked one patient-level split;
* fitted RNA preprocessing on training patients only;
* built a PyTorch ``Dataset`` and event-aware ``DataLoader``;
* created the registered survival task and PS3 method;
* trained and evaluated one fold with the shared survival engine;
* reloaded the checkpoint and exported joint prototype attention;
* generated a categorical prototype map, representative tissue patches,
  prototype proportions, and cross-modal pathway/report rankings for one WSI;
  and
* compared histology-attended report phrases between out-of-fold low- and
  high-risk patients.

The executed example uses fold 0, seed 1, and 50 epochs. Its held-out
DSS C-index is ``0.890909``. The separate
:doc:`COADREAD reproduction page
</tutorials/ps3_coadread_reproduction>` records all five folds and three seeds.

.. note:: Data boundary

   TCGA files and retained feature tensors are not distributed with HistoX.
   Obtain them under their original access terms. This tutorial begins at the
   prepared-feature boundary so the learning path stays focused on the model
   workflow.

.. container:: histox-example-meta

   **Level:** Advanced

   **Example:** TCGA COADREAD, fold 0, seed 1

   **Inputs:** Prepared histology, RNA, and report features

   **Output:** Log-risk, C-index, checkpoint, predictions, prototype map, representative patches, cross-modal attention, and risk-stratified report phrases

.. container:: histox-example-flow

   ``Prepared multimodal features -> Experiment -> PS3 -> held-out evaluation``
"""

# %%
# Install and import
# ------------------
#
# Install the current HistoX source checkout, then restart the Jupyter kernel.
# ``transformers`` supplies the cosine learning-rate schedule used below.
#
# .. code-block:: bash
#
#    python -m pip install -e ".[torch,huggingface]" "wordcloud>=1.9,<2"

from __future__ import annotations

import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import histox as hx
from histox.mil import (
    PS3HistologyPrototypeEncoder,
    PS3PathwayPreprocessor,
    PS3ReportEncoder,
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable training run."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


SEED = 1
seed_everything(SEED)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("HistoX:", Path(hx.__file__).resolve())
print("CUDA available:", torch.cuda.is_available())

# %%
# 1. Configure the cohort
# ------------------------
#
# A fold directory contains ``train.csv``, ``test.csv``, and one retained
# PANTHER file below ``embeddings/``. Report features are one ``.pt`` tensor
# per patient. The RNA table and split files use the same patient identifiers.
#
# Edit these paths for your workstation or server. Keeping them together in
# one cell makes the notebook portable without changing later sections.

DATA_ROOT = Path("data/ps3_crc").expanduser()
SPLIT_ROOT = DATA_ROOT / "splits"
RNA_PATH = DATA_ROOT / "rna_clean.csv"
SIGNATURE_PATH = DATA_ROOT / "hallmarks_signatures.csv"
REPORT_ROOT = DATA_ROOT / "report_embeddings"
RUN_ROOT = Path("runs/ps3_tutorial").expanduser()

FOLD_PATTERN = "TCGA_COADREAD_dss_survival_k={fold}"
EMBEDDING_GLOB = "embeddings/*.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = pd.Series(
    {
        "split_root": SPLIT_ROOT,
        "rna": RNA_PATH,
        "pathway_signatures": SIGNATURE_PATH,
        "report_embeddings": REPORT_ROOT,
        "results": RUN_ROOT,
        "device": DEVICE,
    },
    name="configured value",
)
print(paths.to_string())

# %%
# Expected directory layout
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: text
#
#    data/ps3_crc/
#    |-- rna_clean.csv
#    |-- hallmarks_signatures.csv
#    |-- report_embeddings/{patient_id}.pt
#    `-- splits/
#        |-- TCGA_COADREAD_dss_survival_k=0/
#        |   |-- train.csv
#        |   |-- test.csv
#        |   `-- embeddings/*.pkl
#        |-- ...
#        `-- TCGA_COADREAD_dss_survival_k=4/...
#
# Each split CSV contains ``case_id``, ``dss_survival_days``, and
# ``dss_censorship``. For this cohort, ``dss_censorship=0`` means that the
# disease-specific death was observed; the model therefore uses
# ``event = 1 - dss_censorship``.

# %%
# 2. Load and prepare one real fold
# ----------------------------------
#
# We use fold 0 to explain every tensor passed to PyTorch. Preprocessing is
# fitted on its 214 training patients only. The 54 validation patients are
# transformed with those training-fold statistics.
#
# Pickle and PyTorch files can execute code while loading. Change
# ``TRUST_INPUTS`` to ``True`` only after checking where the files came from.

TRUST_INPUTS = False
FOLD = 0

data_ready = all(path.is_file() for path in (
    RNA_PATH,
    SIGNATURE_PATH,
)) and SPLIT_ROOT.is_dir() and REPORT_ROOT.is_dir()


@dataclass
class PS3Split:
    """The six patient-aligned objects needed by the training loop."""

    patient_ids: List[str]
    histology: torch.Tensor
    genes: torch.Tensor
    reports: List[torch.Tensor]
    duration: torch.Tensor
    event: torch.Tensor


def read_split_table(
    fold_dir: Path,
    name: str,
    rna: pd.DataFrame,
) -> pd.DataFrame:
    """Read one split, sort patient IDs, and verify matching RNA rows."""
    table = pd.read_csv(fold_dir / f"{name}.csv")
    table["case_id"] = table["case_id"].astype(str)
    table = table.drop_duplicates("case_id").set_index("case_id")
    patient_ids = sorted(table.index)
    missing = [patient_id for patient_id in patient_ids if patient_id not in rna.index]
    if missing:
        raise ValueError(f"{name} has {len(missing)} patients missing from RNA")
    return table.loc[patient_ids]


def labels(table: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    duration = torch.as_tensor(
        table["dss_survival_days"].to_numpy(), dtype=torch.float32
    )
    event = torch.as_tensor(
        1 - table["dss_censorship"].to_numpy(), dtype=torch.float32
    )
    return duration, event


def verify_retained_labels(
    retained: Sequence[Sequence[float]],
    duration: torch.Tensor,
    event: torch.Tensor,
) -> None:
    """Confirm that retained PANTHER rows follow the split-table order."""
    retained_event = np.asarray([row[0] for row in retained], dtype=np.float32)
    retained_duration = np.asarray([row[1] for row in retained], dtype=np.float32)
    if not np.array_equal(retained_event, event.numpy()):
        raise RuntimeError("PANTHER event labels do not match the split CSV")
    if not np.allclose(retained_duration, duration.numpy()):
        raise RuntimeError("PANTHER survival times do not match the split CSV")


fold_ready = data_ready and TRUST_INPUTS
if fold_ready:
    fold_dir = SPLIT_ROOT / FOLD_PATTERN.format(fold=FOLD)
    rna = pd.read_csv(RNA_PATH, index_col=0)
    rna.index = rna.index.astype(str)
    signatures = pd.read_csv(SIGNATURE_PATH)
    train_table = read_split_table(fold_dir, "train", rna)
    valid_table = read_split_table(fold_dir, "test", rna)

    train_ids = list(train_table.index)
    valid_ids = list(valid_table.index)
    if set(train_ids) & set(valid_ids):
        raise RuntimeError("Training and validation patients overlap")

    # Fit gene scaling and pathway membership on the training fold only.
    pathway = PS3PathwayPreprocessor().fit(rna.loc[train_ids], signatures)
    train_genes = pathway.transform(rna.loc[train_ids])
    valid_genes = pathway.transform(rna.loc[valid_ids])

    train_duration, train_event = labels(train_table)
    valid_duration, valid_event = labels(valid_table)

    embedding_paths = sorted(fold_dir.glob(EMBEDDING_GLOB))
    if len(embedding_paths) != 1:
        raise FileNotFoundError("Expected exactly one retained PANTHER file")
    with embedding_paths[0].open("rb") as handle:
        retained = pickle.load(handle)

    train_histology = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
        retained["train"]["X"], 16, layout="crc_reproduction"
    )
    valid_histology = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
        retained["test"]["X"], 16, layout="crc_reproduction"
    )
    verify_retained_labels(retained["train"]["y"], train_duration, train_event)
    verify_retained_labels(retained["test"]["y"], valid_duration, valid_event)

    train_reports = [
        torch.load(
            REPORT_ROOT / f"{patient_id}.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for patient_id in train_ids
    ]
    valid_reports = [
        torch.load(
            REPORT_ROOT / f"{patient_id}.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for patient_id in valid_ids
    ]

    train_data = PS3Split(
        train_ids,
        train_histology,
        train_genes,
        train_reports,
        train_duration,
        train_event,
    )
    valid_data = PS3Split(
        valid_ids,
        valid_histology,
        valid_genes,
        valid_reports,
        valid_duration,
        valid_event,
    )

    print("Training patients/events:", len(train_ids), int(train_event.sum()))
    print("Validation patients/events:", len(valid_ids), int(valid_event.sum()))
    print("Histology tensors:", train_histology.shape, valid_histology.shape)
    print("RNA tensors:", train_genes.shape, valid_genes.shape)
    print("Hallmark pathways:", len(pathway.pathway_gene_indices))
    print("First report tensor:", train_reports[0].shape)
else:
    print("Configure the paths and set HISTOX_PS3_TRUST_INPUTS=1 to load fold 0.")

# %%
# The verified fold-0 output was:
#
# .. code-block:: text
#
#    Training patients/events: 214 31
#    Validation patients/events: 54 4
#    Histology tensors: torch.Size([214, 16, 1025]) torch.Size([54, 16, 1025])
#    RNA tensors: torch.Size([214, 4241]) torch.Size([54, 4241])
#    Hallmark pathways: 50
#    First report tensor: torch.Size([20, 512])
#
# The three modalities remain patient-aligned throughout this section:
#
# * ``histology[i]`` is the patient's 16 PANTHER tissue prototypes;
# * ``genes[i]`` is the patient's 4,241 scaled RNA measurements;
# * ``reports[i]`` contains a variable number of 512-dimensional PLIP text
#   segments; and
# * ``duration[i]`` and ``event[i]`` are the DSS target for that patient.
#
# ``layout="crc_reproduction"`` preserves the direct ``allcat`` reshape used
# to train the recorded CRC checkpoints. For a newly trained cohort, keep the
# :meth:`~histox.mil.PS3HistologyPrototypeEncoder.unpack_panther_allcat`
# default, ``prototype_aligned``, so each token contains one component's
# weight, mean, and covariance.

# %%
# 3. Define the Dataset and DataLoader
# -------------------------------------
#
# A ``Dataset`` returns one patient. The collate function stacks fixed-size
# histology/RNA tensors and pads variable-length reports while producing a mask
# so padded positions do not contribute to report attention.


class PS3Dataset(Dataset):
    def __init__(self, data: PS3Split) -> None:
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


class EventAwareBatchSampler(Sampler[List[int]]):
    """Random batches with at least one observed event for the Cox loss."""

    def __init__(self, events: torch.Tensor, batch_size: int) -> None:
        self.events = torch.as_tensor(events).reshape(-1).bool()
        self.batch_size = int(batch_size)
        self.n_batches = int(math.ceil(len(self.events) / self.batch_size))
        if int(self.events.sum()) < self.n_batches:
            raise ValueError("There are not enough events for every batch")

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        order = torch.randperm(len(self.events)).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]
        event_counts = [
            sum(bool(self.events[index]) for index in batch) for batch in batches
        ]
        for empty_index, count in enumerate(event_counts):
            if count:
                continue
            donor_index = next(
                index for index, donor_count in enumerate(event_counts)
                if donor_count > 1
            )
            donor_position = next(
                position
                for position, sample_index in enumerate(batches[donor_index])
                if bool(self.events[sample_index])
            )
            empty_position = next(
                position
                for position, sample_index in enumerate(batches[empty_index])
                if not bool(self.events[sample_index])
            )
            batches[donor_index][donor_position], batches[empty_index][empty_position] = (
                batches[empty_index][empty_position],
                batches[donor_index][donor_position],
            )
            event_counts[donor_index] -= 1
            event_counts[empty_index] += 1
        yield from batches


BATCH_SIZE = 64
if fold_ready:
    train_pad_to = max(int(report.shape[0]) for report in train_reports)
    collate = make_collate(train_pad_to)
    train_loader = DataLoader(
        PS3Dataset(train_data),
        batch_sampler=EventAwareBatchSampler(train_event, BATCH_SIZE),
        num_workers=0,
        collate_fn=collate,
    )
    train_eval_loader = DataLoader(
        PS3Dataset(train_data),
        batch_size=len(train_data.patient_ids),
        shuffle=False,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        PS3Dataset(valid_data),
        batch_size=len(valid_data.patient_ids),
        shuffle=False,
        collate_fn=collate,
    )

    batch = next(iter(train_loader))
    for name in ("histology", "rna", "report", "report_mask", "duration", "event"):
        print(f"{name:12s}", tuple(batch[name].shape))
    print("Observed events in this batch:", int(batch["event"].sum()))
else:
    print("The DataLoaders will be created after fold 0 is loaded.")

# %%
# A fold-0 training batch has the following contract:
#
# .. code-block:: text
#
#    histology    (64, 16, 1025)
#    rna          (64, 4241)
#    report       (64, 32, 512)
#    report_mask  (64, 32)
#    duration     (64,)
#    event        (64,)
#
# Cox partial likelihood is undefined for a batch containing no observed
# event. ``EventAwareBatchSampler`` therefore performs the minimum swap needed
# to place at least one event in every randomly formed batch; it does not drop
# patients or optimizer steps.

# %%
# 4. Create a task and method experiment
# ---------------------------------------
#
# The workflow describes the prediction target separately from the algorithm.
# ``DataSpec`` names the three input modalities and maps the survival fields.
# The registered ``ps3`` method constructs the existing PS3 model; it is not a
# second implementation.


def create_experiment(
    pathway: PS3PathwayPreprocessor,
    output: Path,
) -> hx.Experiment:
    return hx.Experiment(
        task="survival",
        method="ps3",
        data_spec=hx.DataSpec(
            modalities=("histology", "report", "rna"),
            fields={
                "patient_id": PATIENT_COLUMN,
                "duration": DURATION_COLUMN,
                "event": f"1 - {CENSORSHIP_COLUMN}",
            },
        ),
        method_config={
            "n_genes": len(pathway.gene_names),
            "pathway_gene_indices": pathway.pathway_gene_indices,
            "histology_dim": 1025,
            "report_dim": 512,
            "n_histology_prototypes": 16,
            "n_report_prototypes": 10,
            "embedding_dim": 256,
            "modality_embedding_dim": 32,
            "n_fusion_layers": 1,
            "feedforward_dim": 512,
            "dropout": 0.1,
            "report_selection_mode": "crc_reproduction",
        },
        output=output,
        metadata={"cohort": "TCGA COADREAD", "fold": FOLD, "seed": SEED},
    )


selected_dir = RUN_ROOT / f"seed_{SEED}" / f"fold_{FOLD}"
if fold_ready:
    # Batch inspection above consumes the sampler's random permutation.
    # Reset here so seed 1 initializes the model exactly as the batch runner.
    seed_everything(SEED)
    experiment = create_experiment(pathway, selected_dir)
    model = experiment.build_model().to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Task:", experiment.task.name)
    print("Method:", experiment.method.name)
    print(f"Trainable parameters: {parameter_count:,}")
else:
    print("The experiment will be created after fold 0 is loaded.")

# %%
# The fold-0 model contains ``29,459,457`` trainable parameters. The dimensions
# come from the prepared inputs: 4,241 genes assigned to 50 Hallmark pathways,
# 16 histology prototypes of width 1,025, and report segments of width 512.
# Ten report prototypes are selected per patient.

# %%
# 5. Configure training
# ----------------------
#
# PS3 delegates optimization and evaluation to HistoX's shared survival
# engine. The method supplies PS3's forward mapping and Cox loss; the engine
# performs device transfer, backpropagation, scheduling, C-index calculation,
# checkpoint writing, and prediction-table writing. These settings are fixed
# for every fold and seed; held-out outcomes are not used to tune them.

EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

print("Epochs / batch size:", EPOCHS, BATCH_SIZE)
print("Learning rate / weight decay:", LEARNING_RATE, WEIGHT_DECAY)
print("Schedule: cosine decay after one warmup epoch")

# %%
# 6. Train and evaluate fold 0
# -----------------------------
#
# Training is opt-in because this cell allocates the full model and writes its
# outputs. Change ``RUN_TRAINING`` to ``True`` after configuring the paths. The
# returned ``RunResult`` contains metrics, predictions, and typed artifacts.

RUN_TRAINING = False


def report_epoch(row: Dict[str, object]) -> None:
    epoch = int(row["epoch"])
    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
        print(
            f"epoch={epoch:02d}/{EPOCHS} "
            f"loss={row['mean_batch_loss']:.6f} "
            f"lr={row['learning_rate']:.8f}"
        )


if RUN_TRAINING:
    if not fold_ready:
        raise FileNotFoundError("Configure and approve the cohort inputs first")
    fitted = experiment.fit(
        train_loader=train_loader,
        epochs=EPOCHS,
        evaluation_loaders={
            "train": train_eval_loader,
            "test": valid_loader,
        },
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        scheduler="cosine",
        warmup_steps=len(train_loader),
        progress=report_epoch,
    )
    print(f"Training C-index:   {fitted.metrics['train_c_index']:.6f}")
    print(f"Validation C-index: {fitted.metrics['test_c_index']:.6f}")
    for name, artifact in fitted.artifacts.items():
        print(f"{name:18s}", artifact.path)
else:
    print("Set RUN_TRAINING = True to train fold 0 for 50 epochs.")

# %%
# The completed seed-1/fold-0 run produced:
#
# .. code-block:: text
#
#    epoch=01/50 loss=3.340556 lr=0.00010000
#    epoch=05/50 loss=2.171427 lr=0.00009836
#    ...
#    epoch=50/50 loss=0.245202 lr=0.00000000
#    Training C-index:   0.997586
#    Validation C-index: 0.890909
#
# The validation score uses all 54 held-out patients, including four observed
# DSS events. Evaluation occurs after the fixed 50-epoch schedule; the
# held-out score does not select a checkpoint.

# %%
# 7. Reload, evaluate, and extract model evidence
# ------------------------------------------------
#
# The workflow saves the model state, epoch history, and patient predictions.
# Recreate the same experiment before loading ``model_state.pt``. Calling
# ``explain`` exports the joint prototype-attention tensor used by PS3. The
# next section combines that model evidence with the PANTHER assignments and
# patch coordinates saved during WSI preparation.

if fold_ready and selected_dir.is_dir():
    history = pd.read_csv(selected_dir / "history.csv")
    predictions = pd.read_csv(selected_dir / "test_predictions.csv")
    print(history.tail().to_string(index=False))
    print(predictions.head().to_string(index=False))

    restored_experiment = create_experiment(pathway, selected_dir)
    restored_model = restored_experiment.build_model().to(DEVICE)
    restored_state = torch.load(
        selected_dir / "model_state.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    restored_model.load_state_dict(restored_state)
    restored = restored_experiment.evaluate(loader=valid_loader, device=DEVICE)
    explained = restored_experiment.explain(loader=valid_loader, device=DEVICE)
    print(f"Reloaded validation C-index: {restored.metrics['c_index']:.6f}")
    print("Joint attention:", explained.artifacts["joint_attention"].path)

    history.plot(
        x="epoch",
        y="mean_batch_loss",
        figsize=(7.2, 4.0),
        legend=False,
        ylabel="Mean Cox loss",
        title="Seed 1, fold 0 training history",
    )
    plt.tight_layout()
else:
    print("Saved outputs will be read from:", selected_dir)

# %%
# 8. Reconstruct the paper-style prototype explanation
# -----------------------------------------------------
#
# A pretrained bundle keeps the trained model, fold-specific RNA scaler,
# pathway membership, report settings, and the 16 fitted PANTHER centers
# together. ``prototype_explanation`` recreates the 16 histology prototypes
# from the PLIP patch bag, runs PS3, and returns the quantities used by the
# paper's patient-level interpretation figure.
#
# The preparation tutorial writes ``histology_bags/{patient_id}.pt`` and
# ``histology_coordinates/{patient_id}.pt`` together. Do not mix coordinates
# from another WSI extraction run.

BUNDLE_ROOT = Path(
    os.environ.get(
        "HISTOX_PS3_BUNDLE",
        "models/ps3-coadread-seed1-fold0",
    )
).expanduser()
PS3_INPUT_ROOT = Path(
    os.environ.get("HISTOX_PS3_INPUT_ROOT", "ps3_inputs")
).expanduser()
EXAMPLE_PATIENT = os.environ.get("HISTOX_PS3_PATIENT", "PATIENT-001")
EXAMPLE_SLIDE = Path(
    os.environ.get("HISTOX_PS3_SLIDE", "slides/PATIENT-001.svs")
).expanduser()

feature_path = PS3_INPUT_ROOT / "histology_bags" / f"{EXAMPLE_PATIENT}.pt"
coordinate_path = (
    PS3_INPUT_ROOT / "histology_coordinates" / f"{EXAMPLE_PATIENT}.pt"
)
report_path = REPORT_ROOT / f"{EXAMPLE_PATIENT}.pt"
explanation_ready = all(path.is_file() for path in (
    BUNDLE_ROOT / "config.json",
    feature_path,
    coordinate_path,
    report_path,
    RNA_PATH,
    EXAMPLE_SLIDE,
))

if explanation_ready:
    bundle = hx.PS3PretrainedBundle.load(BUNDLE_ROOT, device=DEVICE)
    patch_features = torch.load(
        feature_path, map_location="cpu", weights_only=True
    )
    coordinates = torch.load(
        coordinate_path, map_location="cpu", weights_only=True
    )
    report_embeddings = torch.load(
        report_path, map_location="cpu", weights_only=True
    )
    raw_rna = pd.read_csv(RNA_PATH, index_col=0)
    raw_rna.index = raw_rna.index.astype(str)

    explanation = bundle.prototype_explanation(
        patch_features=patch_features,
        coordinates=coordinates,
        gene_expression=raw_rna.loc[[EXAMPLE_PATIENT]],
        report_embeddings=report_embeddings,
        representative_patches=3,
    )

    prototype_map_path = bundle.save_prototype_map(
        explanation,
        slide=EXAMPLE_SLIDE,
        tile_px=256,
        tile_um=128,
        destination=selected_dir / f"{EXAMPLE_PATIENT}_prototype_map.png",
    )
    representative_path = bundle.save_representative_patches(
        explanation,
        slide=EXAMPLE_SLIDE,
        tile_px=256,
        tile_um=128,
        destination=selected_dir / f"{EXAMPLE_PATIENT}_representative_patches.png",
        top_prototypes=8,
    )
    proportion_path = bundle.save_prototype_proportions(
        explanation,
        destination=selected_dir / f"{EXAMPLE_PATIENT}_prototype_proportions.png",
    )

    myc_pathway = next(
        name for name in explanation["pathway_names"]
        if "MYC_TARGETS_V1" in name.upper().replace(" ", "_")
    )
    dominant_prototype = int(
        explanation["prototype_proportions"].argmax()
    )
    cross_modal_path = bundle.save_cross_modal_attention(
        explanation,
        destination=selected_dir / f"{EXAMPLE_PATIENT}_cross_modal_attention.png",
        pathway=myc_pathway,
        histology_prototype=dominant_prototype,
    )

    print("Patient log-risk:", float(explanation["risk"].item()))
    print("Assigned patches:", len(explanation["prototype_assignments"]))
    print("Dominant prototype:", f"C{dominant_prototype}")
    print("Prototype proportion:", float(
        explanation["prototype_proportions"][dominant_prototype]
    ))
    print("Prototype map:", prototype_map_path)
    print("Representative patches:", representative_path)
    print("Prototype proportions:", proportion_path)
    print("Cross-modal attention:", cross_modal_path)
else:
    print("Configure the bundle, prepared patient inputs, and source WSI above.")

# %%
# In the verified TCGA example, one WSI produced 3,643 hard prototype
# assignments. ``C1`` contained 734 patches (20.15%), followed by ``C3`` with
# 720 patches (19.76%) and ``C7`` with 623 patches (17.10%).
#
# .. code-block:: text
#
#    Patient: TCGA-A6-5660
#    Patches: 3643
#    Joint attention: (1, 76, 76)
#    PANTHER responsibilities: (3643, 16)
#    Patient log-risk: -3.319172
#    Dominant prototype: C1
#    Dominant proportion: 0.201482
#
# Prototype map
# ~~~~~~~~~~~~~
#
# Every retained patch is colored by the prototype with the largest PANTHER
# responsibility. This is a categorical nearest-prototype view, not a
# continuous survival-risk heatmap.
#
# .. image:: /_static/examples/ps3_prototype_map_tcga_a6_5660.png
#    :alt: Categorical PS3 histology prototype map over a TCGA COADREAD whole-slide image
#    :width: 900px
#    :align: center
#
# Representative patches
# ~~~~~~~~~~~~~~~~~~~~~~
#
# For each populated prototype, HistoX ranks assigned patches by that
# prototype's responsibility and re-reads the top coordinates from the source
# SVS. The columns below therefore show actual tissue regions from this WSI.
#
# .. image:: /_static/examples/ps3_representative_patches_tcga_a6_5660.png
#    :alt: Three representative TCGA tissue patches for each leading PS3 histology prototype
#    :width: 1000px
#    :align: center
#
# Prototype proportions
# ~~~~~~~~~~~~~~~~~~~~~
#
# The bar height is the number of patches assigned to a prototype divided by
# all 3,643 retained patches from this slide.
#
# .. image:: /_static/examples/ps3_prototype_proportions_tcga_a6_5660.png
#    :alt: Proportion of TCGA patches assigned to each PS3 histology prototype
#    :width: 760px
#    :align: center
#
# Cross-modal attention
# ~~~~~~~~~~~~~~~~~~~~~
#
# The four panels follow the paper's interpretation scheme: MYC Targets V1 to
# histology, report to histology, the dominant histology prototype to
# pathways, and report to pathways. HistoX applies softmax to the raw attention
# logits within the target modality. For the report panels it then computes
# the standard deviation across the ten selected report prototypes.
#
# .. image:: /_static/examples/ps3_cross_modal_attention_tcga_a6_5660.png
#    :alt: PS3 pathway report and histology cross-modal attention rankings for a TCGA patient
#    :width: 1000px
#    :align: center
#
# Prototype labels are model- and fold-specific. For example, ``C1`` in this
# seed-1/fold-0 model is not assumed to carry the same morphology as ``C1`` in
# another fitted model. A pathologist must review the representative patches
# before attaching a tissue interpretation to a prototype. Attention describes
# a trained association and is not a causal estimate of survival contribution.

# %%
# 9. Compare report language across predicted risk groups
# --------------------------------------------------------
#
# The paper also compares pathology-report phrases between low- and high-risk
# patients. This requires cohort-level out-of-fold predictions; using the
# fold-0 model for all patients would leak training patients into the figure.
# HistoX therefore performs the same steps independently in all five folds:
#
# 1. reload the fold checkpoint and its training-fold RNA preprocessing;
# 2. call ``report_segment_evidence`` for the held-out patients;
# 3. average histology-to-report attention over the 16 histology prototypes;
# 4. map the leading traceable report prototype back to its source 77-token
#    PLIP segment; and
# 5. pool contiguous two-word phrases after splitting patients at the median
#    out-of-fold log-risk.
#
# The core model call is shown below. ``selected_source_segment`` is an index
# into the original report segments, not a generated summary.
#
# .. code-block:: python
#
#    model.eval()
#    with torch.inference_mode():
#        evidence = model.report_segment_evidence(
#            histology_prototypes=held_out_histology,
#            gene_expression=held_out_genes,
#            report_embeddings=held_out_reports,
#            report_mask=held_out_report_mask,
#        )
#
#    for patient_index, patient_id in enumerate(held_out_patient_ids):
#        source_index = int(
#            evidence["selected_source_segment"][patient_index]
#        )
#        source_segments = PS3ReportEncoder.segment_text(
#            raw_reports.loc[patient_id, "text"],
#            tokenizer=plip_processor.tokenizer,
#            max_tokens=77,
#        )
#        selected_text = source_segments[source_index]
#        selected_score = float(evidence["selected_score"][patient_index])
#
# After concatenating all five held-out folds, the grouping and phrase
# extraction are ordinary, inspectable pandas/scikit-learn operations:
#
# .. code-block:: python
#
#    cutoff = patient_evidence["log_risk"].median()
#    patient_evidence["risk_group"] = np.where(
#        patient_evidence["log_risk"] <= cutoff, "low", "high"
#    )
#
#    vectorizer = CountVectorizer(
#        ngram_range=(2, 2),
#        min_df=2,
#        lowercase=True,
#        strip_accents="unicode",
#    )
#    phrase_counts = vectorizer.fit_transform(
#        patient_evidence["selected_text"]
#    )
#
# The complete executable script also checks patient identity, retained labels,
# report-segment counts, checkpoint predictions, and clinical negation before
# writing the figure and its two source tables:
#
# .. code-block:: bash
#
#    python docs/examples/ps3_crc_report_wordclouds.py \
#      --validation-root runs/histox-ps3-validation \
#      --split-root data/ps3_crc/splits \
#      --rna data/ps3_crc/rna_clean.csv \
#      --signatures data/ps3_crc/hallmarks_signatures.csv \
#      --reports-csv data/ps3_crc/TCGA_Reports.csv \
#      --report-embeddings data/ps3_crc/report_embeddings \
#      --output runs/ps3_report_words \
#      --trust-inputs
#
# The executed seed-1 result contains 269 unique held-out patients. The
# out-of-fold median log-risk was ``-7.5766287``, giving 135 low-risk and 134
# high-risk patients. Recomputed risks differed from the saved predictions by
# at most ``2.42e-6``. All 269 selected segment indices mapped to source text;
# 116 reports had fewer than ten segments, so padded selector positions were
# excluded before choosing the source segment.
#
# .. list-table:: Most frequent selected-report phrases
#    :header-rows: 1
#    :widths: 18 28 12 18 28 12
#
#    * - Rank
#      - Low predicted risk
#      - Count
#      - Rank
#      - High predicted risk
#      - Count
#    * - 1
#      - lymph nodes
#      - 27
#      - 1
#      - lymph nodes
#      - 27
#    * - 2
#      - proximal margin
#      - 14
#      - 2
#      - distal margin
#      - 14
#    * - 3
#      - distal margin
#      - 13
#      - 3
#      - pathologic findings
#      - 13
#    * - 4
#      - muscularis propria
#      - 12
#      - 4
#      - proximal margin
#      - 13
#    * - 5
#      - resection margin
#      - 12
#      - 5
#      - adjuvant treatment
#      - 12
#    * - 6
#      - large intestine
#      - 11
#      - 6
#      - neo adjuvant
#      - 12
#
# .. image:: /_static/examples/ps3_report_wordclouds.png
#    :alt: Low- and high-risk pathology-report phrase word clouds selected by PS3 histology-to-report attention
#    :width: 1100px
#    :align: center
#
# Phrase size is a within-group frequency display, not an effect size or a
# hypothesis test. The generated ``ps3_report_segment_evidence.csv`` keeps the
# patient, fold, risk, report prototype, source segment, attention score, and
# exact selected text. ``ps3_report_phrase_frequencies.csv`` provides counts
# and document frequencies behind every displayed phrase.

# %%
# 10. Optional continuous spatial attention
# ------------------------------------------
#
# The categorical prototype map above answers *which morphology prototype is
# closest to each patch*. A continuous spatial map answers a different
# question: *which locations align with the histology prototypes attended by a
# chosen pathway, report prototype, or all model tokens*. Generate it only
# when that second question is relevant.
#
# .. code-block:: python
#
#    spatial = bundle.spatial_attention(
#        patch_features=patch_features,
#        coordinates=coordinates,
#        gene_expression=raw_rna.loc[[EXAMPLE_PATIENT]],
#        report_embeddings=report_embeddings,
#        source="pathway",
#        source_index=myc_pathway,
#    )
#    bundle.save_spatial_heatmap(
#        spatial,
#        slide=EXAMPLE_SLIDE,
#        tile_px=256,
#        tile_um=128,
#        destination=selected_dir / f"{EXAMPLE_PATIENT}_myc_spatial.png",
#    )
#
# Do not use this continuous map as a substitute for the categorical
# prototype map or the representative-patch panel.

# %%
# Next steps
# ----------
#
# You have now completed the full HistoX learning path for one real fold.
# Continue with exactly one of these pages:
#
# * :doc:`Prepare multimodal inputs for PS3
#   </tutorials/ps3_data_preparation>` to start from WSI files, report text,
#   RNA, and outcomes.
# * :doc:`Reproduce PS3 on TCGA COADREAD
#   </tutorials/ps3_coadread_reproduction>` to run all five folds and three
#   random seeds.
# * :doc:`PS3 multimodal survival </methods/multimodal/ps3>` for the model
#   equations, tensor contract, and API reference.
