"""Central configuration for the thesis reproducibility pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Project root: parent of `src/`
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Dataset mode
# -----------------------------------------------------------------------------
# PROCESS2 — final thesis cohort (PROCESS-2 / CognoSpeak layout)
# LEGACY   — previous Data/ or data/ development layout (Archive, Data_AUG, etc.)
DATASET_NAME: str = os.environ.get("THESIS_DATASET", "PROCESS2").strip().upper()

# Optional class_weight='balanced' model variants (robustness only; off by default)
INCLUDE_BALANCED_VARIANTS: bool = (
    os.environ.get("THESIS_INCLUDE_BALANCED", "0").strip() == "1"
)

_DATA_OVERRIDE = os.environ.get("THESIS_DATA_DIR", "").strip()

# PROCESS-2 master table (authoritative CTD-only index)
PROCESS2_MASTER_CSV: Path = (
    PROJECT_ROOT
    / os.environ.get(
        "THESIS_PROCESS2_MASTER_CSV", "outputs/process2_ctd_binary_master.csv"
    )
).resolve()


def resolve_data_dir() -> Path:
    """Return the dataset root directory."""
    if _DATA_OVERRIDE:
        p = Path(_DATA_OVERRIDE).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"THESIS_DATA_DIR is set but not a directory: {p}"
            )
        return p
    if DATASET_NAME == "PROCESS2":
        candidate = PROJECT_ROOT / "PROCESS-2"
        if candidate.is_dir():
            return candidate.resolve()
        raise FileNotFoundError(
            f"PROCESS-2 folder not found at {candidate}. "
            "Download the dataset from Hugging Face / CognoSpeak, extract it there, "
            "or set THESIS_DATA_DIR to the dataset root."
        )
    for name in ("data", "Data"):
        candidate = PROJECT_ROOT / name
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No legacy dataset folder found under {PROJECT_ROOT}. "
        "Create data/ or Data/, set THESIS_DATASET=LEGACY, or use PROCESS-2."
    )


def is_process2() -> bool:
    return DATASET_NAME == "PROCESS2"


DATA_DIR: Path = resolve_data_dir()
OUTPUT_DIR: Path = (PROJECT_ROOT / "outputs").resolve()
FIGURES_DIR: Path = (OUTPUT_DIR / "figures").resolve()
NOTEBOOKS_DIR: Path = (PROJECT_ROOT / "notebooks").resolve()

# PROCESS-2 metadata filename (under DATA_DIR)
PROCESS2_METADATA_FILE: str = "metadata.csv"
PROCESS2_EXPECTED_HC: int = int(os.environ.get("THESIS_EXPECTED_HC", "200"))
PROCESS2_EXPECTED_CI: int = int(os.environ.get("THESIS_EXPECTED_CI", "200"))

# Reproducibility
RANDOM_SEED: int = int(os.environ.get("THESIS_RANDOM_SEED", "42"))

# NLP models
SPACY_MODEL: str = os.environ.get("THESIS_SPACY_MODEL", "en_core_web_sm")
SBERT_MODEL_NAME: str = os.environ.get(
    "THESIS_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Label handling for the *legacy augmented* fallback only
LABEL_MODE: str = os.environ.get("THESIS_LABEL_MODE", "binary_hc_dementia")

# Cross-validation (training set only for PROCESS-2)
CV_SPLITS: int = 5

# SHAP / KernelExplainer
SHAP_BACKGROUND_SIZE: int = int(os.environ.get("THESIS_SHAP_BG", "24"))
KERNEL_SHAP_NSAMPLES: int = int(os.environ.get("THESIS_KERNEL_NSAMPLES", "80"))

# Bootstrap runs for stability (training set only)
STABILITY_BOOTSTRAPS: int = int(os.environ.get("THESIS_BOOTSTRAPS", "15"))

# Feature columns (order matters for SHAP alignment)
FEATURE_COLUMNS: tuple[str, ...] = (
    "type_token_ratio",
    "filler_count",
    "filler_ratio",
    "mean_clause_length",
    "semantic_coherence",
    "content_density",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "determiner_ratio",
)

OUTPUT_FEATURES: Path = OUTPUT_DIR / "features.csv"
OUTPUT_BASELINE: Path = OUTPUT_DIR / "baseline_results.csv"
OUTPUT_OPTIMIZED: Path = OUTPUT_DIR / "optimized_results.csv"
OUTPUT_SHUFFLED: Path = OUTPUT_DIR / "shuffled_label_results.csv"
OUTPUT_SHAP_BASELINE: Path = OUTPUT_DIR / "shap_values_baseline.csv"
OUTPUT_SHAP_OPTIMIZED: Path = OUTPUT_DIR / "shap_values_optimized.csv"
OUTPUT_STABILITY: Path = OUTPUT_DIR / "stability_results.csv"
