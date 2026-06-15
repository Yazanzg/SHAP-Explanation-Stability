"""Central configuration loader for the thesis reproducibility pipeline.

All paths and parameters live in ``config.yaml`` at the repository root. This
module loads that file once and exposes its values as plain module-level
constants and a few small helper functions. There is intentionally no
metaprogramming here: every constant below maps directly to a key in
``config.yaml`` and can be explained line by line.

To run against an alternate configuration (e.g. the 10-row synthetic example
used by the smoke test), set the environment variable ``THESIS_CONFIG`` to the
path of another YAML file before importing the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# -----------------------------------------------------------------------------
# Locate and load config.yaml
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_CONFIG_OVERRIDE = os.environ.get("THESIS_CONFIG", "").strip()
CONFIG_PATH: Path = (
    Path(_CONFIG_OVERRIDE).expanduser().resolve()
    if _CONFIG_OVERRIDE
    else (PROJECT_ROOT / "config.yaml").resolve()
)

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}. "
        "Expected config.yaml at the repository root, or set THESIS_CONFIG."
    )

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG: dict[str, Any] = yaml.safe_load(fh)


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repository root (absolute kept as-is)."""
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_paths = CONFIG["paths"]
DATA_DIR: Path = _resolve(_paths["data_dir"])
MASTER_CSV: Path = _resolve(_paths["master_csv"])
FEATURES_CSV: Path = _resolve(_paths["features_csv"])
OUTPUT_DIR: Path = _resolve(_paths["outputs_dir"])
FIGURES_DIR: Path = _resolve(_paths["figures_dir"])
AUDIT_DIR: Path = _resolve(_paths["audit_dir"])
IMPORTANCE_VECTORS_DIR: Path = _resolve(_paths["importance_vectors_dir"])
NOTEBOOKS_DIR: Path = (PROJECT_ROOT / "notebooks").resolve()

# -----------------------------------------------------------------------------
# Dataset schema (dataset-agnostic column names + PROCESS-2 specifics)
# -----------------------------------------------------------------------------
_dataset = CONFIG["dataset"]
DATASET_NAME: str = _dataset["name"]
_cols = _dataset["columns"]
COL_PARTICIPANT_ID: str = _cols["participant_id"]
COL_TRANSCRIPT_PATH: str = _cols["transcript_path"]
COL_LABEL: str = _cols["label"]
COL_SPLIT: str = _cols["split"]
DIAGNOSIS_COLUMN: str = _dataset.get("diagnosis_column", "diagnosis")
LABEL_MAPPING: dict[str, int] = dict(_dataset["label_mapping"])
POSITIVE_CLASS_NAME: str = _dataset.get("positive_class_name", "CI")
NEGATIVE_CLASS_NAME: str = _dataset.get("negative_class_name", "HC")
TRAIN_SPLIT_VALUE: str = str(_dataset["train_split_value"]).lower()
TEST_SPLIT_VALUE: str = str(_dataset["test_split_value"]).lower()
EXPECTED_TRAIN_POOL: dict[str, int] = dict(_dataset.get("expected_train_pool", {}))
EXPECTED_TEST: dict[str, int] = dict(_dataset.get("expected_test", {}))

# -----------------------------------------------------------------------------
# Features (the 13 modelling features, in SHAP-alignment order)
# -----------------------------------------------------------------------------
FEATURES: list[str] = list(CONFIG["features"])
FEATURE_COLUMNS: tuple[str, ...] = tuple(FEATURES)  # back-compat alias

# -----------------------------------------------------------------------------
# Models and fixed hyperparameters (no tuning)
# -----------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = dict(CONFIG["models"])

# -----------------------------------------------------------------------------
# Learning-curve experiment settings
# -----------------------------------------------------------------------------
_experiment = CONFIG["experiment"]
TRAINING_SIZE_GRID: list[int] = list(_experiment["training_size_grid"])
ITERATIONS_PER_SIZE: int = int(_experiment["iterations_per_size"])
_smoke = _experiment.get("smoke", {})
SMOKE_TRAINING_SIZE_GRID: list[int] = list(
    _smoke.get("training_size_grid", [80, 160, 240])
)
SMOKE_ITERATIONS: int = int(_smoke.get("iterations_per_size", 5))
SMOKE_MODELS: list[str] = list(_smoke.get("models", ["logistic_regression"]))

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
_metrics = CONFIG["metrics"]
PRIMARY_METRIC: str = _metrics.get("primary", "pearson")
JACCARD_KS: list[int] = list(_metrics["jaccard_ks"])
PERCENTILE_INTERVAL: tuple[float, float] = tuple(_metrics["percentile_interval"])  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# SHAP / explainer settings
# -----------------------------------------------------------------------------
_shap = CONFIG["shap"]
SHAP_BACKGROUND_SIZE: int = int(_shap["background_size"])
KERNEL_SHAP_NSAMPLES: int = int(_shap["kernel_nsamples"])

# -----------------------------------------------------------------------------
# Plotting defaults (TASK 8): main figure uses the random-subset regime; the
# full-pool boundary size is reported in the appendix/table only.
# -----------------------------------------------------------------------------
_plotting = CONFIG.get("plotting", {})
MAIN_FIGURE_SIZES: list[int] = list(_plotting.get("main_figure_sizes", []))
BOUNDARY_SIZE: int = int(_plotting.get("boundary_size", 0)) or None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Reproducibility / seeding scheme
# -----------------------------------------------------------------------------
_repro = CONFIG["reproducibility"]
GLOBAL_SEED: int = int(_repro["global_seed"])
SIZE_INDEX_OFFSET: int = int(_repro.get("size_index_offset", 1000))
RANDOM_SEED: int = GLOBAL_SEED  # back-compat alias


def iteration_seed(size_index: int, run_id: int) -> int:
    """Deterministic per-cell seed: global_seed + size_index*offset + run_id.

    Guarantees every (training_size, run_id) draws a distinct, reproducible
    subset. ``size_index`` is the position of the size in TRAINING_SIZE_GRID.
    """
    return GLOBAL_SEED + size_index * SIZE_INDEX_OFFSET + run_id


# -----------------------------------------------------------------------------
# NLP
# -----------------------------------------------------------------------------
SPACY_MODEL: str = CONFIG.get("nlp", {}).get("spacy_model", "en_core_web_sm")


def is_process2() -> bool:
    """True when the active dataset is PROCESS-2 (kept for back-compat)."""
    return DATASET_NAME.upper().startswith("PROCESS")


# -----------------------------------------------------------------------------
# Compatibility aliases used by the data-loading / preprocessing validators.
# -----------------------------------------------------------------------------
PROCESS2_MASTER_CSV: Path = MASTER_CSV
PROCESS2_METADATA_FILE: str = "metadata.csv"
PROCESS2_EXPECTED_HC: int = int(EXPECTED_TRAIN_POOL.get("HC", 0) + EXPECTED_TEST.get("HC", 0)) or 200
PROCESS2_EXPECTED_CI: int = int(EXPECTED_TRAIN_POOL.get("CI", 0) + EXPECTED_TEST.get("CI", 0)) or 200
# Legacy label mode for the optional augmented-cohort fallback in data_loading.
LABEL_MODE: str = "binary_hc_pathological"
