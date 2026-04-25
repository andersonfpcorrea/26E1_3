"""
Central project configuration.

Concentrates path constants, random seed, computational resource limits and
MLflow tracking parameters. Keeping everything in a single module eases
reproducibility and prevents divergence between scripts.

Resource policy
---------------
Training classical models on 70k rows is fast, but cross-validation +
hyperparameter search can saturate the machine. To allow interactive usage
during training, this module:

1. Limits the number of BLAS threads via environment variables (OMP, MKL,
   OPENBLAS, VECLIB, NUMEXPR). These variables work identically on Linux,
   macOS and Windows and must be set before importing numpy/sklearn.
2. Lowers the process priority with `psutil` (cross-platform: nice on
   Linux/macOS, IDLE_PRIORITY_CLASS on Windows).
3. Defines a global N_JOBS consumed by scikit-learn and XGBoost, limiting
   high-level parallelism in addition to the BLAS limit.

Adjustment via environment variables:
    CARDIO_N_JOBS=8       to allow more parallelism
    CARDIO_NICE=0         to skip priority lowering
    CARDIO_SEED=123       for a different seed
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

RAW_DATASET_PATH: Path = RAW_DATA_DIR / "cardio_train.csv"

# In Lambda, /var/task is read-only — skip directory creation.
if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    for _dir in (PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR, MLRUNS_DIR):
        _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED: int = int(os.environ.get("CARDIO_SEED", "42"))

# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------

# Default URI points to ./mlruns at the project root. Can be overridden
# to a remote server via the standard MLflow environment variable.
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    MLRUNS_DIR.as_uri(),
)
MLFLOW_EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "cardio-dcv",
)
MLFLOW_REGISTERED_MODEL_NAME: str = "cardio-dcv-classifier"

# ---------------------------------------------------------------------------
# Resource policy
# ---------------------------------------------------------------------------

N_JOBS: int = int(os.environ.get("CARDIO_N_JOBS", "4"))
NICE_LEVEL: int = int(os.environ.get("CARDIO_NICE", "10"))
_THREAD_LIMIT: str = str(N_JOBS)

# BLAS/thread pool variables — defined here as a fallback, but ideally
# they should come from the Makefile or shell, before numpy is imported.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_policy_applied = False


def apply_resource_policy() -> None:
    """Apply resource limits once per process.

    This function is idempotent and called automatically on package import.
    Exposing it publicly allows notebooks or tests to reapply the policy
    after changing environment variables.
    """

    global _policy_applied
    if _policy_applied:
        return

    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, _THREAD_LIMIT)

    _lower_process_priority()

    _policy_applied = True


def _lower_process_priority() -> None:
    """Lower the current process priority in a cross-platform way.

    We use psutil because it translates the 'nice' concept to the native API
    of each OS:
      - Linux/macOS: calls setpriority with the given nice level (0=normal, 19=lowest).
      - Windows: maps to BELOW_NORMAL_PRIORITY_CLASS or IDLE_PRIORITY_CLASS.
    Silent failures: if psutil is not installed or the OS does not allow it,
    we proceed without breaking execution.
    """

    if NICE_LEVEL <= 0:
        return

    try:
        import psutil

        process = psutil.Process()

        if os.name == "nt":
            # Arbitrary mapping: nice>=15 -> idle, otherwise below normal.
            priority = (
                psutil.IDLE_PRIORITY_CLASS
                if NICE_LEVEL >= 15
                else psutil.BELOW_NORMAL_PRIORITY_CLASS
            )
            process.nice(priority)
        else:
            process.nice(NICE_LEVEL)
    except Exception:
        # We don't want a permission issue to block a training run.
        pass


def summarize_policy() -> dict[str, str | int]:
    """Return a textual summary of the applied policy — useful for logging in MLflow."""

    return {
        "n_jobs": N_JOBS,
        "nice_level": NICE_LEVEL,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "seed": SEED,
    }
