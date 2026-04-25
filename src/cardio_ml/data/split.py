"""
Data partitioning.

Reproducible stratified split. Centralizes the logic to prevent each script
from reimplementing train_test_split with different parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from cardio_ml.config import SEED


@dataclass(frozen=True)
class StratifiedSplit:
    """Result of a stratified train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "test": len(self.X_test)}


def stratified_train_test_split(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    seed: int | None = None,
) -> StratifiedSplit:
    """Split `frame` into train/test while preserving the target proportion.

    Stratification on `target_col` ensures train and test have the same class
    distribution, avoiding metric distortions. The fixed `seed` makes the
    experiment reproducible across runs.
    """

    seed = SEED if seed is None else seed

    X = frame[feature_cols]
    y = frame[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return StratifiedSplit(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
