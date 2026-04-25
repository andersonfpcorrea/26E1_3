"""
Dimensionality reduction.

We chose two techniques: PCA and LDA. Both implement a transformation learned
during `fit` and applicable at inference time via `transform` — a prerequisite
for production use. t-SNE was discarded because it does not offer a deterministic
projection of new points consistently (its `fit_transform` method does not project
external data without retraining), making it unsuitable as a step in a serving
pipeline.

Individual justification:
  - **PCA**: unsupervised, reduces total variance, useful for mitigating
    collinearity between ap_hi/ap_lo/bmi and for controlling overfitting when
    there is a combination of raw + derived features (bmi is a function of
    height/weight).
  - **LDA**: supervised, seeks the projection that maximizes class separation.
    In balanced binary classification like this one, LDA projects to a single
    component optimal for separation. Works well with simple models (logistic
    regression, shallow tree) without losing discriminative power.

The choice of number of components for PCA uses `n_components=0.95` — preserves
95% of the variance, letting the exact number of components be determined by the
data. For LDA the maximum number is `n_classes - 1 = 1`.
"""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DimTechnique = Literal["none", "pca", "lda"]


class DimReducer(BaseEstimator, TransformerMixin):
    """Wrapper that exposes a unified interface for the supported techniques.

    Using a wrapper, instead of scattering `if technique == ...` throughout the
    code, simplifies Pipeline construction and makes it easier to log the same
    artifact in MLflow with homogeneous metadata.
    """

    def __init__(
        self,
        technique: DimTechnique = "none",
        n_components: float | int | None = None,
        random_state: int | None = None,
    ):
        self.technique = technique
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: ANN001 — follows sklearn contract
        if self.technique == "none":
            self._reducer = None
            return self

        if self.technique == "pca":
            n = self.n_components if self.n_components is not None else 0.95
            self._reducer = PCA(n_components=n, random_state=self.random_state)
            self._reducer.fit(X)
        elif self.technique == "lda":
            self._reducer = LinearDiscriminantAnalysis(n_components=self.n_components)
            if y is None:
                raise ValueError("LDA e supervisionada e exige `y` no fit.")
            self._reducer.fit(X, y)
        else:
            raise ValueError(f"Tecnica desconhecida: {self.technique!r}")

        return self

    def transform(self, X):  # noqa: ANN001
        if self._reducer is None:
            return X
        return self._reducer.transform(X)

    def fit_transform(self, X, y=None, **fit_params):  # noqa: ANN001
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int | None:
        """Number of components after fit (None if not applicable)."""

        if self._reducer is None:
            return None
        if self.technique == "pca":
            return self._reducer.n_components_
        if self.technique == "lda":
            return self._reducer.scalings_.shape[1]
        return None


def build_dim_reducer(
    technique: DimTechnique,
    n_components: float | int | None = None,
    random_state: int | None = None,
) -> DimReducer:
    """Factory with explicit signature — used by experiment scripts."""

    return DimReducer(
        technique=technique,
        n_components=n_components,
        random_state=random_state,
    )
