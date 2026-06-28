"""Pitch-velocity modelling and biomechanics z-scores for the OBP dataset.

This module ties the OpenBiomechanics ``poi_metrics.csv`` (per-pitch
biomechanics summary metrics) to pitch release velocity and trains a
:class:`~bayesian_lasso.BayesianLasso` to predict ``pitch_speed_mph`` from the
biomechanics. It also computes per-pitch **z-scores** for every biomechanics
metric (relative to the dataset) and links a pitch to its raw C3D file via
``metadata.csv``.

The pieces here feed the dashboard in ``dashboard.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bayesian_lasso import BayesianLasso

OBP_DATA_BASE = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "main/baseball_pitching/data"
)

# Columns in poi_metrics.csv that are identifiers / target, not features.
_NON_FEATURE_COLS = {
    "session_pitch", "session", "p_throws", "pitch_type", "pitch_speed_mph",
}
TARGET_COL = "pitch_speed_mph"


def download_obp_table(name: str, dest_dir: str = "obp_data") -> str:
    """Fetch ``poi_metrics.csv`` or ``metadata.csv`` via the data-source layer.

    ``name`` is one of ``"poi_metrics"`` or ``"metadata"``.
    """
    import data_sources

    rel = {
        "poi_metrics": "poi/poi_metrics.csv",
        "metadata": "metadata.csv",
    }[name]
    return data_sources.get(rel, dest_dir=dest_dir)


@dataclass
class PitchDataset:
    """Loaded biomechanics features, target velocity, and metadata."""

    poi: pd.DataFrame
    meta: pd.DataFrame
    feature_names: list[str]
    X: np.ndarray          # (n_pitches, n_features), median-imputed
    y: np.ndarray          # (n_pitches,) pitch_speed_mph
    feature_mean: np.ndarray
    feature_std: np.ndarray
    _row_of: dict          # session_pitch -> row index

    def index_of(self, session_pitch: str) -> int:
        return self._row_of[session_pitch]

    def zscores(self, session_pitch: str) -> pd.DataFrame:
        """Per-feature z-scores for one pitch vs. the whole dataset.

        Returns a DataFrame with columns ``feature``, ``value``, ``z`` sorted by
        descending absolute z-score.
        """
        i = self.index_of(session_pitch)
        z = (self.X[i] - self.feature_mean) / self.feature_std
        df = pd.DataFrame(
            {"feature": self.feature_names, "value": self.X[i], "z": z}
        )
        return df.reindex(df.z.abs().sort_values(ascending=False).index).reset_index(
            drop=True
        )

    def c3d_filename(self, session_pitch: str) -> str:
        row = self.meta.loc[self.meta.session_pitch == session_pitch]
        if row.empty:
            raise KeyError(f"session_pitch {session_pitch!r} not in metadata")
        return str(row.iloc[0]["filename_new"])

    def actual_velocity(self, session_pitch: str) -> float:
        return float(self.y[self.index_of(session_pitch)])


def load_dataset(data_dir: str = "obp_data", download: bool = True) -> PitchDataset:
    """Load (downloading if needed) and assemble the modelling dataset."""
    poi_path = os.path.join(data_dir, "poi_metrics.csv")
    meta_path = os.path.join(data_dir, "metadata.csv")
    if download:
        poi_path = download_obp_table("poi_metrics", data_dir)
        meta_path = download_obp_table("metadata", data_dir)

    poi = pd.read_csv(poi_path)
    meta = pd.read_csv(meta_path)

    feature_names = [
        c
        for c in poi.columns
        if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(poi[c])
    ]
    X = poi[feature_names].to_numpy(dtype=float)
    # Median-impute the small number of missing cells.
    col_median = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_median, nan_idx[1])

    y = poi[TARGET_COL].to_numpy(dtype=float)
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    row_of = {sp: i for i, sp in enumerate(poi["session_pitch"].astype(str))}

    return PitchDataset(
        poi=poi,
        meta=meta,
        feature_names=feature_names,
        X=X,
        y=y,
        feature_mean=feature_mean,
        feature_std=feature_std,
        _row_of=row_of,
    )


@dataclass
class TrainedVelocityModel:
    """A fitted Bayesian Lasso plus the data it was trained on."""

    model: BayesianLasso
    dataset: PitchDataset
    test_idx: np.ndarray
    metrics: dict

    def predict_pitch(self, session_pitch: str):
        """Return ``(mean, std)`` predicted velocity for one pitch."""
        i = self.dataset.index_of(session_pitch)
        mean, std = self.model.predict(self.dataset.X[i : i + 1], return_std=True)
        return float(mean[0]), float(std[0])


def train_velocity_model(
    dataset: PitchDataset | None = None,
    test_size: float = 0.2,
    random_state: int = 0,
    **lasso_kwargs,
) -> TrainedVelocityModel:
    """Fit a Bayesian Lasso to predict pitch velocity from biomechanics.

    Holds out a random test split and reports R^2 / RMSE on it.
    """
    if dataset is None:
        dataset = load_dataset()

    rng = np.random.default_rng(random_state)
    n = len(dataset.y)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    model = BayesianLasso(random_state=random_state, **lasso_kwargs)
    model.fit(dataset.X[train_idx], dataset.y[train_idx])

    pred = model.predict(dataset.X[test_idx])
    actual = dataset.y[test_idx]
    ss_res = float(((actual - pred) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    metrics = {
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "rmse": float(np.sqrt(((actual - pred) ** 2).mean())),
        "mae": float(np.abs(actual - pred).mean()),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    return TrainedVelocityModel(
        model=model, dataset=dataset, test_idx=test_idx, metrics=metrics
    )
