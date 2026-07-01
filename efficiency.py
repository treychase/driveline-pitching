"""Mechanical efficiency model for OpenBiomechanics pitching deliveries.

This model scores *how* a pitcher generates velocity: how much mechanical
energy is produced by the big, durable engines — the **torso** and the **lower
body** (hips and knees) — relative to the load placed on the fragile
**throwing elbow**.

The intuition is a classic pitching-development one. Velocity can be bought two
ways: by powering the delivery from the ground up (drive off the rear leg,
brace the lead leg, rotate the pelvis and trunk) and letting that energy flow up
the kinetic chain, or by cranking on the arm and paying for it in elbow valgus
torque (the load the UCL resists). The first pattern is efficient and
arm-friendly; the second is fragile. This model turns that trade-off into a
single, interpretable 0-100 score.

The raw efficiency index for a pitch is

    raw = drive_z  -  elbow_load_z

where

* ``drive_z`` averages the torso group and the lower-body group (each group
  weighted equally) of *standardized* energy-generation / energy-transfer
  metrics — i.e. how much positive mechanical work the trunk, hips, and knees
  put into the throw relative to the rest of the dataset, and
* ``elbow_load_z`` is the standardized elbow varus moment — the peak valgus
  load on the medial elbow (the canonical elbow-injury driver).

``raw`` is high when a pitcher drives hard with the torso and legs **and**
spares the elbow. It is then mapped to a **0-100 percentile within the
dataset**, so a score of 80 means "more mechanically efficient than 80% of the
pitches in this dataset."

All inputs come from ``poi_metrics.csv`` (already loaded by
:mod:`velocity_model`), so the model needs no extra downloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Metric groups
# --------------------------------------------------------------------------- #
# Lower-body engines: net energy the hips and knees *generate* (positive
# concentric work) over the delivery, in Joules.
LOWER_BODY_DRIVERS: list[str] = [
    "rear_hip_generation_pkh_fp",
    "rear_knee_generation_pkh_fp",
    "lead_hip_generation_fp_br",
    "lead_knee_generation_fp_br",
]

# Torso engine: energy transferred through the pelvis→lumbar and the thorax
# (trunk) links of the kinetic chain — the rotational drive of the core.
TORSO_DRIVERS: list[str] = [
    "pelvis_lumbar_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
]

# Throwing-elbow load: peak internal varus moment resisting the valgus torque
# at the medial elbow — the load most associated with UCL injury.
ELBOW_LOAD: list[str] = [
    "elbow_varus_moment",
]

_GROUPS = {
    "lower_body": LOWER_BODY_DRIVERS,
    "torso": TORSO_DRIVERS,
    "elbow_load": ELBOW_LOAD,
}


@dataclass
class EfficiencyScore:
    """Mechanical efficiency result for one pitch.

    Attributes
    ----------
    session_pitch:
        Pitch identifier.
    score:
        0-100 percentile of the raw efficiency index within the dataset. Higher
        = more torso/lower-body drive per unit of elbow load.
    raw:
        Standardized efficiency index (``drive_z - elbow_load_z``).
    drive:
        Combined torso + lower-body drive (mean of the two group z-scores).
    lower_body, torso:
        Group z-scores (average standardized energy generation for each group).
    elbow_load:
        Standardized elbow varus moment (higher = more elbow stress).
    contributions:
        Per-metric breakdown (``metric``, ``group``, ``value``, ``z``,
        ``signed_z`` where drivers keep their sign and the elbow load is
        negated so that "helps efficiency" is always positive).
    """

    session_pitch: str
    score: float
    raw: float
    drive: float
    lower_body: float
    torso: float
    elbow_load: float
    contributions: pd.DataFrame

    def verdict(self) -> str:
        """Short human-readable interpretation of the score."""
        if self.score >= 75:
            return ("Efficient, arm-friendly pattern: strong torso/lower-body "
                    "drive with a comparatively light elbow load.")
        if self.score >= 45:
            return ("Balanced pattern: torso/lower-body drive and elbow load "
                    "are both near the dataset average.")
        return ("Arm-reliant pattern: velocity comes with high elbow load "
                "relative to the torso/lower-body energy generated.")


@dataclass
class MechanicalEfficiencyModel:
    """Dataset-fitted mechanical efficiency scorer.

    Fit once on the pitch POI table; then :meth:`score` any pitch. Column means
    and standard deviations are learned on ``fit`` and reused so every pitch is
    scored on the same scale, and the 0-100 score is the empirical percentile of
    the raw index within the fitted dataset.
    """

    columns: list[str]                 # driver/load columns actually present
    mean: pd.Series                    # per-column mean (dataset)
    std: pd.Series                     # per-column std (dataset, 0 -> 1)
    groups: dict                       # group name -> present columns
    raw_sorted: np.ndarray             # sorted raw index over the dataset
    _row_of: dict                      # session_pitch -> row index
    _poi: pd.DataFrame                 # the POI table (for value lookups)

    # ---- fitting -------------------------------------------------------- #
    @classmethod
    def fit(cls, poi: pd.DataFrame) -> "MechanicalEfficiencyModel":
        """Fit the model on a ``poi_metrics``-style DataFrame."""
        groups = {name: [c for c in cols if c in poi.columns]
                  for name, cols in _GROUPS.items()}
        if not groups["elbow_load"]:
            raise ValueError("poi table lacks 'elbow_varus_moment' for the "
                             "elbow-load term of the efficiency model.")
        if not (groups["lower_body"] or groups["torso"]):
            raise ValueError("poi table lacks any torso/lower-body drive "
                             "columns for the efficiency model.")

        columns = groups["lower_body"] + groups["torso"] + groups["elbow_load"]
        vals = poi[columns].astype(float)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0).replace(0.0, 1.0)

        row_of = {str(sp): i for i, sp in enumerate(poi["session_pitch"])}
        model = cls(columns=columns, mean=mean, std=std, groups=groups,
                    raw_sorted=np.empty(0), _row_of=row_of, _poi=poi)
        # Second pass: raw index over the whole dataset -> percentile lookup.
        raw = np.array([model._raw_from_row(i) for i in range(len(poi))])
        model.raw_sorted = np.sort(raw[np.isfinite(raw)])
        return model

    # ---- scoring -------------------------------------------------------- #
    def _z(self, col: str, value: float) -> float:
        return float((value - self.mean[col]) / self.std[col])

    def _group_z(self, group: str, row: pd.Series) -> float:
        cols = self.groups[group]
        if not cols:
            return 0.0
        zs = [self._z(c, row[c]) for c in cols if np.isfinite(row[c])]
        return float(np.mean(zs)) if zs else 0.0

    def _raw_from_row(self, i: int) -> float:
        row = self._poi.iloc[i]
        lower = self._group_z("lower_body", row)
        torso = self._group_z("torso", row)
        # Equal weight to the two driver groups when both are present.
        present = [g for g in ("lower_body", "torso") if self.groups[g]]
        drive = float(np.mean([{"lower_body": lower, "torso": torso}[g]
                               for g in present])) if present else 0.0
        load = self._group_z("elbow_load", row)
        return drive - load

    def _percentile(self, raw: float) -> float:
        n = len(self.raw_sorted)
        if n == 0 or not np.isfinite(raw):
            return float("nan")
        # Fraction of the dataset with a raw index <= this pitch's, in [0, 100].
        rank = float(np.searchsorted(self.raw_sorted, raw, side="right"))
        return 100.0 * rank / n

    def score(self, session_pitch: str) -> EfficiencyScore:
        """Score a single pitch by its ``session_pitch`` id."""
        sp = str(session_pitch)
        if sp not in self._row_of:
            raise KeyError(f"{sp!r} not in the efficiency model's dataset")
        i = self._row_of[sp]
        row = self._poi.iloc[i]

        lower = self._group_z("lower_body", row)
        torso = self._group_z("torso", row)
        present = [g for g in ("lower_body", "torso") if self.groups[g]]
        drive = float(np.mean([{"lower_body": lower, "torso": torso}[g]
                               for g in present])) if present else 0.0
        load = self._group_z("elbow_load", row)
        raw = drive - load
        score = self._percentile(raw)

        recs = []
        for group in ("lower_body", "torso", "elbow_load"):
            for c in self.groups[group]:
                z = self._z(c, row[c]) if np.isfinite(row[c]) else float("nan")
                recs.append({
                    "metric": c, "group": group,
                    "value": float(row[c]),
                    "z": z,
                    # Positive = helps efficiency (drivers up, elbow load down).
                    "signed_z": (-z if group == "elbow_load" else z),
                })
        contributions = pd.DataFrame(recs)

        return EfficiencyScore(
            session_pitch=sp, score=score, raw=raw, drive=drive,
            lower_body=lower, torso=torso, elbow_load=load,
            contributions=contributions,
        )

    def summary(self) -> pd.DataFrame:
        """Raw index and 0-100 score for every pitch in the fitted dataset."""
        rows = []
        for sp, i in self._row_of.items():
            raw = self._raw_from_row(i)
            rows.append({"session_pitch": sp, "raw": raw,
                         "score": self._percentile(raw)})
        return pd.DataFrame(rows).set_index("session_pitch")


def load_efficiency_model(dataset=None, data_dir: str = "obp_data"
                          ) -> MechanicalEfficiencyModel:
    """Convenience loader: fit the efficiency model on the OBP POI table.

    Parameters
    ----------
    dataset:
        An optional :class:`velocity_model.PitchDataset` (its ``poi`` table is
        reused). If omitted, the POI table is loaded via
        :func:`velocity_model.load_dataset`.
    """
    if dataset is not None:
        poi = dataset.poi
    else:
        from velocity_model import load_dataset
        poi = load_dataset(data_dir=data_dir).poi
    return MechanicalEfficiencyModel.fit(poi)


if __name__ == "__main__":
    import sys

    from velocity_model import load_dataset

    ds = load_dataset()
    model = MechanicalEfficiencyModel.fit(ds.poi)
    sp = sys.argv[1] if len(sys.argv) > 1 else str(ds.poi["session_pitch"].iloc[0])
    res = model.score(sp)
    print(f"Mechanical efficiency for {sp}: {res.score:.0f}/100")
    print(f"  drive (torso+lower body) z = {res.drive:+.2f}  "
          f"(lower body {res.lower_body:+.2f}, torso {res.torso:+.2f})")
    print(f"  elbow load z               = {res.elbow_load:+.2f}")
    print(f"  raw index                  = {res.raw:+.2f}")
    print(f"  {res.verdict()}")
    print("\n  contributions (signed so + = helps efficiency):")
    for _, r in res.contributions.iterrows():
        print(f"    {r['metric']:32s} {r['group']:11s} z={r['z']:+.2f} "
              f"-> {r['signed_z']:+.2f}")
