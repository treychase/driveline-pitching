"""Joint work/energetics and joint-centre velocity vectors for the OBP data.

Two quantities are computed, both consistent with the C3D animation frame:

* **Joint work** — the net mechanical energy generated at each joint, taken from
  OpenBiomechanics' authoritative ``energy_flow.zip`` (``<joint>_energy_generated``,
  in Joules, accumulated over the delivery). Positive = the joint generates
  energy (concentric), negative = it absorbs (eccentric). Reported cumulatively
  and as a total at ball release. (This is preferable to recomputing power as
  M·ω from the raw signals, whose signs are altered by OBP's intuition
  adjustments.)

* **Joint velocity vectors** — the linear velocity of each joint centre, from
  finite-differencing the joint-centre positions in ``landmarks.zip`` (same
  lab/C3D frame). These are the arrows drawn on the joints in the 3D view.

Dataset-wide work-at-release is cached to ``obp_data/joint_work_summary.csv`` so
z-scores of joint work can be shown for any pitch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Joints for which OBP provides net energy generation (Joules).
ENERGY_JOINTS = [
    "shoulder", "elbow", "lead_hip", "lead_knee",
    "rear_hip", "rear_knee", "glove_shoulder", "glove_elbow",
]

# Joint-centre landmark stems for the velocity-vector overlay.
VECTOR_JOINTS = {
    "rear_ankle": "rear_ankle_jc", "rear_knee": "rear_knee_jc",
    "rear_hip": "rear_hip", "lead_ankle": "lead_ankle_jc",
    "lead_knee": "lead_knee_jc", "lead_hip": "lead_hip",
    "shoulder": "shoulder_jc", "elbow": "elbow_jc", "wrist": "wrist_jc",
    "hand": "hand_jc", "glove_shoulder": "glove_shoulder_jc",
    "glove_elbow": "glove_elbow_jc", "glove_wrist": "glove_wrist_jc",
    "glove_hand": "glove_hand_jc",
}

_EVENT_COLS = {"pkh": "pkh_time", "fp": "fp_100_time", "mer": "MER_time",
               "br": "BR_time", "mir": "MIR_time"}

_ENERGY: pd.DataFrame | None = None
_LANDMARKS: pd.DataFrame | None = None


def _axis_cols(stem):
    return [f"{stem}_x", f"{stem}_y", f"{stem}_z"]


def _energy_col(joint):
    return f"{joint}_energy_generated"


def _load_energy(data_dir="obp_data"):
    global _ENERGY
    if _ENERGY is None:
        cols = {"session_pitch", "time", "BR_time", "pkh_time", "fp_100_time",
                "MER_time", "MIR_time"}
        cols.update(_energy_col(j) for j in ENERGY_JOINTS)
        _ENERGY = pd.read_csv(
            os.path.join(data_dir, "energy_flow.zip"),
            usecols=lambda c: c in cols,
        )
    return _ENERGY


def _load_landmarks(data_dir="obp_data"):
    global _LANDMARKS
    if _LANDMARKS is None:
        cols = {"session_pitch", "time"}
        for stem in VECTOR_JOINTS.values():
            cols.update(_axis_cols(stem))
        _LANDMARKS = pd.read_csv(
            os.path.join(data_dir, "landmarks.zip"),
            usecols=lambda c: c in cols,
        )
    return _LANDMARKS


def _clean(series: pd.Series) -> np.ndarray:
    """Fill the leading/trailing NaNs of a cumulative-energy series."""
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(float)


@dataclass
class JointWork:
    """Time-resolved joint work (energy generated) for one pitch."""

    session_pitch: str
    time: np.ndarray
    work: dict                       # joint -> (n,) cumulative joules
    power: dict                      # joint -> (n,) watts (d work / dt)
    events: dict

    def work_at(self, t: float) -> dict:
        return {j: float(np.interp(t, self.time, w)) for j, w in self.work.items()}

    def work_at_release(self) -> dict:
        br = self.events.get("br")
        t = self.time[-1] if br is None else br
        return self.work_at(t)


def load_joint_work(session_pitch, data_dir="obp_data") -> JointWork:
    """Per-joint cumulative work (and power) over the delivery for one pitch."""
    en = _load_energy(data_dir)
    g = en[en.session_pitch.astype(str) == str(session_pitch)].sort_values("time")
    if g.empty:
        raise KeyError(f"No energy-flow data for {session_pitch!r}")
    t = g["time"].to_numpy(float)
    work, power = {}, {}
    for j in ENERGY_JOINTS:
        col = _energy_col(j)
        if col not in g:
            continue
        w = _clean(g[col])
        work[j] = w
        power[j] = np.gradient(w, t)
    events = {k: float(g[c].iloc[0]) for k, c in _EVENT_COLS.items() if c in g}
    return JointWork(str(session_pitch), t, work, power, events)


def load_joint_vectors(session_pitch, frame_times, data_dir="obp_data") -> dict:
    """Joint-centre positions and linear-velocity vectors aligned to frames.

    Returns ``{joint: {"pos": (F,3), "vel": (F,3)}}`` in the lab/C3D frame.
    """
    lm = _load_landmarks(data_dir)
    g = lm[lm.session_pitch.astype(str) == str(session_pitch)].sort_values("time")
    if g.empty:
        raise KeyError(f"No landmarks for {session_pitch!r}")
    t = g["time"].to_numpy(float)
    out = {}
    for joint, stem in VECTOR_JOINTS.items():
        cols = _axis_cols(stem)
        if not all(c in g for c in cols):
            continue
        pos = g[cols].to_numpy(float)
        vel = np.gradient(pos, t, axis=0)
        pos_f = np.vstack([np.interp(frame_times, t, pos[:, k]) for k in range(3)]).T
        vel_f = np.vstack([np.interp(frame_times, t, vel[:, k]) for k in range(3)]).T
        out[joint] = {"pos": pos_f, "vel": vel_f}
    return out


def work_summary(data_dir="obp_data", rebuild=False) -> pd.DataFrame:
    """Per-pitch joint work at ball release for the whole dataset (cached)."""
    cache = os.path.join(data_dir, "joint_work_summary.csv")
    if os.path.exists(cache) and not rebuild:
        return pd.read_csv(cache, index_col="session_pitch")

    en = _load_energy(data_dir)
    rows = {}
    for sp, g in en.groupby("session_pitch"):
        g = g.sort_values("time")
        t = g["time"].to_numpy(float)
        br = float(g["BR_time"].iloc[0])
        rows[sp] = {j: float(np.interp(br, t, _clean(g[_energy_col(j)])))
                    for j in ENERGY_JOINTS if _energy_col(j) in g}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "session_pitch"
    df.to_csv(cache)
    return df


def work_zscores(session_pitch, data_dir="obp_data") -> pd.DataFrame:
    """Z-scores of this pitch's joint work vs. the dataset, sorted by |z|."""
    df = work_summary(data_dir)
    mu, sd = df.mean(), df.std().replace(0, 1.0)
    sp = str(session_pitch)
    if sp not in df.index:
        raise KeyError(f"{sp!r} not in work summary")
    z = (df.loc[sp] - mu) / sd
    out = pd.DataFrame({
        "joint": z.index, "work_J": df.loc[sp].to_numpy(), "z": z.to_numpy(),
    })
    return out.reindex(out.z.abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


if __name__ == "__main__":
    import sys
    sp = sys.argv[1] if len(sys.argv) > 1 else "1097_1"
    jw = load_joint_work(sp)
    print(f"Joint work (energy generated) at release for {sp} (J):")
    for j, w in sorted(jw.work_at_release().items(), key=lambda kv: -abs(kv[1])):
        print(f"  {j:16s} {w:8.1f}")
    print("\nz-scores vs dataset:")
    print(work_zscores(sp).to_string(index=False))
