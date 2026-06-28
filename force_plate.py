"""Per-pitch ground-reaction force-plate signals for the OBP pitching data.

Loads the rear-leg (drive) and lead-leg (stride) force-plate time series from
the OpenBiomechanics ``full_sig/force_plate.zip`` and aligns them to the C3D
marker frames so the forces can be displayed *live* alongside the 3D pose
animation. The C3D markers and the force plates share a common clock (both start
at trial t=0 and span the same duration), so alignment is a simple time
interpolation.

Force-plate sign conventions (from the OBP documentation):
  * rear plate (FP2):  +x = push-off (anterior, toward home plate)
  * lead plate (FP1/3): +x = braking (posterior)
  * both plates:        +y = lateral, +z = superior (vertical, supports weight)
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

OBP_FP_URL = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "main/baseball_pitching/data/full_sig/force_plate.zip"
)

_EVENT_COLS = {
    "pkh": "pkh_time",          # peak knee height (start of stride)
    "fc": "fp_10_time",         # foot contact (10% bodyweight)
    "fp": "fp_100_time",        # foot plant (100% bodyweight)
    "mer": "MER_time",          # max external rotation (layback)
    "br": "BR_time",            # ball release
    "mir": "MIR_time",          # max internal rotation
}

_EVENT_LABELS = {
    "pkh": "Peak knee height",
    "fc": "Foot contact",
    "fp": "Foot plant",
    "mer": "Max ext. rotation",
    "br": "Ball release",
    "mir": "Max int. rotation",
}

# Module-level cache of the (large) combined CSV so repeated pitch lookups are cheap.
_ALL_FP: pd.DataFrame | None = None


def download_force_plate_zip(dest_dir: str = "obp_data") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "force_plate.zip")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(OBP_FP_URL, dest)
    return dest


def _load_all(data_dir: str = "obp_data", download: bool = True) -> pd.DataFrame:
    global _ALL_FP
    if _ALL_FP is None:
        path = os.path.join(data_dir, "force_plate.zip")
        if download and not os.path.exists(path):
            path = download_force_plate_zip(data_dir)
        _ALL_FP = pd.read_csv(path)
    return _ALL_FP


@dataclass
class ForcePlateTrace:
    """Force-plate time series for a single pitch.

    ``rear`` and ``lead`` are ``(n_samples, 3)`` arrays of (x, y, z) force in
    Newtons. ``events`` maps short event keys to times in seconds.
    """

    session_pitch: str
    time: np.ndarray            # (n_samples,) seconds
    rear: np.ndarray            # (n_samples, 3) N
    lead: np.ndarray            # (n_samples, 3) N
    events: dict
    bodyweight_n: float | None = None  # mass*g, if known, for BW normalisation

    @property
    def rear_vertical(self) -> np.ndarray:
        return self.rear[:, 2]

    @property
    def lead_vertical(self) -> np.ndarray:
        return self.lead[:, 2]

    @property
    def rear_mag(self) -> np.ndarray:
        return np.linalg.norm(self.rear, axis=1)

    @property
    def lead_mag(self) -> np.ndarray:
        return np.linalg.norm(self.lead, axis=1)

    def in_bw(self, force_n: np.ndarray) -> np.ndarray:
        """Convert a force array to bodyweight multiples if mass is known."""
        if self.bodyweight_n:
            return force_n / self.bodyweight_n
        return force_n

    def align_to_frames(self, frame_times: np.ndarray) -> dict:
        """Interpolate the forces onto the given ``frame_times`` (seconds).

        Returns a dict of per-frame arrays (rear/lead vertical & magnitude, in N
        and, if mass is known, in bodyweight multiples) plus the event frame
        indices.
        """
        def interp(sig):
            return np.interp(frame_times, self.time, sig, left=0.0, right=0.0)

        out = {
            "rear_vertical": interp(self.rear_vertical),
            "lead_vertical": interp(self.lead_vertical),
            "rear_mag": interp(self.rear_mag),
            "lead_mag": interp(self.lead_mag),
        }
        if self.bodyweight_n:
            for k in list(out):
                out[k + "_bw"] = out[k] / self.bodyweight_n
        # nearest frame index for each event
        ev_frames = {}
        for key, t in self.events.items():
            if np.isfinite(t):
                ev_frames[key] = int(np.argmin(np.abs(frame_times - t)))
        out["event_frames"] = ev_frames
        out["peak_rear"] = float(self.rear_mag.max())
        out["peak_lead"] = float(self.lead_mag.max())
        return out


def load_force_plate(
    session_pitch: str,
    data_dir: str = "obp_data",
    download: bool = True,
    bodyweight_n: float | None = None,
) -> ForcePlateTrace:
    """Load the force-plate trace for one ``session_pitch``."""
    df = _load_all(data_dir, download=download)
    g = df[df["session_pitch"].astype(str) == str(session_pitch)]
    if g.empty:
        raise KeyError(f"No force-plate data for session_pitch {session_pitch!r}")
    g = g.sort_values("time")
    events = {
        key: float(g[col].iloc[0]) for key, col in _EVENT_COLS.items() if col in g
    }
    return ForcePlateTrace(
        session_pitch=str(session_pitch),
        time=g["time"].to_numpy(float),
        rear=g[["rear_force_x", "rear_force_y", "rear_force_z"]].to_numpy(float),
        lead=g[["lead_force_x", "lead_force_y", "lead_force_z"]].to_numpy(float),
        events=events,
        bodyweight_n=bodyweight_n,
    )


def event_label(key: str) -> str:
    return _EVENT_LABELS.get(key, key)


# Light, distinct phase tints (readable behind the GRF traces on a white theme).
_PHASE_SEQ = [
    ("Wind-up", "pkh", "#e3e8ee"),
    ("Stride", "fp", "#d9ecdc"),
    ("Arm cocking", "mer", "#fdf0c9"),
    ("Acceleration", "br", "#ffd9b3"),
]
_PHASE_FINAL = ("Deceleration", "#f3d4d4")


def delivery_phases(events: dict, t_end: float) -> list[dict]:
    """Return the distinct delivery phases as ``{label, t0, t1, color}`` spans.

    Phases are delimited by the standard events: Wind-up (start→peak knee
    height), Stride (→foot plant), Arm cocking (→max external rotation),
    Acceleration (→ball release), and Deceleration/follow-through (→end).
    Phases whose bounding event is missing are skipped.
    """
    phases, prev = [], 0.0
    for label, ev, color in _PHASE_SEQ:
        t = events.get(ev)
        if t is None or not np.isfinite(t) or t <= prev:
            continue
        phases.append({"label": label, "t0": prev, "t1": float(t), "color": color})
        prev = float(t)
    if t_end > prev:
        label, color = _PHASE_FINAL
        phases.append({"label": label, "t0": prev, "t1": float(t_end), "color": color})
    return phases
