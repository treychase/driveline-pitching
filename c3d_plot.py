"""3D plotting utilities for OpenBiomechanics C3D motion-capture files.

This module loads marker (point) data from C3D files and renders them in 3D as
either a single static frame or an animation across frames. It is designed for
the Driveline OpenBiomechanics Project pitching dataset
(https://github.com/drivelineresearch/openbiomechanics), whose C3D files use a
standard Vicon Plug-in-Gait marker set, but it works with any C3D file that
contains 3D point data.

Typical usage
-------------
    # Plot a single frame from a local file
    plot_c3d("000002_003034_73_207_002_FF_809.c3d", frame=300, save_path="frame.png")

    # Animate the whole trial to a gif
    animate_c3d("000002_003034_73_207_002_FF_809.c3d", save_path="pitch.gif")

    # Pull a file straight from the OpenBiomechanics repo and plot it
    path = download_obp_c3d("000002", "000002_003034_73_207_002_FF_809.c3d")
    animate_c3d(path, save_path="pitch.mp4")

Dependencies: numpy, matplotlib, and a C3D reader (``ezc3d`` preferred, with a
fallback to the pure-python ``c3d`` package). See requirements.txt.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# Matplotlib is imported lazily inside the plotting functions so that the data
# loaders can be used in headless / non-plotting contexts without the import.


# Base URL for raw files in the OpenBiomechanics repository.
OBP_RAW_BASE = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "main/baseball_pitching/data/c3d"
)

# Segment connections for the Vicon Plug-in-Gait marker set used by the
# OpenBiomechanics pitching data. Each tuple is a pair of marker labels that
# should be joined by a line to draw a stick figure. Segments whose endpoints
# are not both present in a given file are silently skipped.
PLUG_IN_GAIT_SEGMENTS: list[tuple[str, str]] = [
    # Head
    ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
    # Trunk / spine
    ("LFHD", "C7"), ("RFHD", "C7"), ("C7", "CLAV"), ("CLAV", "STRN"),
    ("C7", "T10"), ("T10", "STRN"),
    # Shoulders
    ("C7", "LSHO"), ("C7", "RSHO"), ("CLAV", "LSHO"), ("CLAV", "RSHO"),
    # Left arm
    ("LSHO", "LUPA"), ("LUPA", "LELB"), ("LSHO", "LELB"),
    ("LELB", "LFRM"), ("LFRM", "LWRA"), ("LFRM", "LWRB"),
    ("LELB", "LWRA"), ("LELB", "LWRB"),
    ("LWRA", "LWRB"), ("LWRA", "LFIN"), ("LWRB", "LFIN"),
    # Right arm
    ("RSHO", "RUPA"), ("RUPA", "RELB"), ("RSHO", "RELB"),
    ("RELB", "RFRM"), ("RFRM", "RWRA"), ("RFRM", "RWRB"),
    ("RELB", "RWRA"), ("RELB", "RWRB"),
    ("RWRA", "RWRB"), ("RWRA", "RFIN"), ("RWRB", "RFIN"),
    # Pelvis
    ("LASI", "RASI"), ("RASI", "RPSI"), ("RPSI", "LPSI"), ("LPSI", "LASI"),
    # Trunk to pelvis
    ("T10", "LPSI"), ("T10", "RPSI"),
    # Left leg
    ("LASI", "LTHI"), ("LTHI", "LKNE"), ("LKNE", "LTIB"),
    ("LTIB", "LANK"), ("LKNE", "LANK"),
    ("LANK", "LHEE"), ("LHEE", "LTOE"), ("LANK", "LTOE"),
    # Right leg
    ("RASI", "RTHI"), ("RTHI", "RKNE"), ("RKNE", "RTIB"),
    ("RTIB", "RANK"), ("RKNE", "RANK"),
    ("RANK", "RHEE"), ("RHEE", "RTOE"), ("RANK", "RTOE"),
]


@dataclass
class C3DMarkers:
    """Container for marker trajectories loaded from a C3D file.

    Attributes
    ----------
    points:
        Array of shape ``(n_frames, n_markers, 3)`` with XYZ coordinates.
        Missing/invalid samples are stored as ``np.nan``.
    labels:
        List of ``n_markers`` marker names (e.g. ``"RWRA"``).
    rate:
        Point sampling rate in Hz.
    units:
        Length unit of the coordinates (e.g. ``"m"`` or ``"mm"``).
    """

    points: np.ndarray
    labels: list[str]
    rate: float
    units: str = "m"
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {label: i for i, label in enumerate(self.labels)}

    @property
    def n_frames(self) -> int:
        return self.points.shape[0]

    @property
    def n_markers(self) -> int:
        return self.points.shape[1]

    def marker(self, label: str) -> np.ndarray:
        """Return the ``(n_frames, 3)`` trajectory for a single marker label."""
        return self.points[:, self._index[label], :]

    def has(self, label: str) -> bool:
        return label in self._index


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_c3d(path: str) -> C3DMarkers:
    """Load marker data from a C3D file.

    Tries the ``ezc3d`` reader first (fast, robust) and falls back to the
    pure-python ``c3d`` package if ``ezc3d`` is not installed.

    Parameters
    ----------
    path:
        Path to a ``.c3d`` file.

    Returns
    -------
    C3DMarkers
        Loaded marker trajectories with NaNs for invalid samples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        return _load_with_ezc3d(path)
    except ImportError:
        return _load_with_c3d(path)


def _load_with_ezc3d(path: str) -> C3DMarkers:
    import ezc3d

    c = ezc3d.c3d(path)
    # ezc3d points: (4, n_markers, n_frames) -> (x, y, z, residual)
    raw = c["data"]["points"]
    xyz = np.asarray(raw[:3], dtype=float)               # (3, m, f)
    points = np.transpose(xyz, (2, 1, 0))                # (f, m, 3)

    params = c["parameters"]["POINT"]
    labels = [str(s).strip() for s in params["LABELS"]["value"]]
    rate = float(params["RATE"]["value"][0])
    units_val = params.get("UNITS", {}).get("value", ["m"])
    units = str(units_val[0]) if units_val else "m"

    points = _clean_invalid(points)
    return C3DMarkers(points=points, labels=labels, rate=rate, units=units)


def _load_with_c3d(path: str) -> C3DMarkers:
    try:
        import c3d as c3d_lib
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(
            "No C3D reader available. Install 'ezc3d' (recommended) or 'c3d':\n"
            "    pip install ezc3d\n"
            "    pip install c3d"
        ) from exc

    with open(path, "rb") as handle:
        reader = c3d_lib.Reader(handle)
        labels = [label.strip() for label in reader.point_labels]
        rate = float(reader.point_rate)
        frames = []
        for _, point, _ in reader.read_frames():
            frames.append(np.asarray(point[:, :3], dtype=float))
    points = np.stack(frames, axis=0) if frames else np.empty((0, len(labels), 3))
    points = _clean_invalid(points)
    return C3DMarkers(points=points, labels=labels, rate=rate, units="m")


def _clean_invalid(points: np.ndarray) -> np.ndarray:
    """Replace exact (0, 0, 0) gap samples and non-finite values with NaN."""
    points = points.astype(float, copy=True)
    zero_gap = np.all(points == 0.0, axis=-1)
    points[zero_gap] = np.nan
    points[~np.isfinite(points)] = np.nan
    return points


def download_obp_c3d(
    session: str,
    filename: str,
    dest_dir: str = "c3d_data",
    base_url: str = OBP_RAW_BASE,
) -> str:
    """Download a single C3D file from the OpenBiomechanics repository.

    Parameters
    ----------
    session:
        The numbered session folder, e.g. ``"000002"``.
    filename:
        The C3D file name within that folder, e.g.
        ``"000002_003034_73_207_002_FF_809.c3d"``.
    dest_dir:
        Local directory to save into (created if needed).
    base_url:
        Unused when the data-source layer is available (kept for compatibility);
        a direct download from this base URL is used as a fallback.

    Returns
    -------
    str
        Local path to the downloaded file.
    """
    try:
        import data_sources
        return data_sources.get(f"c3d/{session}/{filename}", local_name=filename,
                                dest_dir=dest_dir)
    except ImportError:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(f"{base_url}/{session}/{filename}", dest)
        return dest


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def _segments_present(markers: C3DMarkers, segments) -> list[tuple[int, int]]:
    """Resolve label-pair segments to index pairs, keeping only present ones."""
    if segments is None:
        segments = PLUG_IN_GAIT_SEGMENTS
    pairs = []
    for a, b in segments:
        if markers.has(a) and markers.has(b):
            pairs.append((markers._index[a], markers._index[b]))
    return pairs


def _equal_aspect_bounds(points: np.ndarray, extra_points=None):
    """Compute symmetric axis limits so the 3D plot keeps a 1:1:1 aspect.

    ``extra_points`` (e.g. mound vertices) are included in the bounds so added
    scenery is not clipped.
    """
    finite = points[np.isfinite(points).all(axis=-1)]
    if extra_points is not None and len(extra_points):
        extra = np.asarray(extra_points, float)
        extra = extra[np.isfinite(extra).all(axis=-1)]
        finite = np.vstack([finite, extra]) if finite.size else extra
    if finite.size == 0:
        return (-1, 1), (-1, 1), (-1, 1)
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = radius if radius > 0 else 1.0
    return tuple((c - radius, c + radius) for c in centers)


# --------------------------------------------------------------------------- #
# Public plotting API
# --------------------------------------------------------------------------- #

def plot_c3d(
    source,
    frame: int = 0,
    segments=PLUG_IN_GAIT_SEGMENTS,
    show_labels: bool = False,
    elev: float = 12.0,
    azim: float = -60.0,
    title: str | None = None,
    save_path: str | None = None,
    mound: bool = False,
    ax=None,
):
    """Plot a single frame of a C3D file as a 3D stick figure.

    Parameters
    ----------
    source:
        Either a path to a ``.c3d`` file or an already-loaded :class:`C3DMarkers`.
    frame:
        Frame index to render. Negative indices count from the end.
    segments:
        Iterable of ``(label_a, label_b)`` pairs to connect. Pass ``None`` for
        markers only (no skeleton). Defaults to the Plug-in-Gait segment set.
    show_labels:
        If True, annotate each marker with its name.
    elev, azim:
        Initial 3D view angles (degrees).
    title:
        Plot title. Defaults to the file name and frame.
    save_path:
        If given, save the figure to this path instead of (only) returning it.
    mound:
        If True, draw a polygonal dirt pitching mound under the pitcher,
        positioned and sized from the foot markers.
    ax:
        Optional existing 3D axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes3D
        The axes the figure was drawn on.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    markers = source if isinstance(source, C3DMarkers) else load_c3d(source)
    if markers.n_frames == 0:
        raise ValueError("C3D file contains no frames.")
    frame = range(markers.n_frames)[frame]  # normalize / validate index

    pairs = _segments_present(markers, segments)
    coords = markers.points[frame]  # (n_markers, 3)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Optional dirt mound under the pitcher
    mound_verts = None
    if mound:
        from mound import add_mound
        mound_verts = add_mound(ax, markers)

    # Skeleton segments
    for i, j in pairs:
        seg = np.array([coords[i], coords[j]])
        if np.isfinite(seg).all():
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.4", linewidth=1.5)

    # Markers
    finite = np.isfinite(coords).all(axis=-1)
    ax.scatter(
        coords[finite, 0], coords[finite, 1], coords[finite, 2],
        c="tab:red", s=25, depthshade=True,
    )

    if show_labels:
        for idx, label in enumerate(markers.labels):
            if finite[idx]:
                ax.text(*coords[idx], label, fontsize=6, color="0.2")

    _apply_axes_style(ax, markers, elev, azim, extra_points=mound_verts)
    if title is None:
        name = source if isinstance(source, str) else "C3D"
        title = f"{os.path.basename(str(name))}  |  frame {frame}/{markers.n_frames - 1}"
    ax.set_title(title)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def animate_c3d(
    source,
    segments=PLUG_IN_GAIT_SEGMENTS,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    fps: int | None = None,
    elev: float = 12.0,
    azim: float = -60.0,
    title: str | None = None,
    save_path: str | None = None,
    mound: bool = False,
):
    """Animate a C3D file as a rotating-free 3D stick figure over time.

    Parameters
    ----------
    source:
        Path to a ``.c3d`` file or a loaded :class:`C3DMarkers`.
    segments:
        ``(label_a, label_b)`` pairs to connect, or ``None`` for markers only.
    start, stop, step:
        Frame range to animate (``stop`` defaults to the last frame).
    fps:
        Playback frames per second. Defaults to the capture rate / ``step``
        so the animation runs at real time.
    elev, azim:
        3D view angles (degrees).
    title:
        Title; defaults to the file name.
    save_path:
        Output file. ``.gif`` uses the Pillow writer; ``.mp4`` uses ffmpeg.
        If omitted, the animation object is returned without saving (useful in
        notebooks via ``HTML(anim.to_jshtml())``).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (keep a reference to it so it is not garbage
        collected before display/saving).
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    markers = source if isinstance(source, C3DMarkers) else load_c3d(source)
    if markers.n_frames == 0:
        raise ValueError("C3D file contains no frames.")

    stop = markers.n_frames if stop is None else stop
    frames = list(range(start, stop, step))
    if not frames:
        raise ValueError("Empty frame range to animate.")
    if fps is None:
        fps = max(1, int(round(markers.rate / step)))

    pairs = _segments_present(markers, segments)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    mound_verts = None
    if mound:
        from mound import add_mound
        mound_verts = add_mound(ax, markers)

    _apply_axes_style(ax, markers, elev, azim, extra_points=mound_verts)

    name = source if isinstance(source, str) else "C3D"
    base_title = title or os.path.basename(str(name))

    # Pre-create artists and update their data each frame (fast & flicker-free).
    seg_lines = [
        ax.plot([], [], [], color="0.4", linewidth=1.5)[0] for _ in pairs
    ]
    scatter = ax.plot(
        [], [], [], linestyle="none", marker="o", color="tab:red", markersize=4
    )[0]

    def update(frame_idx):
        coords = markers.points[frame_idx]
        for line, (i, j) in zip(seg_lines, pairs):
            seg = np.array([coords[i], coords[j]])
            if np.isfinite(seg).all():
                line.set_data(seg[:, 0], seg[:, 1])
                line.set_3d_properties(seg[:, 2])
            else:
                line.set_data([], [])
                line.set_3d_properties([])
        finite = np.isfinite(coords).all(axis=-1)
        scatter.set_data(coords[finite, 0], coords[finite, 1])
        scatter.set_3d_properties(coords[finite, 2])
        ax.set_title(f"{base_title}  |  frame {frame_idx}/{markers.n_frames - 1}")
        return seg_lines + [scatter]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000.0 / fps, blit=False
    )

    if save_path:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, fps=fps)
        plt.close(fig)
    return anim


def _apply_axes_style(ax, markers: C3DMarkers, elev: float, azim: float,
                      extra_points=None) -> None:
    """Apply consistent labels, equal aspect, and view angle to a 3D axes."""
    xlim, ylim, zlim = _equal_aspect_bounds(
        markers.points.reshape(-1, 3), extra_points=extra_points
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    unit = f" ({markers.units})" if markers.units else ""
    ax.set_xlabel(f"X{unit}")
    ax.set_ylabel(f"Y{unit}")
    ax.set_zlabel(f"Z{unit}  (up)")
    ax.view_init(elev=elev, azim=azim)
    try:  # set_box_aspect requires matplotlib >= 3.3
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="3D plot of an OpenBiomechanics C3D motion-capture file.",
    )
    p.add_argument("path", help="Path to a local .c3d file.")
    p.add_argument(
        "--animate", action="store_true",
        help="Render an animation over all frames instead of a single frame.",
    )
    p.add_argument(
        "--frame", type=int, default=0,
        help="Frame index for a static plot (default: 0; negatives allowed).",
    )
    p.add_argument(
        "--step", type=int, default=2,
        help="Frame step for animation (default: 2).",
    )
    p.add_argument(
        "--no-skeleton", action="store_true",
        help="Plot markers only, without connecting segments.",
    )
    p.add_argument("--labels", action="store_true", help="Annotate marker names.")
    p.add_argument(
        "--mound", action="store_true",
        help="Draw a polygonal dirt pitching mound under the pitcher.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output file (.png for a frame; .gif/.mp4 for animation). "
             "If omitted, attempts an interactive window.",
    )
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    segments = None if args.no_skeleton else PLUG_IN_GAIT_SEGMENTS

    if args.animate:
        out = args.out or "c3d_animation.gif"
        animate_c3d(args.path, segments=segments, step=args.step,
                    save_path=out, mound=args.mound)
        print(f"Saved animation to {out}")
    else:
        plot_c3d(
            args.path, frame=args.frame, segments=segments,
            show_labels=args.labels, save_path=args.out, mound=args.mound,
        )
        if args.out:
            print(f"Saved figure to {args.out}")
        else:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
