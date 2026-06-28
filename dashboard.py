"""Animated biomechanics dashboard for OpenBiomechanics pitching data.

Combines three views for a single pitch into one figure:

1. **3D pose animation** of the C3D motion-capture markers (a pitcher delivery).
2. **Actual vs. predicted release velocity**, where the prediction comes from a
   Bayesian Lasso (Gaussian-prior scale-mixture) trained on the biomechanics
   metrics, shown with its posterior credible interval.
3. **Z-scores of every biomechanics metric** for this pitch relative to the
   dataset, drawn as a colour-coded diverging bar chart (blue = below average,
   red = above average).

Render it to a GIF/MP4 from the command line::

    python dashboard.py --out dashboard.gif

or call :func:`build_dashboard` from Python for more control.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import c3d_plot
from velocity_model import (
    PitchDataset,
    TrainedVelocityModel,
    load_dataset,
    train_velocity_model,
)


def download_c3d_for_pitch(dataset: PitchDataset, session_pitch: str,
                           dest_dir: str = "c3d_data") -> str:
    """Download the raw C3D file backing a given ``session_pitch``."""
    filename = dataset.c3d_filename(session_pitch)
    session_folder = filename.split("_")[0]
    return c3d_plot.download_obp_c3d(session_folder, filename, dest_dir=dest_dir)


def _draw_velocity_panel(ax, predicted, pred_std, actual, lo, hi) -> None:
    """Horizontal gauge comparing predicted (with CI) and actual velocity."""
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Release velocity (mph)")
    ax.set_title("Actual vs. predicted velocity", fontweight="bold")

    # Background track
    ax.axhspan(0.35, 0.65, color="0.92", zorder=0)
    # 95% credible interval band for the prediction
    ci_lo, ci_hi = predicted - 1.96 * pred_std, predicted + 1.96 * pred_std
    ax.axvspan(ci_lo, ci_hi, ymin=0.30, ymax=0.70, color="tab:blue", alpha=0.18,
               zorder=1)
    # Predicted and actual markers
    ax.axvline(predicted, 0.22, 0.78, color="tab:blue", lw=3, zorder=3)
    ax.axvline(actual, 0.22, 0.78, color="tab:green", lw=3, zorder=3)
    ax.plot([predicted], [0.5], "o", color="tab:blue", ms=9, zorder=4)
    ax.plot([actual], [0.5], "D", color="tab:green", ms=9, zorder=4)

    err = predicted - actual
    ax.text(0.02, 0.92,
            f"Predicted: {predicted:.1f}  (95% CI {ci_lo:.1f}–{ci_hi:.1f}) mph",
            transform=ax.transAxes, color="tab:blue", fontsize=10, va="top")
    ax.text(0.02, 0.10,
            f"Actual: {actual:.1f} mph     Error: {err:+.1f} mph",
            transform=ax.transAxes, color="tab:green", fontsize=10, va="bottom")


def _draw_zscore_panel(fig, ax, zdf, top_n: int = 18, zlim: float = 3.0):
    """Color-coded horizontal bars of per-metric z-scores."""
    import matplotlib as mpl
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sub = zdf.head(top_n).iloc[::-1]  # largest |z| on top
    norm = Normalize(vmin=-zlim, vmax=zlim)
    cmap = mpl.colormaps["RdBu_r"]
    colors = cmap(norm(np.clip(sub["z"].to_numpy(), -zlim, zlim)))

    y = np.arange(len(sub))
    ax.barh(y, sub["z"].to_numpy(), color=colors, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].tolist(), fontsize=7)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlim(-zlim - 0.5, zlim + 0.5)
    ax.set_xlabel("z-score (σ from dataset mean)")
    ax.set_title(f"Biomechanics z-scores — top {top_n} by |z|", fontweight="bold")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("z-score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def build_dashboard(
    session_pitch: str | None = None,
    trained: TrainedVelocityModel | None = None,
    step: int = 2,
    top_n: int = 18,
    save_path: str | None = "dashboard.gif",
    fps: int | None = None,
):
    """Build the animated dashboard for one pitch.

    Parameters
    ----------
    session_pitch:
        Which pitch to visualise. Defaults to a pitch from the model's held-out
        test set, so the displayed prediction is genuinely out-of-sample.
    trained:
        A :class:`TrainedVelocityModel`. If omitted, one is trained on the fly.
    step:
        Frame step for the pose animation.
    top_n:
        Number of biomechanics metrics (by absolute z-score) to display.
    save_path:
        Output file (``.gif`` or ``.mp4``). If None, the animation is returned
        without saving.
    fps:
        Playback FPS (defaults to capture rate / step).

    Returns
    -------
    (matplotlib.animation.FuncAnimation, dict)
        The animation and a small info dict (predicted, actual, etc.).
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if trained is None:
        trained = train_velocity_model()
    ds = trained.dataset

    if session_pitch is None:
        # Pick the test-set pitch the model predicts most confidently/typically.
        session_pitch = str(ds.poi["session_pitch"].iloc[int(trained.test_idx[0])])

    predicted, pred_std = trained.predict_pitch(session_pitch)
    actual = ds.actual_velocity(session_pitch)
    zdf = ds.zscores(session_pitch)
    in_test = ds.index_of(session_pitch) in set(trained.test_idx.tolist())

    # Load the pose
    c3d_path = download_c3d_for_pitch(ds, session_pitch)
    markers = c3d_plot.load_c3d(c3d_path)
    pairs = c3d_plot._segments_present(markers, c3d_plot.PLUG_IN_GAIT_SEGMENTS)
    if fps is None:
        fps = max(1, int(round(markers.rate / step)))

    # --- Figure layout ---
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 2.2])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_velo = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 1])

    velo_lo = float(min(ds.y.min(), predicted - 3 * pred_std) - 2)
    velo_hi = float(max(ds.y.max(), predicted + 3 * pred_std) + 2)
    _draw_velocity_panel(ax_velo, predicted, pred_std, actual, velo_lo, velo_hi)
    _draw_zscore_panel(fig, ax_z, zdf, top_n=top_n)

    c3d_plot._apply_axes_style(ax3d, markers, elev=12.0, azim=-60.0)
    tag = "out-of-sample" if in_test else "in-sample"
    fig.suptitle(
        f"Pitch {session_pitch}  ·  Bayesian-Lasso velocity prediction ({tag})  ·  "
        f"test R²={trained.metrics['r2']:.2f}, RMSE={trained.metrics['rmse']:.1f} mph",
        fontsize=12, fontweight="bold",
    )

    # Pose artists
    seg_lines = [ax3d.plot([], [], [], color="0.4", lw=1.5)[0] for _ in pairs]
    scatter = ax3d.plot([], [], [], ls="none", marker="o", color="tab:red",
                        ms=4)[0]

    frames = list(range(0, markers.n_frames, step))

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
        ax3d.set_title(
            f"3D pose — frame {frame_idx}/{markers.n_frames - 1}", fontsize=10
        )
        return seg_lines + [scatter]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000.0 / fps, blit=False
    )

    info = {
        "session_pitch": session_pitch,
        "predicted_mph": predicted,
        "predicted_std": pred_std,
        "actual_mph": actual,
        "error_mph": predicted - actual,
        "out_of_sample": in_test,
        "test_r2": trained.metrics["r2"],
        "test_rmse": trained.metrics["rmse"],
    }

    if save_path:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, fps=fps)
        plt.close(fig)
    return anim, info


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Animated pitching biomechanics dashboard "
                    "(pose + velocity prediction + z-scores).",
    )
    p.add_argument("--pitch", default=None,
                   help="session_pitch to visualise (default: a test-set pitch).")
    p.add_argument("--out", default="dashboard.gif",
                   help="Output file (.gif or .mp4). Default: dashboard.gif")
    p.add_argument("--step", type=int, default=2, help="Pose animation frame step.")
    p.add_argument("--top-n", type=int, default=18,
                   help="Number of z-score bars to show.")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    _, info = build_dashboard(
        session_pitch=args.pitch, step=args.step, top_n=args.top_n,
        save_path=args.out,
    )
    print(f"Saved dashboard to {args.out}")
    print(
        f"  pitch {info['session_pitch']} "
        f"({'out-of-sample' if info['out_of_sample'] else 'in-sample'}): "
        f"predicted {info['predicted_mph']:.1f} ± {info['predicted_std']:.1f} mph, "
        f"actual {info['actual_mph']:.1f} mph, error {info['error_mph']:+.1f} mph"
    )


if __name__ == "__main__":
    main()
