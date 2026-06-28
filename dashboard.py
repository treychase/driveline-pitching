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

import numpy as np

import c3d_plot
import theme
from velocity_model import (
    PitchDataset,
    TrainedVelocityModel,
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
    ax.axhspan(0.35, 0.65, color=theme.PANEL, zorder=0)
    # 95% credible interval band for the prediction
    ci_lo, ci_hi = predicted - 1.96 * pred_std, predicted + 1.96 * pred_std
    ax.axvspan(ci_lo, ci_hi, ymin=0.30, ymax=0.70, color=theme.PLOT_A, alpha=0.18,
               zorder=1)
    # Predicted and actual markers
    ax.axvline(predicted, 0.22, 0.78, color=theme.PLOT_A, lw=3, zorder=3)
    ax.axvline(actual, 0.22, 0.78, color=theme.SLATE, lw=3, zorder=3)
    ax.plot([predicted], [0.5], "o", color=theme.PLOT_A, ms=9, zorder=4)
    ax.plot([actual], [0.5], "D", color=theme.SLATE, ms=9, zorder=4)

    err = predicted - actual
    ax.text(0.02, 0.92,
            f"Predicted: {predicted:.1f}  (95% CI {ci_lo:.1f}–{ci_hi:.1f}) mph",
            transform=ax.transAxes, color=theme.PLOT_A, fontsize=10, va="top")
    ax.text(0.02, 0.10,
            f"Actual: {actual:.1f} mph     Error: {err:+.1f} mph",
            transform=ax.transAxes, color=theme.SLATE, fontsize=10, va="bottom")


def _draw_zscore_panel(fig, ax, zdf, top_n: int = 18, zlim: float = 3.0):
    """Color-coded horizontal bars of per-metric z-scores."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from matplotlib.colors import LinearSegmentedColormap

    sub = zdf.head(top_n).iloc[::-1]  # largest |z| on top
    norm = Normalize(vmin=-zlim, vmax=zlim)
    cmap = LinearSegmentedColormap.from_list(
        "wo_div", theme.DIVERGING_MPL)
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


def _load_force(dataset, session_pitch, frame_times):
    """Load + frame-align force plates for a pitch; None if unavailable."""
    try:
        import force_plate
    except ImportError:
        return None
    meta = dataset.meta.loc[dataset.meta.session_pitch == session_pitch]
    mass_kg = float(meta.iloc[0]["session_mass_kg"]) if not meta.empty else None
    bw_n = mass_kg * 9.81 if mass_kg else None
    try:
        trace = force_plate.load_force_plate(session_pitch, bodyweight_n=bw_n)
    except (KeyError, FileNotFoundError):
        return None
    aligned = trace.align_to_frames(frame_times)
    aligned["has_bw"] = bw_n is not None
    return aligned


_LEAD_C, _REAR_C = theme.PLOT_A, theme.PLOT_B
_HIGHLIGHT = "#d62728"  # colour a foot square flashes to at its peak force


def _square_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "foot", ["#f4f7fb", "#92b7db", theme.PLOT_A])


def _set_square(sq, txt, val, peak, unit):
    """Colour a foot square by its current force; flash red at the max."""
    frac = float(np.clip(val / peak, 0, 1)) if peak > 0 else 0.0
    at_max = peak > 0 and val >= 0.99 * peak
    sq.set_facecolor(_HIGHLIGHT if at_max else _square_cmap()(frac))
    sq.set_edgecolor(_HIGHLIGHT if at_max else "0.5")
    sq.set_linewidth(2.6 if at_max else 1.2)
    txt.set_text(f"{val:.2f}\n{'◀ MAX' if at_max else unit}")
    txt.set_color("white" if (frac > 0.6 or at_max) else theme.TEXT)


def _draw_force_panel(ax_force, ax_bars, frame_times, fp, lead_leg, rear_leg):
    """Draw static force traces + an L/R gauge; return animated artists.

    Returns None if no force data is available (and blanks the axes).
    """
    if fp is None:
        for a in (ax_force, ax_bars):
            a.axis("off")
        ax_force.text(0.5, 0.5, "No force-plate data for this pitch",
                      ha="center", va="center", transform=ax_force.transAxes,
                      color="0.5")
        return None

    use_bw = fp.get("has_bw")
    suffix = "_bw" if use_bw else ""
    unit = "BW" if use_bw else "N"
    lead = fp["lead_vertical" + suffix]
    rear = fp["rear_vertical" + suffix]
    t = frame_times

    import force_plate as _fp
    from matplotlib.patches import Rectangle

    ax_force.plot(t, lead, color=_LEAD_C, lw=1.6, label=f"Lead leg ({lead_leg})")
    ax_force.plot(t, rear, color=_REAR_C, lw=1.6, label=f"Rear leg ({rear_leg})")
    ymax = float(np.nanmax([np.nanmax(lead), np.nanmax(rear), 1.0])) * 1.18
    ax_force.set_ylim(0, ymax)
    ax_force.set_xlim(float(t[0]), float(t[-1]))
    ax_force.set_xlabel("time (s)", fontsize=8)
    ax_force.set_ylabel(f"vertical GRF ({unit})", fontsize=8)
    ax_force.set_title("Live ground reaction force", fontweight="bold", fontsize=10)
    ax_force.tick_params(labelsize=7)

    # Shaded delivery phases behind the traces.
    ev_times = {k: float(t[fr]) for k, fr in fp["event_frames"].items()
                if fr < len(t)}
    for ph in _fp.delivery_phases(ev_times, float(t[-1])):
        ax_force.axvspan(ph["t0"], ph["t1"], color=ph["color"], alpha=0.85,
                         zorder=0, lw=0)
        ax_force.text((ph["t0"] + ph["t1"]) / 2, ymax * 0.97, ph["label"],
                      ha="center", va="top", fontsize=5.5, color="0.4", zorder=1)
    for key in ("fp", "mer", "br"):
        fr = fp["event_frames"].get(key)
        if fr is not None and fr < len(t):
            ax_force.axvline(t[fr], color="0.5", ls="--", lw=0.8, alpha=0.6,
                             zorder=1)
    ax_force.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # Animated trace artists
    cursor = ax_force.axvline(t[0], color="k", lw=1.2, zorder=4)
    lead_dot, = ax_force.plot([t[0]], [lead[0]], "o", color=_LEAD_C, ms=7,
                              zorder=5)
    rear_dot, = ax_force.plot([t[0]], [rear[0]], "o", color=_REAR_C, ms=7,
                              zorder=5)
    live_txt = ax_force.text(
        0.985, 0.96, "", transform=ax_force.transAxes, ha="right", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )

    # Two foot squares (lead/rear) that show the live value and flash at the max.
    ax_bars.set_xlim(0, 1)
    ax_bars.set_ylim(0, 1)
    ax_bars.axis("off")
    ax_bars.set_title(f"Foot vGRF\n({unit})", fontsize=8)
    sq_lead = Rectangle((0.15, 0.55), 0.7, 0.34, facecolor=theme.PANEL,
                        edgecolor="0.5", lw=1.2)
    sq_rear = Rectangle((0.15, 0.09), 0.7, 0.34, facecolor=theme.PANEL,
                        edgecolor="0.5", lw=1.2)
    ax_bars.add_patch(sq_lead)
    ax_bars.add_patch(sq_rear)
    ax_bars.text(0.5, 0.93, f"{lead_leg} lead", ha="center", va="center",
                 fontsize=7, color=_LEAD_C, fontweight="bold")
    ax_bars.text(0.5, 0.47, f"{rear_leg} rear", ha="center", va="center",
                 fontsize=7, color=_REAR_C, fontweight="bold")
    val_lead = ax_bars.text(0.5, 0.72, "", ha="center", va="center",
                            fontsize=10, fontweight="bold")
    val_rear = ax_bars.text(0.5, 0.26, "", ha="center", va="center",
                            fontsize=10, fontweight="bold")

    return {
        "lead": lead, "rear": rear, "t": t, "unit": unit,
        "lead_leg": lead_leg, "rear_leg": rear_leg,
        "cursor": cursor, "lead_dot": lead_dot, "rear_dot": rear_dot,
        "live_txt": live_txt,
        "sq_lead": sq_lead, "sq_rear": sq_rear,
        "val_lead": val_lead, "val_rear": val_rear,
        "peak_lead": float(np.nanmax(lead)), "peak_rear": float(np.nanmax(rear)),
    }


def _update_force_panel(fa, frame_idx):
    """Advance the live force artists to ``frame_idx``; return them for blitting."""
    t = fa["t"]
    tt = float(t[frame_idx])
    lv = float(fa["lead"][frame_idx])
    rv = float(fa["rear"][frame_idx])
    fa["cursor"].set_xdata([tt, tt])
    fa["lead_dot"].set_data([tt], [lv])
    fa["rear_dot"].set_data([tt], [rv])
    fa["live_txt"].set_text(
        f"{fa['lead_leg']} lead: {lv:5.2f} {fa['unit']}\n"
        f"{fa['rear_leg']} rear: {rv:5.2f} {fa['unit']}"
    )
    _set_square(fa["sq_lead"], fa["val_lead"], lv, fa["peak_lead"], fa["unit"])
    _set_square(fa["sq_rear"], fa["val_rear"], rv, fa["peak_rear"], fa["unit"])
    return [fa["cursor"], fa["lead_dot"], fa["rear_dot"], fa["live_txt"],
            fa["sq_lead"], fa["sq_rear"], fa["val_lead"], fa["val_rear"]]


def build_dashboard(
    session_pitch: str | None = None,
    trained: TrainedVelocityModel | None = None,
    step: int = 2,
    top_n: int = 18,
    save_path: str | None = "dashboard.gif",
    fps: int | None = None,
    mound: bool = True,
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
    theme.apply_matplotlib()

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

    # Load + align the force plates (live GRF during the delivery)
    frame_times = np.arange(markers.n_frames) / markers.rate
    fp_aligned = _load_force(ds, session_pitch, frame_times)
    handed = str(ds.poi.iloc[ds.index_of(session_pitch)].get("p_throws", "R"))
    # Map the two force plates to the actual legs by handedness.
    if handed.upper().startswith("R"):
        rear_leg, lead_leg = "R", "L"
    else:
        rear_leg, lead_leg = "L", "R"

    # --- Figure layout ---
    fig = plt.figure(figsize=(15, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0],
                          height_ratios=[1.0, 1.15, 0.95])
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_velo = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1:3, 1])
    # Bottom-left: live force-plate traces + an L/R bar gauge.
    gs_f = gs[2, 0].subgridspec(1, 5)
    ax_force = fig.add_subplot(gs_f[0, 0:4])
    ax_bars = fig.add_subplot(gs_f[0, 4])

    velo_lo = float(min(ds.y.min(), predicted - 3 * pred_std) - 2)
    velo_hi = float(max(ds.y.max(), predicted + 3 * pred_std) + 2)
    _draw_velocity_panel(ax_velo, predicted, pred_std, actual, velo_lo, velo_hi)
    _draw_zscore_panel(fig, ax_z, zdf, top_n=top_n)
    force_art = _draw_force_panel(
        ax_force, ax_bars, frame_times, fp_aligned, lead_leg, rear_leg
    )

    mound_verts = None
    if mound:
        from mound import add_mound
        mound_verts = add_mound(ax3d, markers)
    c3d_plot._apply_axes_style(ax3d, markers, elev=12.0, azim=-60.0,
                               extra_points=mound_verts)
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
        extra = _update_force_panel(force_art, frame_idx) if force_art else []
        return seg_lines + [scatter] + extra

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
    p.add_argument("--no-mound", action="store_true",
                   help="Disable the dirt pitching mound under the pitcher.")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    _, info = build_dashboard(
        session_pitch=args.pitch, step=args.step, top_n=args.top_n,
        save_path=args.out, mound=not args.no_mound,
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
