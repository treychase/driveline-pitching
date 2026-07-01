"""Gradio app for deploying the OpenBiomechanics pitching dashboard.

Designed to run as a HuggingFace Space (``sdk: gradio``, ``app_file: app.py``).
It trains the Bayesian-Lasso velocity model on startup and serves the same
views as the static HTML dashboard — Live delivery, Joint work, Model
diagnostics, and Glossary — sourcing all data directly through
``data_sources`` (the OpenBiomechanics GitHub mirror by default, or a
HuggingFace dataset if ``OBP_HF_DATASET`` is set).

Run locally with ``python app.py`` (serves on http://localhost:7860).
"""

from __future__ import annotations

import numpy as np
import gradio as gr

import c3d_plot
import dashboard_html as dh
import data_sources
import efficiency
import glossary
import joint_kinetics as jk
from dashboard import _load_force, download_c3d_for_pitch
from velocity_model import train_velocity_model

STEP = 6  # frame step for the pose animation (larger = lighter for the Space)

# Train once at startup; reused across requests.
TRAINED = train_velocity_model()
DS = TRAINED.dataset
PITCHES = DS.poi["session_pitch"].astype(str).tolist()
DEFAULT_PITCH = str(DS.poi["session_pitch"].iloc[int(TRAINED.test_idx[0])])
DIAG = dh.build_diagnostics_figures(TRAINED, highlight=[DEFAULT_PITCH])
# Mechanical efficiency model (torso + lower-body drive vs. throwing-elbow load).
EFF = efficiency.MechanicalEfficiencyModel.fit(DS.poi)


def _pitch_header(sp):
    predicted, _ = TRAINED.predict_pitch(sp)
    actual = DS.actual_velocity(sp)
    in_test = DS.index_of(sp) in set(TRAINED.test_idx.tolist())
    handed = str(DS.poi.iloc[DS.index_of(sp)].get("p_throws", "R"))
    tag = "out-of-sample" if in_test else "in-sample"
    return (f"**Pitch {sp}** · {tag} · predicted **{predicted:.1f} mph** "
            f"(actual {actual:.1f}, error {predicted-actual:+.1f}) · {handed}HP")


def live_figures(sp):
    """Build the Live-delivery figures for one pitch."""
    try:
        predicted, std = TRAINED.predict_pitch(sp)
        actual = DS.actual_velocity(sp)
        markers = c3d_plot.load_c3d(download_c3d_for_pitch(DS, sp))
        ft = np.arange(markers.n_frames) / markers.rate
        fp = _load_force(DS, sp, ft)
        handed = str(DS.poi.iloc[DS.index_of(sp)].get("p_throws", "R"))
        rear, lead = ("R", "L") if handed.upper().startswith("R") else ("L", "R")
        try:
            vecs = jk.load_joint_vectors(sp, ft)
        except KeyError:
            vecs = {}
        lo = float(min(DS.y.min(), predicted - 3 * std) - 2)
        hi = float(max(DS.y.max(), predicted + 3 * std) + 2)
        anim = dh.build_pose_force_figure(markers, fp, vecs, ft, lead, rear, STEP,
                                          "anim")
        velo = dh.build_velocity_figure(predicted, std, actual, lo, hi)
        zbio = dh.build_zscore_figure(DS.zscores(sp))
        return _pitch_header(sp), anim, velo, zbio
    except Exception as exc:  # keep the Space responsive on any data hiccup
        return f"Could not build pitch {sp}: {exc}", None, None, None


def work_figures(sp):
    """Build the Joint-work figures for one pitch (colored pose + curves)."""
    try:
        jw = jk.load_joint_work(sp)
        pose = None
        try:
            markers = c3d_plot.load_c3d(download_c3d_for_pitch(DS, sp))
            ft = np.arange(markers.n_frames) / markers.rate
            vecs = jk.load_joint_vectors(sp, ft)
            pose = dh.build_jointwork_pose_figure(markers, jw, vecs, ft, STEP)
        except Exception:
            pose = None
        return (pose,
                dh.build_jointwork_time_figure(jw, None),
                dh.build_jointwork_z_figure(jk.work_zscores(sp)))
    except Exception:
        return None, None, None


def efficiency_figures(sp):
    """Build the mechanical-efficiency gauge + breakdown and a verdict."""
    try:
        res = EFF.score(sp)
        md = (f"**Mechanical efficiency: {res.score:.0f}/100** — "
              f"torso + lower-body drive **z={res.drive:+.2f}**, "
              f"throwing-elbow load **z={res.elbow_load:+.2f}**.  \n{res.verdict()}")
        return md, dh.build_efficiency_figure(res)
    except Exception as exc:
        return f"Could not score pitch {sp}: {exc}", None


HEADER = f"""
# OpenBiomechanics Pitching Dashboard
Bayesian-Lasso velocity prediction · live force plates · joint work & velocity
vectors · mechanical efficiency. Model: **R²={TRAINED.metrics['r2']:.2f}, RMSE={TRAINED.metrics['rmse']:.1f} mph**
(held-out test). Data source: {data_sources.describe()}.
"""


def build_demo():
    with gr.Blocks(title="OBP Pitching Dashboard") as demo:
        gr.Markdown(HEADER)
        pitch = gr.Dropdown(choices=PITCHES, value=DEFAULT_PITCH, label="Pitch",
                            info="Applies to Live delivery and Joint work")

        with gr.Tabs():
            with gr.Tab("Live delivery"):
                live_info = gr.Markdown()
                anim_plot = gr.Plot(label="3D delivery + live force plates")
                with gr.Row():
                    velo_plot = gr.Plot(label="Velocity")
                    zbio_plot = gr.Plot(label="Biomechanics z-scores")
            with gr.Tab("Joint work"):
                jwpose_plot = gr.Plot(label="Delivery colored by joint work")
                with gr.Row():
                    jwt_plot = gr.Plot(label="Joint work accumulated")
                    jwz_plot = gr.Plot(label="Joint work z-scores")
            with gr.Tab("Efficiency"):
                eff_info = gr.Markdown()
                eff_plot = gr.Plot(label="Mechanical efficiency")
            with gr.Tab("Model diagnostics"):
                gr.Plot(DIAG["pva"])
                with gr.Row():
                    gr.Plot(DIAG["resid"])
                    gr.Plot(DIAG["post"])
                gr.Plot(DIAG["coef"])
            with gr.Tab("Glossary"):
                gr.Dataframe(glossary.as_dataframe(), wrap=True,
                             label="Biomechanics glossary")

        live_out = [live_info, anim_plot, velo_plot, zbio_plot]
        work_out = [jwpose_plot, jwt_plot, jwz_plot]
        eff_out = [eff_info, eff_plot]
        pitch.change(live_figures, pitch, live_out)
        pitch.change(work_figures, pitch, work_out)
        pitch.change(efficiency_figures, pitch, eff_out)
        demo.load(live_figures, pitch, live_out)
        demo.load(work_figures, pitch, work_out)
        demo.load(efficiency_figures, pitch, eff_out)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
