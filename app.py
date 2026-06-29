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

import html as _html

import numpy as np
import gradio as gr

import c3d_plot
import dashboard_html as dh
import data_sources
import glossary
import joint_kinetics as jk
from dashboard import _load_force, download_c3d_for_pitch
from velocity_model import train_velocity_model

STEP = 4  # frame step for the pose animation (smaller = smoother playback)
ANIM_HEIGHT = 600  # px height of the animated pose+force panel


def _plotly_iframe(fig, height=ANIM_HEIGHT):
    """Render a Plotly figure as a self-contained iframe.

    Gradio's ``gr.Plot`` does not register Plotly animation *frames* on the
    graph div, so the Play button and frame slider do nothing. Embedding the
    figure as standalone HTML inside an ``<iframe srcdoc>`` runs Plotly in its
    own document — frames register and the animation plays. Plotly.js is loaded
    from the CDN (cached across pitches, so re-selection stays snappy).
    """
    doc = fig.to_html(full_html=True, include_plotlyjs="cdn",
                      default_width="100%", default_height="100%",
                      config={"displaylogo": False, "responsive": True})
    return (f'<iframe srcdoc="{_html.escape(doc, quote=True)}" '
            f'style="width:100%;height:{height}px;border:none" '
            f'loading="lazy"></iframe>')


# Per-pitch result caches so re-selecting a pitch is instant (the first view
# downloads + parses the C3D and builds the figures; later views are cached).
_LIVE_CACHE: dict[str, tuple] = {}
_WORK_CACHE: dict[str, tuple] = {}

# Train once at startup; reused across requests.
TRAINED = train_velocity_model()
DS = TRAINED.dataset


def _build_pitcher_index(ds):
    """Group every pitch by its pitcher so the UI can offer a two-step picker.

    Returns ``(pitchers, sp_to_user)`` where ``pitchers`` maps a pitcher id
    (the ``user`` column from ``metadata.csv``) to a dict with display info and
    that pitcher's list of pitches ``[(session_pitch, velo, pitch_type), ...]``.
    """
    poi = ds.poi
    user_by_sp = dict(zip(ds.meta["session_pitch"].astype(str),
                          ds.meta["user"].astype(str)))
    level_by_user = dict(zip(ds.meta["user"].astype(str),
                             ds.meta["playing_level"].astype(str)))
    pitchers: dict[str, dict] = {}
    for _, r in poi.iterrows():
        sp = str(r["session_pitch"])
        user = user_by_sp.get(sp, "unknown")
        info = pitchers.setdefault(user, {
            "handed": str(r.get("p_throws", "?")),
            "level": level_by_user.get(user, "?"),
            "pitches": [],
        })
        info["pitches"].append((sp, float(r["pitch_speed_mph"]),
                                str(r.get("pitch_type", "?"))))
    for info in pitchers.values():
        info["pitches"].sort(key=lambda t: t[0])
    sp_to_user = {sp: u for u, info in pitchers.items()
                  for sp, _, _ in info["pitches"]}
    return pitchers, sp_to_user


PITCHERS, SP_TO_USER = _build_pitcher_index(DS)
PITCHER_IDS = sorted(PITCHERS, key=lambda u: (0, int(u)) if u.isdigit() else (1, u))


def _pitcher_label(user):
    info = PITCHERS[user]
    velos = [v for _, v, _ in info["pitches"]]
    n = len(velos)
    return (f"Pitcher {user} · {info['handed']}HP · {info['level']} · "
            f"{n} pitch{'es' if n != 1 else ''} · {min(velos):.0f}–{max(velos):.0f} mph")


def _pitcher_choices():
    return [(_pitcher_label(u), u) for u in PITCHER_IDS]


def _pitch_choices(user):
    return [(f"{sp} · {pt} · {v:.1f} mph", sp)
            for sp, v, pt in PITCHERS[user]["pitches"]]


# Default to the first held-out test pitch (falling back to the first pitcher).
DEFAULT_PITCH = str(DS.poi["session_pitch"].iloc[int(TRAINED.test_idx[0])])
if DEFAULT_PITCH in SP_TO_USER:
    DEFAULT_USER = SP_TO_USER[DEFAULT_PITCH]
else:
    DEFAULT_USER = PITCHER_IDS[0]
    DEFAULT_PITCH = PITCHERS[DEFAULT_USER]["pitches"][0][0]

DIAG = dh.build_diagnostics_figures(TRAINED, highlight=[DEFAULT_PITCH])


def _pitch_header(sp):
    predicted, _ = TRAINED.predict_pitch(sp)
    actual = DS.actual_velocity(sp)
    in_test = DS.index_of(sp) in set(TRAINED.test_idx.tolist())
    handed = str(DS.poi.iloc[DS.index_of(sp)].get("p_throws", "R"))
    tag = "out-of-sample" if in_test else "in-sample"
    return (f"**Pitch {sp}** · {tag} · predicted **{predicted:.1f} mph** "
            f"(actual {actual:.1f}, error {predicted-actual:+.1f}) · {handed}HP")


def live_figures(sp):
    """Build the Live-delivery views for one pitch (cached per pitch)."""
    if sp in _LIVE_CACHE:
        return _LIVE_CACHE[sp]
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
        anim_fig = dh.build_pose_force_figure(markers, fp, vecs, ft, lead, rear,
                                              STEP, "anim")
        # Embed as an iframe so the Play button / frame slider actually animate.
        anim = _plotly_iframe(anim_fig, height=ANIM_HEIGHT)
        velo = dh.build_velocity_figure(predicted, std, actual, lo, hi)
        zbio = dh.build_zscore_figure(DS.zscores(sp))
        out = (_pitch_header(sp), anim, velo, zbio)
        _LIVE_CACHE[sp] = out
        return out
    except Exception as exc:  # keep the Space responsive on any data hiccup
        return f"Could not build pitch {sp}: {exc}", "", None, None


def work_figures(sp):
    """Build the Joint-work figures for one pitch (cached per pitch)."""
    if sp in _WORK_CACHE:
        return _WORK_CACHE[sp]
    try:
        jw = jk.load_joint_work(sp)
        zwork = jk.work_zscores(sp)
        # 3D body shaded by joint work at ball release (sided by handedness).
        body = None
        try:
            br = jw.events.get("br")
            t = jw.time[-1] if br is None else br
            pos = jk.joint_positions_at(sp, t)
            handed = str(DS.poi.iloc[DS.index_of(sp)].get("p_throws", "R"))
            body = dh.build_jointwork_body_figure(pos, zwork, handed)
        except Exception:
            body = None
        out = (body, dh.build_jointwork_time_figure(jw, None),
               dh.build_jointwork_z_figure(zwork))
        _WORK_CACHE[sp] = out
        return out
    except Exception:
        return None, None, None


HEADER = f"""
# OpenBiomechanics Pitching Dashboard
Choose any of **{len(PITCHER_IDS)} pitchers** ({len(SP_TO_USER)} pitches) below —
Bayesian-Lasso velocity prediction · live force plates · joint work & velocity
vectors. Model: **R²={TRAINED.metrics['r2']:.2f}, RMSE={TRAINED.metrics['rmse']:.1f} mph**
(held-out test). Data source: {data_sources.describe()}.
"""


def _on_pitcher_change(user):
    """Refresh the pitch dropdown for the newly selected pitcher.

    Setting a new ``value`` here also fires the pitch dropdown's ``change``
    event, which rebuilds the figures — so the selectors stay in sync without
    recomputing the (expensive) pose animation twice.
    """
    choices = _pitch_choices(user)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def build_demo():
    with gr.Blocks(title="OBP Pitching Dashboard") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            pitcher = gr.Dropdown(choices=_pitcher_choices(), value=DEFAULT_USER,
                                  label="Pitcher", filterable=True,
                                  info="Type to search across all pitchers")
            pitch = gr.Dropdown(choices=_pitch_choices(DEFAULT_USER),
                                value=DEFAULT_PITCH, label="Pitch", filterable=True,
                                info="Applies to Live delivery and Joint work")

        with gr.Tabs():
            with gr.Tab("Live delivery"):
                live_info = gr.Markdown()
                anim_plot = gr.HTML(label="3D delivery + live force plates")
                with gr.Row():
                    velo_plot = gr.Plot(label="Velocity")
                    zbio_plot = gr.Plot(label="Biomechanics z-scores")
            with gr.Tab("Joint work"):
                jwbody_plot = gr.Plot(label="Joint work on the 3D body")
                with gr.Row():
                    jwt_plot = gr.Plot(label="Joint work accumulated")
                    jwz_plot = gr.Plot(label="Joint work z-scores")
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
        work_out = [jwbody_plot, jwt_plot, jwz_plot]
        # Picking a pitcher repopulates the pitch list and selects its first
        # pitch; that selection change drives the figure rebuild below.
        pitcher.change(_on_pitcher_change, pitcher, pitch)
        pitch.change(live_figures, pitch, live_out)
        pitch.change(work_figures, pitch, work_out)
        demo.load(live_figures, pitch, live_out)
        demo.load(work_figures, pitch, work_out)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
