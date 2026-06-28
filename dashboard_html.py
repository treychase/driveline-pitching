"""Interactive, self-contained HTML dashboard (Plotly) with tabs.

Tabs:
* **Live delivery** — a **pitch picker** selects the delivery; a Plotly animation
  shows the 3D pose on the dirt mound with a Play button, a frame slider, and a
  **toggle for joint velocity vectors** on every joint. A synchronized live
  ground-reaction-force plot (lead vs. rear leg, by handedness) tracks the
  delivery. Below: the Bayesian-Lasso velocity gauge, the biomechanics z-scores,
  the **joint work accumulated during the delivery**, and the **z-scores of joint
  work** for the selected pitch.
* **Glossary** — searchable full explanations of every biomechanics variable.

Styled with a white & orange theme (``theme.py``).

    python dashboard_html.py --out dashboard.html                 # default pitches
    python dashboard_html.py --pitches 1097_1,1031_2 --out d.html
    python dashboard_html.py --offline --out dashboard.html       # embed Plotly
"""

from __future__ import annotations

import argparse

import numpy as np

import c3d_plot
import glossary
import joint_kinetics as jk
import theme
from dashboard import _load_force, download_c3d_for_pitch
from velocity_model import train_velocity_model

_LEAD_C, _REAR_C = theme.SERIES_A, theme.SERIES_B
_DIRT_SCALE = [[0.0, "#5b3a1e"], [0.5, "#8a5a30"], [1.0, "#b9854f"]]
_VEC_SCALE = 0.03   # metres drawn per (m/s) of joint velocity
_FOOT_STOPS = [(0.0, (255, 247, 239)), (0.5, (255, 176, 102)), (1.0, (255, 122, 0))]
_FOOT_MAX_RGB = (214, 39, 39)   # red flash when a foot is at its peak force


def _foot_state(val, peak, unit):
    """Square colour, text colour, and label for a foot's current force."""
    frac = float(np.clip(val / peak, 0, 1)) if peak > 0 else 0.0
    if peak > 0 and val >= 0.99 * peak:
        return ("rgb(214,39,39)", "white", f"<b>{val:.2f}</b><br>◀ MAX")
    for (f0, c0), (f1, c1) in zip(_FOOT_STOPS, _FOOT_STOPS[1:]):
        if frac <= f1:
            a = (frac - f0) / (f1 - f0) if f1 > f0 else 0
            rgb = tuple(int(c0[k] + a * (c1[k] - c0[k])) for k in range(3))
            break
    else:
        rgb = _FOOT_STOPS[-1][1]
    tcol = "white" if frac > 0.5 else theme.SLATE
    return (f"rgb{rgb}", tcol, f"<b>{val:.2f}</b><br>{unit}")


def _seg_coords(coords, pairs):
    xs, ys, zs = [], [], []
    for i, j in pairs:
        a, b = coords[i], coords[j]
        if np.isfinite(a).all() and np.isfinite(b).all():
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return xs, ys, zs


def _vector_segments(vecs, fi, scale=_VEC_SCALE):
    """Arrow shafts (NaN-separated) and tip points for joint velocity vectors."""
    xs, ys, zs, tx, ty, tz = [], [], [], [], [], []
    for d in vecs.values():
        p, v = d["pos"][fi], d["vel"][fi] * scale
        if np.isfinite(p).all() and np.isfinite(v).all():
            q = p + v
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
            tx.append(q[0]); ty.append(q[1]); tz.append(q[2])
    return (xs, ys, zs), (tx, ty, tz)


def build_pose_force_figure(markers, fp, vecs, frame_times, lead_leg, rear_leg,
                            step, div_id):
    """Animated 3D pose + mound + joint-vector toggle + synced live GRF."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import mound as mound_mod

    pairs = c3d_plot._segments_present(markers, c3d_plot.PLUG_IN_GAIT_SEGMENTS)
    frames_idx = list(range(0, markers.n_frames, step))
    f0 = frames_idx[0]

    use_bw = fp is not None and fp.get("has_bw")
    suffix = "_bw" if use_bw else ""
    unit = "BW" if use_bw else "N"
    if fp is not None:
        lead = fp["lead_vertical" + suffix]
        rear = fp["rear_vertical" + suffix]
        ymax = float(np.nanmax([np.nanmax(lead), np.nanmax(rear), 1.0])) * 1.18
    else:
        lead = rear = np.zeros_like(frame_times)
        ymax = 1.0

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.55, 0.33, 0.12],
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("3D delivery on the mound", f"Live vertical GRF ({unit})",
                        "Foot vGRF"),
        horizontal_spacing=0.04,
    )

    # 0: mound
    mx, my, mz, mi, mj, mk, inten, _ = mound_mod.mound_trimesh(markers)
    fig.add_trace(go.Mesh3d(x=mx, y=my, z=mz, i=mi, j=mj, k=mk, intensity=inten,
                            colorscale=_DIRT_SCALE, showscale=False, opacity=1.0,
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9),
                            hoverinfo="skip"), row=1, col=1)
    # 1,2: skeleton + joints
    c0 = markers.points[f0]
    sx, sy, sz = _seg_coords(c0, pairs)
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                               line=dict(color=theme.SLATE, width=4),
                               name="skeleton", hoverinfo="skip"), row=1, col=1)
    fin = np.isfinite(c0).all(axis=-1)
    fig.add_trace(go.Scatter3d(x=c0[fin, 0], y=c0[fin, 1], z=c0[fin, 2],
                               mode="markers", marker=dict(size=3, color=theme.ORANGE_DARK),
                               name="markers", hoverinfo="skip"), row=1, col=1)
    # 3,4: force traces
    fig.add_trace(go.Scatter(x=frame_times, y=lead, mode="lines",
                             line=dict(color=_LEAD_C, width=2),
                             name=f"Lead leg ({lead_leg})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=frame_times, y=rear, mode="lines",
                             line=dict(color=_REAR_C, width=2),
                             name=f"Rear leg ({rear_leg})"), row=1, col=2)
    # 5,6,7: cursor + dots
    t0 = float(frame_times[f0])
    fig.add_trace(go.Scatter(x=[t0, t0], y=[0, ymax], mode="lines",
                             line=dict(color="#999", width=1.5, dash="dot"),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t0], y=[lead[f0]], mode="markers",
                             marker=dict(color=_LEAD_C, size=11),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t0], y=[rear[f0]], mode="markers",
                             marker=dict(color=_REAR_C, size=11),
                             showlegend=False), row=1, col=2)
    # 8,9: joint velocity vectors (shafts + tips) — visible by default
    (vx, vy, vz), (tx, ty, tz) = _vector_segments(vecs, f0)
    fig.add_trace(go.Scatter3d(x=vx, y=vy, z=vz, mode="lines",
                               line=dict(color=theme.ORANGE, width=5),
                               name="joint velocity", hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="markers",
                               marker=dict(size=3, color=theme.ORANGE),
                               showlegend=False, hoverinfo="skip"), row=1, col=1)
    # 10,11: foot-vGRF squares (lead on top, rear below) in the third subplot
    lc, ltc, ltxt = _foot_state(lead[f0], float(np.nanmax(lead)), unit)
    rc, rtc, rtxt = _foot_state(rear[f0], float(np.nanmax(rear)), unit)
    fig.add_trace(go.Scatter(x=[0.5], y=[0.70], mode="markers+text",
                             marker=dict(symbol="square", size=64, color=lc,
                                         line=dict(color="#888", width=1)),
                             text=[ltxt], textfont=dict(color=ltc, size=12),
                             textposition="middle center", showlegend=False,
                             hoverinfo="skip"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0.5], y=[0.28], mode="markers+text",
                             marker=dict(symbol="square", size=64, color=rc,
                                         line=dict(color="#888", width=1)),
                             text=[rtxt], textfont=dict(color=rtc, size=12),
                             textposition="middle center", showlegend=False,
                             hoverinfo="skip"), row=1, col=3)

    # event lines + shaded delivery phases on the force subplot
    xa = fig.data[3].xaxis or "x"
    ya = fig.data[3].yaxis or "y"
    if fp is not None:
        import force_plate as _fp
        ev_times = {k: float(frame_times[fr]) for k, fr in fp["event_frames"].items()
                    if fr < len(frame_times)}
        for ph in _fp.delivery_phases(ev_times, float(frame_times[-1])):
            fig.add_shape(type="rect", x0=ph["t0"], x1=ph["t1"], y0=0, y1=ymax,
                          xref=xa, yref=ya, line=dict(width=0),
                          fillcolor=ph["color"], opacity=0.85, layer="below")
            fig.add_annotation(x=(ph["t0"] + ph["t1"]) / 2, y=ymax * 0.99,
                               xref=xa, yref=ya, text=ph["label"], showarrow=False,
                               yanchor="top", textangle=0,
                               font=dict(size=8, color="#888"))
        for key, color in [("fp", "#888"), ("mer", "#aaa"), ("br", theme.ORANGE_DARK)]:
            fr = fp["event_frames"].get(key)
            if fr is not None and fr < len(frame_times):
                tx0 = float(frame_times[fr])
                fig.add_shape(type="line", x0=tx0, x1=tx0, y0=0, y1=ymax,
                              xref=xa, yref=ya,
                              line=dict(color=color, dash="dash", width=1))

    # frames
    lead_peak, rear_peak = float(np.nanmax(lead)), float(np.nanmax(rear))
    anim_frames = []
    for f in frames_idx:
        coords = markers.points[f]
        sx, sy, sz = _seg_coords(coords, pairs)
        fin = np.isfinite(coords).all(axis=-1)
        (vx, vy, vz), (tx, ty, tz) = _vector_segments(vecs, f)
        tt = float(frame_times[f])
        lc, ltc, ltxt = _foot_state(lead[f], lead_peak, unit)
        rc, rtc, rtxt = _foot_state(rear[f], rear_peak, unit)
        anim_frames.append(go.Frame(name=str(f), data=[
            go.Scatter3d(x=sx, y=sy, z=sz),
            go.Scatter3d(x=coords[fin, 0], y=coords[fin, 1], z=coords[fin, 2]),
            go.Scatter(x=[tt, tt], y=[0, ymax]),
            go.Scatter(x=[tt], y=[float(lead[f])]),
            go.Scatter(x=[tt], y=[float(rear[f])]),
            go.Scatter3d(x=vx, y=vy, z=vz),
            go.Scatter3d(x=tx, y=ty, z=tz),
            go.Scatter(x=[0.5], y=[0.70],
                       marker=dict(symbol="square", size=64, color=lc,
                                   line=dict(color="#888", width=1)),
                       text=[ltxt], textfont=dict(color=ltc, size=12)),
            go.Scatter(x=[0.5], y=[0.28],
                       marker=dict(symbol="square", size=64, color=rc,
                                   line=dict(color="#888", width=1)),
                       text=[rtxt], textfont=dict(color=rtc, size=12)),
        ], traces=[1, 2, 5, 6, 7, 8, 9, 10, 11]))
    fig.frames = anim_frames

    # Hide the foot-square subplot axes and label the two squares.
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=3)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=3)
    xb = fig.data[10].xaxis or "x3"
    yb = fig.data[10].yaxis or "y3"
    fig.add_annotation(x=0.5, y=0.93, xref=xb, yref=yb, showarrow=False,
                       text=f"<b>{lead_leg}</b> lead", font=dict(size=10, color=_LEAD_C))
    fig.add_annotation(x=0.5, y=0.50, xref=xb, yref=yb, showarrow=False,
                       text=f"<b>{rear_leg}</b> rear", font=dict(size=10, color=_REAR_C))

    fig.update_scenes(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)",
                      zaxis_title="Z up (m)",
                      xaxis=dict(backgroundcolor="#fafafa"),
                      yaxis=dict(backgroundcolor="#fafafa"),
                      zaxis=dict(backgroundcolor="#f2f2f2"),
                      camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)))
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text=f"vertical GRF ({unit})", range=[0, ymax],
                     row=1, col=2)
    fig.update_layout(
        template=theme.plotly_template(), height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", x=0.62, y=1.08),
        updatemenus=[
            dict(type="buttons", showactive=False, x=0.0, y=0, xanchor="right",
                 yanchor="top", buttons=[
                     dict(label="▶ Play", method="animate", args=[None, dict(
                         frame=dict(duration=40, redraw=True), fromcurrent=True,
                         transition=dict(duration=0))]),
                     dict(label="⏸ Pause", method="animate", args=[[None], dict(
                         frame=dict(duration=0, redraw=False), mode="immediate")]),
                 ]),
            dict(type="buttons", direction="right", showactive=True, x=0.0, y=1.08,
                 xanchor="left", yanchor="top", bgcolor="#fff",
                 buttons=[
                     dict(label="Vectors on", method="restyle",
                          args=[{"visible": True}, [8, 9]]),
                     dict(label="Vectors off", method="restyle",
                          args=[{"visible": False}, [8, 9]]),
                 ]),
        ],
        sliders=[dict(active=0, y=0, x=0.05, len=0.55,
                      currentvalue=dict(prefix="frame "),
                      steps=[dict(method="animate", label=str(f),
                                  args=[[str(f)], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=True))])
                             for f in frames_idx])],
    )
    return fig


def build_velocity_figure(predicted, std, actual, velo_lo, velo_hi):
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta", value=predicted,
        delta=dict(reference=actual, suffix=" vs actual"),
        number=dict(suffix=" mph"),
        title=dict(text=f"Predicted velocity<br><span style='font-size:0.8em;"
                        f"color:gray'>95% CI {predicted-1.96*std:.1f}–"
                        f"{predicted+1.96*std:.1f} · actual {actual:.1f}</span>"),
        gauge=dict(axis=dict(range=[velo_lo, velo_hi]),
                   bar=dict(color=theme.ORANGE),
                   steps=[dict(range=[predicted-1.96*std, predicted+1.96*std],
                               color="#ffe6cc")],
                   threshold=dict(line=dict(color=theme.SLATE, width=4),
                                  value=actual)),
    ))
    fig.update_layout(template=theme.plotly_template(), height=260,
                      margin=dict(l=30, r=30, t=60, b=10))
    return fig


def _zbar(features, zvals, title, hover_extra=None):
    import plotly.graph_objects as go

    ht = "%{y}: z=%{x:.2f}<extra></extra>"
    fig = go.Figure(go.Bar(
        x=zvals, y=features, orientation="h",
        marker=dict(color=zvals, colorscale=theme.DIVERGING, cmin=-3, cmax=3,
                    colorbar=dict(title="z")),
        customdata=hover_extra,
        hovertemplate=("%{y}: z=%{x:.2f}<br>%{customdata}<extra></extra>"
                       if hover_extra is not None else ht),
    ))
    fig.update_layout(template=theme.plotly_template(), height=440,
                      margin=dict(l=180, r=10, t=40, b=30), title=title,
                      xaxis_title="z-score (σ from dataset mean)")
    fig.update_yaxes(automargin=True)
    return fig


def build_zscore_figure(zdf, top_n=18):
    sub = zdf.head(top_n).iloc[::-1]
    return _zbar(sub["feature"], sub["z"], f"Biomechanics z-scores — top {top_n}")


def build_jointwork_z_figure(zwork):
    sub = zwork.iloc[::-1]
    return _zbar(sub["joint"], sub["z"], "Joint work z-scores (energy generated)",
                 hover_extra=[f"work = {w:.0f} J" for w in sub["work_J"]])


def build_jointwork_time_figure(jw, frame_times):
    import plotly.graph_objects as go

    fig = go.Figure()
    palette = [theme.ORANGE, theme.SLATE, theme.ORANGE_LIGHT, "#7a8794",
               theme.ORANGE_DARK, "#9aa6b2", "#e0913f", "#1f3b52"]
    for k, (joint, w) in enumerate(jw.work.items()):
        fig.add_trace(go.Scatter(x=jw.time, y=w, mode="lines", name=joint,
                                 line=dict(color=palette[k % len(palette)], width=2)))
    br = jw.events.get("br")
    if br is not None:
        ymax = max((np.nanmax(w) for w in jw.work.values()), default=1)
        ymin = min((np.nanmin(w) for w in jw.work.values()), default=0)
        fig.add_shape(type="line", x0=br, x1=br, y0=ymin, y1=ymax,
                      line=dict(color=theme.ORANGE_DARK, dash="dash", width=1))
        fig.add_annotation(x=br, y=ymax, text="BR", showarrow=False,
                           font=dict(size=10, color=theme.ORANGE_DARK))
    fig.update_layout(template=theme.plotly_template(), height=440,
                      margin=dict(l=10, r=10, t=40, b=30),
                      title="Joint work accumulated during the delivery (J)",
                      xaxis_title="time (s)", yaxis_title="energy generated (J)")
    return fig


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OBP Pitching Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         color: #222; background: #fff; }}
  header {{ background: #fff; color: #cc5f00; padding: 14px 22px;
            border-bottom: 4px solid #ff7a00; }}
  header h1 {{ margin: 0; font-size: 20px; color: #ff7a00; }}
  header .sub {{ color: #555; font-size: 13px; margin-top: 4px; }}
  .tabs {{ display: flex; gap: 4px; background: #fff; padding: 8px 16px 0; }}
  .tablink {{ background: #fff3e6; color: #cc5f00; border: 1px solid #ffd9b3;
             border-bottom: none; padding: 10px 20px; font-size: 14px;
             cursor: pointer; border-radius: 6px 6px 0 0; }}
  .tablink.active {{ background: #ff7a00; color: #fff; font-weight: 600;
             border-color: #ff7a00; }}
  .tabcontent {{ padding: 16px 22px; }}
  .picker {{ margin: 6px 0 14px; font-size: 14px; }}
  .picker select {{ font-size: 14px; padding: 6px 10px; border: 1px solid #ffb066;
             border-radius: 6px; color: #cc5f00; }}
  .pitch-head {{ font-size: 13px; color: #555; margin: 4px 0 10px; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start;
          margin-top: 26px; }}
  .row > div {{ flex: 1; min-width: 320px; }}
  table.gloss {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.gloss th, table.gloss td {{ border: 1px solid #eee; padding: 6px 8px;
         text-align: left; vertical-align: top; }}
  table.gloss th {{ background: #ff7a00; color: #fff; position: sticky; top: 0; }}
  table.gloss tr:nth-child(even) {{ background: #fff7ef; }}
  td.def {{ max-width: 520px; }}
  details summary {{ cursor: pointer; margin: 8px 0; color: #cc5f00; }}
  #gloss-search {{ border: 1px solid #ffb066 !important; }}
</style></head>
<body>
<header>
  <h1>OpenBiomechanics Pitching Dashboard</h1>
  <div class="sub">Bayesian-Lasso velocity · live force plates · joint work &amp;
     velocity vectors · model R²={r2:.2f}, RMSE={rmse:.1f} mph</div>
</header>
<div class="tabs">
  <button class="tablink active" id="b-live" onclick="showTab('live')">
    Live delivery</button>
  <button class="tablink" id="b-gloss" onclick="showTab('gloss')">Glossary</button>
</div>
<div id="live" class="tabcontent">
  <div class="picker">Pitch:
    <select id="pitch-picker" onchange="pickPitch(this.value)">{options}</select>
    <span style="color:#888">— animate, scrub, and toggle joint vectors</span>
  </div>
  {panels}
</div>
<div id="gloss" class="tabcontent" style="display:none">
  {glossary}
</div>
<script>
function showTab(id) {{
  document.getElementById('live').style.display = id==='live' ? 'block':'none';
  document.getElementById('gloss').style.display = id==='gloss' ? 'block':'none';
  document.getElementById('b-live').classList.toggle('active', id==='live');
  document.getElementById('b-gloss').classList.toggle('active', id==='gloss');
  window.dispatchEvent(new Event('resize'));
}}
function pickPitch(sp) {{
  document.querySelectorAll('.pitch-panel').forEach(function(p) {{
    p.style.display = (p.id === 'pitch-' + sp) ? 'block' : 'none';
  }});
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body></html>
"""


def _pitch_figures(ds, trained, sp, step, top_n):
    """Build all figures for one pitch; return (figs_in_order, head_text)."""
    predicted, pred_std = trained.predict_pitch(sp)
    actual = ds.actual_velocity(sp)
    in_test = ds.index_of(sp) in set(trained.test_idx.tolist())
    markers = c3d_plot.load_c3d(download_c3d_for_pitch(ds, sp))
    frame_times = np.arange(markers.n_frames) / markers.rate
    fp = _load_force(ds, sp, frame_times)
    handed = str(ds.poi.iloc[ds.index_of(sp)].get("p_throws", "R"))
    rear_leg, lead_leg = ("R", "L") if handed.upper().startswith("R") else ("L", "R")
    try:
        vecs = jk.load_joint_vectors(sp, frame_times)
    except KeyError:
        vecs = {}
    velo_lo = float(min(ds.y.min(), predicted - 3 * pred_std) - 2)
    velo_hi = float(max(ds.y.max(), predicted + 3 * pred_std) + 2)

    figs = {
        "anim": build_pose_force_figure(markers, fp, vecs, frame_times, lead_leg,
                                        rear_leg, step, f"anim-{sp}"),
        "velo": build_velocity_figure(predicted, pred_std, actual, velo_lo, velo_hi),
        "zbio": build_zscore_figure(ds.zscores(sp), top_n=top_n),
    }
    try:
        jw = jk.load_joint_work(sp)
        figs["jwt"] = build_jointwork_time_figure(jw, frame_times)
        figs["jwz"] = build_jointwork_z_figure(jk.work_zscores(sp))
    except KeyError:
        pass
    tag = "out-of-sample" if in_test else "in-sample"
    head = (f"Pitch {sp} · {tag} · predicted {predicted:.1f} mph "
            f"(actual {actual:.1f}, error {predicted-actual:+.1f}) · "
            f"{handed}HP")
    return figs, head


def build_html(pitches=None, trained=None, step=4, top_n=18,
               out="dashboard.html", offline=False):
    """Build the interactive HTML dashboard with a pitch picker."""
    if trained is None:
        trained = train_velocity_model()
    ds = trained.dataset
    if not pitches:
        seen, pitches = set(), []
        for idx in trained.test_idx:
            sp = str(ds.poi["session_pitch"].iloc[int(idx)])
            if sp not in seen:
                seen.add(sp); pitches.append(sp)
            if len(pitches) >= 4:
                break

    state = {"first": True}

    def emit(fig, div_id):
        inc = (True if offline else "cdn") if state["first"] else False
        state["first"] = False
        return fig.to_html(full_html=False, include_plotlyjs=inc, div_id=div_id)

    panels, options = [], []
    for n, sp in enumerate(pitches):
        figs, head = _pitch_figures(ds, trained, sp, step, top_n)
        anim = emit(figs["anim"], f"anim-{sp}")
        velo = emit(figs["velo"], f"velo-{sp}")
        zbio = emit(figs["zbio"], f"zbio-{sp}")
        jwt = emit(figs["jwt"], f"jwt-{sp}") if "jwt" in figs else ""
        jwz = emit(figs["jwz"], f"jwz-{sp}") if "jwz" in figs else ""
        disp = "block" if n == 0 else "none"
        panels.append(
            f"<div class='pitch-panel' id='pitch-{sp}' style='display:{disp}'>"
            f"<div class='pitch-head'>{head}</div>{anim}"
            f"<div class='row'><div>{velo}</div><div>{zbio}</div></div>"
            f"<div class='row'><div>{jwt}</div><div>{jwz}</div></div>"
            f"</div>"
        )
        options.append(f"<option value='{sp}'>{sp}</option>")

    page = _PAGE.format(
        r2=trained.metrics["r2"], rmse=trained.metrics["rmse"],
        options="".join(options), panels="".join(panels),
        glossary=glossary.render_html(),
    )
    with open(out, "w") as fh:
        fh.write(page)
    return {"out": out, "pitches": pitches}


def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive HTML pitching dashboard.")
    p.add_argument("--pitches", default=None,
                   help="Comma-separated session_pitch ids for the picker.")
    p.add_argument("--out", default="dashboard.html")
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--top-n", type=int, default=18)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args(argv)
    pitches = [s.strip() for s in args.pitches.split(",")] if args.pitches else None
    info = build_html(pitches=pitches, step=args.step, top_n=args.top_n,
                      out=args.out, offline=args.offline)
    print(f"Saved interactive dashboard to {info['out']}")
    print(f"  pitches: {', '.join(info['pitches'])}")


if __name__ == "__main__":
    main()
