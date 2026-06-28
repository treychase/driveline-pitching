"""Interactive, self-contained HTML dashboard (Plotly) with tabs.

Produces a single ``.html`` file with two tabs:

* **Live delivery** — a Plotly animation of the 3D pose on the dirt mound with a
  Play button and a frame slider. Scrubbing the animation drives a synchronized
  **live ground-reaction-force** plot (lead vs. rear leg, labelled by the
  pitcher's actual L/R legs) with a moving time cursor, plus the Bayesian-Lasso
  velocity prediction gauge and the colour-coded biomechanics z-scores.
* **Glossary** — a searchable table with full explanations of every biomechanics
  variable (see ``glossary.py``).

Build it from the command line::

    python dashboard_html.py --pitch 1097_1 --out dashboard.html

then open the HTML file in any browser (needs internet for the Plotly CDN; pass
``--offline`` to embed Plotly for fully offline viewing).
"""

from __future__ import annotations

import argparse

import numpy as np

import c3d_plot
import glossary
from dashboard import _load_force, download_c3d_for_pitch
from velocity_model import train_velocity_model

_LEAD_C, _REAR_C = "#2ca02c", "#ff7f0e"
_DIRT_SCALE = [[0.0, "#5b3a1e"], [0.5, "#8a5a30"], [1.0, "#b9854f"]]


def _seg_coords(coords, pairs):
    """NaN-separated line segments for a Plotly Scatter3d skeleton."""
    xs, ys, zs = [], [], []
    for i, j in pairs:
        a, b = coords[i], coords[j]
        if np.isfinite(a).all() and np.isfinite(b).all():
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return xs, ys, zs


def build_pose_force_figure(markers, fp, frame_times, lead_leg, rear_leg, step):
    """Animated 3D pose + mound with a synchronized live GRF subplot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import mound as mound_mod

    pairs = c3d_plot._segments_present(markers, c3d_plot.PLUG_IN_GAIT_SEGMENTS)
    frames_idx = list(range(0, markers.n_frames, step))

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
        rows=1, cols=2, column_widths=[0.62, 0.38],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("3D delivery on the mound", f"Live vertical GRF ({unit})"),
        horizontal_spacing=0.04,
    )

    # --- trace 0: mound (static) ---
    mx, my, mz, mi, mj, mk, inten, _ = mound_mod.mound_trimesh(markers)
    fig.add_trace(go.Mesh3d(
        x=mx, y=my, z=mz, i=mi, j=mj, k=mk, intensity=inten,
        colorscale=_DIRT_SCALE, showscale=False, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9),
        name="mound", hoverinfo="skip",
    ), row=1, col=1)

    # --- traces 1,2: skeleton + joints (animated) ---
    c0 = markers.points[frames_idx[0]]
    sx, sy, sz = _seg_coords(c0, pairs)
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                               line=dict(color="#444", width=4),
                               name="skeleton", hoverinfo="skip"), row=1, col=1)
    finite = np.isfinite(c0).all(axis=-1)
    fig.add_trace(go.Scatter3d(
        x=c0[finite, 0], y=c0[finite, 1], z=c0[finite, 2], mode="markers",
        marker=dict(size=3, color="#d62728"),
        text=[lab for lab, f in zip(markers.labels, finite) if f],
        name="markers", hoverinfo="text"), row=1, col=1)

    # --- traces 3,4: static force traces ---
    fig.add_trace(go.Scatter(x=frame_times, y=lead, mode="lines",
                             line=dict(color=_LEAD_C, width=2),
                             name=f"Lead leg ({lead_leg})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=frame_times, y=rear, mode="lines",
                             line=dict(color=_REAR_C, width=2),
                             name=f"Rear leg ({rear_leg})"), row=1, col=2)
    # --- traces 5,6,7: animated cursor + dots ---
    t0 = float(frame_times[frames_idx[0]])
    fig.add_trace(go.Scatter(x=[t0, t0], y=[0, ymax], mode="lines",
                             line=dict(color="black", width=1.5, dash="dot"),
                             showlegend=False, name="cursor"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t0], y=[lead[frames_idx[0]]], mode="markers",
                             marker=dict(color=_LEAD_C, size=11),
                             showlegend=False, name="lead_now"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t0], y=[rear[frames_idx[0]]], mode="markers",
                             marker=dict(color=_REAR_C, size=11),
                             showlegend=False, name="rear_now"), row=1, col=2)

    # Event lines on the force subplot. Reference the force axes directly
    # (add_vline with row/col mishandles the mixed scene+xy figure).
    xa = fig.data[3].xaxis or "x"
    ya = fig.data[3].yaxis or "y"
    if fp is not None:
        for key, color in [("fp", "#555"), ("mer", "#888"), ("br", "crimson")]:
            fr = fp["event_frames"].get(key)
            if fr is not None and fr < len(frame_times):
                tx = float(frame_times[fr])
                fig.add_shape(type="line", x0=tx, x1=tx, y0=0, y1=ymax,
                              xref=xa, yref=ya,
                              line=dict(color=color, dash="dash", width=1))
                fig.add_annotation(x=tx, y=ymax, xref=xa, yref=ya,
                                   text=key.upper(), showarrow=False,
                                   font=dict(size=9, color=color),
                                   yanchor="bottom")

    # --- frames ---
    anim_frames = []
    for f in frames_idx:
        coords = markers.points[f]
        sx, sy, sz = _seg_coords(coords, pairs)
        fin = np.isfinite(coords).all(axis=-1)
        tt = float(frame_times[f])
        anim_frames.append(go.Frame(name=str(f), data=[
            go.Scatter3d(x=sx, y=sy, z=sz),
            go.Scatter3d(x=coords[fin, 0], y=coords[fin, 1], z=coords[fin, 2]),
            go.Scatter(x=[tt, tt], y=[0, ymax]),
            go.Scatter(x=[tt], y=[float(lead[f])]),
            go.Scatter(x=[tt], y=[float(rear[f])]),
        ], traces=[1, 2, 5, 6, 7]))
    fig.frames = anim_frames

    # Layout: scene aspect, camera, play/slider
    fig.update_scenes(
        aspectmode="data",
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z up (m)",
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    )
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text=f"vertical GRF ({unit})", range=[0, ymax],
                     row=1, col=2)

    frame_ms = 40
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", x=0.62, y=1.08),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=0, xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=frame_ms, redraw=True),
                    fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.05, len=0.55,
            currentvalue=dict(prefix="frame "),
            steps=[dict(method="animate", label=str(f),
                        args=[[str(f)], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))])
                   for f in frames_idx],
        )],
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
        gauge=dict(
            axis=dict(range=[velo_lo, velo_hi]),
            bar=dict(color="#1f77b4"),
            steps=[dict(range=[predicted-1.96*std, predicted+1.96*std],
                        color="#cfe3f3")],
            threshold=dict(line=dict(color="#2ca02c", width=4), value=actual),
        ),
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def build_zscore_figure(zdf, top_n=18):
    import plotly.graph_objects as go

    sub = zdf.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=sub["z"], y=sub["feature"], orientation="h",
        marker=dict(color=sub["z"], colorscale="RdBu", reversescale=True,
                    cmin=-3, cmax=3, colorbar=dict(title="z")),
        hovertemplate="%{y}: z=%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=40, b=30),
        title=f"Biomechanics z-scores — top {top_n} by |z|",
        xaxis_title="z-score (σ from dataset mean)",
    )
    return fig


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OBP Pitching Dashboard — {pitch}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         color: #222; }}
  header {{ background: #0d2b45; color: #fff; padding: 14px 22px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .sub {{ opacity: .8; font-size: 13px; margin-top: 4px; }}
  .tabs {{ display: flex; gap: 4px; background: #0d2b45; padding: 0 16px; }}
  .tablink {{ background: #16456b; color: #fff; border: none; padding: 10px 20px;
             font-size: 14px; cursor: pointer; border-radius: 6px 6px 0 0; }}
  .tablink.active {{ background: #fff; color: #0d2b45; font-weight: 600; }}
  .tabcontent {{ padding: 16px 22px; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
  .row > div {{ flex: 1; min-width: 320px; }}
  table.gloss {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.gloss th, table.gloss td {{ border: 1px solid #ddd; padding: 6px 8px;
         text-align: left; vertical-align: top; }}
  table.gloss th {{ background: #0d2b45; color: #fff; position: sticky; top: 0; }}
  table.gloss tr:nth-child(even) {{ background: #f6f8fa; }}
  td.def {{ max-width: 520px; }}
  details summary {{ cursor: pointer; margin: 8px 0; }}
</style></head>
<body>
<header>
  <h1>OpenBiomechanics Pitching Dashboard</h1>
  <div class="sub">Pitch {pitch} · {tag} · predicted {pred:.1f} mph
     (actual {actual:.1f}, error {err:+.1f}) · model R²={r2:.2f},
     RMSE={rmse:.1f} mph</div>
</header>
<div class="tabs">
  <button class="tablink active" id="b-live" onclick="showTab('live')">
    Live delivery</button>
  <button class="tablink" id="b-gloss" onclick="showTab('gloss')">
    Glossary</button>
</div>
<div id="live" class="tabcontent">
  {anim}
  <div class="row">
    <div>{velo}</div>
    <div>{zfig}</div>
  </div>
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
</script>
</body></html>
"""


def build_html(session_pitch=None, trained=None, step=3, top_n=18,
               out="dashboard.html", offline=False):
    """Build the interactive HTML dashboard for one pitch; return info dict."""
    if trained is None:
        trained = train_velocity_model()
    ds = trained.dataset
    if session_pitch is None:
        session_pitch = str(ds.poi["session_pitch"].iloc[int(trained.test_idx[0])])

    predicted, pred_std = trained.predict_pitch(session_pitch)
    actual = ds.actual_velocity(session_pitch)
    zdf = ds.zscores(session_pitch)
    in_test = ds.index_of(session_pitch) in set(trained.test_idx.tolist())

    markers = c3d_plot.load_c3d(download_c3d_for_pitch(ds, session_pitch))
    frame_times = np.arange(markers.n_frames) / markers.rate
    fp = _load_force(ds, session_pitch, frame_times)
    handed = str(ds.poi.iloc[ds.index_of(session_pitch)].get("p_throws", "R"))
    rear_leg, lead_leg = ("R", "L") if handed.upper().startswith("R") else ("L", "R")

    velo_lo = float(min(ds.y.min(), predicted - 3 * pred_std) - 2)
    velo_hi = float(max(ds.y.max(), predicted + 3 * pred_std) + 2)

    fig_anim = build_pose_force_figure(
        markers, fp, frame_times, lead_leg, rear_leg, step)
    fig_velo = build_velocity_figure(predicted, pred_std, actual, velo_lo, velo_hi)
    fig_z = build_zscore_figure(zdf, top_n=top_n)

    plotlyjs = True if offline else "cdn"
    anim_html = fig_anim.to_html(full_html=False, include_plotlyjs=plotlyjs,
                                 div_id="anim")
    velo_html = fig_velo.to_html(full_html=False, include_plotlyjs=False)
    z_html = fig_z.to_html(full_html=False, include_plotlyjs=False)

    tag = "out-of-sample" if in_test else "in-sample"
    page = _PAGE.format(
        pitch=session_pitch, tag=tag, pred=predicted, actual=actual,
        err=predicted - actual, r2=trained.metrics["r2"],
        rmse=trained.metrics["rmse"], anim=anim_html, velo=velo_html,
        zfig=z_html, glossary=glossary.render_html(),
    )
    with open(out, "w") as fh:
        fh.write(page)

    return {
        "out": out, "session_pitch": session_pitch, "predicted_mph": predicted,
        "actual_mph": actual, "out_of_sample": in_test,
        "has_force": fp is not None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive HTML pitching dashboard.")
    p.add_argument("--pitch", default=None)
    p.add_argument("--out", default="dashboard.html")
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--top-n", type=int, default=18)
    p.add_argument("--offline", action="store_true",
                   help="Embed Plotly.js for fully offline viewing (larger file).")
    args = p.parse_args(argv)
    info = build_html(session_pitch=args.pitch, step=args.step, top_n=args.top_n,
                      out=args.out, offline=args.offline)
    print(f"Saved interactive dashboard to {info['out']}")
    print(f"  pitch {info['session_pitch']}: predicted "
          f"{info['predicted_mph']:.1f} mph, actual {info['actual_mph']:.1f} mph, "
          f"force plates: {'yes' if info['has_force'] else 'no'}")


if __name__ == "__main__":
    main()
