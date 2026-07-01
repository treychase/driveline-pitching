"""Interactive, self-contained HTML dashboard (Plotly) with tabs.

Tabs:
* **Live delivery** — a **pitch picker** selects the delivery; a Plotly animation
  shows the 3D pose on the dirt mound with a Play button, a frame slider, and a
  **toggle for joint velocity vectors** on every joint. A synchronized live
  ground-reaction-force plot (lead vs. rear leg, by handedness) tracks the
  delivery. Below: the Bayesian-Lasso velocity gauge, the biomechanics z-scores,
  the **joint work accumulated during the delivery**, and the **z-scores of joint
  work** for the selected pitch.
* **Joint work** — a **3D body coloured by joint work** (red generates, blue
  absorbs), with no plain stick figure overlaid, plus the work-vs-time curves and
  z-scores.
* **Efficiency** — a **mechanical efficiency score** (torso + lower-body drive
  vs. throwing-elbow load) with a gauge and a signed contribution breakdown.
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

_LEAD_C, _REAR_C = theme.PLOT_A, theme.PLOT_B
_DIRT_SCALE = [[0.0, "#5b3a1e"], [0.5, "#8a5a30"], [1.0, "#b9854f"]]
_VEC_SCALE = 0.03   # metres drawn per (m/s) of joint velocity
_VEC_HEAD = 0.06    # arrowhead cone size (m, absolute)
_FOOT_STOPS = theme.FOOT_STOPS
_FOOT_MAX = theme.FOOT_MAX   # red flash when a foot is at its peak force


def _foot_state(val, peak, unit):
    """Square colour, text colour, and label for a foot's current force."""
    frac = float(np.clip(val / peak, 0, 1)) if peak > 0 else 0.0
    if peak > 0 and val >= 0.99 * peak:
        return (_FOOT_MAX, "white", f"<b>{val:.2f}</b><br>◀ MAX")
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


def _vector_arrows(vecs, fi, scale=_VEC_SCALE):
    """Shaft segments (NaN-separated) and arrowhead cones for joint vectors.

    Returns ``((xs, ys, zs), (cx, cy, cz, cu, cv, cw))`` — the shaft polyline
    and the arrowhead cone positions (at each shaft tip) with unit direction
    components, so the vectors read as arrows pointing along the motion.
    """
    xs, ys, zs = [], [], []
    cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
    for d in vecs.values():
        p, raw = d["pos"][fi], d["vel"][fi]
        v = raw * scale
        n = float(np.linalg.norm(raw))
        if np.isfinite(p).all() and np.isfinite(v).all() and n > 1e-6:
            q = p + v
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
            u = raw / n  # unit direction for a uniform arrowhead
            cx.append(q[0]); cy.append(q[1]); cz.append(q[2])
            cu.append(u[0]); cv.append(u[1]); cw.append(u[2])
    return (xs, ys, zs), (cx, cy, cz, cu, cv, cw)


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

    # 0: mound (coarser tessellation than the static plots — it is re-rasterised
    # every animation frame, so fewer faces = faster, smoother playback).
    mx, my, mz, mi, mj, mk, inten, _ = mound_mod.mound_trimesh(markers, sectors=32,
                                                               rings=10)
    fig.add_trace(go.Mesh3d(x=mx, y=my, z=mz, i=mi, j=mj, k=mk, intensity=inten,
                            colorscale=_DIRT_SCALE, showscale=False, opacity=1.0,
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9),
                            hoverinfo="skip"), row=1, col=1)
    # 1,2: skeleton + joints
    c0 = markers.points[f0]
    sx, sy, sz = _seg_coords(c0, pairs)
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                               line=dict(color=theme.SLATE, width=4),
                               name="skeleton", showlegend=False,
                               hoverinfo="skip"), row=1, col=1)
    fin = np.isfinite(c0).all(axis=-1)
    fig.add_trace(go.Scatter3d(x=c0[fin, 0], y=c0[fin, 1], z=c0[fin, 2],
                               mode="markers", marker=dict(size=3, color=theme.PLOT_MUTED),
                               name="markers", showlegend=False,
                               hoverinfo="skip"), row=1, col=1)
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
    # 8,9: joint velocity vectors as arrows (shafts + arrowhead cones)
    (vx, vy, vz), (cx, cy, cz, cu, cv, cw) = _vector_arrows(vecs, f0)
    fig.add_trace(go.Scatter3d(x=vx, y=vy, z=vz, mode="lines",
                               line=dict(color=theme.PLOT_ACCENT, width=5),
                               name="joint velocity", showlegend=False,
                               hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Cone(x=cx, y=cy, z=cz, u=cu, v=cv, w=cw, anchor="tip",
                          sizemode="absolute", sizeref=_VEC_HEAD, showscale=False,
                          colorscale=[[0, theme.PLOT_ACCENT], [1, theme.PLOT_ACCENT]],
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

    # Delivery-phase spans, used to shade the GRF plot AND to tint the live 3D
    # backdrop frame-by-frame as the delivery plays.
    import force_plate as _fp
    _BG_NEUTRAL = "#dfe3e8"
    # Bold wall tints so the phase colour clearly reads on the 3D box.
    _BG_STRONG = {"Wind-up": "#9bb9e6", "Stride": "#8ed49f",
                  "Arm cocking": "#f1d24f", "Acceleration": "#ff9e54",
                  "Deceleration": "#ec8585"}
    phase_spans = []
    if fp is not None:
        ev_times = {k: float(frame_times[fr]) for k, fr in fp["event_frames"].items()
                    if fr < len(frame_times)}
        phase_spans = _fp.delivery_phases(ev_times, float(frame_times[-1]))

    def _phase_bg(t):
        for ph in phase_spans:
            if ph["t0"] <= t <= ph["t1"]:
                return _BG_STRONG.get(ph["label"], ph["color"])
        return _BG_NEUTRAL

    # event lines + shaded delivery phases on the force subplot
    xa = fig.data[3].xaxis or "x"
    ya = fig.data[3].yaxis or "y"
    if fp is not None:
        for ph in phase_spans:
            fig.add_shape(type="rect", x0=ph["t0"], x1=ph["t1"], y0=0, y1=ymax,
                          xref=xa, yref=ya, line=dict(width=0),
                          fillcolor=ph["color"], opacity=0.85, layer="below")
            # Rotate labels vertical so the narrow late phases don't overlap.
            fig.add_annotation(x=(ph["t0"] + ph["t1"]) / 2, y=ymax * 0.98,
                               xref=xa, yref=ya, text=ph["label"], showarrow=False,
                               yanchor="top", textangle=-90,
                               font=dict(size=8, color="#888"))
        for key, color in [("fp", "#888"), ("mer", "#aaa"), ("br", theme.PLOT_ACCENT)]:
            fr = fp["event_frames"].get(key)
            if fr is not None and fr < len(frame_times):
                tx0 = float(frame_times[fr])
                fig.add_shape(type="line", x0=tx0, x1=tx0, y0=0, y1=ymax,
                              xref=xa, yref=ya,
                              line=dict(color=color, dash="dash", width=1))

    # A bold phase-colour bar across the top of the 3D panel. It is drawn on the
    # SVG layer (above the WebGL scene) so, unlike the wall tint, it is clearly
    # visible, and it updates every frame so the live phase reads at a glance.
    base_shapes = [s.to_plotly_json() for s in fig.layout.shapes]

    def _phase_bar(color):
        return dict(type="rect", xref="paper", yref="paper",
                    x0=0.0, x1=0.50, y0=0.90, y1=0.95, layer="above",
                    fillcolor=color, line=dict(color="#bbb", width=1))

    fig.add_shape(**_phase_bar(_phase_bg(float(frame_times[f0]))))

    # frames
    lead_peak, rear_peak = float(np.nanmax(lead)), float(np.nanmax(rear))
    anim_frames = []
    for f in frames_idx:
        coords = markers.points[f]
        sx, sy, sz = _seg_coords(coords, pairs)
        fin = np.isfinite(coords).all(axis=-1)
        (vx, vy, vz), (cx, cy, cz, cu, cv, cw) = _vector_arrows(vecs, f)
        tt = float(frame_times[f])
        lc, ltc, ltxt = _foot_state(lead[f], lead_peak, unit)
        rc, rtc, rtxt = _foot_state(rear[f], rear_peak, unit)
        bg = _phase_bg(tt)
        frame_data = [
            go.Scatter3d(x=sx, y=sy, z=sz),
            go.Scatter3d(x=coords[fin, 0], y=coords[fin, 1], z=coords[fin, 2]),
            go.Scatter(x=[tt, tt], y=[0, ymax]),
            go.Scatter(x=[tt], y=[float(lead[f])]),
            go.Scatter(x=[tt], y=[float(rear[f])]),
            go.Scatter3d(x=vx, y=vy, z=vz),
            go.Cone(x=cx, y=cy, z=cz, u=cu, v=cv, w=cw),
            go.Scatter(x=[0.5], y=[0.70],
                       marker=dict(symbol="square", size=64, color=lc,
                                   line=dict(color="#888", width=1)),
                       text=[ltxt], textfont=dict(color=ltc, size=12)),
            go.Scatter(x=[0.5], y=[0.28],
                       marker=dict(symbol="square", size=64, color=rc,
                                   line=dict(color="#888", width=1)),
                       text=[rtxt], textfont=dict(color=rtc, size=12)),
        ]
        frame_traces = [1, 2, 5, 6, 7, 8, 9, 10, 11]
        anim_frames.append(go.Frame(name=str(f), layout=dict(
            scene=dict(xaxis=dict(backgroundcolor=bg),
                       yaxis=dict(backgroundcolor=bg),
                       zaxis=dict(backgroundcolor=bg)),
            shapes=base_shapes + [_phase_bar(bg)]),
            data=frame_data, traces=frame_traces))
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

    bg0 = _phase_bg(float(frame_times[f0]))
    fig.update_scenes(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)",
                      zaxis_title="Z up (m)",
                      # uirevision keeps the user's camera while the backdrop
                      # colour animates with the delivery phase.
                      uirevision="pose",
                      xaxis=dict(backgroundcolor=bg0),
                      yaxis=dict(backgroundcolor=bg0),
                      zaxis=dict(backgroundcolor=bg0),
                      camera=dict(eye=dict(x=1.35, y=-1.35, z=0.75)))
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text=f"vertical GRF ({unit})", range=[0, ymax],
                     row=1, col=2)
    # One animation frame spans ``step`` captured frames, so this many ms per
    # frame plays the delivery back at real time; "Fast" plays at ~2x.
    rt_ms = max(1, int(round(1000 * step / markers.rate)))
    fig.update_layout(
        template=theme.plotly_template(), height=620,
        margin=dict(l=0, r=0, t=95, b=70),
        # Only the two GRF leg traces appear in the legend (the 3D traces are
        # self-evident); place it top-centre, between the corner button groups
        # and above the subplot titles, so it never collides with the slider.
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.16,
                    yanchor="top"),
        updatemenus=[
            # Play / Pause, top-left, clear of the titles.
            dict(type="buttons", direction="right", showactive=False,
                 x=0.0, y=1.16, xanchor="left", yanchor="top", bgcolor="#fff",
                 buttons=[
                     dict(label=lbl, method="animate", args=[None, dict(
                         frame=dict(duration=dur, redraw=True), fromcurrent=True,
                         transition=dict(duration=0))])
                     for lbl, dur in (("▶ Play", rt_ms),
                                      ("▶▶ Fast", max(1, rt_ms // 2)))
                 ] + [
                     dict(label="⏸ Pause", method="animate", args=[[None], dict(
                         frame=dict(duration=0, redraw=False), mode="immediate")]),
                 ],
                 ),
            # Joint-vector toggle, top-right, clear of the titles.
            dict(type="buttons", direction="right", showactive=True,
                 x=1.0, y=1.16, xanchor="right", yanchor="top", bgcolor="#fff",
                 buttons=[
                     dict(label="Vectors on", method="restyle",
                          args=[{"visible": True}, [8, 9]]),
                     dict(label="Vectors off", method="restyle",
                          args=[{"visible": False}, [8, 9]]),
                 ]),
        ],
        sliders=[dict(active=0, y=0, yanchor="top", x=0.05, len=0.55,
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
                   bar=dict(color=theme.PLOT_A),
                   steps=[dict(range=[predicted-1.96*std, predicted+1.96*std],
                               color="#d7e6f2")],
                   threshold=dict(line=dict(color=theme.PLOT_ACCENT, width=4),
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


# Limb segments shaded by the work generated at their driving joint, plus the
# neutral structural segments that give the body its shape. Joint names match
# joint_kinetics: ``shoulder``/``elbow`` are the throwing arm, ``glove_*`` the
# glove arm, and ``lead``/``rear`` the legs — all in real lab-frame positions,
# so the body is correctly sided for right- and left-handed pitchers.
_BODY_LIMBS = [
    ("shoulder", "elbow", "shoulder"),                 # throwing upper arm
    ("elbow", "wrist", "elbow"),                        # throwing forearm
    ("glove_shoulder", "glove_elbow", "glove_shoulder"),
    ("glove_elbow", "glove_wrist", "glove_elbow"),
    ("lead_hip", "lead_knee", "lead_hip"),
    ("lead_knee", "lead_ankle", "lead_knee"),
    ("rear_hip", "rear_knee", "rear_hip"),
    ("rear_knee", "rear_ankle", "rear_knee"),
]
_BODY_NEUTRAL = [("shoulder", "glove_shoulder"), ("lead_hip", "rear_hip"),
                 ("wrist", "hand"), ("glove_wrist", "glove_hand")]


def build_jointwork_body_figure(positions, zwork, handed):
    """A 3D body with each limb shaded by its joint's work z-score.

    ``positions`` maps joint -> ``(3,)`` lab-frame coordinates (e.g. the pose at
    ball release); ``zwork`` is the ``work_zscores`` DataFrame; ``handed`` is the
    pitcher's throwing hand (``'R'``/``'L'``). Warm limbs generated more energy
    than the dataset average at that joint, cool limbs less.

    The body is drawn from the joint centres so the **work colour carries the
    figure** — the limb segments and joint spheres are shaded by energy
    generated, with only thin neutral links (shoulders, hips, hands) and a head
    for shape. The plain grey C3D stick figure is intentionally **not** overlaid.
    """
    import plotly.graph_objects as go

    zmap = dict(zip(zwork["joint"].astype(str), zwork["z"].astype(float)))
    wmap = dict(zip(zwork["joint"].astype(str), zwork["work_J"].astype(float)))

    def P(j):
        p = positions.get(j)
        return p if p is not None and np.isfinite(p).all() else None

    fig = go.Figure()

    # Joint-centre body with only thin neutral links + a head for shape (no
    # grey C3D stick figure): the work-shaded limbs and joints carry the colour.
    nx, ny, nz = [], [], []
    for a, b in _BODY_NEUTRAL:
        pa, pb = P(a), P(b)
        if pa is not None and pb is not None:
            nx += [pa[0], pb[0], None]; ny += [pa[1], pb[1], None]
            nz += [pa[2], pb[2], None]
    head_pt = None
    sh = [P("shoulder"), P("glove_shoulder")]
    hp = [P("lead_hip"), P("rear_hip")]
    if all(p is not None for p in sh + hp):
        smid, hmid = (sh[0] + sh[1]) / 2, (hp[0] + hp[1]) / 2
        nx += [smid[0], hmid[0], None]; ny += [smid[1], hmid[1], None]
        nz += [smid[2], hmid[2], None]
        head_pt = smid + 0.4 * (smid - hmid)
        nx += [smid[0], head_pt[0], None]; ny += [smid[1], head_pt[1], None]
        nz += [smid[2], head_pt[2], None]
    if nx:
        fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode="lines",
                                   line=dict(color=theme.PLOT_MUTED, width=6),
                                   showlegend=False, hoverinfo="skip"))
    if head_pt is not None:
        fig.add_trace(go.Scatter3d(
            x=[head_pt[0]], y=[head_pt[1]], z=[head_pt[2]], mode="markers",
            marker=dict(size=18, color=theme.PLOT_MUTED, opacity=0.95,
                        line=dict(color="#fff", width=1)),
            showlegend=False, hoverinfo="skip"))

    # Work-shaded limb segments (thick lines coloured by the driving joint's z).
    for a, b, drv in _BODY_LIMBS:
        pa, pb = P(a), P(b)
        if pa is None or pb is None:
            continue
        z = zmap.get(drv)
        if z is None:
            fig.add_trace(go.Scatter3d(
                x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]], mode="lines",
                line=dict(color=theme.PLOT_MUTED, width=12), showlegend=False,
                hoverinfo="skip"))
        else:
            w = wmap.get(drv, float("nan"))
            fig.add_trace(go.Scatter3d(
                x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]], mode="lines",
                line=dict(color=[z, z], coloraxis="coloraxis", width=12),
                hovertemplate=f"{drv}: {w:.0f} J (z={z:+.2f})<extra></extra>",
                showlegend=False))

    # Joint spheres coloured by work, carrying the shared colour axis.
    jx, jy, jz, jc, jt = [], [], [], [], []
    for j, z in zmap.items():
        p = P(j)
        if p is None:
            continue
        jx.append(p[0]); jy.append(p[1]); jz.append(p[2]); jc.append(z)
        jt.append(f"{j}: {wmap.get(j, float('nan')):.0f} J (z={z:+.2f})")
    if jx:
        fig.add_trace(go.Scatter3d(
            x=jx, y=jy, z=jz, mode="markers",
            marker=dict(size=8, color=jc, coloraxis="coloraxis",
                        line=dict(color="#fff", width=1)),
            text=jt, hovertemplate="%{text}<extra></extra>", showlegend=False))

    arm = "right" if str(handed).upper().startswith("R") else "left"
    fig.update_layout(
        template=theme.plotly_template(), height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        title=(f"Joint work on the body — {arm}-handed pitcher "
               f"(warm = more energy generated vs dataset)"),
        coloraxis=dict(colorscale=theme.DIVERGING, cmin=-3, cmax=3,
                       colorbar=dict(title="work z")),
        scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)",
                   zaxis_title="Z up (m)",
                   xaxis=dict(backgroundcolor="#fafafa"),
                   yaxis=dict(backgroundcolor="#fafafa"),
                   zaxis=dict(backgroundcolor="#f2f2f2"),
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
    )
    return fig


def build_jointwork_time_figure(jw, frame_times):
    import plotly.graph_objects as go

    fig = go.Figure()
    palette = theme.PLOT_COLORWAY
    for k, (joint, w) in enumerate(jw.work.items()):
        fig.add_trace(go.Scatter(x=jw.time, y=w, mode="lines", name=joint,
                                 line=dict(color=palette[k % len(palette)], width=2)))
    br = jw.events.get("br")
    if br is not None:
        ymax = max((np.nanmax(w) for w in jw.work.values()), default=1)
        ymin = min((np.nanmin(w) for w in jw.work.values()), default=0)
        fig.add_shape(type="line", x0=br, x1=br, y0=ymin, y1=ymax,
                      line=dict(color=theme.PLOT_ACCENT, dash="dash", width=1))
        fig.add_annotation(x=br, y=ymax, text="BR", showarrow=False,
                           font=dict(size=10, color=theme.PLOT_ACCENT))
    fig.update_layout(template=theme.plotly_template(), height=440,
                      margin=dict(l=10, r=10, t=40, b=30),
                      title="Joint work accumulated during the delivery (J)",
                      xaxis_title="time (s)", yaxis_title="energy generated (J)")
    return fig


# Green (helps efficiency) -> white -> red (hurts) diverging scale for the
# efficiency contribution breakdown.
_EFF_SCALE = [[0.0, "#b2182b"], [0.5, "#f7f7f7"], [1.0, "#1a9850"]]
_EFF_LABELS = {
    "rear_hip_generation_pkh_fp": "rear hip generation",
    "rear_knee_generation_pkh_fp": "rear knee generation",
    "lead_hip_generation_fp_br": "lead hip generation",
    "lead_knee_generation_fp_br": "lead knee generation",
    "pelvis_lumbar_transfer_fp_br": "pelvis→lumbar transfer",
    "thorax_distal_transfer_fp_br": "thorax (trunk) transfer",
    "elbow_varus_moment": "elbow varus moment (load)",
}
_GROUP_LABEL = {"lower_body": "lower body", "torso": "torso",
                "elbow_load": "elbow load"}


def build_efficiency_figure(res):
    """Gauge (0-100 score) + signed contribution breakdown for one pitch.

    ``res`` is an :class:`efficiency.EfficiencyScore`. The gauge shows the
    dataset percentile; the bars show each metric's standardized contribution,
    signed so that green/positive always *helps* efficiency (drivers up, elbow
    load down).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.42, 0.58], vertical_spacing=0.14,
        specs=[[{"type": "indicator"}], [{"type": "xy"}]],
        subplot_titles=("", "Contributions (green = helps, red = hurts)"),
    )

    # Colour the gauge bar by how good the score is.
    bar_color = (theme.ORANGE if res.score >= 45 else theme.PLOT_ACCENT)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=res.score,
        number=dict(suffix=" / 100", font=dict(size=34)),
        title=dict(text="Mechanical efficiency<br>"
                        "<span style='font-size:0.75em;color:gray'>"
                        "torso + lower-body drive vs. throwing-elbow load "
                        "(dataset percentile)</span>"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=bar_color),
            steps=[dict(range=[0, 45], color="#fde3e1"),
                   dict(range=[45, 75], color="#fff3e6"),
                   dict(range=[75, 100], color="#e5f4e6")],
            threshold=dict(line=dict(color=theme.SLATE, width=3), value=50),
        ),
    ), row=1, col=1)

    # Contribution bars, ordered drivers (top) then elbow load (bottom).
    contrib = res.contributions
    order = {"lower_body": 0, "torso": 1, "elbow_load": 2}
    contrib = contrib.iloc[
        contrib["group"].map(order).argsort(kind="stable")
    ].iloc[::-1]
    labels = [f"{_EFF_LABELS.get(m, m)}" for m in contrib["metric"]]
    signed = contrib["signed_z"].to_numpy()
    fig.add_trace(go.Bar(
        x=signed, y=labels, orientation="h",
        marker=dict(color=signed, colorscale=_EFF_SCALE, cmin=-3, cmax=3,
                    line=dict(color="#ccc", width=0.5)),
        customdata=np.stack([contrib["group"].map(_GROUP_LABEL),
                             contrib["z"], contrib["value"]], axis=-1),
        hovertemplate=("%{y} (%{customdata[0]})<br>z=%{customdata[1]:+.2f}, "
                       "value=%{customdata[2]:.1f}<br>"
                       "contribution=%{x:+.2f}<extra></extra>"),
    ), row=2, col=1)
    fig.update_xaxes(title_text="standardized contribution (σ)", range=[-3, 3],
                     zeroline=True, zerolinecolor="#999", zerolinewidth=1,
                     row=2, col=1)
    fig.update_yaxes(automargin=True, row=2, col=1)
    fig.update_layout(template=theme.plotly_template(), height=560,
                      margin=dict(l=10, r=10, t=60, b=40), showlegend=False)
    return fig


def build_diagnostics_figures(trained, highlight=None):
    """Build Bayesian-Lasso model-diagnostic figures (global to the model)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model, ds = trained.model, trained.dataset
    n = len(ds.y)
    test_idx = {int(i) for i in trained.test_idx.tolist()}
    pred = model.predict(ds.X)
    actual = ds.y
    sp_all = ds.poi["session_pitch"].astype(str).to_numpy()
    is_test = np.array([i in test_idx for i in range(n)])
    hl = set(highlight or [])
    tmpl = theme.plotly_template()

    # 1) Predicted vs actual
    lo = float(min(actual.min(), pred.min())) - 1
    hi = float(max(actual.max(), pred.max())) + 1
    pva = go.Figure()
    pva.add_trace(go.Scatter(
        x=actual[~is_test], y=pred[~is_test], mode="markers", name="train",
        marker=dict(color=theme.PLOT_MUTED, size=6, opacity=0.6),
        text=sp_all[~is_test],
        hovertemplate="train %{text}<br>actual %{x:.1f}, pred %{y:.1f}<extra></extra>"))
    pva.add_trace(go.Scatter(
        x=actual[is_test], y=pred[is_test], mode="markers", name="test (held out)",
        marker=dict(color=theme.PLOT_A, size=7), text=sp_all[is_test],
        hovertemplate="test %{text}<br>actual %{x:.1f}, pred %{y:.1f}<extra></extra>"))
    pva.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal",
                             line=dict(color="#999", dash="dash")))
    if hl:
        m = np.array([s in hl for s in sp_all])
        pva.add_trace(go.Scatter(
            x=actual[m], y=pred[m], mode="markers+text", name="selected",
            marker=dict(color=theme.PLOT_ACCENT, size=13, symbol="circle-open",
                        line=dict(width=2)),
            text=sp_all[m], textposition="top center"))
    pva.update_layout(
        template=tmpl, height=420, margin=dict(l=10, r=10, t=50, b=40),
        title=(f"Predicted vs actual — test R²={trained.metrics['r2']:.2f}, "
               f"RMSE={trained.metrics['rmse']:.1f}, MAE={trained.metrics['mae']:.1f} mph "
               f"(n_train={trained.metrics['n_train']}, n_test={trained.metrics['n_test']})"),
        xaxis_title="actual velocity (mph)", yaxis_title="predicted velocity (mph)")

    # 2) Residuals vs predicted (test)
    resid = actual - pred
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(
        x=pred[is_test], y=resid[is_test], mode="markers",
        marker=dict(color=theme.PLOT_A, size=7), text=sp_all[is_test], name="test",
        hovertemplate="%{text}<br>pred %{x:.1f}, resid %{y:+.1f}<extra></extra>"))
    rfig.add_hline(y=0, line=dict(color="#999", dash="dash"))
    rfig.update_layout(template=tmpl, height=360, margin=dict(l=10, r=10, t=40, b=40),
                       title="Residuals vs predicted (held-out test)",
                       xaxis_title="predicted (mph)", yaxis_title="residual (mph)")

    # 3) Posterior coefficients (top 20 by |mean|, 95% CI)
    rows = model.coef_summary(ds.feature_names)[:20][::-1]
    names = [r["feature"] for r in rows]
    means = np.array([r["mean"] for r in rows])
    err_lo = means - np.array([r["ci_low"] for r in rows])
    err_hi = np.array([r["ci_high"] for r in rows]) - means
    cmax = float(np.abs(means).max()) or 1.0
    coef = go.Figure(go.Bar(
        x=means, y=names, orientation="h",
        marker=dict(color=means, colorscale=theme.DIVERGING, cmin=-cmax, cmax=cmax),
        error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                     color="#888", thickness=1)))
    coef.update_layout(
        template=tmpl, height=560, margin=dict(l=210, r=10, t=50, b=40),
        title="Posterior coefficients (standardized) — top 20 by |mean|, 95% CI",
        xaxis_title="standardized effect on velocity")
    coef.update_yaxes(automargin=True)

    # 4) Posterior distributions of σ (noise) and λ² (shrinkage)
    sigma = np.sqrt(model.sigma2_samples_)
    post = make_subplots(rows=1, cols=2,
                         subplot_titles=("Residual noise σ (mph)", "Shrinkage λ²"))
    post.add_trace(go.Histogram(x=sigma, marker=dict(color=theme.PLOT_A),
                                nbinsx=40), row=1, col=1)
    post.add_trace(go.Histogram(x=model.lambda2_samples_,
                                marker=dict(color=theme.PLOT_B), nbinsx=40),
                   row=1, col=2)
    post.update_layout(template=tmpl, height=360, showlegend=False,
                       margin=dict(l=10, r=10, t=50, b=30),
                       title="Posterior distributions (Gibbs samples)")

    return {"pva": pva, "resid": rfig, "coef": coef, "post": post}


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
  .picker {{ padding: 10px 22px 0; font-size: 14px; background: #fff7ef;
             border-bottom: 1px solid #ffe0c2; }}
  .picker select {{ font-size: 14px; padding: 6px 10px; border: 1px solid #ffb066;
             border-radius: 6px; color: #cc5f00; }}
  .pitch-head {{ font-size: 13px; color: #555; margin: 4px 0 10px; }}
  h2 {{ color: #cc5f00; }}
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
  <button class="tablink" id="b-work" onclick="showTab('work')">Joint work</button>
  <button class="tablink" id="b-eff" onclick="showTab('eff')">Efficiency</button>
  <button class="tablink" id="b-diag" onclick="showTab('diag')">
    Model diagnostics</button>
  <button class="tablink" id="b-gloss" onclick="showTab('gloss')">Glossary</button>
</div>
<div class="picker" id="pickerbar">Pitch:
  <select id="pitch-picker" onchange="pickPitch(this.value)">{options}</select>
  <span style="color:#888">— applies to the Live delivery, Joint work, and
    Efficiency tabs</span>
</div>
<div id="live" class="tabcontent">{live_panels}</div>
<div id="work" class="tabcontent" style="display:none">{work_panels}</div>
<div id="eff" class="tabcontent" style="display:none">{eff_panels}</div>
<div id="diag" class="tabcontent" style="display:none">{diag}</div>
<div id="gloss" class="tabcontent" style="display:none">{glossary}</div>
<script>
var TABS = ['live', 'work', 'eff', 'diag', 'gloss'];
function showTab(id) {{
  TABS.forEach(function(t) {{
    document.getElementById(t).style.display = (t===id) ? 'block' : 'none';
    document.getElementById('b-'+t).classList.toggle('active', t===id);
  }});
  document.getElementById('pickerbar').style.display =
    (id==='live' || id==='work' || id==='eff') ? 'block' : 'none';
  window.dispatchEvent(new Event('resize'));
}}
function pickPitch(sp) {{
  document.querySelectorAll('.pitch-panel').forEach(function(p) {{
    p.style.display = (p.dataset.sp === sp) ? 'block' : 'none';
  }});
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body></html>
"""


def _pitch_figures(ds, trained, sp, step, top_n, eff_model=None):
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
        zwork = jk.work_zscores(sp)
        figs["jwt"] = build_jointwork_time_figure(jw, frame_times)
        figs["jwz"] = build_jointwork_z_figure(zwork)
        # 3D body coloured by joint work at ball release (no stick figure).
        try:
            br = jw.events.get("br")
            t = jw.time[-1] if br is None else br
            figs["jwbody"] = build_jointwork_body_figure(
                jk.joint_positions_at(sp, t), zwork, handed)
        except Exception:
            pass
    except KeyError:
        pass
    if eff_model is not None:
        try:
            figs["eff"] = build_efficiency_figure(eff_model.score(sp))
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
    import efficiency
    eff_model = efficiency.MechanicalEfficiencyModel.fit(ds.poi)
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

    live_panels, work_panels, eff_panels, options = [], [], [], []
    for n, sp in enumerate(pitches):
        figs, head = _pitch_figures(ds, trained, sp, step, top_n, eff_model)
        anim = emit(figs["anim"], f"anim-{sp}")
        velo = emit(figs["velo"], f"velo-{sp}")
        zbio = emit(figs["zbio"], f"zbio-{sp}")
        disp = "block" if n == 0 else "none"
        live_panels.append(
            f"<div class='pitch-panel' data-sp='{sp}' id='live-{sp}' "
            f"style='display:{disp}'><div class='pitch-head'>{head}</div>{anim}"
            f"<div class='row'><div>{velo}</div><div>{zbio}</div></div></div>"
        )
        if "jwt" in figs:
            body_html = (emit(figs["jwbody"], f"jwbody-{sp}")
                         if figs.get("jwbody") is not None else "")
            jwt = emit(figs["jwt"], f"jwt-{sp}")
            jwz = emit(figs["jwz"], f"jwz-{sp}")
            wbody = (f"{body_html}"
                     f"<div class='row'><div>{jwt}</div><div>{jwz}</div></div>")
        else:
            wbody = "<p>No joint-work data for this pitch.</p>"
        work_panels.append(
            f"<div class='pitch-panel' data-sp='{sp}' id='work-{sp}' "
            f"style='display:{disp}'><div class='pitch-head'>{head}</div>"
            f"<h2>Joint work — energy generated per joint</h2>{wbody}</div>"
        )
        if "eff" in figs:
            eff = emit(figs["eff"], f"eff-{sp}")
            ebody = (f"<div class='pitch-head'>{head}</div>"
                     f"<h2>Mechanical efficiency — torso &amp; lower-body drive "
                     f"vs. throwing-elbow load</h2>{eff}")
        else:
            ebody = "<p>No efficiency data for this pitch.</p>"
        eff_panels.append(
            f"<div class='pitch-panel' data-sp='{sp}' id='eff-{sp}' "
            f"style='display:{disp}'>{ebody}</div>"
        )
        options.append(f"<option value='{sp}'>{sp}</option>")

    # Model diagnostics (global to the fitted model; selected pitches highlighted)
    d = build_diagnostics_figures(trained, highlight=pitches)
    diag = (
        "<h2>Model diagnostics — Bayesian Lasso</h2>"
        + emit(d["pva"], "diag-pva")
        + f"<div class='row'><div>{emit(d['resid'], 'diag-resid')}</div>"
        + f"<div>{emit(d['post'], 'diag-post')}</div></div>"
        + emit(d["coef"], "diag-coef")
    )

    page = _PAGE.format(
        r2=trained.metrics["r2"], rmse=trained.metrics["rmse"],
        options="".join(options), live_panels="".join(live_panels),
        work_panels="".join(work_panels), eff_panels="".join(eff_panels),
        diag=diag, glossary=glossary.render_html(),
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
