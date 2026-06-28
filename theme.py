"""Visual theme: orange shell (page chrome), neutral plots.

The orange palette is used for the dashboard *shell* — header, tabs, the pitch
picker, buttons. The plots themselves use a neutral palette so the colour
carries data, not branding.
"""

from __future__ import annotations

# --- Shell (chrome) palette: orange -------------------------------------- #
WHITE = "#ffffff"
PANEL = "#fff7ef"          # very light orange tint for shell panels
ORANGE = "#ff7a00"         # primary shell accent
ORANGE_DARK = "#cc5f00"
ORANGE_LIGHT = "#ffb066"
SLATE = "#33414f"
GRID = "#ececec"
TEXT = "#222222"

# --- Plot palette: neutral ----------------------------------------------- #
PLOT_A = "#1f77b4"         # primary series (e.g. lead leg / predicted)
PLOT_B = "#5a6470"         # secondary series (e.g. rear leg / actual)
PLOT_ACCENT = "#d62728"    # emphasis (joint vectors, peak/MAX)
PLOT_MUTED = "#9aa3ad"
PLOT_COLORWAY = [PLOT_A, PLOT_B, "#2ca02c", "#9467bd", "#8c564b", PLOT_ACCENT,
                 "#17becf", "#bcbd22"]

# Neutral diverging scales (blue -> white -> red) for z-scores.
DIVERGING = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]
DIVERGING_MPL = ["#2166ac", "#f7f7f7", "#b2182b"]

# Sequential ramp for the foot-vGRF squares (light -> blue); peak flashes red.
FOOT_STOPS = [(0.0, (244, 247, 251)), (0.5, (146, 183, 219)), (1.0, (31, 119, 180))]
FOOT_MAX = "#d62728"


def plotly_template():
    """Return a clean white Plotly template with a neutral colourway."""
    import plotly.graph_objects as go

    return go.layout.Template(layout=dict(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Roboto, sans-serif"),
        colorway=PLOT_COLORWAY,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#ccc"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#ccc"),
        title=dict(font=dict(color=SLATE)),
    ))


def apply_matplotlib():
    """Apply a clean white background with a neutral colour cycle."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": SLATE,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "axes.prop_cycle": mpl.cycler(color=PLOT_COLORWAY),
    })
