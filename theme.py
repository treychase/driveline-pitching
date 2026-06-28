"""Shared white & orange visual theme for the dashboards."""

from __future__ import annotations

# Primary palette
WHITE = "#ffffff"
PANEL = "#fff7ef"          # very light orange tint for panels
ORANGE = "#ff7a00"         # primary accent
ORANGE_DARK = "#cc5f00"
ORANGE_LIGHT = "#ffb066"
SLATE = "#33414f"          # dark neutral for the secondary series / text
GRID = "#ececec"
TEXT = "#222222"

# Two-series colours (e.g. lead vs rear leg)
SERIES_A = ORANGE          # primary
SERIES_B = SLATE           # secondary

# Diverging colourscale (negative -> white -> positive) on-theme for z-scores.
DIVERGING = [[0.0, "#2c5f8a"], [0.5, "#ffffff"], [1.0, ORANGE]]
# Sequential dirt scale is defined where the mound is drawn (kept brown).


def plotly_template():
    """Return a Plotly template implementing the white/orange theme."""
    import plotly.graph_objects as go

    return go.layout.Template(layout=dict(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Roboto, sans-serif"),
        colorway=[ORANGE, SLATE, ORANGE_LIGHT, "#7a8794", ORANGE_DARK],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#ccc"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#ccc"),
        title=dict(font=dict(color=SLATE)),
    ))


def apply_matplotlib():
    """Apply the theme to Matplotlib rcParams."""
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
        "axes.prop_cycle": mpl.cycler(color=[ORANGE, SLATE, ORANGE_LIGHT,
                                             "#7a8794", ORANGE_DARK]),
    })
