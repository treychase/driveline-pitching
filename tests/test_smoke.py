"""Offline smoke tests for the OBP pitching dashboard (no network needed)."""

import numpy as np


def test_all_modules_import():
    import bayesian_lasso, c3d_plot, dashboard, dashboard_html  # noqa: F401
    import data_sources, force_plate, glossary, joint_kinetics  # noqa: F401
    import mound, theme, velocity_model  # noqa: F401


def test_bayesian_lasso_recovers_sparse_signal():
    from bayesian_lasso import BayesianLasso

    rng = np.random.default_rng(0)
    n, p = 200, 12
    X = rng.standard_normal((n, p))
    true = np.zeros(p)
    true[[0, 3, 7]] = [4.0, -3.0, 2.0]
    y = X @ true + rng.standard_normal(n) * 0.5

    model = BayesianLasso(n_iter=800, burn_in=400, random_state=0).fit(X, y)
    top = {r["feature"] for r in model.coef_summary()[:3]}
    assert top == {"x0", "x3", "x7"}
    mean, std = model.predict(X[:5], return_std=True)
    assert mean.shape == (5,) and np.all(std > 0)


def test_glossary_covers_structure():
    import glossary

    assert len(glossary.TERMS) >= 80
    df = glossary.as_dataframe()
    assert {"variable", "category", "units", "event", "definition"} <= set(df.columns)
    assert glossary.lookup("pitch_speed_mph") is not None
    assert "<table" in glossary.render_html()
    assert glossary.render_markdown().startswith("# Biomechanics Glossary")


def test_delivery_phases():
    import force_plate

    events = {"pkh": 0.1, "fp": 0.8, "mer": 0.95, "br": 1.0}
    phases = force_plate.delivery_phases(events, t_end=1.2)
    labels = [p["label"] for p in phases]
    assert labels == ["Wind-up", "Stride", "Arm cocking", "Acceleration",
                      "Deceleration"]
    # spans are contiguous and ordered
    for a, b in zip(phases, phases[1:]):
        assert a["t1"] == b["t0"]


def test_mound_geometry_from_synthetic_markers():
    import c3d_plot
    import mound

    rng = np.random.default_rng(1)
    labels = ["RTOE", "LTOE", "RHEE", "LHEE", "RANK", "LANK"]
    n = 60
    pts = np.zeros((n, len(labels), 3))
    for i in range(len(labels)):
        pts[:, i, :] = rng.normal(0, 0.2, (n, 3)) + np.array([i * 0.1, 0, 0.05])
    markers = c3d_plot.C3DMarkers(points=pts, labels=labels, rate=360.0)
    params = mound.estimate_mound(markers)
    assert params["r_outer"] > params["r_flat"] > 0
    assert params["table_z"] > params["field_z"]
    faces, colors, params2, verts = mound.build_mound_mesh(markers)
    assert verts.shape[1] == 3 and len(faces) == len(colors)


def test_theme_palette_and_template():
    import theme

    for c in (theme.ORANGE, theme.PLOT_A, theme.PLOT_B, theme.PLOT_ACCENT):
        assert c.startswith("#")
    assert len(theme.DIVERGING) == 3
    tmpl = theme.plotly_template()      # requires plotly; installed in CI
    assert tmpl.layout.paper_bgcolor == theme.WHITE


def test_data_sources_describe():
    import data_sources

    assert "cache" in data_sources.describe()
    assert data_sources.OBP_BASE.startswith("https://")
