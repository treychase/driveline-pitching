"""Offline smoke tests for the OBP pitching dashboard (no network needed)."""

import numpy as np


def test_all_modules_import():
    import bayesian_lasso, c3d_plot, dashboard, dashboard_html  # noqa: F401
    import data_sources, efficiency, force_plate, glossary  # noqa: F401
    import joint_kinetics, mound, theme, velocity_model  # noqa: F401


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


def _synthetic_poi(n=60, seed=0):
    import pandas as pd

    import efficiency

    rng = np.random.default_rng(seed)
    cols = (efficiency.LOWER_BODY_DRIVERS + efficiency.TORSO_DRIVERS
            + efficiency.ELBOW_LOAD)
    data = {c: rng.normal(100, 20, n) for c in cols}
    data["session_pitch"] = [f"p{i}" for i in range(n)]
    return pd.DataFrame(data)


def test_efficiency_model_scores_and_ranks():
    import efficiency

    poi = _synthetic_poi()
    model = efficiency.MechanicalEfficiencyModel.fit(poi)

    res = model.score("p0")
    assert 0.0 <= res.score <= 100.0
    # signed_z negates the elbow-load term so "+ helps efficiency" holds.
    load_row = res.contributions[res.contributions.group == "elbow_load"].iloc[0]
    assert load_row["signed_z"] == -load_row["z"]
    # raw index equals torso/lower-body drive minus elbow load.
    assert abs(res.raw - (res.drive - res.elbow_load)) < 1e-9

    summary = model.summary()
    assert len(summary) == len(poi)
    assert summary["score"].between(0, 100).all()


def test_efficiency_rewards_drive_and_penalizes_elbow():
    import pandas as pd

    import efficiency

    poi = _synthetic_poi()
    # Craft two contrasting pitches: an efficient one (high drive, low elbow
    # load) and an arm-reliant one (low drive, high elbow load).
    hi_drive = {c: 200.0 for c in efficiency.LOWER_BODY_DRIVERS
                + efficiency.TORSO_DRIVERS}
    hi_drive[efficiency.ELBOW_LOAD[0]] = 40.0
    hi_drive["session_pitch"] = "efficient"
    lo_drive = {c: 40.0 for c in efficiency.LOWER_BODY_DRIVERS
                + efficiency.TORSO_DRIVERS}
    lo_drive[efficiency.ELBOW_LOAD[0]] = 200.0
    lo_drive["session_pitch"] = "arm_reliant"
    poi = pd.concat([poi, pd.DataFrame([hi_drive, lo_drive])], ignore_index=True)

    model = efficiency.MechanicalEfficiencyModel.fit(poi)
    assert model.score("efficient").score > model.score("arm_reliant").score


def test_jointwork_body_figure_is_work_colored_without_stick_figure():
    import pandas as pd

    import dashboard_html as dh
    import theme

    rng = np.random.default_rng(1)
    joints = ["shoulder", "elbow", "wrist", "hand", "glove_shoulder",
              "glove_elbow", "glove_wrist", "glove_hand", "lead_hip", "lead_knee",
              "lead_ankle", "rear_hip", "rear_knee", "rear_ankle"]
    positions = {j: rng.normal(0, 0.5, 3) + np.array([0, 0, 1.0])
                 for j in joints}
    zwork = pd.DataFrame({
        "joint": ["shoulder", "elbow", "lead_hip", "lead_knee", "rear_hip",
                  "rear_knee", "glove_shoulder", "glove_elbow"],
        "work_J": rng.normal(500, 200, 8),
        "z": rng.normal(0, 1, 8),
    })

    fig = dh.build_jointwork_body_figure(positions, zwork, "R")
    assert fig is not None and len(fig.data) > 0
    # The work colour axis carries the figure (limbs/joints shaded by work z)...
    assert fig.layout.coloraxis.colorscale is not None
    # ...and no full C3D stick-figure skeleton (theme.SLATE lines) is overlaid.
    slate_lines = [t for t in fig.data
                   if getattr(getattr(t, "line", None), "color", None) == theme.SLATE]
    assert slate_lines == []
