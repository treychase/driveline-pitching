# Driveline Pitching — 3D C3D Plotter

A small, dependency-light Python module for **loading and plotting C3D
motion-capture files in 3D**, built for the
[Driveline OpenBiomechanics Project (OBP)](https://github.com/drivelineresearch/openbiomechanics)
baseball pitching dataset.

It reads the marker (point) trajectories from a `.c3d` file and renders them as
a 3D stick figure — either a single static frame or an animation across the
whole pitch — using the standard Vicon Plug-in-Gait marker set that the OBP
pitching data uses.

![Example frame](examples/example_frame.png)

## Install

```bash
pip install -r requirements.txt
```

`ezc3d` is the preferred C3D reader. If it is not installed, the module
automatically falls back to the pure-python `c3d` package.

## Quick start

```python
import c3d_plot

# 1. (optional) Pull a C3D straight from the OpenBiomechanics repo
path = c3d_plot.download_obp_c3d(
    session="000002",
    filename="000002_003034_73_207_002_FF_809.c3d",
)

# 2. Plot a single frame as a 3D stick figure
c3d_plot.plot_c3d(path, frame=350, save_path="frame.png")

# 3. Animate the whole pitch to a gif (or .mp4 if ffmpeg is available)
c3d_plot.animate_c3d(path, step=2, save_path="pitch.gif")
```

The animation renders at real time by default (the capture rate, 360 Hz,
divided by `step`).

![Example animation](examples/example_pitch.gif)

## Command line

```bash
# Static frame -> PNG
python c3d_plot.py path/to/file.c3d --frame 350 --out frame.png

# Animation -> GIF
python c3d_plot.py path/to/file.c3d --animate --step 2 --out pitch.gif

# Markers only (no skeleton), with marker-name labels
python c3d_plot.py path/to/file.c3d --no-skeleton --labels --out markers.png

# Add a polygonal dirt pitching mound under the pitcher
python c3d_plot.py path/to/file.c3d --animate --mound --out pitch_on_mound.gif
```

### Dirt pitching mound

Passing `mound=True` to `plot_c3d`/`animate_c3d` (or `--mound` on the CLI, and
on by default in the dashboard) draws a **polygonal dirt mound** under the
pitcher. The mound is derived entirely from the C3D foot markers
(`mound.py`): the center, radius, ground level, and the downhill heading toward
home plate are estimated from where the feet travel during the delivery, so the
dirt sits naturally under the pitcher for any pitch or capture orientation. It
is rendered as a polar grid of shaded dirt-colored quads — a flat plateau over
the footwork, a cosine-eased slope to the surrounding field, a slight crown, and
a white pitching rubber under the pivot foot.

## API

| Function | Purpose |
| --- | --- |
| `load_c3d(path)` | Load a `.c3d` into a `C3DMarkers` object (`points`, `labels`, `rate`, `units`). |
| `plot_c3d(source, frame=…, …)` | Render one frame as a 3D scatter + skeleton. |
| `animate_c3d(source, …)` | Animate a frame range; save to `.gif`/`.mp4` or return the animation. |
| `download_obp_c3d(session, filename)` | Fetch a single C3D from the OpenBiomechanics repo. |

`source` may be a file path or an already-loaded `C3DMarkers` instance.

### `C3DMarkers`

```python
mk = c3d_plot.load_c3d("file.c3d")
mk.n_frames          # number of frames
mk.n_markers         # number of markers
mk.rate              # sampling rate (Hz)
mk.units             # coordinate units, e.g. "m"
mk.labels            # list of marker names
mk.marker("RWRA")    # (n_frames, 3) trajectory for one marker
mk.points            # (n_frames, n_markers, 3) array; gaps are NaN
```

## Notes

- Coordinates are plotted with **Z up**, matching the OBP convention, and the
  axes use a 1:1:1 aspect so proportions are preserved.
- Marker gaps (stored as `(0,0,0)` in some files) and non-finite samples are
  converted to `NaN` and skipped when drawing.
- The skeleton is defined by `PLUG_IN_GAIT_SEGMENTS`. Segments are only drawn
  when both endpoint markers are present, so files with a reduced marker set
  still render. Pass `segments=None` to plot markers only, or supply your own
  list of `(label_a, label_b)` pairs.

## Velocity prediction + biomechanics dashboard

Beyond plotting the raw motion, the repo can **predict pitch release velocity
from the biomechanics** and present everything as a single animated dashboard:

1. the 3D pose animation,
2. the **actual vs. predicted release velocity** (with a posterior credible
   interval), and
3. **z-scores of every biomechanics metric**, colour-coded blue→red
   (below→above the dataset average).

![Dashboard](examples/dashboard.gif)

```bash
# Train the model, fetch the pitch's C3D, and render the dashboard
python dashboard.py --out dashboard.gif

# Pick a specific pitch and show more z-score bars
python dashboard.py --pitch 1097_1 --top-n 24 --out dashboard.mp4
```

```python
from dashboard import build_dashboard
anim, info = build_dashboard(session_pitch="1097_1", save_path="dashboard.gif")
print(info)   # predicted/actual mph, error, out-of-sample flag, test R²/RMSE
```

### The model: a Bayesian Lasso with Gaussian priors

`bayesian_lasso.py` implements the Bayesian Lasso (Park & Casella, 2008) from
scratch in NumPy via Gibbs sampling. It uses the **scale-mixture-of-Gaussians**
representation of the Laplace prior: each coefficient gets a *Gaussian prior*
conditional on its own variance,

```
beta_j | sigma^2, tau_j^2  ~  Normal(0, sigma^2 * tau_j^2)
tau_j^2                    ~  Exponential(lambda^2 / 2)
```

Marginalising over `tau_j^2` recovers the double-exponential (Lasso) prior that
shrinks weak predictors toward zero, while every Gibbs full-conditional stays a
clean Gaussian / inverse-Gaussian / gamma draw. A Gamma hyper-prior on
`lambda^2` lets the data choose the shrinkage strength.

Trained on the 76 OpenBiomechanics POI metrics to predict `pitch_speed_mph`, it
reaches roughly **R² ≈ 0.8, RMSE ≈ 2 mph** out-of-sample.

```python
from bayesian_lasso import BayesianLasso
from velocity_model import load_dataset, train_velocity_model

trained = train_velocity_model()          # downloads POI data, fits, holds out a test set
print(trained.metrics)                     # {'r2': ..., 'rmse': ..., ...}
mean, std = trained.predict_pitch("1097_1")

# Inspect which biomechanics drive the prediction (posterior mean + 95% CI)
for row in trained.model.coef_summary(trained.dataset.feature_names)[:10]:
    print(row["feature"], round(row["mean"], 3), row["nonzero"])
```

`velocity_model.py` also computes per-pitch **z-scores** for every metric
(`dataset.zscores(session_pitch)`) and links a pitch to its raw C3D file through
`metadata.csv`.

> Note: POI metrics are end-of-pitch summary values, so the velocity prediction
> and z-scores describe the whole delivery; the pose panel animates the motion
> they summarise. Pitchers are a mix of left- and right-handed; the model uses
> biomechanics magnitudes and is not mirrored by handedness.

## Interactive dashboard with tabs

`dashboard_html.py` builds a **self-contained interactive HTML dashboard**. The
**shell** (header, tabs, pitch picker, buttons) uses an orange theme; the
**plots stay neutral** so colour carries data, not branding (`theme.py`). A
**pitch picker** (applies to the Live delivery and Joint work tabs) selects the
delivery. There are four tabs:

* **Live delivery** — a Plotly animation of the 3D pose on the dirt mound with a
  Play button, a frame slider, and a **toggle for joint velocity vectors** on
  every joint. Scrubbing drives:
  * a synchronized **live ground-reaction-force** plot of the lead vs. rear leg
    (by handedness), with the delivery **phases shaded** (wind-up, stride, arm
    cocking, acceleration, deceleration) and a moving cursor;
  * two **foot-vGRF squares** (L/R) that show each foot's force every frame and
    **flash red ("◀ MAX") at that foot's peak**;
  * the velocity gauge and the biomechanics z-scores.
* **Joint work** — joint work accumulated during the delivery and the
  **z-scores of joint work** (energy generated) for the selected pitch.
* **Model diagnostics** — Bayesian-Lasso diagnostics: predicted vs. actual
  (train/test, selected pitches highlighted), residuals, the **posterior
  coefficients with 95% credible intervals**, and the posterior distributions of
  the noise σ and shrinkage λ².
* **Glossary** — a searchable table with **full explanations of every
  biomechanics variable**.

![Live delivery tab](examples/dashboard_html_live.png)
![Model diagnostics tab](examples/dashboard_html_diagnostics.png)
![Glossary tab](examples/dashboard_html_glossary.png)

```bash
# Build with the default pitch picker (Plotly from CDN; needs internet to view)
python dashboard_html.py --out dashboard.html

# Choose the pitches in the picker
python dashboard_html.py --pitches 1097_1,1031_2,1031_3 --out dashboard.html

# Fully self-contained / offline (embeds Plotly.js, larger file)
python dashboard_html.py --offline --out dashboard.html
```

The animated **matplotlib** dashboard (`dashboard.py`) carries the same neutral
plot styling and the same live force panel: phase-shaded lead/rear GRF traces
with a moving cursor plus the two color-changing L/R foot squares next to the
pose animation.

### Joint work, force plates, and the glossary as modules

```python
import force_plate, glossary, joint_kinetics

# Per-pitch ground reaction forces + delivery phases, aligned to the C3D frames
trace = force_plate.load_force_plate("1097_1", bodyweight_n=76 * 9.81)
aligned = trace.align_to_frames(frame_times)   # rear/lead vertical & magnitude per frame
force_plate.delivery_phases(trace.events, t_end=1.2)   # wind-up → deceleration spans

# Joint work (net energy generated, J), z-scores, and joint velocity vectors
jw = joint_kinetics.load_joint_work("1097_1")          # cumulative work per joint
jw.work_at_release()                                    # {'shoulder': 980, 'elbow': -742, ...}
joint_kinetics.work_zscores("1097_1")                   # z-scores vs the dataset
joint_kinetics.load_joint_vectors("1097_1", frame_times)  # joint-centre velocity vectors

# Glossary of every variable (definition, units, event, category)
glossary.lookup("elbow_varus_moment").definition
glossary.as_dataframe()                                 # all 81 variables as a table
```

Force plates (`rear_force_*`, `lead_force_*`, ~1080 Hz) come from
`full_sig/force_plate.zip`; joint work uses OBP's authoritative
`<joint>_energy_generated` from `full_sig/energy_flow.zip`; joint-centre velocity
vectors are finite-differenced from `full_sig/landmarks.zip`. All share the C3D
clock, so they align to the pose by time interpolation. A full text glossary is
also rendered to [`GLOSSARY.md`](GLOSSARY.md).

## Data & license

The C3D files belong to the OpenBiomechanics Project and are licensed
CC BY-NC-SA 4.0 (non-commercial). This repository contains only plotting code
plus small rendered demo assets; it does not redistribute the dataset. See the
[OBP repository](https://github.com/drivelineresearch/openbiomechanics) for the
data and its terms.
