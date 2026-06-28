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
```

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

## Data & license

The C3D files belong to the OpenBiomechanics Project and are licensed
CC BY-NC-SA 4.0 (non-commercial). This repository contains only plotting code
plus small rendered demo assets; it does not redistribute the dataset. See the
[OBP repository](https://github.com/drivelineresearch/openbiomechanics) for the
data and its terms.
