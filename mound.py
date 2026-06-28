"""Polygonal dirt pitching-mound geometry derived from C3D foot markers.

The mound is built *from the data*: its center, radius, ground height, and the
downhill heading (toward home plate) are all estimated from where the pitcher's
foot markers go during the delivery, so the dirt sits naturally under the feet
for any pitch / capture orientation.

Geometry: a raised circular dirt platform with a flat plateau covering the
pitcher's footwork, a cosine-eased slope down to the surrounding field at the
rim, a slight crown, a mild forward downhill toward home plate, and a small
random mottle for a dirt-like surface. A white pitching rubber is placed under
the pivot foot's starting position.

Use :func:`add_mound` to draw it onto a Matplotlib 3D axes.
"""

from __future__ import annotations

import numpy as np

# Foot / lower-leg markers used to locate and size the mound.
_FOOT_LABELS = [
    "RTOE", "LTOE", "RHEE", "LHEE", "RANK", "LANK", "RMANK", "LMANK",
]

# Dirt colour ramp (low/shadowed slope -> high/sunlit plateau), linear RGB.
_DIRT_LOW = np.array([0.42, 0.27, 0.15])
_DIRT_HIGH = np.array([0.69, 0.48, 0.30])
_RUBBER_COLOR = "#f2f2ef"


def _present_feet(markers):
    return [lab for lab in _FOOT_LABELS if markers.has(lab)]


def _foot_points(markers, labels, frames=slice(None)):
    """Stack valid foot marker XYZ samples over ``frames`` into (k, 3)."""
    pts = []
    for lab in labels:
        traj = markers.marker(lab)[frames]          # (f, 3)
        pts.append(traj[np.isfinite(traj).all(axis=-1)])
    return np.concatenate(pts, axis=0) if pts else np.empty((0, 3))


def estimate_mound(markers, mound_height: float = 0.28, margin: float = 0.30):
    """Estimate mound placement parameters from the foot markers.

    Returns a dict with center, radii, ground heights, heading, and rubber pose.
    """
    labels = _present_feet(markers)
    cloud = _foot_points(markers, labels)
    if cloud.shape[0] < 4:
        # Fallback: use all markers' horizontal footprint.
        cloud = markers.points.reshape(-1, 3)
        cloud = cloud[np.isfinite(cloud).all(axis=-1)]

    center_xy = cloud[:, :2].mean(axis=0)
    foot_reach = np.linalg.norm(cloud[:, :2] - center_xy, axis=1).max()
    foot_z = float(np.percentile(cloud[:, 2], 1.0))

    # Heading: from early-stance foot position toward late (landing) position.
    n_frames = markers.n_frames
    early = _foot_points(markers, labels, slice(0, max(1, n_frames // 10)))
    late = _foot_points(markers, labels, slice(int(n_frames * 0.82), n_frames))
    start_xy = early[:, :2].mean(axis=0) if early.size else center_xy
    end_xy = late[:, :2].mean(axis=0) if late.size else center_xy + np.array([1.0, 0.0])
    heading = end_xy - start_xy
    norm = np.linalg.norm(heading)
    heading = heading / norm if norm > 1e-6 else np.array([1.0, 0.0])

    r_flat = float(foot_reach + margin)             # plateau covers the footwork
    r_outer = float(r_flat + 0.85)                  # slope ring out to the field
    table_z = foot_z - 0.02                         # dirt surface just below feet
    field_z = table_z - mound_height

    return {
        "center_xy": center_xy,
        "r_flat": r_flat,
        "r_outer": r_outer,
        "table_z": table_z,
        "field_z": field_z,
        "heading": heading,
        "rubber_xy": start_xy,                      # pivot foot pushes off here
    }


def _surface_height(dx, dy, params, rng):
    """Vectorised mound height at horizontal offsets (dx, dy) from center."""
    r = np.hypot(dx, dy)
    r_flat, r_outer = params["r_flat"], params["r_outer"]
    table_z, field_z = params["table_z"], params["field_z"]

    frac = np.clip((r - r_flat) / (r_outer - r_flat), 0.0, 1.0)
    ease = 0.5 - 0.5 * np.cos(np.pi * frac)         # smooth plateau -> rim
    z = table_z + (field_z - table_z) * ease

    # Mild forward downhill toward home plate (beyond the plateau only).
    fwd = dx * params["heading"][0] + dy * params["heading"][1]
    z -= 0.05 * np.clip(fwd - r_flat, 0.0, None)

    # Gentle crown + dirt mottle.
    z += 0.025 * np.clip(1.0 - r / r_outer, 0.0, 1.0)
    z += rng.normal(0.0, 0.006, size=z.shape)
    return np.maximum(z, field_z - 0.02)


def _mound_grid(markers, sectors: int, rings: int, mound_height: float, seed: int):
    """Build the (rings+1, sectors+1, 3) polar vertex grid and mound params."""
    rng = np.random.default_rng(seed)
    params = estimate_mound(markers, mound_height=mound_height)
    cx, cy = params["center_xy"]

    thetas = np.linspace(0.0, 2.0 * np.pi, sectors + 1)
    # Denser rings near the slope for a cleaner rim.
    radii = np.linspace(0.0, params["r_outer"], rings + 1) ** 1.15
    radii = radii / radii.max() * params["r_outer"]

    R, TH = np.meshgrid(radii, thetas, indexing="ij")
    DX, DY = R * np.cos(TH), R * np.sin(TH)
    Z = _surface_height(DX, DY, params, rng)
    X, Y = cx + DX, cy + DY
    grid = np.stack([X, Y, Z], axis=-1)
    return grid, params


def build_mound_mesh(markers, sectors: int = 56, rings: int = 16,
                     mound_height: float = 0.28, seed: int = 0):
    """Build the mound as polar-grid quad faces.

    Returns ``(faces, face_colors, params, all_vertices)`` where ``faces`` is a
    list of (4, 3) quads suitable for ``Poly3DCollection``.
    """
    rng = np.random.default_rng(seed + 1)
    grid, params = _mound_grid(markers, sectors, rings, mound_height, seed)

    faces, colors = [], []
    span = params["table_z"] - params["field_z"] or 1.0
    for i in range(rings):
        for j in range(sectors):
            quad = np.array([
                grid[i, j], grid[i, j + 1], grid[i + 1, j + 1], grid[i + 1, j],
            ])
            faces.append(quad)
            t = np.clip((quad[:, 2].mean() - params["field_z"]) / span, 0, 1)
            base = _DIRT_LOW + (_DIRT_HIGH - _DIRT_LOW) * t
            base = np.clip(base + rng.normal(0.0, 0.025, 3), 0.0, 1.0)
            colors.append(base)

    return faces, np.array(colors), params, grid.reshape(-1, 3)


def mound_trimesh(markers, sectors: int = 56, rings: int = 16,
                  mound_height: float = 0.28, seed: int = 0):
    """Triangulated mound for Plotly ``Mesh3d``.

    Returns ``(x, y, z, i, j, k, intensity, params)`` where ``i/j/k`` are
    triangle vertex indices and ``intensity`` is per-vertex height (for a dirt
    colorscale).
    """
    grid, params = _mound_grid(markers, sectors, rings, mound_height, seed)
    verts = grid.reshape(-1, 3)
    n_col = sectors + 1

    def vid(ri, sj):
        return ri * n_col + sj

    I, J, K = [], [], []
    for ri in range(rings):
        for sj in range(sectors):
            a, b = vid(ri, sj), vid(ri, sj + 1)
            c, d = vid(ri + 1, sj + 1), vid(ri + 1, sj)
            I += [a, a]
            J += [b, c]
            K += [c, d]
    return (verts[:, 0], verts[:, 1], verts[:, 2],
            np.array(I), np.array(J), np.array(K), verts[:, 2], params)


def _rubber_quad(params, half_len=0.305, half_wid=0.076):
    """Pitching-rubber polygon (regulation 24x6 in), perpendicular to heading."""
    hx, hy = params["heading"]
    along = np.array([-hy, hx])                     # perpendicular = rubber long axis
    across = np.array([hx, hy])
    c = np.array(params["rubber_xy"])
    z = params["table_z"] + 0.006
    corners = [
        c + along * half_len + across * half_wid,
        c + along * half_len - across * half_wid,
        c - along * half_len - across * half_wid,
        c - along * half_len + across * half_wid,
    ]
    return np.array([[p[0], p[1], z] for p in corners])


def add_mound(ax, markers, sectors: int = 56, rings: int = 16,
              mound_height: float = 0.28, seed: int = 0,
              show_rubber: bool = True):
    """Draw a polygonal dirt mound under the pitcher onto a 3D axes.

    Parameters
    ----------
    ax:
        A Matplotlib 3D axes.
    markers:
        A :class:`c3d_plot.C3DMarkers` instance.
    sectors, rings:
        Polar-grid resolution of the mound surface.
    mound_height:
        Height of the plateau above the surrounding field (metres).
    seed:
        Seed for the dirt mottle/texture.
    show_rubber:
        Whether to draw the white pitching rubber.

    Returns
    -------
    np.ndarray
        ``(n_vertices, 3)`` mound vertices, so callers can expand axis limits.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces, colors, params, verts = build_mound_mesh(
        markers, sectors=sectors, rings=rings, mound_height=mound_height, seed=seed
    )
    coll = Poly3DCollection(
        faces, facecolors=colors, edgecolors="none", linewidths=0,
        zsort="average",
    )
    coll.set_zorder(0)
    ax.add_collection3d(coll)

    if show_rubber:
        rubber = _rubber_quad(params)
        rub = Poly3DCollection(
            [rubber], facecolors=_RUBBER_COLOR, edgecolors="0.5", linewidths=0.5,
        )
        rub.set_zorder(1)
        ax.add_collection3d(rub)
        verts = np.vstack([verts, rubber])

    return verts
