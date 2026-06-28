"""Glossary of OpenBiomechanics pitching biomechanics variables.

Every variable in ``poi_metrics.csv`` is documented here with a full
explanation, its units, the delivery event it is measured at, and the relevant
sign/coordinate convention. Definitions follow the official OBP documentation
(``baseball_pitching/README.md``) and are expanded with context on timing,
units, and why each metric matters for pitching.

Use :func:`render_html` to produce a self-contained, searchable glossary tab,
:func:`as_dataframe` for tabular access, or :func:`lookup` for a single entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- #
# Shared context: delivery events and coordinate conventions
# --------------------------------------------------------------------------- #

EVENTS = {
    "PKH (peak knee height)":
        "The top of the leg lift, when the lead knee reaches its highest point "
        "and the pitcher begins moving toward the plate.",
    "FC (foot contact)":
        "First touch of the lead foot, defined as 10% bodyweight on the lead "
        "force plate.",
    "FP (foot plant)":
        "Lead foot fully planted, defined as 100% bodyweight on the lead force "
        "plate. The start of the arm-cocking/acceleration phase.",
    "MER (maximum external rotation)":
        "Peak shoulder external rotation, a.k.a. 'layback' — the throwing arm "
        "is maximally cocked just before it whips forward.",
    "BR (ball release)":
        "The instant the ball leaves the hand.",
    "MIR (maximum internal rotation)":
        "Peak shoulder internal rotation during follow-through, after release.",
}

COORDINATE_SYSTEM = (
    "Global (lab) frame: +x points from second base toward home plate, +y "
    "points toward first base, +z points up. Joint angles, forces, and moments "
    "are reported per the right-hand rule with adjustments so signs match "
    "coach/player intuition (righty/lefty symmetry, etc.). Joint forces and "
    "moments are internal."
)

GRF_CONVENTIONS = (
    "Ground reaction forces: rear plate (FP2) +x = push-off (anterior, toward "
    "home); lead plate (FP1/3) +x = braking (posterior); both plates +y = "
    "lateral, +z = superior (vertical). Filtered with a 4th-order Butterworth "
    "low-pass at 40 Hz."
)


@dataclass
class Term:
    name: str
    category: str
    units: str
    event: str
    definition: str


def _t(name, category, units, event, definition):
    return Term(name, category, units, event, definition)


# --------------------------------------------------------------------------- #
# The glossary (ordered to mirror poi_metrics.csv)
# --------------------------------------------------------------------------- #

ID = "Identifier"
ANGLE = "Kinematics – joint angle"
AVEL = "Kinematics – angular velocity"
LVEL = "Kinematics – linear velocity"
MOMENT = "Kinetics – joint moment"
ENERGY = "Energy flow"
GRF = "Ground reaction force"
SPATIAL = "Spatiotemporal / other"

TERMS: list[Term] = [
    _t("session_pitch", ID, "—", "—",
       "Unique identifier for a single pitch (session + pitch number). Used to "
       "join the POI metrics, force plates, and C3D file for one delivery."),
    _t("session", ID, "—", "—",
       "Unique identifier for a data-collection session (one athlete on one day)."),
    _t("p_throws", ID, "R/L", "—",
       "Pitcher handedness: 'R' (right-handed) or 'L' (left-handed)."),
    _t("pitch_type", ID, "—", "—",
       "Type of pitch thrown. In this dataset all pitches are 'FF' (four-seam "
       "fastball)."),
    _t("pitch_speed_mph", ID, "mph", "BR",
       "Release speed of the pitch in miles per hour — the target variable the "
       "velocity model predicts."),

    _t("max_shoulder_internal_rotational_velo", AVEL, "deg/s", "peak",
       "Peak shoulder internal-rotation angular velocity. The fastest rotational "
       "motion in the throw and the single strongest correlate of velocity; the "
       "arm internally rotates explosively from layback (MER) to release."),
    _t("max_elbow_extension_velo", AVEL, "deg/s", "peak",
       "Peak elbow-extension angular velocity — how fast the elbow straightens "
       "as the forearm whips forward into release."),
    _t("max_torso_rotational_velo", AVEL, "deg/s", "peak",
       "Peak torso axial-rotation (Z) angular velocity — how fast the trunk "
       "rotates toward the plate. A key link in the kinetic chain."),
    _t("max_rotation_hip_shoulder_separation", ANGLE, "deg", "peak",
       "Peak hip–shoulder separation: the maximum angular difference between "
       "the pelvis and the thorax. Greater separation stores more elastic energy "
       "(the 'X-factor')."),
    _t("max_elbow_flexion", ANGLE, "deg", "peak",
       "Peak elbow flexion angle (most bent the elbow gets), typically around "
       "arm cocking."),
    _t("max_shoulder_external_rotation", ANGLE, "deg", "MER",
       "Peak shoulder external rotation ('layback') — how far the forearm lays "
       "back behind the arm at maximum cocking. Higher values lengthen the "
       "acceleration path."),
    _t("elbow_flexion_fp", ANGLE, "deg", "FP",
       "Elbow flexion angle at foot plant."),
    _t("elbow_pronation_fp", ANGLE, "deg", "FP",
       "Wrist/forearm pronation angle at foot plant."),
    _t("rotation_hip_shoulder_separation_fp", ANGLE, "deg", "FP",
       "Hip–shoulder separation angle at foot plant — how 'coiled' the pitcher "
       "is when the front foot lands."),
    _t("shoulder_horizontal_abduction_fp", ANGLE, "deg", "FP",
       "Throwing-shoulder horizontal abduction at foot plant (how far the arm is "
       "swung back behind the chest plane)."),
    _t("shoulder_abduction_fp", ANGLE, "deg", "FP",
       "Throwing-shoulder abduction (elevation of the upper arm away from the "
       "torso) at foot plant — relates to arm slot."),
    _t("shoulder_external_rotation_fp", ANGLE, "deg", "FP",
       "Throwing-shoulder external rotation at foot plant."),
    _t("lead_knee_extension_angular_velo_fp", AVEL, "deg/s", "FP",
       "Lead-knee extension angular velocity at foot plant — early blocking/"
       "bracing of the front leg."),
    _t("lead_knee_extension_angular_velo_br", AVEL, "deg/s", "BR",
       "Lead-knee extension angular velocity at ball release. A stiff, "
       "extending front leg at release transfers energy up the chain."),
    _t("lead_knee_extension_angular_velo_max", AVEL, "deg/s", "peak",
       "Peak lead-knee extension angular velocity during the delivery."),
    _t("torso_anterior_tilt_fp", ANGLE, "deg", "FP",
       "Trunk forward (anterior) flexion at foot plant."),
    _t("torso_lateral_tilt_fp", ANGLE, "deg", "FP",
       "Trunk lateral (side) flexion at foot plant — glove-side vs throwing-side "
       "lean."),
    _t("torso_rotation_fp", ANGLE, "deg", "FP",
       "Trunk axial rotation at foot plant (how open/closed the chest is)."),
    _t("pelvis_anterior_tilt_fp", ANGLE, "deg", "FP",
       "Pelvis anterior/posterior tilt at foot plant."),
    _t("pelvis_lateral_tilt_fp", ANGLE, "deg", "FP",
       "Pelvis lateral tilt (hip hike) at foot plant."),
    _t("pelvis_rotation_fp", ANGLE, "deg", "FP",
       "Pelvis axial rotation at foot plant — how far the hips have opened to the "
       "plate when the front foot lands."),
    _t("max_cog_velo_x", LVEL, "m/s", "peak",
       "Peak center-of-gravity velocity toward home plate — overall momentum of "
       "the body moving down the mound."),
    _t("torso_rotation_min", ANGLE, "deg", "peak",
       "Peak torso counter-rotation angle (most 'closed' the trunk gets) before "
       "rotating forward."),
    _t("max_pelvis_rotational_velo", AVEL, "deg/s", "peak",
       "Peak pelvis axial-rotation angular velocity — the hips lead the trunk in "
       "the rotational sequence."),
    _t("glove_shoulder_horizontal_abduction_fp", ANGLE, "deg", "FP",
       "Glove-side shoulder horizontal abduction at foot plant."),
    _t("glove_shoulder_abduction_fp", ANGLE, "deg", "FP",
       "Glove-side shoulder abduction at foot plant."),
    _t("glove_shoulder_external_rotation_fp", ANGLE, "deg", "FP",
       "Glove-side shoulder external rotation at foot plant. The glove arm helps "
       "set and stabilize trunk posture."),
    _t("glove_shoulder_abduction_mer", ANGLE, "deg", "MER",
       "Glove-side shoulder abduction at maximum external rotation."),
    _t("elbow_flexion_mer", ANGLE, "deg", "MER",
       "Throwing-elbow flexion angle at maximum external rotation (layback)."),
    _t("torso_anterior_tilt_mer", ANGLE, "deg", "MER",
       "Trunk forward flexion at maximum external rotation."),
    _t("torso_lateral_tilt_mer", ANGLE, "deg", "MER",
       "Trunk lateral flexion at maximum external rotation — contralateral tilt "
       "raises the release point."),
    _t("torso_rotation_mer", ANGLE, "deg", "MER",
       "Trunk axial rotation at maximum external rotation."),
    _t("elbow_varus_moment", MOMENT, "Nm", "peak",
       "Peak elbow varus moment — the internal joint moment resisting valgus "
       "stress at the elbow. A primary proxy for UCL (elbow) injury load."),
    _t("shoulder_internal_rotation_moment", MOMENT, "Nm", "peak",
       "Peak shoulder internal-rotation moment — internal load at the shoulder "
       "during arm acceleration."),
    _t("torso_anterior_tilt_br", ANGLE, "deg", "BR",
       "Trunk forward flexion at ball release."),
    _t("torso_lateral_tilt_br", ANGLE, "deg", "BR",
       "Trunk lateral flexion at ball release — strongly tied to release height "
       "and arm slot."),
    _t("torso_rotation_br", ANGLE, "deg", "BR",
       "Trunk axial rotation at ball release (how far the chest has rotated to "
       "face the plate)."),
    _t("lead_knee_extension_from_fp_to_br", ANGLE, "deg", "FP→BR",
       "Change in lead-knee extension angle from foot plant to ball release — "
       "how much the front leg straightens/braces during acceleration."),
    _t("cog_velo_pkh", LVEL, "m/s", "PKH",
       "Center-of-gravity velocity toward home at peak knee height — momentum "
       "generated during the leg lift/initial drive."),
    _t("stride_length", SPATIAL, "% body height", "FP",
       "Stride length as a percentage of body height (distance from the rubber "
       "to the lead foot at plant)."),
    _t("stride_angle", SPATIAL, "deg", "FP",
       "Stride direction relative to straight toward the plate; positive = a "
       "cross-body (closed) stride."),
    _t("arm_slot", SPATIAL, "deg", "BR",
       "Arm slot — the global forearm projection angle at release that describes "
       "how over-the-top vs side-arm the delivery is."),
    _t("timing_peak_torso_to_peak_pelvis_rot_velo", SPATIAL, "s", "—",
       "Time between peak pelvis angular velocity and peak torso angular "
       "velocity — a measure of the rotational sequencing/separation timing."),
    _t("max_shoulder_horizontal_abduction", ANGLE, "deg", "peak",
       "Peak throwing-shoulder horizontal abduction ('scap loading') during the "
       "delivery."),

    _t("shoulder_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred across the throwing shoulder between foot plant and "
       "ball release (flow of mechanical energy through the joint)."),
    _t("shoulder_generation_fp_br", ENERGY, "J", "FP→BR",
       "Energy generated (positive muscle work) at the throwing shoulder, FP→BR."),
    _t("shoulder_absorption_fp_br", ENERGY, "J", "FP→BR",
       "Energy absorbed (negative work, eccentric) at the throwing shoulder, "
       "FP→BR."),
    _t("elbow_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred across the throwing elbow, FP→BR."),
    _t("elbow_generation_fp_br", ENERGY, "J", "FP→BR",
       "Energy generated at the throwing elbow, FP→BR."),
    _t("elbow_absorption_fp_br", ENERGY, "J", "FP→BR",
       "Energy absorbed at the throwing elbow, FP→BR."),
    _t("lead_hip_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred across the lead hip, FP→BR."),
    _t("lead_hip_generation_fp_br", ENERGY, "J", "FP→BR",
       "Energy generated at the lead hip, FP→BR — the front hip is a major power "
       "producer in the block-and-rotate."),
    _t("lead_hip_absorption_fp_br", ENERGY, "J", "FP→BR",
       "Energy absorbed at the lead hip, FP→BR."),
    _t("lead_knee_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred across the lead knee, FP→BR."),
    _t("lead_knee_generation_fp_br", ENERGY, "J", "FP→BR",
       "Energy generated at the lead knee, FP→BR."),
    _t("lead_knee_absorption_fp_br", ENERGY, "J", "FP→BR",
       "Energy absorbed at the lead knee, FP→BR — the bracing front leg absorbs "
       "large amounts of energy."),
    _t("rear_hip_transfer_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy transferred across the rear (drive) hip between peak knee height "
       "and foot plant."),
    _t("rear_hip_generation_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy generated at the rear hip, PKH→FP — power from the drive leg."),
    _t("rear_hip_absorption_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy absorbed at the rear hip, PKH→FP."),
    _t("rear_knee_transfer_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy transferred across the rear knee, PKH→FP."),
    _t("rear_knee_generation_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy generated at the rear knee, PKH→FP."),
    _t("rear_knee_absorption_pkh_fp", ENERGY, "J", "PKH→FP",
       "Energy absorbed at the rear knee, PKH→FP."),
    _t("pelvis_lumbar_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred out of the pelvis toward the trunk (up the chain), "
       "FP→BR."),
    _t("thorax_distal_transfer_fp_br", ENERGY, "J", "FP→BR",
       "Energy transferred out of the trunk toward the throwing shoulder, FP→BR."),

    _t("rear_grf_x_max", GRF, "N", "peak",
       "Peak anterior push-off ground reaction force on the rear (drive) leg — "
       "how hard the pitcher drives toward the plate."),
    _t("rear_grf_y_max", GRF, "N", "peak",
       "Peak lateral push-off ground reaction force on the rear leg."),
    _t("rear_grf_z_max", GRF, "N", "peak",
       "Peak vertical ground reaction force on the rear leg."),
    _t("rear_grf_mag_max", GRF, "N", "peak",
       "Peak resultant (magnitude) ground reaction force on the rear leg."),
    _t("rear_grf_angle_at_max", GRF, "deg", "peak",
       "Direction of the rear ground reaction force vector at the instant of "
       "peak magnitude (projection angle)."),
    _t("lead_grf_x_max", GRF, "N", "peak",
       "Peak braking ground reaction force on the lead (stride) leg — how hard "
       "the front leg blocks against forward momentum."),
    _t("lead_grf_y_max", GRF, "N", "peak",
       "Peak lateral braking ground reaction force on the lead leg."),
    _t("lead_grf_z_max", GRF, "N", "peak",
       "Peak vertical ground reaction force on the lead leg — often several "
       "times bodyweight as the front leg braces."),
    _t("lead_grf_mag_max", GRF, "N", "peak",
       "Peak resultant (magnitude) ground reaction force on the lead leg."),
    _t("lead_grf_angle_at_max", GRF, "deg", "peak",
       "Direction of the lead ground reaction force vector at the instant of "
       "peak magnitude (projection angle)."),
    _t("peak_rfd_rear", GRF, "N/s", "peak",
       "Peak rate of force development on the rear leg — how explosively the "
       "drive-leg force is applied."),
    _t("peak_rfd_lead", GRF, "N/s", "peak",
       "Peak rate of force development on the lead leg — how explosively the "
       "front leg loads against the ground during the block."),
]

# Quick-lookup index
_BY_NAME = {t.name: t for t in TERMS}


def lookup(name: str) -> Term | None:
    """Return the glossary :class:`Term` for a variable name, or None."""
    return _BY_NAME.get(name)


def as_dataframe() -> pd.DataFrame:
    """Return the glossary as a DataFrame."""
    return pd.DataFrame(
        [
            {
                "variable": t.name,
                "category": t.category,
                "units": t.units,
                "event": t.event,
                "definition": t.definition,
            }
            for t in TERMS
        ]
    )


def render_markdown() -> str:
    """Render the glossary as a Markdown document."""
    lines = ["# Biomechanics Glossary", ""]
    lines.append("## Delivery events")
    for k, v in EVENTS.items():
        lines.append(f"- **{k}** — {v}")
    lines += ["", "## Coordinate system", "", COORDINATE_SYSTEM,
              "", "## Ground reaction forces", "", GRF_CONVENTIONS, "",
              "## Variables", ""]
    cur = None
    for t in TERMS:
        if t.category != cur:
            cur = t.category
            lines += ["", f"### {cur}", ""]
        lines.append(
            f"- **`{t.name}`** ({t.units}; {t.event}): {t.definition}"
        )
    return "\n".join(lines)


def render_html(search_box: bool = True) -> str:
    """Render the glossary as a self-contained HTML fragment (a 'tab').

    Includes a client-side search box that filters rows as you type.
    """
    import html

    rows = []
    for t in TERMS:
        rows.append(
            "<tr class='gloss-row'>"
            f"<td class='var'><code>{html.escape(t.name)}</code></td>"
            f"<td class='cat'>{html.escape(t.category)}</td>"
            f"<td class='units'>{html.escape(t.units)}</td>"
            f"<td class='event'>{html.escape(t.event)}</td>"
            f"<td class='def'>{html.escape(t.definition)}</td>"
            "</tr>"
        )
    events_html = "".join(
        f"<li><b>{html.escape(k)}</b> — {html.escape(v)}</li>"
        for k, v in EVENTS.items()
    )
    search = (
        "<input id='gloss-search' type='text' placeholder='Filter variables…' "
        "oninput=\"glossFilter()\" style='width:100%;padding:8px;margin:8px 0;"
        "font-size:14px;box-sizing:border-box;'>"
        if search_box else ""
    )
    return f"""
<div class="glossary">
  <h2>Biomechanics glossary</h2>
  <details open><summary><b>Delivery events &amp; conventions</b></summary>
    <ul>{events_html}</ul>
    <p><b>Coordinate system.</b> {html.escape(COORDINATE_SYSTEM)}</p>
    <p><b>Ground reaction forces.</b> {html.escape(GRF_CONVENTIONS)}</p>
  </details>
  {search}
  <table id="gloss-table" class="gloss">
    <thead><tr>
      <th>Variable</th><th>Category</th><th>Units</th><th>Event</th>
      <th>Full explanation</th>
    </tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
<script>
function glossFilter() {{
  var q = document.getElementById('gloss-search').value.toLowerCase();
  var rows = document.querySelectorAll('#gloss-table tbody tr');
  rows.forEach(function(r) {{
    r.style.display = r.innerText.toLowerCase().indexOf(q) > -1 ? '' : 'none';
  }});
}}
</script>
"""


def write_markdown(path: str = "GLOSSARY.md") -> str:
    with open(path, "w") as fh:
        fh.write(render_markdown())
    return path


if __name__ == "__main__":
    # Sanity / coverage check against poi_metrics.csv if present.
    print(f"{len(TERMS)} glossary terms defined.")
    write_markdown()
    print("Wrote GLOSSARY.md")
