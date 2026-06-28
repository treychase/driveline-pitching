# Biomechanics Glossary

## Delivery events
- **PKH (peak knee height)** — The top of the leg lift, when the lead knee reaches its highest point and the pitcher begins moving toward the plate.
- **FC (foot contact)** — First touch of the lead foot, defined as 10% bodyweight on the lead force plate.
- **FP (foot plant)** — Lead foot fully planted, defined as 100% bodyweight on the lead force plate. The start of the arm-cocking/acceleration phase.
- **MER (maximum external rotation)** — Peak shoulder external rotation, a.k.a. 'layback' — the throwing arm is maximally cocked just before it whips forward.
- **BR (ball release)** — The instant the ball leaves the hand.
- **MIR (maximum internal rotation)** — Peak shoulder internal rotation during follow-through, after release.

## Coordinate system

Global (lab) frame: +x points from second base toward home plate, +y points toward first base, +z points up. Joint angles, forces, and moments are reported per the right-hand rule with adjustments so signs match coach/player intuition (righty/lefty symmetry, etc.). Joint forces and moments are internal.

## Ground reaction forces

Ground reaction forces: rear plate (FP2) +x = push-off (anterior, toward home); lead plate (FP1/3) +x = braking (posterior); both plates +y = lateral, +z = superior (vertical). Filtered with a 4th-order Butterworth low-pass at 40 Hz.

## Variables


### Identifier

- **`session_pitch`** (—; —): Unique identifier for a single pitch (session + pitch number). Used to join the POI metrics, force plates, and C3D file for one delivery.
- **`session`** (—; —): Unique identifier for a data-collection session (one athlete on one day).
- **`p_throws`** (R/L; —): Pitcher handedness: 'R' (right-handed) or 'L' (left-handed).
- **`pitch_type`** (—; —): Type of pitch thrown. In this dataset all pitches are 'FF' (four-seam fastball).
- **`pitch_speed_mph`** (mph; BR): Release speed of the pitch in miles per hour — the target variable the velocity model predicts.

### Kinematics – angular velocity

- **`max_shoulder_internal_rotational_velo`** (deg/s; peak): Peak shoulder internal-rotation angular velocity. The fastest rotational motion in the throw and the single strongest correlate of velocity; the arm internally rotates explosively from layback (MER) to release.
- **`max_elbow_extension_velo`** (deg/s; peak): Peak elbow-extension angular velocity — how fast the elbow straightens as the forearm whips forward into release.
- **`max_torso_rotational_velo`** (deg/s; peak): Peak torso axial-rotation (Z) angular velocity — how fast the trunk rotates toward the plate. A key link in the kinetic chain.

### Kinematics – joint angle

- **`max_rotation_hip_shoulder_separation`** (deg; peak): Peak hip–shoulder separation: the maximum angular difference between the pelvis and the thorax. Greater separation stores more elastic energy (the 'X-factor').
- **`max_elbow_flexion`** (deg; peak): Peak elbow flexion angle (most bent the elbow gets), typically around arm cocking.
- **`max_shoulder_external_rotation`** (deg; MER): Peak shoulder external rotation ('layback') — how far the forearm lays back behind the arm at maximum cocking. Higher values lengthen the acceleration path.
- **`elbow_flexion_fp`** (deg; FP): Elbow flexion angle at foot plant.
- **`elbow_pronation_fp`** (deg; FP): Wrist/forearm pronation angle at foot plant.
- **`rotation_hip_shoulder_separation_fp`** (deg; FP): Hip–shoulder separation angle at foot plant — how 'coiled' the pitcher is when the front foot lands.
- **`shoulder_horizontal_abduction_fp`** (deg; FP): Throwing-shoulder horizontal abduction at foot plant (how far the arm is swung back behind the chest plane).
- **`shoulder_abduction_fp`** (deg; FP): Throwing-shoulder abduction (elevation of the upper arm away from the torso) at foot plant — relates to arm slot.
- **`shoulder_external_rotation_fp`** (deg; FP): Throwing-shoulder external rotation at foot plant.

### Kinematics – angular velocity

- **`lead_knee_extension_angular_velo_fp`** (deg/s; FP): Lead-knee extension angular velocity at foot plant — early blocking/bracing of the front leg.
- **`lead_knee_extension_angular_velo_br`** (deg/s; BR): Lead-knee extension angular velocity at ball release. A stiff, extending front leg at release transfers energy up the chain.
- **`lead_knee_extension_angular_velo_max`** (deg/s; peak): Peak lead-knee extension angular velocity during the delivery.

### Kinematics – joint angle

- **`torso_anterior_tilt_fp`** (deg; FP): Trunk forward (anterior) flexion at foot plant.
- **`torso_lateral_tilt_fp`** (deg; FP): Trunk lateral (side) flexion at foot plant — glove-side vs throwing-side lean.
- **`torso_rotation_fp`** (deg; FP): Trunk axial rotation at foot plant (how open/closed the chest is).
- **`pelvis_anterior_tilt_fp`** (deg; FP): Pelvis anterior/posterior tilt at foot plant.
- **`pelvis_lateral_tilt_fp`** (deg; FP): Pelvis lateral tilt (hip hike) at foot plant.
- **`pelvis_rotation_fp`** (deg; FP): Pelvis axial rotation at foot plant — how far the hips have opened to the plate when the front foot lands.

### Kinematics – linear velocity

- **`max_cog_velo_x`** (m/s; peak): Peak center-of-gravity velocity toward home plate — overall momentum of the body moving down the mound.

### Kinematics – joint angle

- **`torso_rotation_min`** (deg; peak): Peak torso counter-rotation angle (most 'closed' the trunk gets) before rotating forward.

### Kinematics – angular velocity

- **`max_pelvis_rotational_velo`** (deg/s; peak): Peak pelvis axial-rotation angular velocity — the hips lead the trunk in the rotational sequence.

### Kinematics – joint angle

- **`glove_shoulder_horizontal_abduction_fp`** (deg; FP): Glove-side shoulder horizontal abduction at foot plant.
- **`glove_shoulder_abduction_fp`** (deg; FP): Glove-side shoulder abduction at foot plant.
- **`glove_shoulder_external_rotation_fp`** (deg; FP): Glove-side shoulder external rotation at foot plant. The glove arm helps set and stabilize trunk posture.
- **`glove_shoulder_abduction_mer`** (deg; MER): Glove-side shoulder abduction at maximum external rotation.
- **`elbow_flexion_mer`** (deg; MER): Throwing-elbow flexion angle at maximum external rotation (layback).
- **`torso_anterior_tilt_mer`** (deg; MER): Trunk forward flexion at maximum external rotation.
- **`torso_lateral_tilt_mer`** (deg; MER): Trunk lateral flexion at maximum external rotation — contralateral tilt raises the release point.
- **`torso_rotation_mer`** (deg; MER): Trunk axial rotation at maximum external rotation.

### Kinetics – joint moment

- **`elbow_varus_moment`** (Nm; peak): Peak elbow varus moment — the internal joint moment resisting valgus stress at the elbow. A primary proxy for UCL (elbow) injury load.
- **`shoulder_internal_rotation_moment`** (Nm; peak): Peak shoulder internal-rotation moment — internal load at the shoulder during arm acceleration.

### Kinematics – joint angle

- **`torso_anterior_tilt_br`** (deg; BR): Trunk forward flexion at ball release.
- **`torso_lateral_tilt_br`** (deg; BR): Trunk lateral flexion at ball release — strongly tied to release height and arm slot.
- **`torso_rotation_br`** (deg; BR): Trunk axial rotation at ball release (how far the chest has rotated to face the plate).
- **`lead_knee_extension_from_fp_to_br`** (deg; FP→BR): Change in lead-knee extension angle from foot plant to ball release — how much the front leg straightens/braces during acceleration.

### Kinematics – linear velocity

- **`cog_velo_pkh`** (m/s; PKH): Center-of-gravity velocity toward home at peak knee height — momentum generated during the leg lift/initial drive.

### Spatiotemporal / other

- **`stride_length`** (% body height; FP): Stride length as a percentage of body height (distance from the rubber to the lead foot at plant).
- **`stride_angle`** (deg; FP): Stride direction relative to straight toward the plate; positive = a cross-body (closed) stride.
- **`arm_slot`** (deg; BR): Arm slot — the global forearm projection angle at release that describes how over-the-top vs side-arm the delivery is.
- **`timing_peak_torso_to_peak_pelvis_rot_velo`** (s; —): Time between peak pelvis angular velocity and peak torso angular velocity — a measure of the rotational sequencing/separation timing.

### Kinematics – joint angle

- **`max_shoulder_horizontal_abduction`** (deg; peak): Peak throwing-shoulder horizontal abduction ('scap loading') during the delivery.

### Energy flow

- **`shoulder_transfer_fp_br`** (J; FP→BR): Energy transferred across the throwing shoulder between foot plant and ball release (flow of mechanical energy through the joint).
- **`shoulder_generation_fp_br`** (J; FP→BR): Energy generated (positive muscle work) at the throwing shoulder, FP→BR.
- **`shoulder_absorption_fp_br`** (J; FP→BR): Energy absorbed (negative work, eccentric) at the throwing shoulder, FP→BR.
- **`elbow_transfer_fp_br`** (J; FP→BR): Energy transferred across the throwing elbow, FP→BR.
- **`elbow_generation_fp_br`** (J; FP→BR): Energy generated at the throwing elbow, FP→BR.
- **`elbow_absorption_fp_br`** (J; FP→BR): Energy absorbed at the throwing elbow, FP→BR.
- **`lead_hip_transfer_fp_br`** (J; FP→BR): Energy transferred across the lead hip, FP→BR.
- **`lead_hip_generation_fp_br`** (J; FP→BR): Energy generated at the lead hip, FP→BR — the front hip is a major power producer in the block-and-rotate.
- **`lead_hip_absorption_fp_br`** (J; FP→BR): Energy absorbed at the lead hip, FP→BR.
- **`lead_knee_transfer_fp_br`** (J; FP→BR): Energy transferred across the lead knee, FP→BR.
- **`lead_knee_generation_fp_br`** (J; FP→BR): Energy generated at the lead knee, FP→BR.
- **`lead_knee_absorption_fp_br`** (J; FP→BR): Energy absorbed at the lead knee, FP→BR — the bracing front leg absorbs large amounts of energy.
- **`rear_hip_transfer_pkh_fp`** (J; PKH→FP): Energy transferred across the rear (drive) hip between peak knee height and foot plant.
- **`rear_hip_generation_pkh_fp`** (J; PKH→FP): Energy generated at the rear hip, PKH→FP — power from the drive leg.
- **`rear_hip_absorption_pkh_fp`** (J; PKH→FP): Energy absorbed at the rear hip, PKH→FP.
- **`rear_knee_transfer_pkh_fp`** (J; PKH→FP): Energy transferred across the rear knee, PKH→FP.
- **`rear_knee_generation_pkh_fp`** (J; PKH→FP): Energy generated at the rear knee, PKH→FP.
- **`rear_knee_absorption_pkh_fp`** (J; PKH→FP): Energy absorbed at the rear knee, PKH→FP.
- **`pelvis_lumbar_transfer_fp_br`** (J; FP→BR): Energy transferred out of the pelvis toward the trunk (up the chain), FP→BR.
- **`thorax_distal_transfer_fp_br`** (J; FP→BR): Energy transferred out of the trunk toward the throwing shoulder, FP→BR.

### Ground reaction force

- **`rear_grf_x_max`** (N; peak): Peak anterior push-off ground reaction force on the rear (drive) leg — how hard the pitcher drives toward the plate.
- **`rear_grf_y_max`** (N; peak): Peak lateral push-off ground reaction force on the rear leg.
- **`rear_grf_z_max`** (N; peak): Peak vertical ground reaction force on the rear leg.
- **`rear_grf_mag_max`** (N; peak): Peak resultant (magnitude) ground reaction force on the rear leg.
- **`rear_grf_angle_at_max`** (deg; peak): Direction of the rear ground reaction force vector at the instant of peak magnitude (projection angle).
- **`lead_grf_x_max`** (N; peak): Peak braking ground reaction force on the lead (stride) leg — how hard the front leg blocks against forward momentum.
- **`lead_grf_y_max`** (N; peak): Peak lateral braking ground reaction force on the lead leg.
- **`lead_grf_z_max`** (N; peak): Peak vertical ground reaction force on the lead leg — often several times bodyweight as the front leg braces.
- **`lead_grf_mag_max`** (N; peak): Peak resultant (magnitude) ground reaction force on the lead leg.
- **`lead_grf_angle_at_max`** (deg; peak): Direction of the lead ground reaction force vector at the instant of peak magnitude (projection angle).
- **`peak_rfd_rear`** (N/s; peak): Peak rate of force development on the rear leg — how explosively the drive-leg force is applied.
- **`peak_rfd_lead`** (N/s; peak): Peak rate of force development on the lead leg — how explosively the front leg loads against the ground during the block.