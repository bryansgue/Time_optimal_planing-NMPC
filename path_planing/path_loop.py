"""
path_loop.py — Acrobatic vertical-loop circuit.

8 gates forcing a vertical loop in the XZ plane with hand-tuned normals so
that at the apex (G3) the drone is INVERTED (thrust vector points DOWN in
world frame). This is the scenario where NMPC-Att with yaw-only reference
cannot work: q_att = [cos(psi/2), 0, 0, sin(psi/2)] cannot represent an
inverted attitude, so only NMPC-Full with complete flatness inversion
tracks the reference.

Geometry:
  - Loop plane: XZ, centered at (0, 0, 3), radius ~2 m
  - Apex gate G3 at (0, 0, 5.0) with normal (-1, 0, 0) forcing v_apex in -X
  - Inversion condition: v_apex^2 > g*R  =>  v > 4.43 m/s at R=2
  - PMM with v_max=10 m/s will pass at ~6-8 m/s => drone must be upside down

Workspace: X in [-2, 4], Y in [-2, 3], Z in [0.8, 5.0]

Output: gates_loop.npz
"""

import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

GATE_RADIUS   = 0.40
SAFETY_MARGIN = 0.08

# -- 8 gates: approach, vertical loop (XZ plane), return lobe ------------------
gate_positions = np.array([
    [ 4.0, -2.0, 1.5],   # G0 - entry, level flight
    [ 3.0,  0.0, 2.0],   # G1 - pre-loop, starting to pitch up
    [ 2.0,  0.0, 4.0],   # G2 - quarter loop (climbing + rotating)
    [ 0.0,  0.0, 5.0],   # G3 - APEX - drone INVERTED, v = -X
    [-2.0,  0.0, 4.0],   # G4 - three-quarter loop (descending)
    [ 0.0,  0.0, 1.0],   # G5 - loop bottom exit, level
    [ 3.0,  2.0, 4.0],   # G6 - post-loop climb, uses loop-exit inertia
    [-1.0,  3.0, 2.0],   # G7 - return leg
    [ 1.5,  0.5, 0.8],   # G8 - FINISH gate, lowered below return path
], dtype=float)

# -- Hand-tuned normals for the loop; auto-tangent for approach/return ---------
# Loop gates (G1..G5) have normals derived from circular motion in XZ plane.
# Approach/return gates (G0, G6, G7) use direction-to-next-gate.
n_gates = len(gate_positions)
gate_normals = np.zeros_like(gate_positions)

# Auto-tangent for non-loop gates (direction to next gate modulo N)
for i in range(n_gates):
    d = gate_positions[(i + 1) % n_gates] - gate_positions[i]
    gate_normals[i] = d / np.linalg.norm(d)

# Override loop-critical gates with hand-tuned normals forcing circular flow
# in the XZ plane (center ~ (0,0,3), radius ~ 2).
# Parametrize by angle theta measured from +X axis in XZ plane:
#   G1 at theta ~ -45 deg : v direction ~ (sin(-45), 0, -(-cos(-45))) hmm
# Simpler — tangent to circle at each apex point, CCW when viewed from +Y:
#   Gk: pos = (R cos th, 0, 3 + R sin th),  tangent = (-sin th, 0, cos th)
# Assign:
#   G1 (transition in): keep auto — connects level flight to loop entry
#   G2 at (2, 0, 4): th = atan2(4-3, 2) = atan2(1, 2) ~ 26.6 deg
#                    tangent = (-sin(26.6), 0, cos(26.6)) = (-0.447, 0, +0.894)
#   G3 at (0, 0, 5): th = 90 deg, tangent = (-1, 0, 0)  -- APEX, pure -X
#   G4 at (-2, 0, 4): th = atan2(1, -2) = 153.4 deg
#                    tangent = (-sin(153.4), 0, cos(153.4)) = (-0.447, 0, -0.894)
#   G5 at (0, 0, 1): th = 270 deg, tangent = (+1, 0, 0)  -- bottom exit, pure +X
gate_normals[2] = np.array([-0.4472, 0.0,  0.8944])
gate_normals[3] = np.array([-1.0,    0.0,  0.0])
gate_normals[4] = np.array([-0.4472, 0.0, -0.8944])
gate_normals[5] = np.array([ 1.0,    0.0,  0.0])

# G6 and G7: smooth tangents via centered difference (G_{k+1} - G_{k-1}) so the
# arrows follow the natural curvature of the path, not the chord to the next gate.
for k in (6, 7):
    tangent = gate_positions[(k + 1) % n_gates] - gate_positions[k - 1]
    gate_normals[k] = tangent / np.linalg.norm(tangent)

# re-normalize (defensive)
for i in range(n_gates):
    gate_normals[i] /= np.linalg.norm(gate_normals[i])

out = os.path.join(script_dir, 'gates_loop.npz')
np.savez(out,
    gate_positions=gate_positions,
    gate_normals=gate_normals,
    gate_radius=np.array([GATE_RADIUS]),
    safety_margin=np.array([SAFETY_MARGIN]),
)

arc = sum(np.linalg.norm(gate_positions[(i+1) % n_gates] - gate_positions[i])
          for i in range(n_gates))
print("=" * 60)
print("VERTICAL LOOP CIRCUIT - 8 gates, XZ-plane loop")
print("=" * 60)
for i, (p, nv) in enumerate(zip(gate_positions, gate_normals)):
    tag = ""
    if i == 3: tag = "  <-- APEX (inverted)"
    if i == 2: tag = "  <-- loop entry climbing"
    if i == 4: tag = "  <-- loop descending"
    if i == 5: tag = "  <-- loop exit level"
    print(f"  G{i}: p={np.round(p,2)}  n={np.round(nv,3)}{tag}")
print(f"\nApprox arc: {arc:.1f} m  ->  {out}")
print(f"Inversion condition at apex: v > sqrt(g*R) = sqrt(9.81*2) = 4.43 m/s")
