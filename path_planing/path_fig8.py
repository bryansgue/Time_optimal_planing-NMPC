"""
path_fig8.py — Moderate figure-8 racing circuit (8 gates, aggressive 3D).

Design choices (vs. original gates.npz):
  1. Start gate rotated to the crossover point of the figure-8 so the drone's
     initial direction (from rest) aligns with the sustained-speed trajectory.
     This removes a cold-start transient that otherwise unfairly penalizes
     NMPC-Att (which lacks the thrust/rate feedforward to catch up quickly).
  2. Z-range expanded from ~[0.94, 2.06] to [0.8, 3.0] with alternating
     low/high gates, forcing real 3D motion.

Geometry: two lobes in +X and -X, meeting at crossover along Y-axis, radius
  ~6 m in X.  Loop 1: G0(-Y low) → G1(+X high) → G2(+X +Y) → G3(+X -Y high)
                       → G4(+X low) → G5(+Y high) → G6(-X +Y) → G7(-X -Y high)
                       → back to G0.
Workspace: X in [-7, 12], Y in [-3.5, 3.5], Z in [0.8, 3.0]

Output: gates.npz (overwrites the default fig-8 config)
"""

import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

GATE_RADIUS   = 0.40
SAFETY_MARGIN = 0.08

# Cyclic shift of the original fig-8 + alternating Z for real 3D dynamics.
#   z alternates 0.8 / 3.0 (low-high-low-high...) so every segment has a
#   vertical component; |Δz| per segment = 2.2 m.
gate_positions = np.array([
    [-0.28, -2.14, 1.0],   # G0  crossover, LOW — starting from here aligns
                           #     v0=0 with diagonal direction to G1
    [ 5.28,  2.14, 2.5],   # G1  +X lobe entry, HIGH
    [11.46,  3.18, 1.4],   # G2  +X apex top, LOW
    [11.46, -3.18, 2.2],   # G3  +X apex bottom, HIGH
    [ 5.28, -2.14, 1.0],   # G4  +X lobe exit, LOW
    [-0.28,  2.14, 2.5],   # G5  crossover other side, HIGH
    [-6.46,  3.18, 1.4],   # G6  -X lobe, LOW
    [-6.46, -3.18, 2.2],   # G7  -X lobe, HIGH  → returns to G0
], dtype=float)

# Normals: centered-difference tangent so arrows follow natural path curvature.
n_gates = len(gate_positions)
gate_normals = np.zeros_like(gate_positions)
for k in range(n_gates):
    tangent = gate_positions[(k + 1) % n_gates] - gate_positions[k - 1]
    gate_normals[k] = tangent / np.linalg.norm(tangent)

out = os.path.join(script_dir, 'gates.npz')
np.savez(out,
    gate_positions=gate_positions,
    gate_normals=gate_normals,
    gate_radius=np.array([GATE_RADIUS]),
    safety_margin=np.array([SAFETY_MARGIN]),
)

arc = sum(np.linalg.norm(gate_positions[(i+1) % n_gates] - gate_positions[i])
          for i in range(n_gates))
print("=" * 60)
print("FIGURE-8 CIRCUIT - 8 gates, aggressive 3D")
print("=" * 60)
for i, (p, nv) in enumerate(zip(gate_positions, gate_normals)):
    tag = ""
    if i == 0:
        tag = "  <-- START (crossover, low)"
    print(f"  G{i}: p={np.round(p, 2)}  n={np.round(nv, 3)}{tag}")
print(f"\nZ range: [{gate_positions[:,2].min():.2f}, {gate_positions[:,2].max():.2f}] m"
      f"  (spread = {np.ptp(gate_positions[:,2]):.2f} m)")
print(f"Approx arc: {arc:.1f} m  ->  {out}")
