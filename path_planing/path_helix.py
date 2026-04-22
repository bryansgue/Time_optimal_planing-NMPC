"""
path_helix.py — Large-radius ascending helix circuit.

8 gates on a helix of radius 7 m, rising from Z=0.6 m to Z=3.0 m over
one full revolution.  The large radius fills the same XY footprint as
the figure-8 and sprint circuits; the 2.4 m Z rise makes the 3D
corkscrew shape clearly visible in the isometric view.

  X ∈ [-7, 7] m   Y ∈ [-7, 7] m   Z ∈ [0.6, 3.0] m

Output: gates_helix.npz
"""

import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

GATE_RADIUS   = 0.40
SAFETY_MARGIN = 0.08
N_GATES = 8
RADIUS  = 7.0    # m
Z_START = 0.6    # m
Z_END   = 3.0    # m

angles = np.linspace(0.0, 2*np.pi, N_GATES, endpoint=False)
z_vals = np.linspace(Z_START, Z_END, N_GATES)

gate_positions = np.column_stack([
    RADIUS * np.cos(angles),
    RADIUS * np.sin(angles),
    z_vals,
])

# Tangent to the helix: d/dθ [R cosθ, R sinθ, z(θ)]
dz_dtheta = (Z_END - Z_START) / (2*np.pi)
gate_normals = np.zeros_like(gate_positions)
for i, th in enumerate(angles):
    t = np.array([-RADIUS*np.sin(th), RADIUS*np.cos(th), dz_dtheta])
    gate_normals[i] = t / np.linalg.norm(t)

out = os.path.join(script_dir, 'gates_helix.npz')
np.savez(out,
    gate_positions=gate_positions,
    gate_normals=gate_normals,
    gate_radius=np.array([GATE_RADIUS]),
    safety_margin=np.array([SAFETY_MARGIN]),
)

arc = sum(np.linalg.norm(gate_positions[(i+1)%N_GATES] - gate_positions[i])
          for i in range(N_GATES))
print("=" * 55)
print(f"HELIX CIRCUIT — R={RADIUS}m, Z {Z_START}→{Z_END}m, {N_GATES} gates")
print("=" * 55)
for i, (p, n) in enumerate(zip(gate_positions, gate_normals)):
    print(f"  G{i}: p={np.round(p,2)}  n={np.round(n,3)}")
print(f"\nApprox arc: {arc:.1f} m  →  {out}")
