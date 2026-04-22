"""
path_sprint.py — Wide oval race circuit.

8 gates forming a proper oval: two long parallel straights (~18 m) connected
by banked hairpins.  Designed to fill the same XY footprint as the figure-8
circuit so all three panels in the multi-circuit figure look proportional.

  X ∈ [0, 18] m   Y ∈ [-4, 4] m   Z ∈ [1.0, 2.0] m

Output: gates_sprint.npz
"""

import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

GATE_RADIUS   = 0.40
SAFETY_MARGIN = 0.08

# ── 8 gates — wide oval ───────────────────────────────────────────────────────
#
#   G7 ──── G6 ──── G5 ──── G4
#   |    (top straight, +Y)    |
#   G0                        G3   ← right hairpin
#   |   (bottom straight, -Y) |
#   G1 ──── G2 ──── G3 ... wait, let me redo
#
#  Clockwise when viewed from above:
#  G0 (left hairpin) → G1,G2 (bottom) → G3 (right hairpin) → G4,G5,G6 (top) → G7 (near G0)
#
gate_positions = np.array([
    [ 0.0,  0.0,  1.5],   # G0 — left hairpin centre
    [ 4.5, -4.0,  1.2],   # G1 — bottom straight, left
    [ 9.0, -4.0,  1.0],   # G2 — bottom straight, centre
    [13.5, -4.0,  1.2],   # G3 — bottom straight, right
    [18.0,  0.0,  1.5],   # G4 — right hairpin centre
    [13.5,  4.0,  1.8],   # G5 — top straight, right
    [ 9.0,  4.0,  2.0],   # G6 — top straight, centre
    [ 4.5,  4.0,  1.8],   # G7 — top straight, left
], dtype=float)

# normals: direction of travel gate[i] → gate[i+1 mod N]
n = len(gate_positions)
gate_normals = np.zeros_like(gate_positions)
for i in range(n):
    d = gate_positions[(i+1) % n] - gate_positions[i]
    gate_normals[i] = d / np.linalg.norm(d)

out = os.path.join(script_dir, 'gates_sprint.npz')
np.savez(out,
    gate_positions=gate_positions,
    gate_normals=gate_normals,
    gate_radius=np.array([GATE_RADIUS]),
    safety_margin=np.array([SAFETY_MARGIN]),
)

arc = sum(np.linalg.norm(gate_positions[(i+1)%n] - gate_positions[i]) for i in range(n))
print("=" * 55)
print("SPRINT CIRCUIT — wide oval, 8 gates")
print("=" * 55)
for i, (p, nv) in enumerate(zip(gate_positions, gate_normals)):
    print(f"  G{i}: p={np.round(p,2)}  n={np.round(nv,3)}")
print(f"\nApprox arc: {arc:.1f} m  →  {out}")
