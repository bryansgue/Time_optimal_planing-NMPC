"""
path_lissajous.py — Analytical Lissajous reference trajectory for gate navigation.

Trajectory (3D Lissajous figure):
    x(t) = 5.00 * sin(1*w*t) + 2.5
    y(t) = 1.25 * sin(2*w*t)
    z(t) = 0.75 * sin(3*w*t) + 1.5

Peak speed at t=0 (all cosines = 1 simultaneously):
    |v|_max = w * sqrt(5^2 + 2.5^2 + 2.25^2) = w * 6.026

Usage: set V_MAX_TARGET [m/s], script computes w automatically.
Gates are placed at N_GATES equally-spaced arc-length samples; each gate
normal is the normalised velocity at that instant.

Outputs (saved to this directory):
    xref_optimo_3D_PMM.npy   (6, N+1) — [p; v]
    uref_optimo_3D_PMM.npy   (3, N)   — [ax; ay; az]
    tref_optimo_3D_PMM.npy   (N+1,)   — time vector
    gate_config.npz           — gate_positions, gate_normals, gate_radius, …
"""

import os
import sys
import numpy as np
from scipy.io import savemat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import sys as _sys
for _k in list(_sys.modules.keys()):
    if "mpl_toolkits" in _k:
        del _sys.modules[_k]
import mpl_toolkits
mpl_toolkits.__path__ = [
    p for p in ["/home/bryansgue/.local/lib/python3.10/site-packages/mpl_toolkits"]
    if os.path.isdir(p)
] or mpl_toolkits.__path__
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETERS — adjust here
# =============================================================================
V_MAX_TARGET = 10.0     # [m/s]  desired peak speed (theoretical max of Lissajous)
N_GATES      = 8        # number of gates (even → half-interval offset avoids center crossing)
N_LAPS       = 1        # number of complete periods to simulate
DT_REF       = 0.01     # [s]  reference time step (= 1/f_c, 100 Hz)
GATE_RADIUS  = 0.40     # [m]  gate inner radius
SAFETY_MARGIN = 0.08    # [m]  safety margin

# =============================================================================
# COMPUTE w
# =============================================================================
_speed_scale = np.sqrt(5.0**2 + (2*2.0)**2 + (3*0.75)**2)   # amplitudes × freqs
w = V_MAX_TARGET / _speed_scale
T_period = 2.0 * np.pi / w
T_total  = N_LAPS * T_period

print("=" * 60)
print("LISSAJOUS PATH GENERATOR")
print("=" * 60)
print(f"V_max target  = {V_MAX_TARGET:.2f} m/s")
print(f"w             = {w:.4f} rad/s")
print(f"T_period      = {T_period:.3f} s")
print(f"T_total       = {T_total:.3f} s")
print(f"N_gates       = {N_GATES}  (over {N_LAPS} lap)")

# =============================================================================
# TRAJECTORY FUNCTIONS (analytical)
# =============================================================================
def pos(t):
    return np.array([
        5.00 * np.sin(1.0 * w * t) + 2.5,
        2.00 * np.sin(2.0 * w * t),
        0.75 * np.sin(3.0 * w * t) + 1.5,
    ])

def vel(t):
    return np.array([
        5.00 * 1.0 * w * np.cos(1.0 * w * t),
        2.00 * 2.0 * w * np.cos(2.0 * w * t),
        0.75 * 3.0 * w * np.cos(3.0 * w * t),
    ])

def acc(t):
    return np.array([
        -5.00 * (1.0 * w)**2 * np.sin(1.0 * w * t),
        -2.00 * (2.0 * w)**2 * np.sin(2.0 * w * t),
        -0.75 * (3.0 * w)**2 * np.sin(3.0 * w * t),
    ])

# =============================================================================
# BUILD REFERENCE ARRAYS (100 Hz)
# =============================================================================
N_steps = int(round(T_total / DT_REF))
t_vec   = np.linspace(0.0, T_total, N_steps + 1)

p_arr = np.array([pos(t) for t in t_vec]).T   # (3, N+1)
v_arr = np.array([vel(t) for t in t_vec]).T   # (3, N+1)
a_arr = np.array([acc(t) for t in t_vec[:N_steps]]).T  # (3, N)

X_opt = np.vstack([p_arr, v_arr])   # (6, N+1)
U_opt = a_arr                        # (3, N)

v_mag = np.linalg.norm(v_arr, axis=0)
print(f"\nTrajectory: {N_steps+1} points  |  v_max={v_mag.max():.3f} m/s  |  v_mean={v_mag.mean():.3f} m/s")

# =============================================================================
# GATE PLACEMENT — N_GATES equally-spaced arc-length samples over one period
# =============================================================================
# Compute cumulative arc length over one full period at high resolution
N_arc = 10000
t_arc  = np.linspace(0.0, T_period, N_arc)
v_arc  = np.array([vel(t) for t in t_arc])          # (N_arc, 3)
ds     = np.linalg.norm(v_arc, axis=1) * (T_period / N_arc)
s_cum  = np.concatenate([[0.0], np.cumsum(ds)])      # (N_arc+1,)
arc_total = s_cum[-1]
print(f"Arc length (1 lap) = {arc_total:.2f} m")

# Target arc positions: evenly spaced with half-interval offset to avoid
# the figure-8 center crossings at t=0 and t=T/2
arc_step   = arc_total / N_GATES
arc_offset = arc_step / 2.0                                    # shift away from crossings
s_targets  = np.arange(N_GATES) * arc_step + arc_offset       # (N_GATES,)

# Find time for each target arc length (interpolation)
t_gates = np.interp(s_targets, s_cum, np.linspace(0, T_period, N_arc + 1))

gate_positions  = np.array([pos(t) for t in t_gates])    # (N_gates, 3)
gate_velocities = np.array([vel(t) for t in t_gates])    # (N_gates, 3)
gate_normals    = gate_velocities / np.linalg.norm(gate_velocities, axis=1, keepdims=True)

t_gates_full = t_gates
k_gate       = np.round(t_gates_full / DT_REF).astype(int)

print(f"\nGate layout ({len(gate_positions)} gates):")
for i, (gp, gn, tg) in enumerate(zip(gate_positions, gate_normals, t_gates_full)):
    v_at = np.linalg.norm(vel(tg))
    print(f"  G{i}: p={np.round(gp,2)}  n={np.round(gn,3)}  t={tg:.3f}s  |v|={v_at:.2f} m/s")

# =============================================================================
# VERIFY CROSSINGS (post-hoc check)
# =============================================================================
print("\nVerification — gate crossings in reference trajectory:")
all_ok = True
for i in range(len(gate_positions)):
    k_i   = k_gate[i]
    p_ref = p_arr[:, k_i]
    gpos  = gate_positions[i]
    gnorm = gate_normals[i]
    d     = p_ref - gpos
    perp  = np.dot(d, gnorm)
    radial= np.linalg.norm(d - perp * gnorm)
    ok    = radial < GATE_RADIUS
    print(f"  G{i}: r={radial:.4f}m  perp={perp:.4f}m  {'✓' if ok else '✗ MISS'}")
    if not ok:
        all_ok = False
print("  All OK ✓" if all_ok else "  Some gates missed — increase N_arc resolution")

# =============================================================================
# SAVE
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# Only save gate geometry — PMM will generate xref/uref/tref from these gates
np.savez(os.path.join(script_dir, 'gate_config.npz'),
    gate_positions = gate_positions,
    gate_normals   = gate_normals,
    gate_radius    = np.array([GATE_RADIUS]),
    safety_margin  = np.array([SAFETY_MARGIN]),
)
print(f"\nSaved: gate_config.npz  →  {script_dir}/")
print("Run path_time_3D_PMM.py next to compute time-optimal reference.")

# =============================================================================
# PLOTS
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# 3D trajectory
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
vel_norm = v_mag / v_mag.max()
for k in range(len(vel_norm) - 1):
    c = plt.cm.plasma(vel_norm[k])
    ax1.plot(p_arr[0, k:k+2], p_arr[1, k:k+2], p_arr[2, k:k+2],
             color=c, linewidth=2.5, alpha=0.85)

theta_c = np.linspace(0, 2*np.pi, 60)
for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    ref_v = np.array([0.,0.,1.]) if abs(gn[2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gn, ref_v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gn, t1)
    circ = np.array([gp + GATE_RADIUS*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta_c])
    color_g = 'limegreen'
    ax1.plot(circ[:,0], circ[:,1], circ[:,2], color=color_g, lw=2)
    ax1.scatter(*gp, color='red', s=50, zorder=5)
    ax1.text(gp[0], gp[1], gp[2]+0.25, f'G{gi}', fontsize=7, ha='center')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, v_mag.max()))
sm.set_array([])
fig.colorbar(sm, ax=ax1, shrink=0.5, label='Speed [m/s]', pad=0.1)
ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
ax1.set_title(f'Lissajous 3D  (v_max={v_mag.max():.1f} m/s)', fontsize=9)
ax1.view_init(elev=25, azim=-55)

# Speed profile
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_vec, v_mag, 'k-', lw=1.5)
for i, tg in enumerate(t_gates_full):
    ax2.axvline(tg, color='#2ECC71', lw=0.8, ls=':', alpha=0.8)
    ax2.text(tg, v_mag.max()*0.92, f'G{i}', fontsize=7, ha='center', color='green')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Speed [m/s]')
ax2.set_title('Speed profile')
ax2.set_ylim(0, None)
ax2.grid(True, alpha=0.4)

# XY top view
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(p_arr[0], p_arr[1], 'b-', lw=1.5, alpha=0.7)
for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    ref_v = np.array([0.,0.,1.]) if abs(gn[2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gn, ref_v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gn, t1)
    circ = np.array([gp + GATE_RADIUS*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta_c])
    ax3.plot(circ[:,0], circ[:,1], color='limegreen', lw=2)
    ax3.text(gp[0], gp[1]+0.18, f'G{gi}', fontsize=7, ha='center')
ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]')
ax3.set_title('Top view (XY)')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# XZ side view
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(p_arr[0], p_arr[2], 'b-', lw=1.5, alpha=0.7)
for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    ref_v = np.array([0.,0.,1.]) if abs(gn[2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gn, ref_v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gn, t1)
    circ = np.array([gp + GATE_RADIUS*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta_c])
    ax4.plot(circ[:,0], circ[:,2], color='limegreen' if gi>0 else 'orange', lw=2)
    ax4.text(gp[0], gp[2]+0.1, f'G{gi}', fontsize=7, ha='center')
ax4.set_xlabel('X [m]'); ax4.set_ylabel('Z [m]')
ax4.set_title('Side view (XZ)')
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Lissajous Reference  |  w={w:.3f} rad/s  |  v_max={v_mag.max():.1f} m/s  |  T={T_period:.2f}s', fontsize=10)
plt.tight_layout()
out_png = os.path.join(script_dir, 'path_lissajous.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {out_png}")
