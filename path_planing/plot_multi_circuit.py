"""
plot_multi_circuit.py — 1×3 isometric panel for the three PMM circuits.

All three subplots share the SAME xlim / ylim / zlim (computed from the
global bounding box of all circuits combined), so 1 m looks identical
across panels.  set_box_aspect([rx, ry, rz]) keeps physical proportions.

Output:  ACCESS_latex/figs/fig_pmm_circuits.pdf
"""

import os
import numpy as np

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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

D   = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, '..', 'ACCESS_latex', 'figs', 'fig_pmm_circuits.pdf')

CIRCUITS = [
    dict(title='(a) Sprint',
         xref='xref_optimo_3D_PMM_sprint.npy',
         tref='tref_optimo_3D_PMM_sprint.npy',
         gates='gates_sprint.npz',
         elev=30, azim=-55),
    dict(title='(b) Figure-8',
         xref='xref_optimo_3D_PMM.npy',
         tref='tref_optimo_3D_PMM.npy',
         gates='gates.npz',
         elev=28, azim=-52),
    dict(title='(c) Helix',
         xref='xref_optimo_3D_PMM_helix.npy',
         tref='tref_optimo_3D_PMM_helix.npy',
         gates='gates_helix.npz',
         elev=32, azim=-48),
]

GATE_R = 0.40
THETA  = np.linspace(0, 2*np.pi, 72)
MARGIN = 1.0

# ── Load ──────────────────────────────────────────────────────────────────────
for c in CIRCUITS:
    X   = np.load(os.path.join(D, c['xref']))
    t   = np.load(os.path.join(D, c['tref']))
    cfg = np.load(os.path.join(D, c['gates']))
    vel = np.linalg.norm(X[3:6, :], axis=0)
    c.update(X=X, t=t, vel=vel,
             T_f=float(t[-1]), vmax=float(vel.max()),
             gpos=cfg['gate_positions'], gnor=cfg['gate_normals'])
    print(f"{c['title']}: T_f={c['T_f']:.2f}s  v_peak={c['vmax']:.2f} m/s")

# ── Global bounding box — per axis independently ──────────────────────────────
all_x = np.concatenate([np.r_[c['X'][0,:], c['gpos'][:,0]] for c in CIRCUITS])
all_y = np.concatenate([np.r_[c['X'][1,:], c['gpos'][:,1]] for c in CIRCUITS])
all_z = np.concatenate([np.r_[c['X'][2,:], c['gpos'][:,2]] for c in CIRCUITS])

# Per-axis global limits — same for all three panels
x0, x1 = all_x.min() - MARGIN, all_x.max() + MARGIN
y0, y1 = all_y.min() - MARGIN, all_y.max() + MARGIN
z0, z1 = all_z.min() - MARGIN, all_z.max() + MARGIN
rx, ry, rz = x1-x0, y1-y0, z1-z0

print(f"\nGlobal X=[{x0:.1f},{x1:.1f}]  Y=[{y0:.1f},{y1:.1f}]  Z=[{z0:.2f},{z1:.2f}]")
print(f"Box aspect  rx={rx:.1f}  ry={ry:.1f}  rz={rz:.2f}")

# ── Shared speed colourmap (global v_max) ─────────────────────────────────────
global_vmax = max(c['vmax'] for c in CIRCUITS)
norm = Normalize(vmin=0, vmax=global_vmax)
cmap = plt.cm.plasma
sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

def seg_color(v): return cmap(norm(v))

def gate_circle(pos, n, r):
    ref = np.array([0.,0.,1.]) if abs(n[2]) < 0.9 else np.array([1.,0.,0.])
    e1  = np.cross(n, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)
    return np.array([pos + r*(np.cos(th)*e1 + np.sin(th)*e2) for th in THETA])

# ── Figure 1×3 ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.5))
fig.subplots_adjust(left=0.02, right=0.87, bottom=0.04, top=0.91, wspace=0.05)

for i, c in enumerate(CIRCUITS):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')

    X   = c['X']
    vel = c['vel']
    N   = X.shape[1] - 1

    # speed-coloured trajectory
    for k in range(N):
        ax.plot(X[0,k:k+2], X[1,k:k+2], X[2,k:k+2],
                color=seg_color(vel[k]), linewidth=2.2, alpha=0.90)

    # gate circles + labels
    for idx, (pos, n) in enumerate(zip(c['gpos'], c['gnor'])):
        pts = gate_circle(pos, n, GATE_R)
        col = '#FF6B35' if idx == 0 else '#2ECC71'
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=1.8, zorder=4)
        ax.scatter(*pos, color='#C0392B', s=35, zorder=5)
        ax.text(pos[0], pos[1], pos[2]+0.22, f'G{idx}',
                fontsize=7, color='#1A252F', ha='center', fontweight='bold')

    # ── SAME limits for all three panels ────────────────────────────────────
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect([rx, ry, rz])   # same physical scale on every panel

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.view_init(elev=c['elev'], azim=c['azim'])
    ax.grid(True, alpha=0.20)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('X [m]', fontsize=8, labelpad=3)
    ax.set_ylabel('Y [m]', fontsize=8, labelpad=3)
    ax.set_zlabel('Z [m]', fontsize=8, labelpad=2)
    ax.set_title(
        f"{c['title']}\n"
        r"$T_f^*=" + f"{c['T_f']:.2f}" + r"$~s,  $v_{\rm peak}=" + f"{c['vmax']:.2f}" + r"$~m/s",
        fontsize=9, pad=5)

# shared colourbar
cax = fig.add_axes([0.895, 0.12, 0.012, 0.72])
cb  = fig.colorbar(sm, cax=cax)
cb.set_label('Speed [m/s]', fontsize=9)
cb.ax.tick_params(labelsize=8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\n✓  Saved → {OUT}')
