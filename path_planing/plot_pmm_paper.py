"""
Publication-quality PMM figures for IEEE Access — all single-column.

Outputs (ACCESS_latex/figs/):
  fig_pmm_3d.pdf      — standalone 3-D perspective, speed colourmap
  fig_pmm_views.pdf   — XY (top) + XZ (side) side-by-side
  fig_pmm_speed.pdf   — speed profile with gate-crossing instants
"""

import os, sys
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Load data ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
X    = np.load(os.path.join(script_dir, 'xref_optimo_3D_PMM.npy'))
U    = np.load(os.path.join(script_dir, 'uref_optimo_3D_PMM.npy'))
t    = np.load(os.path.join(script_dir, 'tref_optimo_3D_PMM.npy'))
cfg  = np.load(os.path.join(script_dir, 'gates.npz'))

gate_pos = cfg['gate_positions']
gate_nor = cfg['gate_normals']
gate_r   = float(cfg['gate_radius'])

# Compute k_gate: index of PMM step closest to each gate centre
_p = X[:3, :]
k_gate = np.array([int(np.argmin(np.linalg.norm(_p - gp[:, None], axis=0)))
                   for gp in gate_pos])

T_opt    = t[-1]
N        = X.shape[1] - 1
vel      = np.linalg.norm(X[3:6, :], axis=0)
vmax     = vel.max()
t_gate   = t[k_gate]
labels   = [f'G{i}' for i in range(len(gate_pos))]

out_dir = os.path.join(script_dir, '..', 'ACCESS_latex', 'figs')
os.makedirs(out_dir, exist_ok=True)

THETA = np.linspace(0, 2 * np.pi, 72)
cmap  = plt.cm.plasma
norm  = Normalize(vmin=0, vmax=vmax)
sm    = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

def seg_color(k): return cmap(norm(vel[k]))

def gate_circle_3d(pos, n, r):
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.array([pos + r*(np.cos(th)*e1 + np.sin(th)*e2) for th in THETA])

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — standalone 3-D view
# ─────────────────────────────────────────────────────────────────────────────
fig3d, ax3d = plt.subplots(1, 1, figsize=(8.5, 6.5),
                            subplot_kw={'projection': '3d'})
fig3d.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.93)

for k in range(N):
    ax3d.plot(X[0,k:k+2], X[1,k:k+2], X[2,k:k+2],
              color=seg_color(k), linewidth=2.5, alpha=0.92)

for idx, (pos, n) in enumerate(zip(gate_pos, gate_nor)):
    pts = gate_circle_3d(pos, n, gate_r)
    col = '#FF6B35' if idx == 0 else '#2ECC71'
    ax3d.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=2.2, zorder=4)
    ax3d.scatter(*pos, color='#C0392B', s=50, zorder=5)
    ax3d.quiver(pos[0], pos[1], pos[2],
                n[0]*0.50, n[1]*0.50, n[2]*0.50,
                color='#7D3C98', arrow_length_ratio=0.30,
                linewidth=1.0, alpha=0.75)
    off = pos + n*0.38 + np.array([0, 0, 0.22])
    ax3d.text(off[0], off[1], off[2], labels[idx],
              fontsize=9, color='#1A252F', ha='center', fontweight='bold')

ax3d.scatter(*gate_pos[0], color='#FF6B35', s=130, marker='*', zorder=6)

# ── Equal physical scale: box aspect proportional to data ranges ──────────
_margin = 0.7
_all_x = np.concatenate([X[0, :], gate_pos[:, 0]])
_all_y = np.concatenate([X[1, :], gate_pos[:, 1]])
_all_z = np.concatenate([X[2, :], gate_pos[:, 2]])
x0, x1 = _all_x.min() - _margin, _all_x.max() + _margin
y0, y1 = _all_y.min() - _margin, _all_y.max() + _margin
z0, z1 = _all_z.min() - _margin, _all_z.max() + _margin
rx, ry, rz = x1 - x0, y1 - y0, z1 - z0

ax3d.set_xlim(x0, x1)
ax3d.set_ylim(y0, y1)
ax3d.set_zlim(z0, z1)
ax3d.set_box_aspect([rx, ry, rz])   # 1 m in X = 1 m in Y = 1 m in Z

# ── Uniform tick step: 4 m on X and Y, 0.5 m on Z ────────────────────────
from matplotlib.ticker import MultipleLocator
ax3d.xaxis.set_major_locator(MultipleLocator(4))
ax3d.yaxis.set_major_locator(MultipleLocator(4))
ax3d.zaxis.set_major_locator(MultipleLocator(0.5))

cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.50, pad=0.04, aspect=22)
cbar.set_label('Speed [m/s]', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
ax3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
ax3d.set_zlabel('Z [m]', fontsize=9, labelpad=4)
ax3d.tick_params(labelsize=8)
ax3d.view_init(elev=28, azim=-52)
ax3d.grid(True, alpha=0.25)

p3d = os.path.join(out_dir, 'fig_pmm_3d.pdf')
fig3d.savefig(p3d, dpi=300, bbox_inches='tight')
plt.close(fig3d)
print(f'✓ {p3d}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — XY (top) stacked over XZ (bottom), shared X axis
# ─────────────────────────────────────────────────────────────────────────────
fig_views, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(8.5, 6.0),
                                          sharex=True)
fig_views.subplots_adjust(left=0.10, right=0.90, bottom=0.09, top=0.93,
                          hspace=0.08)

# --- XY (top panel) ---
for k in range(N):
    ax_xy.plot(X[0,k:k+2], X[1,k:k+2], color=seg_color(k),
               linewidth=2.0, alpha=0.9)
for idx, (pos, n) in enumerate(zip(gate_pos, gate_nor)):
    pts = gate_circle_3d(pos, n, gate_r)
    col = '#FF6B35' if idx == 0 else '#2ECC71'
    ax_xy.plot(pts[:,0], pts[:,1], color=col, linewidth=1.8, alpha=0.85)
    ax_xy.scatter(pos[0], pos[1], color='#C0392B', s=40, zorder=5)
    ax_xy.annotate('', xy=(pos[0]+n[0]*0.42, pos[1]+n[1]*0.42),
                   xytext=(pos[0], pos[1]),
                   arrowprops=dict(arrowstyle='->', color='#7D3C98',
                                   lw=0.9, shrinkA=0, shrinkB=0))
    ax_xy.text(pos[0], pos[1]+0.34, labels[idx],
               fontsize=8.5, ha='center', color='#1A252F', fontweight='bold')
ax_xy.scatter(gate_pos[0,0], gate_pos[0,1], color='#FF6B35',
              s=100, marker='*', zorder=6)
ax_xy.set_ylabel('Y [m]', fontsize=10)
ax_xy.tick_params(labelsize=9, labelbottom=False)
ax_xy.set_aspect('equal')
ax_xy.grid(True, alpha=0.3, linestyle='--')
ax_xy.text(0.02, 0.96, '(a) Top view (XY)', transform=ax_xy.transAxes,
           fontsize=10, fontweight='bold', va='top')

# --- XZ (bottom panel) ---
for k in range(N):
    ax_xz.plot(X[0,k:k+2], X[2,k:k+2], color=seg_color(k),
               linewidth=2.0, alpha=0.9)
for idx, (pos, n) in enumerate(zip(gate_pos, gate_nor)):
    pts = gate_circle_3d(pos, n, gate_r)
    col = '#FF6B35' if idx == 0 else '#2ECC71'
    ax_xz.plot(pts[:,0], pts[:,2], color=col, linewidth=1.8, alpha=0.85)
    ax_xz.scatter(pos[0], pos[2], color='#C0392B', s=40, zorder=5)
    ax_xz.annotate('', xy=(pos[0]+n[0]*0.42, pos[2]+n[2]*0.42),
                   xytext=(pos[0], pos[2]),
                   arrowprops=dict(arrowstyle='->', color='#7D3C98',
                                   lw=0.9, shrinkA=0, shrinkB=0))
    ax_xz.text(pos[0], pos[2]+0.27, labels[idx],
               fontsize=8.5, ha='center', color='#1A252F', fontweight='bold')
ax_xz.scatter(gate_pos[0,0], gate_pos[0,2], color='#FF6B35',
              s=100, marker='*', zorder=6)
ax_xz.set_xlabel('X [m]', fontsize=10)
ax_xz.set_ylabel('Z [m]', fontsize=10)
ax_xz.tick_params(labelsize=9)
ax_xz.set_aspect('equal')
ax_xz.grid(True, alpha=0.3, linestyle='--')
ax_xz.text(0.02, 0.96, '(b) Side view (XZ)', transform=ax_xz.transAxes,
           fontsize=10, fontweight='bold', va='top')

# shared colorbar on the right
cbar2 = fig_views.colorbar(sm, ax=[ax_xy, ax_xz], shrink=0.95,
                            pad=0.02, aspect=35, location='right')
cbar2.set_label('Speed [m/s]', fontsize=10)
cbar2.ax.tick_params(labelsize=9)

p_views = os.path.join(out_dir, 'fig_pmm_views.pdf')
fig_views.savefig(p_views, dpi=300, bbox_inches='tight')
plt.close(fig_views)
print(f'✓ {p_views}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — speed profile
# ─────────────────────────────────────────────────────────────────────────────
fig_sp, ax_sp = plt.subplots(1, 1, figsize=(8.5, 3.8))
fig_sp.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.88)

ax_sp.plot(t, vel,           color='#2980B9', linewidth=2.2,
           label=r'$\|\mathbf{v}\|$')
ax_sp.plot(t, X[3,:],        color='#E74C3C', linewidth=1.2,
           alpha=0.75, linestyle='--', label=r'$v_x$')
ax_sp.plot(t, X[4,:],        color='#27AE60', linewidth=1.2,
           alpha=0.75, linestyle='--', label=r'$v_y$')
ax_sp.plot(t, X[5,:],        color='#F39C12', linewidth=1.2,
           alpha=0.75, linestyle='--', label=r'$v_z$')

for i in range(1, len(gate_pos)):
    tk = t_gate[i]
    ax_sp.axvline(tk, color='#27AE60', linewidth=0.9, linestyle=':', alpha=0.85)
    ax_sp.text(tk + 0.04, vmax * 0.96, labels[i],
               fontsize=7.5, color='#1E8449', va='top', ha='left', rotation=90)

k_peak = np.argmax(vel)
ax_sp.annotate(f'$v_{{\\rm peak}} = {vmax:.2f}$~m/s',
               xy=(t[k_peak], vmax),
               xytext=(t[k_peak] + 0.7, vmax - 0.55),
               fontsize=9, color='#1A252F',
               arrowprops=dict(arrowstyle='->', color='#1A252F', lw=0.9))

ax_sp.axvline(T_opt, color='#7D3C98', linewidth=1.2, linestyle='--', alpha=0.85)
ax_sp.text(T_opt - 1.35, 0.18, f'$T_f^* = {T_opt:.2f}$~s',
           fontsize=9, color='#7D3C98')

ax_sp.set_xlabel('Time [s]', fontsize=10)
ax_sp.set_ylabel('Speed [m/s]', fontsize=10)
ax_sp.set_xlim([0, T_opt + 0.25])
ax_sp.set_ylim([vel.min() - 0.3, vmax * 1.22])
ax_sp.tick_params(labelsize=9)
ax_sp.legend(fontsize=9, loc='upper left', ncol=4, framealpha=0.85)
ax_sp.grid(True, alpha=0.3)

p_sp = os.path.join(out_dir, 'fig_pmm_speed.pdf')
fig_sp.savefig(p_sp, dpi=300, bbox_inches='tight')
plt.close(fig_sp)
print(f'✓ {p_sp}')
