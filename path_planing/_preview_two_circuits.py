"""3-D PMM reference + NMPC-Att / NMPC-Full tracked trajectories overlaid
on a single two-panel 3-D figure (one panel per circuit)."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import sys as _sys
for _k in list(_sys.modules.keys()):
    if "mpl_toolkits" in _k: del _sys.modules[_k]
import mpl_toolkits
mpl_toolkits.__path__ = [p for p in ["/home/bryansgue/.local/lib/python3.10/site-packages/mpl_toolkits"] if os.path.isdir(p)] or mpl_toolkits.__path__
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

script = os.path.dirname(os.path.abspath(__file__))
root   = os.path.dirname(script)

S = [loadmat(os.path.join(script, 'path_PMM_results_fig8.mat')),
     loadmat(os.path.join(script, 'path_PMM_results_loop.mat'))]

R = [np.load(os.path.join(root, 'experiments', 'results',
             f'gate_experiment_results_{name}.npy'), allow_pickle=True).item()
     for name in ('fig8', 'loop')]

def speed(s):
    return np.sqrt(s['vx'].ravel()**2 + s['vy'].ravel()**2 + s['vz'].ravel()**2)

v_all_max = float(np.ceil(max(speed(S[0]).max(), speed(S[1]).max())))

titles = [
    f"(a) Figure-8   $T_{{opt}}={float(S[0]['T_opt']):.2f}$ s,  $v_{{max}}={speed(S[0]).max():.2f}$ m/s",
    f"(b) Vertical Loop   $T_{{opt}}={float(S[1]['T_opt']):.2f}$ s,  $v_{{max}}={speed(S[1]).max():.2f}$ m/s",
]

m = 0.6
fig = plt.figure(figsize=(14, 6.4))
cmap = plt.cm.viridis

for k, (s, r) in enumerate(zip(S, R)):
    ax = fig.add_subplot(1, 2, k+1, projection='3d')
    px = s['px'].ravel(); py = s['py'].ravel(); pz = s['pz'].ravel()
    v  = speed(s)
    gp = s['gate_positions']; gn = s['gate_normals']
    gr = float(s['gate_radius'])

    # Per-circuit axis limits
    all_pos_k = np.vstack([
        np.column_stack([px, py, pz]),
        gp,
    ])
    xL = (all_pos_k[:,0].min()-m, all_pos_k[:,0].max()+m)
    yL = (all_pos_k[:,1].min()-m, all_pos_k[:,1].max()+m)
    zL = (0.0,                    all_pos_k[:,2].max()+m)

    # PMM reference: coloured by speed
    v_n = v / v_all_max
    for i in range(len(px)-1):
        ax.plot(px[i:i+2], py[i:i+2], pz[i:i+2],
                color=cmap(v_n[i]), linewidth=1.6, alpha=0.55)

    # Tracked NMPC trajectories (first trial of each)
    for ctrl, col, lbl, lw in [('att',  (0.15, 0.40, 0.95), 'NMPC-Att',  1.7),
                                ('full', (0.15, 0.75, 0.25), 'NMPC-Full', 1.7)]:
        x = r[ctrl]['x']
        ax.plot(x[0], x[1], x[2], color=col, linewidth=lw,
                label=lbl, zorder=6)

    # Gates
    th = np.linspace(0, 2*np.pi, 60)
    for g, (p, n) in enumerate(zip(gp, gn)):
        n = n/np.linalg.norm(n)
        ref = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([1.0,0,0])
        e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        pts = np.array([p + gr*(np.cos(t)*e1 + np.sin(t)*e2) for t in th])
        ax.plot(pts[:,0], pts[:,1], pts[:,2],
                color=(0.35, 0.35, 0.35), linewidth=1.2)
        ax.scatter(*p, color='red', s=22, zorder=5)
        ax.text(p[0], p[1], p[2]+0.30, f'G{g}', fontsize=7, ha='center',
                fontweight='bold')

    ax.set_xlim(xL); ax.set_ylim(yL); ax.set_zlim(zL)
    ax.set_xlabel('X [m]', fontsize=9)
    ax.set_ylabel('Y [m]', fontsize=9)
    ax.set_zlabel('Z [m]', fontsize=9)
    ax.view_init(elev=22, azim=-55)
    ax.set_title(titles[k], fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if k == 0:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Shared speed colorbar (reference colouring)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, v_all_max))
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.18, 0.014, 0.65])
fig.colorbar(sm, cax=cbar_ax, label='PMM reference speed [m/s]')

plt.subplots_adjust(left=0.02, right=0.91, wspace=0.06, top=0.93, bottom=0.04)
out_png = os.path.join(script, 'pmm_circuits_preview.png')
out_pdf = os.path.join(script, '..', 'ACCESS_latex', 'figs', 'fig_pmm_circuits.pdf')
fig.savefig(out_png, dpi=160, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_png}\nSaved: {out_pdf}")
