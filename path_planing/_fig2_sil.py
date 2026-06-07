"""Figure 2 (resubmission) — evaluation circuits in 3-D, plotted from the
ACTUAL data the paper reports:
  - PMM reference : xref_optimo_3D_PMM{,_loop}.npy   (Table III source)
  - gates         : gates.npz / gates_loop.npz
  - tracked traj  : sil_gate_results_{fig8,loop}.npy  (SiL, first valid trial)
Output goes to a NEW pdf (fig_pmm_circuits_v2.pdf) for review; the old
preview script is left untouched."""
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
from matplotlib.lines import Line2D
import numpy as np

script = os.path.dirname(os.path.abspath(__file__))
root   = os.path.dirname(script)

# --- data sources per circuit -------------------------------------------
CIRC = [
    dict(lbl='fig8', ref='xref_optimo_3D_PMM.npy',
         gates='gates.npz',      sil='sil_gate_results_fig8.npy',
         title='(a) Figure-8'),
    dict(lbl='loop', ref='xref_optimo_3D_PMM_loop.npy',
         gates='gates_loop.npz', sil='sil_gate_results_loop.npy',
         title='(b) Vertical Loop'),
]

def first_valid(sil, th=5.0):
    """First trial where both controllers stayed below the blow-up bound."""
    keys = sorted(sil.keys(), key=lambda s: int(s.split('_')[1]))
    for k in keys:
        if sil[k]['att']['pos_rmse'] <= th and sil[k]['full']['pos_rmse'] <= th:
            return k
    return keys[0]

m = 0.6
fig = plt.figure(figsize=(14, 6.4))

for k, C in enumerate(CIRC):
    ax = fig.add_subplot(1, 2, k+1, projection='3d')

    xref = np.load(os.path.join(script, C['ref']))           # 6 x T
    px, py, pz = xref[0], xref[1], xref[2]

    g = np.load(os.path.join(script, C['gates']))
    gp = g['gate_positions']; gn = g['gate_normals']
    gr = float(g['gate_radius'])

    sil = np.load(os.path.join(root, 'experiments', 'results', C['sil']),
                  allow_pickle=True).item()
    tk = first_valid(sil)

    # axis limits from reference + gates
    allp = np.vstack([np.column_stack([px, py, pz]), gp])
    ax.set_xlim(allp[:,0].min()-m, allp[:,0].max()+m)
    ax.set_ylim(allp[:,1].min()-m, allp[:,1].max()+m)
    ax.set_zlim(0.0,               allp[:,2].max()+m)

    # tracked NMPC trajectories (SiL, first valid trial)
    for ctrl, col in [('att', (0.15, 0.40, 0.95)),
                      ('full',(0.15, 0.75, 0.25))]:
        x = sil[tk][ctrl]['x']                                # 13 x T
        ax.plot(x[0], x[1], x[2], color=col, linewidth=1.7, zorder=3)

    # PMM reference: thick black dashed (on top)
    ax.plot(px, py, pz, color='black', linestyle='--', linewidth=2.0,
            alpha=0.9, zorder=4)

    # gates
    th = np.linspace(0, 2*np.pi, 60)
    for gi, (p, n) in enumerate(zip(gp, gn)):
        n = n/np.linalg.norm(n)
        ref = np.array([0,0,1.0]) if abs(n[2]) < 0.9 else np.array([1.0,0,0])
        e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        pts = np.array([p + gr*(np.cos(t)*e1 + np.sin(t)*e2) for t in th])
        ax.plot(pts[:,0], pts[:,1], pts[:,2],
                color=(0.35,0.35,0.35), linewidth=1.2)
        ax.scatter(*p, color='red', s=22, zorder=5)
        ax.text(p[0], p[1], p[2]+0.30, f'G{gi}', fontsize=7, ha='center',
                fontweight='bold')

    ax.set_xlabel('X [m]', fontsize=9)
    ax.set_ylabel('Y [m]', fontsize=9)
    ax.set_zlabel('Z [m]', fontsize=9)
    ax.view_init(elev=22, azim=-55)
    ax.set_title(C['title'], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# shared legend centred between the two panels
legend_handles = [
    Line2D([0],[0], color='black', lw=2.0, linestyle='--', label='PMM reference'),
    Line2D([0],[0], color=(0.15,0.40,0.95), lw=1.8, label='NMPC-Att'),
    Line2D([0],[0], color=(0.15,0.75,0.25), lw=1.8, label='NMPC-Full'),
]
fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.47, 0.97), ncol=3, fontsize=9,
           framealpha=0.9, columnspacing=1.8, handlelength=2.0)

plt.subplots_adjust(left=0.02, right=0.98, wspace=0.06, top=0.93, bottom=0.04)
out_png = os.path.join(script, 'fig2_sil_preview.png')
out_pdf = os.path.join(root, 'ACCESS_latex', 'figs', 'fig_pmm_circuits_v2.pdf')
fig.savefig(out_png, dpi=160, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_png}\nSaved: {out_pdf}")
