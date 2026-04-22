"""
plot_gate_results.py — Publication-quality figures for the IEEE Access paper.

Figures generated
-----------------
  fig1_traj3d.pdf       — 3D trajectories (PMM + 3 NMPC modes)
  fig2_pos_error.pdf    — Position error norm over time
  fig3_gate_error.pdf   — Radial error per gate (grouped bar chart)
  fig4_timing.pdf       — NMPC solver time per mode (box plot)
  fig5_velocity.pdf     — Speed profile: PMM reference vs tracked
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys as _sys, os as _os
for _k in list(_sys.modules.keys()):
    if 'mpl_toolkits' in _k:
        del _sys.modules[_k]
import mpl_toolkits
mpl_toolkits.__path__ = [
    p for p in ['/home/bryansgue/.local/lib/python3.10/site-packages/mpl_toolkits']
    if _os.path.isdir(p)
] or mpl_toolkits.__path__
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

_CIRCUIT   = os.environ.get('CIRCUIT', 'fig8')
_SUFFIX    = '' if _CIRCUIT == 'fig8' else f'_{_CIRCUIT}'
_GATE_F    = 'gates.npz' if _CIRCUIT == 'fig8' else f'gates_{_CIRCUIT}.npz'
_RES_FILE  = os.path.join(os.path.dirname(__file__), 'results',
                           f'sil_gate_results_{_CIRCUIT}.npy')
_XREF_FILE = os.path.join(_root, 'path_planing', f'xref_optimo_3D_PMM{_SUFFIX}.npy')
_UREF_FILE = os.path.join(_root, 'path_planing', f'uref_optimo_3D_PMM{_SUFFIX}.npy')
_TREF_FILE = os.path.join(_root, 'path_planing', f'tref_optimo_3D_PMM{_SUFFIX}.npy')
_GATE_FILE = os.path.join(_root, 'path_planing', _GATE_F)
_OUT_DIR   = os.path.join(os.path.dirname(__file__), 'results', f'figures_{_CIRCUIT}')
os.makedirs(_OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Style
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {'att': '#3A86FF', 'full': '#2DC653'}
LABELS = {'att': 'NMPC-Att', 'full': 'NMPC-Full'}
MODES  = ('att', 'full')
LW     = 1.8
ALPHA  = 0.85

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':       150,
    'pdf.fonttype':      42,   # TrueType fonts in PDF
    'ps.fonttype':       42,
})


def savefig(fig, name: str):
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 1 — 3D trajectory
# ══════════════════════════════════════════════════════════════════════════════

def fig1_traj3d(results, pmm_p, gate_cfg, all_trials=None):
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection='3d')

    gpos   = gate_cfg['gate_positions']
    gnorm  = gate_cfg['gate_normals']
    radius = float(gate_cfg['gate_radius'])
    theta  = np.linspace(0, 2 * np.pi, 80)

    # ── PMM time-optimal reference ────────────────────────────────────────
    ax.plot(pmm_p[0], pmm_p[1], pmm_p[2],
            'k--', lw=1.5, alpha=0.5, label='PMM reference', zorder=1)

    # ── NMPC trajectories: overlay all trials + bold mean ────────────────
    for mode in MODES:
        if all_trials is not None and len(all_trials) > 1:
            stack = _stack_trials(all_trials, mode, 'x',
                                  slicer=lambda a: a[0:3])  # (N,3,T)
            for k in range(stack.shape[0]):
                ax.plot(stack[k, 0], stack[k, 1], stack[k, 2],
                        color=COLORS[mode], lw=0.6, alpha=0.12, zorder=2)
            mu = stack.mean(axis=0)
            ax.plot(mu[0], mu[1], mu[2],
                    color=COLORS[mode], lw=LW, alpha=0.95,
                    label=f'{LABELS[mode]} (mean, N={stack.shape[0]})', zorder=3)
        else:
            x_log = results[mode]['x']
            ax.plot(x_log[0], x_log[1], x_log[2],
                    color=COLORS[mode], lw=LW, alpha=ALPHA,
                    label=LABELS[mode], zorder=3)

    # ── Gates — force vertical plane (nxy only) so circles don't look like lines ──
    for gi, (pos, n) in enumerate(zip(gpos, gnorm)):
        # Project normal to XY plane for visualization; gates are rendered vertical
        nxy = np.array([n[0], n[1], 0.0])
        nxy_norm = np.linalg.norm(nxy)
        n_vis = nxy / nxy_norm if nxy_norm > 0.1 else np.array([1., 0., 0.])
        t1 = np.cross(n_vis, np.array([0., 0., 1.]))
        t1_norm = np.linalg.norm(t1)
        t1 = t1 / t1_norm if t1_norm > 1e-6 else np.array([0., 1., 0.])
        t2 = np.array([0., 0., 1.])   # vertical axis of gate
        circ = np.array([pos + radius * (np.cos(t) * t1 + np.sin(t) * t2)
                         for t in theta])
        color = '#FF595E'
        ax.plot(circ[:, 0], circ[:, 1], circ[:, 2],
                color=color, lw=2.0, zorder=4)
        ax.text(pos[0], pos[1], pos[2] + 0.28, f'G{gi}',
                fontsize=7, ha='center', color='black', fontweight='bold')

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=28, azim=-50)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'fig1_traj3d.pdf')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 2 — Position error over time
# ══════════════════════════════════════════════════════════════════════════════

def fig2_pos_error(results, all_trials=None):
    from path_planing.reference_conversion import load_and_convert
    from config.experiment_config import T_S
    t_pmm = np.load(_TREF_FILE)
    T_opt = float(t_pmm[-1])
    T_sim = T_opt + 0.5
    ref   = load_and_convert(
        _XREF_FILE, _UREF_FILE, _TREF_FILE, T_s=T_S, T_sim=T_sim)
    p_ref_arr = ref['p']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    T_clip = T_opt - 0.5

    for mode in MODES:
        if all_trials is not None and len(all_trials) > 1:
            x_stack = _stack_trials(all_trials, mode, 'x',
                                    slicer=lambda a: a[0:3])     # (N,3,T)
            t        = all_trials[0][mode]['t'][:x_stack.shape[-1]]
            mask     = t <= T_clip
            t_c      = t[mask]
            T_use    = min(p_ref_arr.shape[1], mask.sum())
            err_stack = np.linalg.norm(
                x_stack[..., mask][..., :T_use] - p_ref_arr[None, :, :T_use],
                axis=1)                                          # (N,T_use)
            mu, lo, hi = _mean_band(err_stack)
            ax.fill_between(t_c[:T_use], lo, hi,
                            color=COLORS[mode], alpha=0.18, linewidth=0)
            ax.plot(t_c[:T_use], mu, color=COLORS[mode], lw=LW,
                    alpha=0.95,
                    label=f'{LABELS[mode]} (mean $\\pm$ 95\\% band)')
        else:
            t     = results[mode]['t']
            x_log = results[mode]['x']
            mask  = t <= T_clip
            t_c   = t[mask]
            N_min = min(p_ref_arr.shape[1], mask.sum())
            err   = np.linalg.norm(x_log[0:3, mask] - p_ref_arr[:, :N_min], axis=0)
            ax.plot(t_c[:N_min], err, color=COLORS[mode], lw=LW,
                    alpha=ALPHA, label=LABELS[mode])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position error [m]')
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, 'fig2_pos_error.pdf')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 3 — Radial error per gate (grouped bars)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_gate_error(results, gate_radius: float, all_trials=None):
    n_gates  = len(results['att']['crossings'])
    gate_ids = np.arange(1, n_gates + 1)

    fig, ax = plt.subplots(figsize=(8, 3.8))

    if all_trials is not None and len(all_trials) > 1:
        # Box plot per gate × mode (side-by-side)
        box_w = 0.35
        for i, mode in enumerate(MODES):
            # (N_trials, n_gates)  radial errors
            data = np.array([[c['radial_error'] if not np.isinf(c['radial_error'])
                              else gate_radius
                              for c in tr[mode]['crossings']]
                             for tr in all_trials])
            positions = gate_ids + (i - 0.5) * box_w
            bp = ax.boxplot(data, positions=positions, widths=box_w * 0.9,
                            patch_artist=True, showfliers=True,
                            manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS[mode])
                patch.set_alpha(0.55)
                patch.set_edgecolor(COLORS[mode])
            for elem in ('medians',):
                for line in bp[elem]:
                    line.set_color('black')
                    line.set_linewidth(1.2)
            for elem in ('whiskers', 'caps'):
                for line in bp[elem]:
                    line.set_color(COLORS[mode])
            for flier in bp['fliers']:
                flier.set(marker='o', markersize=3,
                          markerfacecolor=COLORS[mode],
                          markeredgecolor=COLORS[mode], alpha=0.5)
            ax.plot([], [], color=COLORS[mode], lw=6, alpha=0.55,
                    label=LABELS[mode])
    else:
        bar_w = 0.35
        for i, mode in enumerate(MODES):
            r_errs = [c['radial_error'] if not np.isinf(c['radial_error'])
                      else gate_radius
                      for c in results[mode]['crossings']]
            offsets = gate_ids + (i - 0.5) * bar_w
            ax.bar(offsets, r_errs, bar_w,
                   color=COLORS[mode], alpha=0.85, label=LABELS[mode],
                   edgecolor='white', linewidth=0.5)

    ax.axhline(gate_radius, color='red', lw=1.5, ls='--',
               label=f'Gate radius ({gate_radius:.2f} m)')
    ax.set_xlabel('Gate index')
    ax.set_ylabel('Radial crossing error [m]')
    ax.set_xticks(gate_ids)
    ax.set_xticklabels([f'G{i}' for i in gate_ids])
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, 'fig3_gate_error.pdf')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 4 — Solver timing (box plot)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_timing(results, all_trials=None):
    if all_trials is not None and len(all_trials) > 1:
        pooled = {m: np.concatenate([tr[m]['solver_ms'] for tr in all_trials])
                  for m in MODES}
    else:
        pooled = {m: results[m]['solver_ms'] for m in MODES}
    stats  = {m: {'median': np.median(pooled[m]),
                  'p95':    np.percentile(pooled[m], 95),
                  'p99':    np.percentile(pooled[m], 99)}
              for m in MODES}

    x      = np.arange(len(MODES))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.bar(x - width, [stats[m]['median'] for m in MODES],
           width, label='Median', color=[COLORS[m] for m in MODES],
           alpha=0.95, edgecolor='white')
    ax.bar(x,         [stats[m]['p95']    for m in MODES],
           width, label='P95',    color=[COLORS[m] for m in MODES],
           alpha=0.60, edgecolor='white', hatch='//')
    ax.bar(x + width, [stats[m]['p99']    for m in MODES],
           width, label='P99',    color=[COLORS[m] for m in MODES],
           alpha=0.35, edgecolor='white', hatch='xx')

    ax.axhline(10.0, color='#C0392B', lw=1.5, ls='--', label='Deadline (10 ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODES])
    ax.set_ylabel('Solver time [ms]')
    ax.set_ylim(0, 12)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, axis='y', alpha=0.35)
    fig.tight_layout()
    savefig(fig, 'fig4_timing.pdf')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Speed profile
# ══════════════════════════════════════════════════════════════════════════════

def fig5_speed(results, all_trials=None):
    X_pmm = np.load(_XREF_FILE)
    t_pmm = np.load(_TREF_FILE)
    T_opt = float(t_pmm[-1])
    spd_pmm = np.linalg.norm(X_pmm[3:6, :], axis=0)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t_pmm, spd_pmm, 'k--', lw=2.0, alpha=0.6, label='PMM reference')

    T_clip = T_opt - 0.5
    for mode in MODES:
        if all_trials is not None and len(all_trials) > 1:
            x_stack = _stack_trials(all_trials, mode, 'x',
                                    slicer=lambda a: a[3:6])    # (N,3,T)
            t   = all_trials[0][mode]['t'][:x_stack.shape[-1]]
            msk = t <= T_clip
            spd_stack = np.linalg.norm(x_stack[..., msk], axis=1)  # (N,T')
            mu, lo, hi = _mean_band(spd_stack)
            ax.fill_between(t[msk], lo, hi, color=COLORS[mode],
                            alpha=0.18, linewidth=0)
            ax.plot(t[msk], mu, color=COLORS[mode], lw=LW, alpha=0.95,
                    label=f'{LABELS[mode]} (mean $\\pm$ 95\\% band)')
        else:
            x_log = results[mode]['x']
            t     = results[mode]['t']
            msk   = t <= T_clip
            spd   = np.linalg.norm(x_log[3:6, msk], axis=0)
            ax.plot(t[msk], spd, color=COLORS[mode], lw=LW,
                    alpha=ALPHA, label=LABELS[mode])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed [m/s]')
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    savefig(fig, 'fig5_speed.pdf')


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — load flatness reference (deterministic, no MuJoCo needed)
# ══════════════════════════════════════════════════════════════════════════════

def _load_flat_ref(results):
    from path_planing.reference_conversion import load_and_convert
    t_pmm = np.load(_TREF_FILE)
    T_sim = float(results['full']['t'][-1]) + 0.01
    ref = load_and_convert(
        _XREF_FILE,
        _UREF_FILE,
        _TREF_FILE,
        T_s=0.01, T_sim=T_sim
    )
    N = results['att']['x'].shape[1]
    for k in ref:
        if hasattr(ref[k], 'shape') and ref[k].ndim >= 1:
            ref[k] = ref[k][..., :N]
    return ref


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 6 — Attitude error SO(3) norm vs time (3 configs)
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_R(q):
    """q = [qw, qx, qy, qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),     2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])

def _so3_log_norm(R):
    """‖log(R)‖_F = |θ| where θ = acos((tr(R)-1)/2)."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return abs(np.arccos(cos_theta))

def _attitude_error_deg(q_act_stack, q_ref):
    """q_act_stack: (N,4,T) or (4,T);  q_ref: (4,T) -> error in degrees."""
    single = q_act_stack.ndim == 2
    if single:
        q_act_stack = q_act_stack[None]
    N, _, T = q_act_stack.shape
    out = np.empty((N, T))
    for n in range(N):
        for k in range(T):
            out[n, k] = _so3_log_norm(_quat_to_R(q_act_stack[n, :, k]) @
                                      _quat_to_R(q_ref[:, k]).T)
    out = np.degrees(out)
    return out[0] if single else out


def fig6_att_error(results, ref, gate_cfg, all_trials=None):
    t_cross = [c['t_cross'] for c in results['full']['crossings']
               if c.get('crossed', False)]
    N = min(results['full']['x'].shape[1], ref['q'].shape[1])
    t = results['full']['t'][:N]
    q_ref = ref['q'][:, :N]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for mode in MODES:
        if all_trials is not None and len(all_trials) > 1:
            q_stack = _stack_trials(all_trials, mode, 'x',
                                    slicer=lambda a: a[6:10])      # (M,4,T)
            Tlim = min(q_stack.shape[-1], N)
            err_stack = _attitude_error_deg(q_stack[..., :Tlim],
                                            q_ref[:, :Tlim])       # (M,Tlim)
            mu, lo, hi = _mean_band(err_stack)
            ax.fill_between(t[:Tlim], lo, hi, color=COLORS[mode],
                            alpha=0.18, linewidth=0)
            ax.plot(t[:Tlim], mu, color=COLORS[mode], lw=LW,
                    alpha=0.95,
                    label=f'{LABELS[mode]} (mean $\\pm$ 95\\% band)')
        else:
            q_act = results[mode]['x'][6:10, :N]
            err = _attitude_error_deg(q_act, q_ref)
            ax.plot(t, err, color=COLORS[mode], lw=LW,
                    alpha=ALPHA, label=LABELS[mode])

    for tc in t_cross:
        ax.axvline(tc, color='#2ECC71', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, 'fig6_att_error.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 7 — Angular rate tracking ω(t) vs ω*(t) — NMPC-Full only
# ══════════════════════════════════════════════════════════════════════════════

def fig7_omega_tracking(results, ref, gate_cfg):
    t_cross = [c['t_cross'] for c in results['full']['crossings'] if c.get('crossed', False)]
    N = min(results['full']['x'].shape[1], ref['omega'].shape[1])
    t = results['full']['t'][:N]
    omega_act = results['full']['x'][10:13, :N]
    omega_ref = ref['omega'][:, :N]

    labels_ax = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for i, ax in enumerate(axes):
        ax.plot(t, omega_act[i], color='#2C3E50', lw=1.6,
                alpha=0.9, label='Actual')
        ax.plot(t, omega_ref[i], color='#E74C3C', lw=1.2,
                ls='--', alpha=0.85, label='Reference $\\omega^*$')
        ax.axhline( 9.0, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax.axhline(-9.0, color='gray', lw=0.8, ls=':', alpha=0.6)
        for tc in t_cross:
            ax.axvline(tc, color='#2ECC71', lw=0.7, ls=':', alpha=0.6)
        ax.set_ylabel(f'{labels_ax[i]} [rad/s]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', ncol=2)

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, 'fig7_omega_tracking.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  Fig 8 — XY and XZ projections of tracking trajectories (3 configs + PMM)
# ══════════════════════════════════════════════════════════════════════════════

def fig8_traj_views(results, pmm_p, gate_cfg):
    gate_pos = gate_cfg['gate_positions']
    gate_nor = gate_cfg['gate_normals']
    radius   = float(gate_cfg['gate_radius'])
    n_gates  = len(gate_pos)
    theta    = np.linspace(0, 2 * np.pi, 80)

    # Color gates by NMPC-Att crossing (Full=all green, story is where Att fails)
    crossed_att = {c['gate']: c.get('crossed', False)
                   for c in results['att']['crossings']}

    def gate_ellipse_proj(pos, n, axes=(0,1)):
        ref = np.array([0.,0.,1.]) if abs(n[2]) < 0.9 else np.array([1.,0.,0.])
        t1  = np.cross(n, ref); t1 /= np.linalg.norm(t1)
        t2  = np.cross(n, t1)
        pts = np.array([pos + radius*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta])
        return pts[:, axes[0]], pts[:, axes[1]]

    fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08, left=0.10, right=0.97, top=0.97, bottom=0.09)

    # PMM reference
    ax_xy.plot(pmm_p[0], pmm_p[1], 'k--', lw=1.4, alpha=0.55, label='PMM ref')
    ax_xz.plot(pmm_p[0], pmm_p[2], 'k--', lw=1.4, alpha=0.55, label='PMM ref')

    # NMPC trajectories
    for mode in MODES:
        x_log = results[mode]['x']
        ax_xy.plot(x_log[0], x_log[1], color=COLORS[mode], lw=LW,
                   alpha=ALPHA, label=LABELS[mode])
        ax_xz.plot(x_log[0], x_log[2], color=COLORS[mode], lw=LW,
                   alpha=ALPHA, label=LABELS[mode])

    # Gate circles: G0 = gray (start, not evaluated),
    #               G1-G7 = green if NMPC-Att crossed, red+thick if missed
    for gi, (pos, n) in enumerate(zip(gate_pos, gate_nor)):
        if gi == 0:
            col, lw_g, zo = '#888888', 1.8, 4
        elif crossed_att.get(gi, False):
            col, lw_g, zo = '#2ECC71', 2.0, 4
        else:
            col, lw_g, zo = '#E74C3C', 3.5, 6   # missed: thicker, on top

        gx_xy, gy_xy = gate_ellipse_proj(pos, n, (0, 1))
        gx_xz, gz_xz = gate_ellipse_proj(pos, n, (0, 2))
        ax_xy.plot(gx_xy, gy_xy, color=col, lw=lw_g, zorder=zo)
        ax_xz.plot(gx_xz, gz_xz, color=col, lw=lw_g, zorder=zo)

        ax_xy.text(pos[0], pos[1] + 0.28, f'G{gi}',
                   fontsize=8 if not crossed_att.get(gi, gi==0) else 7,
                   ha='center', color=col, fontweight='bold', zorder=zo+1)
        ax_xz.text(pos[0], pos[2] + 0.20, f'G{gi}',
                   fontsize=7, ha='center', color=col, fontweight='bold', zorder=zo+1)

    import matplotlib.patches as mpatches
    p_ok   = mpatches.Patch(color='#2ECC71', label='Gate crossed (Att)')
    p_miss = mpatches.Patch(color='#E74C3C', label='Gate missed (Att)')
    p_start= mpatches.Patch(color='#888888', label='Start gate')

    ax_xy.set_ylabel('Y [m]')
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3, ls='--')
    handles, lbls = ax_xy.get_legend_handles_labels()
    ax_xy.legend(handles + [p_ok, p_miss, p_start],
                 lbls + ['Gate crossed (Att)', 'Gate missed (Att)', 'Start gate'],
                 loc='upper right', ncol=3, fontsize=7.5)
    ax_xy.text(0.01, 0.97, '(a) Top view (XY)', transform=ax_xy.transAxes,
               fontsize=9, fontweight='bold', va='top')

    ax_xz.set_xlabel('X [m]')
    ax_xz.set_ylabel('Z [m]')
    ax_xz.set_aspect('equal')
    ax_xz.grid(True, alpha=0.3, ls='--')
    ax_xz.text(0.01, 0.97, '(b) Side view (XZ)', transform=ax_xz.transAxes,
               fontsize=9, fontweight='bold', va='top')

    path = os.path.join(_OUT_DIR, 'fig8_traj_views.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Fig flatness_analysis — ω*(t) per axis + T*(t) vs T_hover
#  Shows that differential flatness feedforward is non-trivial (peaks ≈ ±9 rad/s
#  at gate transitions) — the key contribution of NMPC-Full over NMPC-Att.
# ══════════════════════════════════════════════════════════════════════════════

def fig_flatness_analysis(ref):
    from config.experiment_config import MASS_MUJOCO as MASS, G
    T_hover = MASS * G

    t       = ref['t']
    omega   = ref['omega']       # (3, N)
    T_ref   = ref['T']           # (N,)

    omega_labels = [r'$\omega^*_x$ [rad/s]',
                    r'$\omega^*_y$ [rad/s]',
                    r'$\omega^*_z$ [rad/s]']
    omega_colors = ['#C0392B', '#1A7FC1', '#229954']

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8), sharex=True)
    fig.subplots_adjust(hspace=0.10, left=0.12, right=0.97, top=0.96, bottom=0.08)

    # ── ω*(t) per axis ────────────────────────────────────────────────────────
    for i in range(3):
        ax = axes[i]
        ax.plot(t, omega[i], color=omega_colors[i], lw=1.6, alpha=0.9)
        ax.axhline( 9.0, color='gray', lw=0.8, ls=':', alpha=0.55, label='±9 rad/s clip')
        ax.axhline(-9.0, color='gray', lw=0.8, ls=':', alpha=0.55)
        ax.set_ylabel(omega_labels[i], fontsize=9)
        ax.set_ylim(-11, 11)
        ax.grid(True, alpha=0.30)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        # Highlight regions where |ω*| > 5 rad/s (highly non-trivial feedforward)
        mask = np.abs(omega[i]) > 5.0
        ax.fill_between(t, -11, 11, where=mask,
                        color=omega_colors[i], alpha=0.08)

    # ── T*(t) vs T_hover ─────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, T_ref, color='#6C3483', lw=1.8, alpha=0.9, label='$T^*(t)$')
    ax.axhline(T_hover, color='#E67E22', lw=1.4, ls='--', alpha=0.85,
               label=f'$T_{{hover}}$ = {T_hover:.1f} N')
    ax.set_ylabel('Thrust $T^*$ [N]', fontsize=9)
    ax.set_xlabel('Time [s]', fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.30)

    fig.suptitle('Differential Flatness Reference — NMPC-Full feedforward signals',
                 fontsize=10, fontweight='bold')

    path = os.path.join(_OUT_DIR, 'fig_flatness_analysis.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _select_representative(raw):
    """SiL runs produce {trial_k: {att:..., full:...}}.  Return the
    representative trial (closest to median Full-RMSE) and the list of
    all valid trials for statistical aggregation."""
    if 'att' in raw and 'full' in raw:
        return raw, [raw]   # legacy single-trial layout
    items = list(raw.items())
    valid = [(k, v) for k, v in items if v['full']['pos_rmse'] < 5.0
                                     and v['att']['pos_rmse']  < 5.0]
    rmses = np.array([v['full']['pos_rmse'] for _, v in valid])
    med   = np.median(rmses)
    idx   = int(np.argmin(np.abs(rmses - med)))
    key, rep = valid[idx]
    print(f'  [SiL] {len(valid)}/{len(items)} valid trials — rep={key} '
          f'(Full RMSE={rep["full"]["pos_rmse"]:.3f} m, median={med:.3f})')
    return rep, [v for _, v in valid]


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stack_trials(all_trials, mode, field, slicer=None):
    """Stack a per-trial array across trials → shape (N_trials, ..., N_time).

    Trials may have slightly different lengths; we truncate to the shortest.
    """
    arrs = []
    for tr in all_trials:
        a = tr[mode][field]
        if slicer is not None:
            a = slicer(a)
        arrs.append(a)
    # shortest time dim
    Tmin = min(a.shape[-1] for a in arrs)
    arrs = [a[..., :Tmin] for a in arrs]
    return np.stack(arrs, axis=0)


def _mean_band(stack, q_lo=2.5, q_hi=97.5):
    """Given (N_trials, ..., T), return (mean, lo, hi) bands across trials."""
    mu = stack.mean(axis=0)
    lo = np.percentile(stack, q_lo, axis=0)
    hi = np.percentile(stack, q_hi, axis=0)
    return mu, lo, hi


_PAPER_FIG_DIR = os.path.join(_root, 'ACCESS_latex', 'figs')


def _copy_to_paper(src_name: str, dst_name: str):
    """Copy a generated PDF to ACCESS_latex/figs/ with the given final name."""
    import shutil
    src = os.path.join(_OUT_DIR, src_name)
    dst = os.path.join(_PAPER_FIG_DIR, dst_name)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f'  → {dst}')


def main():
    """Regenerate ONLY the figures referenced by access.tex for this circuit.

    Per-circuit outputs:
      - fig3_gate_error_{circuit}.pdf       (box plot over N_trials)
      - fig6_att_error_{circuit}.pdf        (mean ± 95% band)
      - fig_flatness_analysis_{circuit}.pdf (deterministic feedforward)

    Combined (fig8+loop, requires both .npy files):
      - fig_pos_error_combined.pdf          (run once from any circuit)

    Note: fig_pmm_circuits.pdf is regenerated by
    path_planing/_preview_two_circuits.py (deterministic PMM, no SiL data).
    """
    print('Loading results...')
    raw                  = np.load(_RES_FILE, allow_pickle=True).item()
    results, all_trials  = _select_representative(raw)
    gate_cfg             = np.load(_GATE_FILE)

    print('Generating paper figures...')
    fig3_gate_error(results, float(gate_cfg['gate_radius']),
                    all_trials=all_trials)
    _copy_to_paper('fig3_gate_error.pdf', f'fig3_gate_error_{_CIRCUIT}.pdf')

    print('  Loading flatness reference...')
    ref = _load_flat_ref(results)
    gate_info = {'gate_positions': gate_cfg['gate_positions'],
                 'gate_normals':   gate_cfg['gate_normals'],
                 'gate_radius':    float(gate_cfg['gate_radius'])}
    fig6_att_error(results, ref, gate_info, all_trials=all_trials)
    _copy_to_paper('fig6_att_error.pdf', f'fig6_att_error_{_CIRCUIT}.pdf')

    fig_flatness_analysis(ref)
    _copy_to_paper('fig_flatness_analysis.pdf',
                   f'fig_flatness_analysis_{_CIRCUIT}.pdf')

    # The combined pos-error figure needs both circuits' data.
    if os.environ.get('COMBINED', '1') == '1':
        try:
            fig_pos_error_combined()
            print(f'  → {os.path.join(_PAPER_FIG_DIR, "fig_pos_error_combined.pdf")}')
        except FileNotFoundError as e:
            print(f'  [skip combined] {e}')

    print(f'\nPaper figures updated in {_PAPER_FIG_DIR}/')


# ══════════════════════════════════════════════════════════════════════════════
#  fig_pos_error_combined — both circuits on a single axis (mean ± 95% band)
# ══════════════════════════════════════════════════════════════════════════════

def fig_pos_error_combined():
    """Combine fig-8 and loop position-error curves on one axis.

    Style: fig-8 solid, loop dotted; Att blue, Full green.  Mean ± 95 % band
    computed over valid SiL trials (RMSE < 5 m).  Output written directly
    to ACCESS_latex/figs/fig_pos_error_combined.pdf.
    """
    from path_planing.reference_conversion import load_and_convert
    from config.experiment_config import T_S

    circuits = ('fig8', 'loop')
    line_style = {'fig8': '-', 'loop': ':'}
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for circ in circuits:
        suf  = '' if circ == 'fig8' else f'_{circ}'
        fres = os.path.join(results_dir, f'sil_gate_results_{circ}.npy')
        if not os.path.exists(fres):
            raise FileNotFoundError(fres)
        raw       = np.load(fres, allow_pickle=True).item()
        valid     = [v for v in raw.values()
                     if v['att']['pos_rmse'] < 5.0
                     and v['full']['pos_rmse'] < 5.0]

        xref = os.path.join(_root, 'path_planing', f'xref_optimo_3D_PMM{suf}.npy')
        uref = os.path.join(_root, 'path_planing', f'uref_optimo_3D_PMM{suf}.npy')
        tref = os.path.join(_root, 'path_planing', f'tref_optimo_3D_PMM{suf}.npy')
        t_pmm = np.load(tref)
        T_opt = float(t_pmm[-1])
        T_sim = T_opt + 0.5
        T_clip = T_opt - 0.5
        ref   = load_and_convert(xref, uref, tref, T_s=T_S, T_sim=T_sim)
        p_ref = ref['p']

        for mode in MODES:
            x_stack = _stack_trials(valid, mode, 'x',
                                    slicer=lambda a: a[0:3])
            t_full  = valid[0][mode]['t'][:x_stack.shape[-1]]
            mask    = t_full <= T_clip
            T_use   = min(p_ref.shape[1], mask.sum())
            err     = np.linalg.norm(
                x_stack[..., mask][..., :T_use] - p_ref[None, :, :T_use],
                axis=1)
            mu, lo, hi = _mean_band(err)
            tc = t_full[mask][:T_use]
            ax.fill_between(tc, lo, hi, color=COLORS[mode],
                            alpha=0.14, linewidth=0)
            ax.plot(tc, mu, color=COLORS[mode], lw=LW, alpha=0.95,
                    linestyle=line_style[circ],
                    label=f'{LABELS[mode]} ({circ})')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position error [m]')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(_PAPER_FIG_DIR, 'fig_pos_error_combined.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
