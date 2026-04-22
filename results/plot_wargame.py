"""
plot_wargame.py — Publication-quality plots for wargame paper results.

Loads .npy log files produced by wargame_mil_sim.py or wargame_mujoco_node.py
and generates the figures needed for Section V of the RAL paper.

Figures produced
----------------
  fig1_separation_{scenario}.pdf   — separation distance + d_min line
  fig2_barriers_{scenario}.pdf     — ψ₀, φ₀ barrier values (≥0 = safe)
  fig3_solvetime_{scenario}.pdf    — evader NMPC solve time vs 10 ms budget
  fig4_trajectory_{scenario}.pdf   — 3D trajectories of both drones
  fig5_comparison_{scenario}.pdf   — NMPC+HOCBF vs CBF-QP vs no-CBF

Usage
-----
    # Plot MiL results
    python results/plot_wargame.py --scenario cornering --source mil

    # Plot SiL results (all three modes overlaid)
    python results/plot_wargame.py --scenario cornering --source sil --compare

    # Generate all figures for all scenarios
    python results/plot_wargame.py --all
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path

# ── Plot style ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'legend.fontsize':  8,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       150,
    'lines.linewidth':  1.4,
})

RESULTS_DIR = Path(__file__).parent.parent / 'results_sim'
OUT_DIR     = Path(__file__).parent.parent / 'results_sim' / 'paper_figs'

D_MIN  = 0.5   # minimum separation [m]
R_WS   = 4.0   # workspace radius [m]
P_C    = np.array([0.0, 0.0, 1.5])

COLORS = {
    'hocbf': '#1f77b4',   # blue
    'cbfqp': '#d62728',   # red
    'nocbf': '#ff7f0e',   # orange
    'dmin':  '#2ca02c',   # green
}
LABELS = {
    'hocbf': 'NMPC+HOCBF (proposed)',
    'cbfqp': 'CBF-QP (post-hoc)',
    'nocbf': 'NMPC (no safety)',
}

SCENARIOS = ['free', 'cornering', 'speed_advantage']
MODES     = ['hocbf', 'cbfqp', 'nocbf']


# ══════════════════════════════════════════════════════════════════════════
#  Load helpers
# ══════════════════════════════════════════════════════════════════════════

def load(scenario: str, mode: str, source: str) -> dict | None:
    """Load log for (scenario, mode, source). Returns None if file missing."""
    fname = RESULTS_DIR / f'wargame_{scenario}_{mode}_{source}.npy'
    if not fname.exists():
        print(f'[warn] not found: {fname}')
        return None
    log = np.load(fname, allow_pickle=True).item()
    # Ensure arrays
    for key in ['t', 'dist', 'psi0', 'phi0', 'solve_ms']:
        if key in log:
            log[key] = np.asarray(log[key])
    return log


# ══════════════════════════════════════════════════════════════════════════
#  Individual figures
# ══════════════════════════════════════════════════════════════════════════

def fig_separation(logs: dict, scenario: str, source: str, out_dir: Path):
    """Fig 1 – separation distance vs time for all modes."""
    fig, ax = plt.subplots(figsize=(6, 3))

    for mode, log in logs.items():
        if log is None:
            continue
        ax.plot(log['t'], log['dist'],
                color=COLORS[mode], label=LABELS[mode],
                alpha=0.9)

    ax.axhline(D_MIN, color=COLORS['dmin'], linestyle='--',
               linewidth=1.2, label=f'$d_{{\\min}} = {D_MIN}$ m')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Separation $d(t)$ [m]')
    ax.set_title(f'Pursuer–Evader Separation ({_scen_label(scenario)})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    _save(fig, out_dir / f'fig1_separation_{scenario}_{source}.pdf')


def fig_barriers(logs: dict, scenario: str, source: str, out_dir: Path):
    """Fig 2 – HOCBF barrier values ψ₀ and φ₀ over time."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)

    for mode, log in logs.items():
        if log is None:
            continue
        axes[0].plot(log['t'], log['psi0'],
                     color=COLORS[mode], label=LABELS[mode], alpha=0.9)
        axes[1].plot(log['t'], log['phi0'],
                     color=COLORS[mode], label=LABELS[mode], alpha=0.9)

    for ax, title in zip(axes,
                         [r'Separation barrier $\psi_0(t)$',
                          r'Workspace barrier $\phi_0(t)$']):
        ax.axhline(0, color='k', linestyle=':', linewidth=1.0)
        ax.fill_between([0, logs.get('hocbf', {}).get('t', [0, 1])[-1]],
                        0, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.2,
                        color='red', alpha=0.06, label='_nolegend_')
        ax.set_ylabel('Barrier value')
        ax.set_title(title + '  (must remain $\\geq 0$)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Time [s]')
    fig.suptitle(f'HOCBF Barrier Values ({_scen_label(scenario)})',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / f'fig2_barriers_{scenario}_{source}.pdf')


def fig_solvetime(log_hocbf: dict | None, scenario: str,
                  source: str, out_dir: Path):
    """Fig 3 – evader NMPC+HOCBF solve time histogram + time series."""
    if log_hocbf is None:
        return
    ms = log_hocbf['solve_ms']

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Time series
    ax = axes[0]
    ax.plot(log_hocbf['t'], ms, color=COLORS['hocbf'], linewidth=0.8)
    ax.axhline(10.0, color='r', linestyle='--', linewidth=1.1,
               label='10 ms budget')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Solve time [ms]')
    ax.set_title('Solve time vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(ms, bins=40, color=COLORS['hocbf'], edgecolor='white',
            linewidth=0.4)
    ax.axvline(10.0, color='r', linestyle='--', linewidth=1.1,
               label='10 ms budget')
    ax.axvline(np.mean(ms), color='navy', linestyle='-', linewidth=1.2,
               label=f'mean = {np.mean(ms):.2f} ms')
    ax.set_xlabel('Solve time [ms]')
    ax.set_ylabel('Count')
    ax.set_title('Solve time distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'NMPC+HOCBF Solver Timing ({_scen_label(scenario)})',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / f'fig3_solvetime_{scenario}_{source}.pdf')


def fig_trajectory_3d(logs: dict, scenario: str, source: str, out_dir: Path):
    """Fig 4 – 3D trajectories of evader and pursuer."""
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection='3d')

    # Workspace sphere (wireframe)
    u_s = np.linspace(0, 2*np.pi, 40)
    v_s = np.linspace(0, np.pi, 20)
    xs  = P_C[0] + R_WS * np.outer(np.cos(u_s), np.sin(v_s))
    ys  = P_C[1] + R_WS * np.outer(np.sin(u_s), np.sin(v_s))
    zs  = P_C[2] + R_WS * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.08, linewidth=0.4)

    for mode, log in logs.items():
        if log is None or 'p_e' not in log:
            continue
        p_e = np.array(log['p_e'])
        p_p = np.array(log['p_p'])
        ax.plot(p_e[:, 0], p_e[:, 1], p_e[:, 2],
                color=COLORS[mode], label=f'Evader ({LABELS[mode]})',
                linewidth=1.2, alpha=0.85)
        if mode == 'hocbf':   # plot pursuer trajectory once
            ax.plot(p_p[:, 0], p_p[:, 1], p_p[:, 2],
                    color='k', linestyle='--', linewidth=0.8,
                    label='Pursuer', alpha=0.6)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f'3D Trajectories ({_scen_label(scenario)})')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    _save(fig, out_dir / f'fig4_trajectory_{scenario}_{source}.pdf')


def fig_metrics_table(scenario_logs: dict, source: str, out_dir: Path):
    """Generate a summary metrics table (printed + saved as text)."""
    lines = [
        f'{"Scenario":<18}{"Mode":<12}{"Sep.Viol%":>10}'
        f'{"WS.Viol%":>10}{"MinDist(m)":>12}{"MeanMs":>9}{"MaxMs":>9}',
        '-' * 80,
    ]
    for scenario, logs in scenario_logs.items():
        for mode, log in logs.items():
            if log is None:
                continue
            n   = len(log['t'])
            v_s = float(np.sum(np.array(log['psi0']) < 0)) / n * 100
            v_w = float(np.sum(np.array(log['phi0']) < 0)) / n * 100
            md  = float(np.min(log['dist']))
            ms  = float(np.array(log['solve_ms']).mean()) if 'solve_ms' in log else float('nan')
            mx  = float(np.array(log['solve_ms']).max())  if 'solve_ms' in log else float('nan')
            lines.append(
                f'{scenario:<18}{mode:<12}{v_s:>10.2f}'
                f'{v_w:>10.2f}{md:>12.3f}{ms:>9.2f}{mx:>9.2f}'
            )
        lines.append('')

    text = '\n'.join(lines)
    print('\n' + text)
    (out_dir / f'metrics_table_{source}.txt').write_text(text)


# ══════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════

def _scen_label(s: str) -> str:
    return {'free': 'Free Pursuit',
            'cornering': 'Cornering',
            'speed_advantage': 'Speed Advantage'}[s]


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f'  Saved → {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def plot_scenario(scenario: str, source: str, compare: bool):
    out_dir = OUT_DIR
    logs = {mode: load(scenario, mode, source) for mode in MODES}

    if not compare:
        # Only plot the HOCBF result
        logs = {'hocbf': logs['hocbf']}

    fig_separation(logs, scenario, source, out_dir)
    fig_barriers(logs, scenario, source, out_dir)
    fig_solvetime(logs.get('hocbf'), scenario, source, out_dir)
    fig_trajectory_3d(logs, scenario, source, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='cornering',
                        choices=SCENARIOS)
    parser.add_argument('--source', default='mil',
                        choices=['mil', 'sil'],
                        help='Log source: mil (MiL) or sil (MuJoCo SiL)')
    parser.add_argument('--compare', action='store_true',
                        help='Overlay all three modes (hocbf, cbfqp, nocbf)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all figures for all scenarios')
    args = parser.parse_args()

    if args.all:
        scenario_logs = {}
        for sc in SCENARIOS:
            logs = {mode: load(sc, mode, args.source) for mode in MODES}
            scenario_logs[sc] = logs
            plot_scenario(sc, args.source, compare=True)
        fig_metrics_table(scenario_logs, args.source, OUT_DIR)
    else:
        plot_scenario(args.scenario, args.source, args.compare)
