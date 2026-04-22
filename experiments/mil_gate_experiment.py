"""
mil_gate_experiment.py — Model-in-the-Loop gate navigation experiment.

Two NMPC configurations (ablation study):

  'att'  — p*(t) + v*(t) + yaw-derived q*(t).  Common practice baseline.
            Solver: Drone_gate_att  (Qw=0, no angular-rate cost)

  'full' — Full differential flatness: p*, v*, q*, ω*, T*.  Contribution.
            Solver: Drone_gate_full (all cost terms active)

MuJoCo integrates at 500 Hz; NMPC runs at 100 Hz.
"""

import os
import sys
import time
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

from config.experiment_config import (
    T_S, N_PREDICTION, T_PREDICTION, G, MASS_MUJOCO as MASS, T_MAX,
)
from models.quadrotor_model_rate import f_system_model
from ocp.nmpc_gate_tracker import build_gate_solver_att, build_gate_solver_full
from utils.numpy_utils import rk4_step_quadrotor
from path_planing.reference_conversion import (
    load_and_convert,
    build_param_att,
    build_param_full,
)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(_RESULTS_DIR, exist_ok=True)

_CIRCUIT = os.environ.get('CIRCUIT', 'fig8')   # 'fig8' or 'loop'
_SUFFIX  = '' if _CIRCUIT == 'fig8' else f'_{_CIRCUIT}'
_GATE_F  = 'gates.npz' if _CIRCUIT == 'fig8' else f'gates_{_CIRCUIT}.npz'
_XREF = os.path.join(_root, 'path_planing', f'xref_optimo_3D_PMM{_SUFFIX}.npy')
_UREF = os.path.join(_root, 'path_planing', f'uref_optimo_3D_PMM{_SUFFIX}.npy')
_TREF = os.path.join(_root, 'path_planing', f'tref_optimo_3D_PMM{_SUFFIX}.npy')
_GATE = os.path.join(_root, 'path_planing', _GATE_F)

T_SIM_EXTRA = 0.5
RK4_STEPS   = 10

# ── Noise levels for statistical experiment (Fase 2) ──────────────────────────
N_TRIALS   = 10
NOISE_STD  = {
    'p':     0.02,    # position   [m]
    'v':     0.05,    # velocity   [m/s]
    'q':     0.01,    # quaternion [rad equiv.]
    'omega': 0.05,    # body rates [rad/s]
}


# ══════════════════════════════════════════════════════════════════════════════
#  Gate crossing detection
# ══════════════════════════════════════════════════════════════════════════════

def check_gate_crossings(p_hist, t_hist, gate_positions, gate_normals, gate_radius):
    n_gates  = gate_positions.shape[0]
    crossings = []
    for gi in range(1, n_gates):
        gpos  = gate_positions[gi]
        gnorm = gate_normals[gi]
        dists = np.linalg.norm(p_hist - gpos[:, None], axis=0)
        k_min = int(np.argmin(dists))
        r_err = float(dists[k_min])
        p_cross  = p_hist[:, k_min]
        dp       = p_cross - gpos
        dp_plane = dp - np.dot(dp, gnorm) * gnorm
        crossings.append({
            'gate':         gi,
            'crossed':      r_err < gate_radius,
            'radial_error': float(np.linalg.norm(dp_plane)),
            'dist_3d':      r_err,
            't_cross':      float(t_hist[k_min]),
            'p_cross':      p_cross,
        })
    return crossings


# ══════════════════════════════════════════════════════════════════════════════
#  Single-mode simulation
# ══════════════════════════════════════════════════════════════════════════════

def warm_start_from_ref(solver, ref_interp, k_start, x0, N):
    N_sim   = ref_interp['t'].shape[0]
    T_hover = MASS * G
    for stage in range(N + 1):
        k_ref = min(k_start + stage, N_sim - 1)
        x_ref = np.concatenate([
            ref_interp['p'][:, k_ref],
            ref_interp['v'][:, k_ref],
            ref_interp['q'][:, k_ref],
            np.zeros(3),
        ])
        solver.set(stage, 'x', x_ref)
    for stage in range(N):
        k_ref = min(k_start + stage, N_sim - 1)
        T_ref = float(np.clip(ref_interp['T'][k_ref], 0.0, T_MAX))
        solver.set(stage, 'u', np.array([T_ref, 0.0, 0.0, 0.0]))


def _make_param(mode, ref, k):
    if mode == 'att':
        return build_param_att(
            ref['p'][:, k], ref['v'][:, k], ref['q_att'][:, k])
    else:
        return build_param_full(
            ref['p'][:, k], ref['v'][:, k],
            ref['q'][:, k], ref['omega'][:, k],
            float(ref['T'][k]))


def run_mode(mode, ref_interp, solver, f_system, gate_cfg, x0,
             noise_std=None, rng=None):
    """
    Simulate one trial.  If noise_std is given, Gaussian noise is added to the
    measured state (what the NMPC sees) at every step; the true state evolves
    noise-free via RK4.
    """
    N_sim     = ref_interp['t'].shape[0]
    t_vec     = ref_interp['t']
    dt_rk4    = T_S / RK4_STEPS

    x_true    = np.zeros((13, N_sim))   # noise-free true state
    u_log     = np.zeros((4,  N_sim - 1))
    solver_ms = np.zeros(N_sim - 1)
    x_true[:, 0] = x0

    warm_start_from_ref(solver, ref_interp, 0, x0, N_PREDICTION)
    if noise_std is None:
        print(f'  [mode={mode}] simulating {N_sim} steps (no noise)...')
    else:
        print(f'  [mode={mode}] simulating {N_sim} steps (with noise, seed={rng.bit_generator.state["state"]["state"]})...')

    for k in range(N_sim - 1):
        x_meas = x_true[:, k].copy()
        if noise_std is not None and rng is not None:
            x_meas[0:3]  += rng.normal(0, noise_std['p'],     3)
            x_meas[3:6]  += rng.normal(0, noise_std['v'],     3)
            x_meas[6:10] += rng.normal(0, noise_std['q'],     4)
            x_meas[10:13]+= rng.normal(0, noise_std['omega'], 3)
            qn = np.linalg.norm(x_meas[6:10])
            if qn > 1e-6:
                x_meas[6:10] /= qn

        solver.set(0, 'lbx', x_meas)
        solver.set(0, 'ubx', x_meas)
        for j in range(N_PREDICTION + 1):
            k_ref = min(k + j, N_sim - 1)
            solver.set(j, 'p', _make_param(mode, ref_interp, k_ref))

        tic = time.perf_counter()
        status = solver.solve()
        solver_ms[k] = 1e3 * (time.perf_counter() - tic)

        if status not in (0, 2):
            warm_start_from_ref(solver, ref_interp, k, x_true[:, k], N_PREDICTION)

        u_log[:, k] = solver.get(0, 'u')

        x_next = x_true[:, k].copy()
        for _ in range(RK4_STEPS):
            x_next = rk4_step_quadrotor(x_next, u_log[:, k], dt_rk4, f_system)
        q_norm = np.linalg.norm(x_next[6:10])
        if q_norm > 1e-6:
            x_next[6:10] /= q_norm
        x_true[:, k + 1] = x_next

    crossings  = check_gate_crossings(
        x_true[0:3, :], t_vec,
        gate_cfg['gate_positions'], gate_cfg['gate_normals'],
        float(gate_cfg['gate_radius']),
    )
    pos_rmse   = float(np.sqrt(np.mean(
        np.sum((x_true[0:3, :] - ref_interp['p'])**2, axis=0))))
    gates_ok   = sum(c['crossed'] for c in crossings)
    mean_r_err = float(np.mean([c['dist_3d'] for c in crossings]))

    print(f'    RMSE pos = {pos_rmse:.4f} m  |  '
          f'gates {gates_ok}/{len(crossings)}  |  '
          f'mean radial err = {mean_r_err:.4f} m  |  '
          f'solver mean = {solver_ms.mean():.2f} ms')

    return {
        'x':          x_true,
        'u':          u_log,
        't':          t_vec,
        'solver_ms':  solver_ms,
        'crossings':  crossings,
        'pos_rmse':   pos_rmse,
        'gates_ok':   gates_ok,
        'mean_r_err': mean_r_err,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Fase 2 — Statistical experiment: N_TRIALS with Gaussian sensor noise
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_experiment(ref_interp, solvers, f_system, gate_cfg, x0,
                               n_trials=N_TRIALS):
    """
    Run N_TRIALS per mode with independent Gaussian noise seeds.
    Returns per-mode arrays of shape (n_trials,) for each scalar metric.
    """
    n_gates = gate_cfg['gate_positions'].shape[0] - 1   # gates 1..n-1
    stats   = {m: {'pos_rmse': [], 'gates_ok': [], 'mean_r_err': [],
                   'solver_p95': [], 'solver_p99': []}
               for m in ('att', 'full')}

    for trial in range(n_trials):
        seed = 42 + trial
        print(f'\n  ── Trial {trial+1}/{n_trials}  seed={seed} ──')
        for mode in ('att', 'full'):
            rng    = np.random.default_rng(seed)
            solver = solvers[mode]
            res    = run_mode(mode, ref_interp, solver, f_system, gate_cfg, x0,
                              noise_std=NOISE_STD, rng=rng)
            stats[mode]['pos_rmse'  ].append(res['pos_rmse'])
            stats[mode]['gates_ok'  ].append(res['gates_ok'])
            stats[mode]['mean_r_err'].append(res['mean_r_err'])
            stats[mode]['solver_p95'].append(float(np.percentile(res['solver_ms'], 95)))
            stats[mode]['solver_p99'].append(float(np.percentile(res['solver_ms'], 99)))

    # Convert to arrays
    for m in stats:
        for k in stats[m]:
            stats[m][k] = np.array(stats[m][k])

    return stats, n_gates


def print_stats(stats, n_gates, n_trials):
    total_gates = n_trials * n_gates
    print('\n' + '═' * 62)
    print(f'STATISTICAL RESULTS  (N={n_trials} trials, noise ON)')
    print(f'{"METRIC":<32} {"ATT":>14} {"FULL":>14}')
    print('─' * 62)
    for m in ('att', 'full'):
        pass   # printed per-metric below

    def row(label, key, fmt='.4f'):
        print(f'  {label:<30}', end='')
        for m in ('att', 'full'):
            arr = stats[m][key]
            print(f'  {arr.mean():{fmt}} ± {arr.std():{fmt}}', end='')
        print()

    row('Pos RMSE [m]',           'pos_rmse')
    print(f'  {"Gate crossing rate":<30}', end='')
    for m in ('att', 'full'):
        total_ok = int(stats[m]['gates_ok'].sum())
        print(f'  {total_ok:>6}/{total_gates}         ', end='')
    print()
    row('Mean radial error [m]',  'mean_r_err')
    row('Solver P95 [ms]',        'solver_p95', '.2f')
    row('Solver P99 [ms]',        'solver_p99', '.2f')
    print('═' * 62)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_pmm  = np.load(_TREF)
    T_opt  = float(t_pmm[-1])
    T_sim  = T_opt + T_SIM_EXTRA
    print(f'PMM T_opt = {T_opt:.2f} s  |  T_sim = {T_sim:.2f} s')

    ref_interp = load_and_convert(
        xref_path=_XREF, uref_path=_UREF, tref_path=_TREF,
        T_s=T_S, T_sim=T_sim,
    )
    N_sim = ref_interp['t'].shape[0]
    print(f'Reference: {N_sim} NMPC steps at {1/T_S:.0f} Hz')

    gate_cfg_raw = np.load(_GATE)
    gate_cfg = {
        'gate_positions': gate_cfg_raw['gate_positions'],
        'gate_normals':   gate_cfg_raw['gate_normals'],
        'gate_radius':    float(gate_cfg_raw['gate_radius']),
    }
    print(f'Gates: {gate_cfg["gate_positions"].shape[0]}  '
          f'(radius = {gate_cfg["gate_radius"]:.2f} m)')

    p0 = gate_cfg['gate_positions'][0]
    x0 = np.concatenate([p0, np.zeros(3), np.array([1., 0., 0., 0.]), np.zeros(3)])
    print(f'x0 = p={np.round(p0, 2)}')

    # Build both solvers (compile once, reuse .so)
    builders = {'att': build_gate_solver_att, 'full': build_gate_solver_full}
    _, _, f_system = build_gate_solver_att(x0, rebuild=False)
    solvers = {}
    for mode in ('att', 'full'):
        solvers[mode], _, _ = builders[mode](x0, rebuild=False)

    # ── Deterministic trial (no noise) ───────────────────────────────────────
    results = {}
    for mode in ('att', 'full'):
        print(f'\n{"─"*60}\nMODE: {mode.upper()}\n{"─"*60}')
        results[mode] = run_mode(mode, ref_interp, solvers[mode],
                                 f_system, gate_cfg, x0)

    save_path = os.path.join(_RESULTS_DIR, f'gate_experiment_results_{_CIRCUIT}.npy')
    np.save(save_path, results, allow_pickle=True)
    print(f'\nResults saved → {save_path}')

    n_gates = len(results['att']['crossings'])
    print('\n' + '═' * 50)
    print(f'{"METRIC":<28} {"ATT":>10} {"FULL":>10}')
    print('─' * 50)
    for metric, key in [('Pos RMSE [m]', 'pos_rmse'),
                        ('Mean radial error [m]', 'mean_r_err'),
                        ('Solver mean [ms]', None)]:
        print(f'{metric:<28}', end='')
        for m in ('att', 'full'):
            val = results[m][key] if key else results[m]['solver_ms'].mean()
            print(f' {val:>10.4f}', end='')
        print()
    print(f'{"Gates crossed":<28}', end='')
    for m in ('att', 'full'):
        print(f' {results[m]["gates_ok"]:>9d}/{n_gates}', end='')
    print()
    print('═' * 50)

    # ── Statistical experiment (Fase 2) ──────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'FASE 2 — Statistical experiment  (N={N_TRIALS} trials, noise ON)')
    print(f'{"═"*60}')
    stat_results, n_stat_gates = run_statistical_experiment(
        ref_interp, solvers, f_system, gate_cfg, x0, n_trials=N_TRIALS)

    stat_save = os.path.join(_RESULTS_DIR, f'gate_experiment_stats_{_CIRCUIT}.npy')
    np.save(stat_save, stat_results, allow_pickle=True)
    print(f'Statistical results saved → {stat_save}')
    print_stats(stat_results, n_stat_gates, N_TRIALS)

    return results, stat_results


if __name__ == '__main__':
    main()
