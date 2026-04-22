"""
War-game Model-in-the-Loop simulation.

Both drones are integrated with RK4 using the quadrotor ODE directly —
no MuJoCo, no ROS 2.  This is the first validation step.

Two NMPC controllers run at 100 Hz:
  Pursuer  → standard NMPC, minimises distance to evader
  Evader   → NMPC + 2× degree-2 HOCBF (separation + workspace)

Scenarios
---------
  'free'      : pursuer chases evader with no special strategy
  'cornering' : pursuer tries to pin evader against workspace boundary

Usage
-----
    python wargame_mil_sim.py [--scenario free|cornering] [--no-cbf-qp]
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.quadrotor_wargame_model import wargame_evader_model
from models.quadrotor_model_rate    import f_system_model
from ocp.nmpc_evader_cbf import (
    build_evader_solver, solve_evader, warm_start_hover,
    D_MIN, Z_FLOOR, Z_CEIL,
    N_PREDICTION as _N_PRED,
)
from ocp.nmpc_pursuer import build_pursuer_solver, solve_pursuer
from config.experiment_config import (
    T_FINAL, T_S, N_PREDICTION, T_PREDICTION, T_MAX, T_MIN, W_MAX, G,
)

RESULTS_DIR = Path('results_sim')
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  RK4 plant integration
# ══════════════════════════════════════════════════════════════════════════

def make_rk4(f_casadi):
    """Return a numpy RK4 step closure around a CasADi Function f(x, u)→ẋ."""
    def _rk4(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = np.array(f_casadi(x, u)).flatten()
        k2 = np.array(f_casadi(x + dt / 2 * k1, u)).flatten()
        k3 = np.array(f_casadi(x + dt / 2 * k2, u)).flatten()
        k4 = np.array(f_casadi(x + dt * k3, u)).flatten()
        x_next = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        # Re-normalise quaternion (indices 6:10)
        q = x_next[6:10]
        x_next[6:10] = q / np.linalg.norm(q)
        return x_next
    return _rk4


# ══════════════════════════════════════════════════════════════════════════
#  Initial conditions per scenario
# ══════════════════════════════════════════════════════════════════════════

def make_init(scenario: str):
    """Return (x0_evader, x0_pursuer) for the chosen scenario."""
    # Neutral hover attitude
    q_hover = np.array([1.0, 0.0, 0.0, 0.0])
    w_zero  = np.zeros(3)

    if scenario == 'free':
        # Evader near centre; pursuer 3 m away
        p_e = np.array([0.0,  0.0, 1.5])
        p_p = np.array([3.0,  0.0, 1.5])
    elif scenario == 'cornering':
        # Evader already near workspace boundary; pursuer inside pushing it out
        p_e = np.array([3.2,  0.0, 1.5])   # close to r_ws=4 m boundary
        p_p = np.array([1.5,  0.0, 1.5])   # pursuer between centre and evader
    elif scenario == 'speed_advantage':
        # Pursuer already close and approaching fast — hardest for the evader
        p_e = np.array([0.0,  0.0, 1.5])
        p_p = np.array([1.8,  0.0, 1.5])   # only 1.3 m away (d_min = 0.5)
        v_p_init = np.array([-2.5, 0.0, 0.0])  # fast approach toward evader
        x_e = np.concatenate([p_e, np.zeros(3), q_hover, w_zero])
        x_p = np.concatenate([p_p, v_p_init,   q_hover, w_zero])
        return x_e, x_p
    else:
        raise ValueError(f'Unknown scenario: {scenario}')

    x_e = np.concatenate([p_e, np.zeros(3), q_hover, w_zero])
    x_p = np.concatenate([p_p, np.zeros(3), q_hover, w_zero])
    return x_e, x_p


# ══════════════════════════════════════════════════════════════════════════
#  Nominal evasion policy (used as input to the CBF-QP filter)
# ══════════════════════════════════════════════════════════════════════════

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """q = [qw, qx, qy, qz] → 3×3 rotation matrix (world ← body)."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ])


def nominal_evader(x_e: np.ndarray, x_p: np.ndarray) -> np.ndarray:
    """Greedy nominal evasion: tilt body toward escape direction + near-max thrust.

    This is the nominal policy fed into the CBF-QP filter.  It is intentionally
    simple (no receding-horizon) so that any failure in the cornering / speed-
    advantage scenarios is attributable to the post-hoc CBF architecture, not
    to a bad nominal.

    Returns u_nom = [T, wx_cmd, wy_cmd, wz_cmd].
    """
    from config.experiment_config import G, T_MAX, W_MAX, MASS_MUJOCO as m
    p_e = x_e[0:3]
    p_p = x_p[0:3]
    q_e = x_e[6:10]
    w_e = x_e[10:13]

    delta   = p_e - p_p
    dist    = max(np.linalg.norm(delta), 1e-3)
    esc_dir = delta / dist          # unit vector away from pursuer

    # Tilt the body toward esc_dir: command body rates to align body-z with esc_dir
    R   = _quat_to_rot(q_e)
    bz  = R[:, 2]                   # current body-z in world frame
    axis = np.cross(bz, esc_dir)    # rotation axis (body frame correction)
    Kp  = 8.0                       # proportional gain for rate command
    w_cmd = Kp * axis
    w_cmd = np.clip(w_cmd, -W_MAX, W_MAX)

    # Thrust: hover + escape boost
    T_nom = min(m * (G + 3.0), T_MAX)  # gravity comp + 3 m/s² escape accel

    return np.array([T_nom, w_cmd[0], w_cmd[1], w_cmd[2]])


# ══════════════════════════════════════════════════════════════════════════
#  Post-hoc CBF-QP baseline
# ══════════════════════════════════════════════════════════════════════════

def cbf_qp_filter(u_nom: np.ndarray, x_e: np.ndarray, x_p: np.ndarray,
                  d_min: float) -> np.ndarray:
    """Minimal-norm CBF-QP safety filter (post-hoc, separation only).

    Solves:  min ||u - u_nom||²
             s.t.  ψ̇₀_sep ≥ -γ·ψ₀_sep   [separation, relative degree 1 approx]
    Condition is linear in u[0]=T.
    """
    from config.experiment_config import MASS_MUJOCO as m

    p_e = x_e[0:3]; v_e = x_e[3:6]
    p_p = x_p[0:3]; v_p = x_p[3:6]

    delta_p = p_e - p_p
    psi0 = delta_p @ delta_p - d_min**2
    dpsi0_dt_free = 2.0 * delta_p @ (v_e - v_p)

    gamma = 1.0
    a1 = (2.0 / m) * delta_p[2]
    b1 = -gamma * psi0 - dpsi0_dt_free

    T_nom  = u_nom[0]
    T_safe = T_nom

    if abs(a1) > 1e-6:
        if a1 > 0:
            T_safe = max(T_safe, b1 / a1)
        else:
            T_safe = min(T_safe, b1 / a1)

    T_safe = np.clip(T_safe, T_MIN, T_MAX)
    u_safe = u_nom.copy()
    u_safe[0] = T_safe
    return u_safe


# ══════════════════════════════════════════════════════════════════════════
#  Main simulation
# ══════════════════════════════════════════════════════════════════════════

def run(scenario: str = 'free', compare_cbf_qp: bool = True):
    print(f'\n[MiL] Scenario: {scenario}')

    # ── Build solvers ─────────────────────────────────────────────────────
    x0_e, x0_p = make_init(scenario)
    z0 = np.concatenate([x0_e, x0_p[0:3], x0_p[3:6]])

    print('[MiL] Building evader solver (NMPC+HOCBF)…')
    solver_e, _ = build_evader_solver(z0)
    warm_start_hover(solver_e, z0, _N_PRED)

    print('[MiL] Building pursuer solver…')
    solver_p, _ = build_pursuer_solver(x0_p)

    # ── RK4 plant functions ───────────────────────────────────────────────
    _, f_e, _, _ = f_system_model()
    _, f_p, _, _ = f_system_model()
    rk4_e = make_rk4(f_e)
    rk4_p = make_rk4(f_p)

    # ── Logging ───────────────────────────────────────────────────────────
    n_steps = int(T_FINAL / T_S)
    log = {k: [] for k in ['t', 'dist', 'p_e', 'p_p', 'u_e', 'u_p',
                            'psi0', 'alt_e', 'safe_sep', 'safe_alt',
                            'solve_ms']}
    if compare_cbf_qp:
        log_qp = {k: [] for k in ['t', 'dist', 'psi0', 'alt_e',
                                   'safe_sep', 'safe_alt']}

    # ── Recover model CBF function for separation logging ─────────────────
    _, _, cbf_fns = wargame_evader_model()
    f_psi0 = cbf_fns['psi0']

    # ── State copies ──────────────────────────────────────────────────────
    x_e  = x0_e.copy()
    x_p  = x0_p.copy()
    if compare_cbf_qp:
        x_e_qp = x0_e.copy()
        x_p_qp = x0_p.copy()

    print(f'[MiL] Running {n_steps} steps…')
    t0 = time.time()

    for k in range(n_steps):
        t_sim = k * T_S

        # ── Evader NMPC+HOCBF ─────────────────────────────────────────────
        z_meas = np.concatenate([x_e, x_p[0:3], x_p[3:6]])
        t_solve = time.perf_counter()
        u_e = solve_evader(solver_e, z_meas)
        solve_ms = 1e3 * (time.perf_counter() - t_solve)

        # ── Pursuer NMPC ──────────────────────────────────────────────────
        u_p = solve_pursuer(solver_p, x_p, x_e)

        # ── Clip & integrate ──────────────────────────────────────────────
        u_e = np.clip(u_e, [T_MIN, -W_MAX, -W_MAX, -W_MAX],
                           [T_MAX,  W_MAX,  W_MAX,  W_MAX])
        u_p = np.clip(u_p, [T_MIN, -W_MAX, -W_MAX, -W_MAX],
                           [T_MAX,  W_MAX,  W_MAX,  W_MAX])

        x_e_next = rk4_e(x_e, u_e, T_S)
        x_p_next = rk4_p(x_p, u_p, T_S)

        # ── Log ───────────────────────────────────────────────────────────
        z_log    = np.concatenate([x_e, x_p[0:3], x_p[3:6]])
        psi0_val = float(f_psi0(z_log))
        alt_e    = x_e[2]            # evader altitude
        dist     = np.linalg.norm(x_e[0:3] - x_p[0:3])

        log['t'].append(t_sim)
        log['dist'].append(dist)
        log['p_e'].append(x_e[0:3].copy())
        log['p_p'].append(x_p[0:3].copy())
        log['u_e'].append(u_e.copy())
        log['u_p'].append(u_p.copy())
        log['psi0'].append(psi0_val)
        log['alt_e'].append(alt_e)
        log['safe_sep'].append(psi0_val >= 0)
        log['safe_alt'].append(Z_FLOOR <= alt_e <= Z_CEIL)
        log['solve_ms'].append(solve_ms)

        # ── CBF-QP baseline ───────────────────────────────────────────────
        if compare_cbf_qp:
            u_p_qp  = solve_pursuer(solver_p, x_p_qp, x_e_qp)
            u_e_nom = nominal_evader(x_e_qp, x_p_qp)
            u_e_qp  = cbf_qp_filter(u_e_nom, x_e_qp, x_p_qp, D_MIN)

            x_e_qp_next = rk4_e(x_e_qp, u_e_qp, T_S)
            x_p_qp_next = rk4_p(x_p_qp, u_p_qp, T_S)

            z_qp     = np.concatenate([x_e_qp, x_p_qp[0:3], x_p_qp[3:6]])
            dist_qp  = np.linalg.norm(x_e_qp[0:3] - x_p_qp[0:3])
            psi0_qp  = float(f_psi0(z_qp))
            alt_qp   = x_e_qp[2]

            log_qp['t'].append(t_sim)
            log_qp['dist'].append(dist_qp)
            log_qp['psi0'].append(psi0_qp)
            log_qp['alt_e'].append(alt_qp)
            log_qp['safe_sep'].append(psi0_qp >= 0)
            log_qp['safe_alt'].append(Z_FLOOR <= alt_qp <= Z_CEIL)

            x_e_qp = x_e_qp_next
            x_p_qp = x_p_qp_next

        x_e = x_e_next
        x_p = x_p_next

    wall = time.time() - t0
    rtf  = T_FINAL / wall

    # ── Summary ───────────────────────────────────────────────────────────
    viol_sep = sum(1 for v in log['safe_sep'] if not v)
    viol_alt = sum(1 for v in log['safe_alt'] if not v)
    mean_ms  = np.mean(log['solve_ms'])
    max_ms   = np.max(log['solve_ms'])
    min_dist = np.min(log['dist'])

    print(f'\n[MiL] ── Results ({scenario}) ──')
    print(f'  Wall time:         {wall:.1f} s  (RTF={rtf:.1f}x)')
    print(f'  NMPC+HOCBF sep.  violations: {viol_sep}/{n_steps} '
          f'({100*viol_sep/n_steps:.1f}%)  min_dist={min_dist:.3f} m')
    print(f'  NMPC+HOCBF alt.  violations: {viol_alt}/{n_steps}  '
          f'({100*viol_alt/n_steps:.1f}%)')
    print(f'  Solve time: mean={mean_ms:.2f} ms, max={max_ms:.2f} ms')

    if compare_cbf_qp:
        v_sep_qp = sum(1 for v in log_qp['safe_sep'] if not v)
        v_alt_qp = sum(1 for v in log_qp['safe_alt'] if not v)
        min_dist_qp = np.min(log_qp['dist'])
        print(f'  CBF-QP   sep.  violations: {v_sep_qp}/{n_steps} '
              f'({100*v_sep_qp/n_steps:.1f}%)  min_dist={min_dist_qp:.3f} m')
        print(f'  CBF-QP   alt.  violations: {v_alt_qp}/{n_steps}  '
              f'({100*v_alt_qp/n_steps:.1f}%)')

    # ── Plots ─────────────────────────────────────────────────────────────
    ts = log['t']
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # ── Plot 1: separation distance ───────────────────────────────────────
    ax = axes[0]
    ax.plot(ts, log['dist'], 'b', label='NMPC+HOCBF')
    if compare_cbf_qp:
        ax.plot(log_qp['t'], log_qp['dist'], 'r--', alpha=0.7, label='CBF-QP')
    ax.axhline(D_MIN, color='k', linestyle=':', label=f'$d_{{\\min}}={D_MIN}$ m')
    ax.set_ylabel('Separation [m]')
    ax.set_title(f'Scenario: {scenario} — Pursuer–Evader Separation')
    ax.legend(); ax.grid(True)

    # ── Plot 2: separation barrier ψ₀ ────────────────────────────────────
    ax = axes[1]
    ax.plot(ts, log['psi0'], 'b', label=r'$\psi_0$ NMPC+HOCBF')
    if compare_cbf_qp:
        ax.plot(log_qp['t'], log_qp['psi0'], 'r--', alpha=0.7,
                label=r'$\psi_0$ CBF-QP')
    ax.axhline(0, color='k', linestyle=':', linewidth=1.0)
    ax.set_ylabel(r'$\psi_0 = d^2 - d_{\min}^2$')
    ax.set_title(r'Separation barrier $\psi_0$ (must remain $\geq 0$)')
    ax.legend(fontsize=8); ax.grid(True)

    # ── Plot 3: solve time ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(ts, log['solve_ms'], 'b', linewidth=0.8)
    ax.axhline(10.0, color='r', linestyle='--', label='10 ms budget')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Solve time [ms]')
    ax.set_title('Evader NMPC+HOCBF solve time')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    fname = RESULTS_DIR / f'wargame_{scenario}_mil.png'
    plt.savefig(fname, dpi=150)
    print(f'[MiL] Saved → {fname}')
    plt.close(fig)

    # ── Save .npy log ─────────────────────────────────────────────────────
    for key in ['dist', 'psi0', 'alt_e', 'solve_ms']:
        log[key] = np.array(log[key])
    np.save(RESULTS_DIR / f'wargame_{scenario}_hocbf_mil.npy', log)
    if compare_cbf_qp:
        for key in ['dist', 'psi0', 'alt_e']:
            log_qp[key] = np.array(log_qp[key])
        np.save(RESULTS_DIR / f'wargame_{scenario}_cbfqp_mil.npy', log_qp)
        print(f'[MiL] Logs saved to results_sim/')

    return log, (log_qp if compare_cbf_qp else None)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='free',
                        choices=['free', 'cornering', 'speed_advantage'])
    parser.add_argument('--no-cbf-qp', action='store_true',
                        help='Skip CBF-QP baseline comparison')
    args = parser.parse_args()

    run(scenario=args.scenario, compare_cbf_qp=not args.no_cbf_qp)
