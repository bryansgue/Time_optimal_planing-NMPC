"""
wargame_mujoco_node.py — Pursuer vs. Evader Software-in-the-Loop (MuJoCo).

Physics engine : MuJoCo 3  (rigid-body dynamics + contact detection)
Control        : two independent acados SQP-RTI solvers at 100 Hz
                   Evader  → NMPC + degree-2 HOCBF  (proposed)
                   Pursuer → standard NMPC           (adversary)

Forces / torques are injected via xfrc_applied (world frame).
MuJoCo handles gravity, inertia, and drone–drone contact detection.

Scenarios
---------
  free            : pursuer chases evader, no special strategy
  cornering       : pursuer drives evader toward workspace boundary
  speed_advantage : pursuer starts close with fast approach velocity

Modes  (--mode flag)
-----
  hocbf   : Evader NMPC+HOCBF     [proposed method]
  cbfqp   : Evader greedy nominal + post-hoc CBF-QP filter
  nocbf   : Evader NMPC without any safety constraint

Usage
-----
    python wargame_mujoco_node.py --scenario cornering --mode hocbf
    python wargame_mujoco_node.py --scenario cornering --mode cbfqp
    python wargame_mujoco_node.py --scenario free --no-render
"""

import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from config.experiment_config import (
    T_FINAL, T_S, N_PREDICTION, T_PREDICTION,
    T_MAX, T_MIN, W_MAX, G, MASS_MUJOCO as MASS, TAU_RC,
)
from ocp.nmpc_evader_cbf  import build_evader_solver,  solve_evader,  D_MIN, R_WS, P_C
from ocp.nmpc_pursuer     import build_pursuer_solver, solve_pursuer
from ocp.nmpc_evader_nocbf import build_evader_nocbf_solver, solve_evader_nocbf
from wargame_mil_sim      import cbf_qp_filter, nominal_evader

RESULTS_DIR  = Path('results_sim')
MUJOCO_XML   = Path(__file__).parent / 'mujoco_model' / 'two_drones.xml'

# Inertia tensor (matches XML diaginertia)
J_DIAG = np.array([0.010, 0.010, 0.018])


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] → 3×3 rotation matrix (world ← body)."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ])


def extract_state(data: mujoco.MjData, drone_id: int) -> np.ndarray:
    """Extract 13-DOF state from MuJoCo for drone drone_id (0=evader, 1=pursuer).

    Returns x = [p(3), v(3), q(4), ω(3)]
    MuJoCo freejoint layout:
      qpos[7*id : 7*id+7] = [px, py, pz, qw, qx, qy, qz]
      qvel[6*id : 6*id+6] = [vx, vy, vz, wx, wy, wz]
    """
    qo = 7 * drone_id
    vo = 6 * drone_id
    p = data.qpos[qo:qo+3].copy()
    q = data.qpos[qo+3:qo+7].copy()  # [qw, qx, qy, qz]
    v = data.qvel[vo:vo+3].copy()
    w = data.qvel[vo+3:vo+6].copy()
    return np.concatenate([p, v, q, w])


def apply_wrench(data: mujoco.MjData, body_id: int,
                 x_state: np.ndarray, u: np.ndarray) -> None:
    """Compute and apply world-frame wrench to a drone body.

    The body-rate dynamics are:
        v̇ = -g·e₃ + (T/m)·R·e₃          (MuJoCo gravity handles -g·e₃)
        ω̇ = (ω_cmd - ω) / τ_rc

    We apply:
        F_world = T · R · e₃              (thrust in world frame)
        τ_world = R · (J · (ω_cmd - ω) / τ_rc)
    """
    T, wx_cmd, wy_cmd, wz_cmd = u
    q     = x_state[6:10]
    omega = x_state[10:13]
    R     = quat_to_rot(q)

    # Thrust force (world frame)
    F = T * (R @ np.array([0.0, 0.0, 1.0]))

    # Rate-controller torque (body → world)
    omega_cmd = np.array([wx_cmd, wy_cmd, wz_cmd])
    tau_body  = J_DIAG * (omega_cmd - omega) / TAU_RC
    tau_world = R @ tau_body

    data.xfrc_applied[body_id, 0:3] = F
    data.xfrc_applied[body_id, 3:6] = tau_world


def set_initial_state(data: mujoco.MjData, drone_id: int,
                      x: np.ndarray) -> None:
    """Write x = [p, v, q, ω] into MuJoCo qpos/qvel."""
    qo = 7 * drone_id
    vo = 6 * drone_id
    data.qpos[qo:qo+3]   = x[0:3]   # position
    data.qpos[qo+3:qo+7] = x[6:10]  # quaternion [qw, qx, qy, qz]
    data.qvel[vo:vo+3]   = x[3:6]   # linear velocity
    data.qvel[vo+3:vo+6] = x[10:13] # angular velocity


def check_contact(data: mujoco.MjData) -> bool:
    """Return True if any contact between evader and pursuer is detected."""
    return data.ncon > 0


# ══════════════════════════════════════════════════════════════════════════
#  Scenario initial conditions
# ══════════════════════════════════════════════════════════════════════════

def make_init(scenario: str):
    """Return (x0_evader, x0_pursuer) for the chosen scenario."""
    q_hover = np.array([1.0, 0.0, 0.0, 0.0])
    w_zero  = np.zeros(3)

    if scenario == 'free':
        p_e = np.array([0.0,  0.0, 1.5])
        p_p = np.array([3.0,  0.0, 1.5])
        v_e = np.zeros(3); v_p = np.zeros(3)

    elif scenario == 'cornering':
        # Evader near boundary; pursuer drives it into the wall
        p_e = np.array([3.2,  0.0, 1.5])
        p_p = np.array([1.5,  0.0, 1.5])
        v_e = np.zeros(3); v_p = np.zeros(3)

    elif scenario == 'speed_advantage':
        # Pursuer already close and approaching fast
        p_e = np.array([0.0,  0.0, 1.5])
        p_p = np.array([1.8,  0.0, 1.5])
        v_e = np.zeros(3)
        v_p = np.array([-2.5, 0.0, 0.0])   # fast approach

    else:
        raise ValueError(f'Unknown scenario: {scenario!r}')

    x_e = np.concatenate([p_e, v_e, q_hover, w_zero])
    x_p = np.concatenate([p_p, v_p, q_hover, w_zero])
    return x_e, x_p


# ══════════════════════════════════════════════════════════════════════════
#  Main simulation
# ══════════════════════════════════════════════════════════════════════════

def run(scenario: str = 'free',
        mode:     str = 'hocbf',
        render:   bool = True,
        save_dir: Path = RESULTS_DIR) -> dict:
    """Run the wargame SiL and return a metrics dict.

    Parameters
    ----------
    scenario : 'free' | 'cornering' | 'speed_advantage'
    mode     : 'hocbf' | 'cbfqp' | 'nocbf'
    render   : show MuJoCo viewer
    save_dir : directory for saved .npy results
    """
    save_dir.mkdir(exist_ok=True)
    print(f'\n[SiL] scenario={scenario}  mode={mode}')

    x0_e, x0_p = make_init(scenario)

    # ── Build acados solvers ──────────────────────────────────────────────
    z0  = np.concatenate([x0_e, x0_p[0:3], x0_p[3:6]])

    print('[SiL] Building pursuer solver…')
    solver_p, _ = build_pursuer_solver(x0_p)

    if mode == 'hocbf':
        print('[SiL] Building evader NMPC+HOCBF solver…')
        solver_e, _ = build_evader_solver(z0)
    elif mode == 'nocbf':
        print('[SiL] Building evader NMPC (no CBF) solver…')
        solver_e, _ = build_evader_nocbf_solver(z0)
    else:  # cbfqp — pursuer solver already built, no separate evader solver needed
        solver_e = None
        print('[SiL] CBF-QP mode: using greedy nominal + post-hoc filter.')

    # ── Load MuJoCo ───────────────────────────────────────────────────────
    model_mj = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    data_mj  = mujoco.MjData(model_mj)

    evader_id  = model_mj.body('evader').id
    pursuer_id = model_mj.body('pursuer').id

    set_initial_state(data_mj, 0, x0_e)
    set_initial_state(data_mj, 1, x0_p)
    mujoco.mj_forward(model_mj, data_mj)

    # ── Logging ───────────────────────────────────────────────────────────
    n_steps  = int(T_FINAL / T_S)
    p_c_np   = np.array(P_C)
    log = {k: [] for k in [
        't', 'dist', 'p_e', 'p_p', 'u_e', 'u_p',
        'psi0', 'phi0', 'contact', 'solve_ms',
    ]}

    # CBF functions for logging
    from models.quadrotor_wargame_model import wargame_evader_model
    _, _, cbf_fns = wargame_evader_model()
    f_psi0 = cbf_fns['psi0']
    f_phi0 = cbf_fns['phi0']

    # ── Simulation loop ───────────────────────────────────────────────────
    viewer_ctx = None
    if render:
        viewer_ctx = mujoco.viewer.launch_passive(model_mj, data_mj)

    print(f'[SiL] Running {n_steps} steps ({T_FINAL} s)…')
    t_wall0 = time.time()

    for k in range(n_steps):
        t_sim  = k * T_S
        x_e    = extract_state(data_mj, 0)
        x_p    = extract_state(data_mj, 1)

        # ── Evader control ────────────────────────────────────────────────
        t0_solve = time.perf_counter()
        if mode == 'hocbf':
            z_meas = np.concatenate([x_e, x_p[0:3], x_p[3:6]])
            u_e    = solve_evader(solver_e, z_meas)
        elif mode == 'nocbf':
            z_meas = np.concatenate([x_e, x_p[0:3], x_p[3:6]])
            u_e    = solve_evader_nocbf(solver_e, z_meas)
        else:  # cbfqp
            u_nom = nominal_evader(x_e, x_p)
            u_e   = cbf_qp_filter(u_nom, x_e, x_p, D_MIN, R_WS, p_c_np)
        solve_ms = 1e3 * (time.perf_counter() - t0_solve)

        # ── Pursuer control ───────────────────────────────────────────────
        u_p = solve_pursuer(solver_p, x_p, x_e)

        # ── Clip inputs ───────────────────────────────────────────────────
        u_e = np.clip(u_e, [T_MIN, -W_MAX, -W_MAX, -W_MAX],
                           [T_MAX,  W_MAX,  W_MAX,  W_MAX])
        u_p = np.clip(u_p, [T_MIN, -W_MAX, -W_MAX, -W_MAX],
                           [T_MAX,  W_MAX,  W_MAX,  W_MAX])

        # ── Apply forces and step physics ─────────────────────────────────
        apply_wrench(data_mj, evader_id,  x_e, u_e)
        apply_wrench(data_mj, pursuer_id, x_p, u_p)

        # MuJoCo timestep is 2 ms; control runs at 10 ms → 5 physics steps
        n_phys = max(1, round(T_S / model_mj.opt.timestep))
        for _ in range(n_phys):
            mujoco.mj_step(model_mj, data_mj)
            # Re-apply wrench each physics step (xfrc_applied resets each step)
            apply_wrench(data_mj, evader_id,  x_e, u_e)
            apply_wrench(data_mj, pursuer_id, x_p, u_p)

        # ── Log ───────────────────────────────────────────────────────────
        z_log  = np.concatenate([x_e, x_p[0:3], x_p[3:6]])
        dist   = float(np.linalg.norm(x_e[0:3] - x_p[0:3]))
        psi0_v = float(f_psi0(z_log))
        phi0_v = float(f_phi0(z_log))

        log['t'].append(t_sim)
        log['dist'].append(dist)
        log['p_e'].append(x_e[0:3].copy())
        log['p_p'].append(x_p[0:3].copy())
        log['u_e'].append(u_e.copy())
        log['u_p'].append(u_p.copy())
        log['psi0'].append(psi0_v)
        log['phi0'].append(phi0_v)
        log['contact'].append(check_contact(data_mj))
        log['solve_ms'].append(solve_ms)

        if render and viewer_ctx is not None:
            viewer_ctx.sync()

    wall = time.time() - t_wall0

    if viewer_ctx is not None:
        viewer_ctx.close()

    # ── Convert lists to arrays ───────────────────────────────────────────
    for k in ['dist', 'psi0', 'phi0', 'solve_ms']:
        log[k] = np.array(log[k])
    log['t']       = np.array(log['t'])
    log['contact'] = np.array(log['contact'])
    log['p_e']     = np.array(log['p_e'])
    log['p_p']     = np.array(log['p_p'])
    log['u_e']     = np.array(log['u_e'])
    log['u_p']     = np.array(log['u_p'])

    # ── Summary ───────────────────────────────────────────────────────────
    n_viol_sep = int(np.sum(log['psi0'] < 0))
    n_viol_ws  = int(np.sum(log['phi0'] < 0))
    n_contact  = int(np.sum(log['contact']))
    mean_ms    = float(np.mean(log['solve_ms']))
    max_ms     = float(np.max(log['solve_ms']))
    min_dist   = float(np.min(log['dist']))

    print(f'\n[SiL] ── Results ({scenario} / {mode}) ──')
    print(f'  Wall time     : {wall:.1f} s  (RTF = {T_FINAL/wall:.1f}×)')
    print(f'  Sep. violations: {n_viol_sep}/{n_steps}  ({100*n_viol_sep/n_steps:.1f}%)')
    print(f'  WS  violations : {n_viol_ws}/{n_steps}   ({100*n_viol_ws/n_steps:.1f}%)')
    print(f'  Physical contacts: {n_contact}')
    print(f'  Min separation   : {min_dist:.3f} m  (d_min={D_MIN} m)')
    print(f'  Solve time [evader]: mean={mean_ms:.2f} ms, max={max_ms:.2f} ms')

    # ── Save ──────────────────────────────────────────────────────────────
    fname = save_dir / f'wargame_{scenario}_{mode}_sil.npy'
    np.save(fname, log)
    print(f'[SiL] Saved → {fname}')

    return log


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wargame MuJoCo SiL: pursuer vs evader')
    parser.add_argument('--scenario', default='free',
                        choices=['free', 'cornering', 'speed_advantage'])
    parser.add_argument('--mode', default='hocbf',
                        choices=['hocbf', 'cbfqp', 'nocbf'])
    parser.add_argument('--no-render', action='store_true',
                        help='Disable MuJoCo viewer (faster, headless)')
    args = parser.parse_args()

    run(scenario=args.scenario,
        mode=args.mode,
        render=not args.no_render)
