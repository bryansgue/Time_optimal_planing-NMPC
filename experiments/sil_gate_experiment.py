"""
sil_gate_experiment.py — Software-in-the-Loop gate navigation.

Replaces the internal RK4 of mil_gate_experiment.py with a real MuJoCo
simulator reached through ROS 2 (/quadrotor/odom for state,
/quadrotor/trpy_cmd for thrust+body-rate commands).  Scene reset between
runs is delegated to SimControl.reset() — the MuJoCo plugin re-seeds the
external-force disturbance each reload.

Two controllers (NMPC-Att, NMPC-Full) are evaluated on the circuit selected
via env var CIRCUIT ('fig8' | 'loop').  Metrics match mil_gate_experiment
(pos RMSE, gate crossings, solver timing).

Usage:
    # Terminal 1:  ros2 launch <mujoco quadrotor sim>
    # Terminal 2:
    #     source ~/uav_ws/install/setup.bash
    #     CIRCUIT=fig8  python3 experiments/sil_gate_experiment.py
"""

import os
import sys
import time
import threading
import numpy as np

import rclpy

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

from config.experiment_config import (
    T_S, N_PREDICTION, T_PREDICTION, G, MASS_MUJOCO as MASS, T_MAX,
)

# Stage duration of the MPC (may be coarser than control period T_S).
# Used to map prediction-horizon stages onto the fine reference grid.
STAGE_DT       = T_PREDICTION / N_PREDICTION
STAGE_DT_RATIO = STAGE_DT / T_S                 # ref-samples per MPC stage
from ocp.nmpc_gate_tracker import build_gate_solver_att, build_gate_solver_full
from path_planing.reference_conversion import (
    load_and_convert, build_param_att, build_param_full,
)
from ros2_interface.mujoco_interface import MujocoInterface, wait_for_connection
from ros2_interface.reset_sim import SimControl


# ── Paths (circuit-aware) ─────────────────────────────────────────────────────
_CIRCUIT = os.environ.get('CIRCUIT', 'fig8')
_SUFFIX  = '' if _CIRCUIT == 'fig8' else f'_{_CIRCUIT}'
_GATE_F  = 'gates.npz' if _CIRCUIT == 'fig8' else f'gates_{_CIRCUIT}.npz'
_XREF = os.path.join(_root, 'path_planing', f'xref_optimo_3D_PMM{_SUFFIX}.npy')
_UREF = os.path.join(_root, 'path_planing', f'uref_optimo_3D_PMM{_SUFFIX}.npy')
_TREF = os.path.join(_root, 'path_planing', f'tref_optimo_3D_PMM{_SUFFIX}.npy')
_GATE = os.path.join(_root, 'path_planing', _GATE_F)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(_RESULTS_DIR, exist_ok=True)

T_SIM_EXTRA = 0.5
N_TRIALS    = int(os.environ.get('N_TRIALS', '1'))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def check_gate_crossings(p_hist, t_hist, gate_positions, gate_normals, gate_radius):
    crossings = []
    for gi in range(1, gate_positions.shape[0]):
        gpos, gnorm = gate_positions[gi], gate_normals[gi]
        dists = np.linalg.norm(p_hist - gpos[:, None], axis=0)
        k_min = int(np.argmin(dists))
        r_err = float(dists[k_min])
        dp = p_hist[:, k_min] - gpos
        dp_plane = dp - np.dot(dp, gnorm) * gnorm
        crossings.append({
            'gate':         gi,
            'crossed':      r_err < gate_radius,
            'radial_error': float(np.linalg.norm(dp_plane)),
            'dist_3d':      r_err,
            't_cross':      float(t_hist[k_min]),
            'p_cross':      p_hist[:, k_min],
        })
    return crossings


def warm_start_from_ref(solver, ref_interp, k_start, N):
    N_sim = ref_interp['t'].shape[0]
    for stage in range(N + 1):
        k_ref = min(k_start + int(round(stage * STAGE_DT_RATIO)), N_sim - 1)
        x_ref = np.concatenate([
            ref_interp['p'][:, k_ref], ref_interp['v'][:, k_ref],
            ref_interp['q'][:, k_ref], np.zeros(3),
        ])
        solver.set(stage, 'x', x_ref)
    for stage in range(N):
        k_ref = min(k_start + int(round(stage * STAGE_DT_RATIO)), N_sim - 1)
        T_ref = float(np.clip(ref_interp['T'][k_ref], 0.0, T_MAX))
        solver.set(stage, 'u', np.array([T_ref, 0.0, 0.0, 0.0]))


def _make_param(mode, ref, k):
    if mode == 'att':
        return build_param_att(ref['p'][:, k], ref['v'][:, k], ref['q_att'][:, k])
    return build_param_full(
        ref['p'][:, k], ref['v'][:, k],
        ref['q'][:, k], ref['omega'][:, k], float(ref['T'][k]))


# ══════════════════════════════════════════════════════════════════════════════
#  Single trial on the real MuJoCo sim
# ══════════════════════════════════════════════════════════════════════════════

def run_mode_sil(mode, ref_interp, solver, gate_cfg, muj, sim):
    """Run one closed-loop trial talking to MuJoCo over ROS 2."""
    print(f'\n  [mode={mode}] reset + PD-hold …')
    # Kill any leftover PD thread from the previous mode and zero commands
    # BEFORE resetting the sim, otherwise two publishers race during reset.
    muj.stop_pd_hold()
    muj.send_cmd(0.0, 0.0, 0.0, 0.0)
    time.sleep(0.1)

    # Invalidate cached odom so we wait for a fresh post-reset message.
    with muj._lock:
        muj.connected = False

    sim.reset()
    time.sleep(1.0)
    if not wait_for_connection(muj):
        raise RuntimeError('no odom after reset')

    # Fly to starting gate and hold
    p0 = gate_cfg['gate_positions'][0]
    muj.start_pd_hold(target=p0, mass=MASS, g=G)
    time.sleep(2.0)   # settle

    N_sim = ref_interp['t'].shape[0]
    t_vec = ref_interp['t']

    x_log     = np.zeros((13, N_sim))
    u_log     = np.zeros((4,  N_sim - 1))
    solver_ms = np.zeros(N_sim - 1)

    # Initial state from MuJoCo odom
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)
    x_log[:, 0] = np.concatenate([pos0, vel0, quat0, omega0])

    warm_start_from_ref(solver, ref_interp, 0, N_PREDICTION)
    muj.stop_pd_hold()

    print(f'  [mode={mode}] SiL loop: {N_sim} steps @ {1/T_S:.0f} Hz')
    for k in range(N_sim - 1):
        tic = time.perf_counter()

        x_meas = x_log[:, k]
        solver.set(0, 'lbx', x_meas)
        solver.set(0, 'ubx', x_meas)
        for j in range(N_PREDICTION + 1):
            k_ref = min(k + int(round(j * STAGE_DT_RATIO)), N_sim - 1)
            solver.set(j, 'p', _make_param(mode, ref_interp, k_ref))

        t_solve = time.perf_counter()
        status = solver.solve()
        solver_ms[k] = 1e3 * (time.perf_counter() - t_solve)

        if status not in (0, 2):
            warm_start_from_ref(solver, ref_interp, k, N_PREDICTION)

        u = solver.get(0, 'u')
        u_log[:, k] = u
        T_send = float(np.clip(u[0], 0.0, T_MAX))
        muj.send_cmd(T_send, float(u[1]), float(u[2]), float(u[3]))

        # Wait until T_S elapsed, then read next state
        elapsed = time.perf_counter() - tic
        if elapsed < T_S:
            time.sleep(T_S - elapsed)

        pos, vel, quat, omega = muj.get_state()
        qn = np.linalg.norm(quat) + 1e-12
        x_log[:, k + 1] = np.concatenate([pos, vel, quat / qn, omega])

    # Safety hover
    p_final = muj.get_state()[0]
    muj.start_pd_hold(target=p_final, mass=MASS, g=G)

    # Metrics
    crossings = check_gate_crossings(
        x_log[0:3, :], t_vec,
        gate_cfg['gate_positions'], gate_cfg['gate_normals'],
        float(gate_cfg['gate_radius']))
    pos_rmse   = float(np.sqrt(np.mean(
        np.sum((x_log[0:3, :] - ref_interp['p'])**2, axis=0))))
    gates_ok   = sum(c['crossed'] for c in crossings)
    mean_r_err = float(np.mean([c['dist_3d'] for c in crossings]))

    print(f'    RMSE={pos_rmse:.4f} m | gates {gates_ok}/{len(crossings)} | '
          f'mean dist={mean_r_err:.4f} m | solver mean={solver_ms.mean():.2f} ms')

    return {
        'x': x_log, 'u': u_log, 't': t_vec,
        'solver_ms': solver_ms, 'crossings': crossings,
        'pos_rmse': pos_rmse, 'gates_ok': gates_ok,
        'mean_r_err': mean_r_err,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_pmm = np.load(_TREF)
    T_opt = float(t_pmm[-1])
    T_sim = T_opt + T_SIM_EXTRA
    print(f'[CIRCUIT={_CIRCUIT}]  PMM T_opt={T_opt:.2f} s  T_sim={T_sim:.2f} s')

    ref_interp = load_and_convert(
        xref_path=_XREF, uref_path=_UREF, tref_path=_TREF,
        T_s=T_S, T_sim=T_sim,
    )
    gate_cfg_raw = np.load(_GATE)
    gate_cfg = {
        'gate_positions': gate_cfg_raw['gate_positions'],
        'gate_normals':   gate_cfg_raw['gate_normals'],
        'gate_radius':    float(gate_cfg_raw['gate_radius']),
    }

    # ROS 2 init + MuJoCo interface
    rclpy.init()
    muj = MujocoInterface(node_name='sil_gate_experiment')
    threading.Thread(target=rclpy.spin, args=(muj,), daemon=True).start()
    sim = SimControl(node=muj)

    if not wait_for_connection(muj):
        rclpy.shutdown()
        return

    # Build both solvers (compile once)
    x0 = np.concatenate([
        gate_cfg['gate_positions'][0], np.zeros(3),
        np.array([1., 0., 0., 0.]), np.zeros(3)])
    solvers = {}
    solvers['att'],  _, _ = build_gate_solver_att(x0,  rebuild=False)
    solvers['full'], _, _ = build_gate_solver_full(x0, rebuild=False)

    results = {}
    for trial in range(N_TRIALS):
        print(f'\n{"═"*60}\n  TRIAL {trial+1}/{N_TRIALS}\n{"═"*60}')
        trial_results = {}
        for mode in ('att', 'full'):
            print(f'\n─── mode: {mode.upper()} ───')
            trial_results[mode] = run_mode_sil(
                mode, ref_interp, solvers[mode], gate_cfg, muj, sim)
        if N_TRIALS == 1:
            results = trial_results
        else:
            results[f'trial_{trial}'] = trial_results

    save_path = os.path.join(_RESULTS_DIR, f'sil_gate_results_{_CIRCUIT}.npy')
    np.save(save_path, results, allow_pickle=True)
    print(f'\nSiL results saved → {save_path}')

    muj.stop_pd_hold()
    muj.stop()
    muj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        try: rclpy.shutdown()
        except Exception: pass
    except Exception as e:
        import traceback
        print(f'\nError: {type(e).__name__}: {e}')
        traceback.print_exc()
        try: rclpy.shutdown()
        except Exception: pass
