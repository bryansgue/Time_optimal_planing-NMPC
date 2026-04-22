"""
Pursuer NMPC — standard tracking controller that chases the evader.

State   x ∈ ℝ¹³ = [p(3), v(3), q(4), ω(3)]
Input   u ∈ ℝ⁴  = [T, wx_cmd, wy_cmd, wz_cmd]

Cost:
    ℓ(x, u) = ||p_p - p_e||²_Q + ||log(q_p ⊖ q_e)||²_K + u^T R u
    V_f(x)  = ||p_p - p_e||²_Q_N

The evader's current state is passed as a parameter (constant reference
over the horizon — updated every control step).

No CBF constraints: the pursuer acts as an unconstrained adversary
within its physical limits.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_model_rate import f_system_model
from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi   as log_quat,
)
from config.experiment_config import (
    T_MAX, T_MIN, W_MAX,
    T_S, N_PREDICTION, T_PREDICTION,
)

# ── Pursuer cost weights ────────────────────────────────────────────────────
Q_POS    = [50.0, 50.0, 50.0]   # position pursuit weight
Q_ORI    = [5.0,  5.0,  5.0]    # orientation weight
R_CTRL   = [0.5,  5.0,  5.0, 5.0]


# ──────────────────────────────────────────────────────────────────────────────
#  OCP builder
# ──────────────────────────────────────────────────────────────────────────────

def create_pursuer_ocp(
    x0:  np.ndarray,
    N:   int   = N_PREDICTION,
    T_h: float = T_PREDICTION,
) -> AcadosOcp:
    """Build the pursuer NMPC OCP (standard tracking toward evader).

    Parameters
    ----------
    x0  : initial state (13,) — pursuer state
    N   : horizon steps
    T_h : horizon length [s]
    """
    ocp = AcadosOcp()

    model, _, _, _ = f_system_model()
    ocp.model = model

    nx = model.x.size()[0]   # 13
    nu = model.u.size()[0]   # 4
    np_par = model.p.size()[0]  # 17 (reference state + control)

    # ── Horizon ───────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N

    # ── Cost matrices ─────────────────────────────────────────────────────
    Q_mat = np.diag(Q_POS)
    K_mat = np.diag(Q_ORI)
    R_mat = np.diag(R_CTRL)

    # ── External cost (reuse existing model parameter p = reference state) ─
    ocp.parameter_values = np.zeros(np_par)
    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # The model already has p[0:3] as reference position, p[6:10] as ref quat
    pos_err  = model.p[0:3] - model.x[0:3]
    quat_err = quaternion_error(model.x[6:10], model.p[6:10])
    log_q    = log_quat(quat_err)

    ocp.model.cost_expr_ext_cost = (
        pos_err.T @ Q_mat @ pos_err
        + log_q.T  @ K_mat @ log_q
        + model.u.T @ R_mat @ model.u
    )
    ocp.model.cost_expr_ext_cost_e = (
        pos_err.T @ Q_mat @ pos_err
        + log_q.T  @ K_mat @ log_q
    )

    # ── Input box constraints ─────────────────────────────────────────────
    ocp.constraints.lbu   = np.array([T_MIN, -W_MAX, -W_MAX, -W_MAX])
    ocp.constraints.ubu   = np.array([T_MAX,  W_MAX,  W_MAX,  W_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ── Initial state ─────────────────────────────────────────────────────
    ocp.constraints.x0 = x0

    # ── Solver options ────────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type  = 'ERK'
    ocp.solver_options.nlp_solver_type  = 'SQP_RTI'
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = T_h

    return ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Solver factory
# ──────────────────────────────────────────────────────────────────────────────

def build_pursuer_solver(
    x0:      np.ndarray,
    N:       int   = N_PREDICTION,
    T_h:     float = T_PREDICTION,
    rebuild: bool  = False,
) -> tuple:
    """Code-generate, compile and return the pursuer solver.

    Uses model name 'Drone_pursuer' and exports to 'c_generated_code_pursuer/'
    to avoid collisions with the baseline Drone_ode_rate_ctrl solver.
    """
    import os
    ocp = create_pursuer_ocp(x0, N=N, T_h=T_h)
    ocp.model.name            = 'Drone_pursuer'
    ocp.code_export_directory = 'c_generated_code_pursuer'
    solver_json               = 'acados_ocp_Drone_pursuer.json'

    so_path = os.path.join(ocp.code_export_directory,
                           'libacados_ocp_solver_Drone_pursuer.so')

    if rebuild or not os.path.exists(so_path):
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    else:
        print(f'[pursuer] using cached solver ({so_path})')

    solver = AcadosOcpSolver.create_cython_solver(solver_json)
    return solver, ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Online solve step
# ──────────────────────────────────────────────────────────────────────────────

def solve_pursuer(
    solver:     AcadosOcpSolver,
    x_pursuer:  np.ndarray,
    x_evader:   np.ndarray,
) -> np.ndarray:
    """Run one SQP-RTI step and return the first control action u*(0).

    Parameters
    ----------
    solver      : compiled pursuer solver
    x_pursuer   : current pursuer state (13,)
    x_evader    : current evader state (13,) — used as constant reference

    Returns
    -------
    u_opt : (4,)  [T, wx_cmd, wy_cmd, wz_cmd]
    """
    # Set evader state as reference parameter at all stages
    p_ref = np.zeros(17)
    p_ref[0:13] = x_evader       # reference state = evader position/attitude
    p_ref[13:17] = 0.0           # reference control = 0 (hover-like)

    for k in range(N_PREDICTION + 1):
        solver.set(k, 'p', p_ref)

    # Fix initial state
    solver.set(0, 'lbx', x_pursuer)
    solver.set(0, 'ubx', x_pursuer)

    status = solver.solve()
    if status not in (0, 2):
        print(f'[pursuer NMPC] solver status {status}')

    return solver.get(0, 'u')
