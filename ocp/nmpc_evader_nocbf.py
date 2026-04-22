"""
nmpc_evader_nocbf.py — Evader NMPC without any safety constraint.

Same augmented state z ∈ ℝ¹⁹ and evasion cost as nmpc_evader_cbf.py,
but with NO HOCBF constraints.  Used as the third comparison curve in
the paper results (alongside NMPC+HOCBF and CBF-QP post-hoc).

State   z ∈ ℝ¹⁹ = [x_e(13), p_p(3), v_p(3)]
Input   u ∈ ℝ⁴  = [T, wx_cmd, wy_cmd, wz_cmd]

Stage cost: ℓ(z, u) = -β·ψ₀(z) + u^T R u
Terminal:   V_f(z)  = -β_N·ψ₀(z)

No nonlinear stage constraints — only box input constraints.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat

from models.quadrotor_wargame_model import wargame_evader_model
from config.experiment_config import (
    T_MAX, T_MIN, W_MAX,
    T_S, N_PREDICTION, T_PREDICTION,
)

# ── Cost defaults (same as HOCBF variant) ─────────────────────────────────
BETA     = 5.0
BETA_N   = 10.0
R_T      = 0.5
R_W      = 5.0


# ──────────────────────────────────────────────────────────────────────────────
#  OCP builder
# ──────────────────────────────────────────────────────────────────────────────

def create_evader_nocbf_ocp(
    z0:     np.ndarray,
    N:      int   = N_PREDICTION,
    T_h:    float = T_PREDICTION,
    beta:   float = BETA,
    beta_N: float = BETA_N,
) -> AcadosOcp:
    """Build evader OCP with evasion cost but NO HOCBF constraints."""
    ocp = AcadosOcp()

    model, _, cbf = wargame_evader_model()
    ocp.model = model

    ocp.solver_options.N_horizon = N

    # ── Cost ─────────────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    psi0_expr = cbf['psi0_expr']
    R_mat     = np.diag([R_T, R_W, R_W, R_W])

    ocp.model.cost_expr_ext_cost   = (
        -beta * psi0_expr
        + model.u.T @ R_mat @ model.u
    )
    ocp.model.cost_expr_ext_cost_e = -beta_N * psi0_expr

    # ── Input box constraints (only) ──────────────────────────────────────
    ocp.constraints.lbu   = np.array([T_MIN, -W_MAX, -W_MAX, -W_MAX])
    ocp.constraints.ubu   = np.array([T_MAX,  W_MAX,  W_MAX,  W_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ── Initial state ─────────────────────────────────────────────────────
    ocp.constraints.x0 = z0

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

def build_evader_nocbf_solver(
    z0:      np.ndarray,
    N:       int   = N_PREDICTION,
    T_h:     float = T_PREDICTION,
    rebuild: bool  = False,
    **kwargs,
) -> tuple:
    """Code-generate, compile and return the no-CBF evader solver."""
    import os
    ocp = create_evader_nocbf_ocp(z0, N=N, T_h=T_h, **kwargs)
    ocp.model.name            = 'Drone_wargame_evader_nocbf'
    ocp.code_export_directory = 'c_generated_code_nocbf'
    solver_json               = f'acados_ocp_{ocp.model.name}.json'

    so_path = os.path.join(ocp.code_export_directory,
                           f'libacados_ocp_solver_{ocp.model.name}.so')

    if rebuild or not os.path.exists(so_path):
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    else:
        print(f'[evader-nocbf] using cached solver ({so_path})')

    solver = AcadosOcpSolver.create_cython_solver(solver_json)
    return solver, ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Online solve step
# ──────────────────────────────────────────────────────────────────────────────

def solve_evader_nocbf(
    solver: AcadosOcpSolver,
    z_meas: np.ndarray,
) -> np.ndarray:
    """Run one SQP-RTI step and return the first control action u*(0).

    Parameters
    ----------
    solver  : compiled no-CBF evader solver
    z_meas  : augmented state [p_e, v_e, q_e, ω_e, p_p, v_p] (19,)

    Returns
    -------
    u_opt : (4,)  [T, wx_cmd, wy_cmd, wz_cmd]
    """
    solver.set(0, 'lbx', z_meas)
    solver.set(0, 'ubx', z_meas)

    status = solver.solve()
    if status not in (0, 2):
        print(f'[evader no-CBF] solver status {status}')

    return solver.get(0, 'u')
