"""
NMPC gate tracker — two mode-specific solvers for ablation study.

  NMPC-Att  : ℓ = ||p-p*||²_Qp + ||v-v*||²_Qv + ||log(q*⊖q)||²_Qq + ||u-u*||²_R
              q* = yaw-derived (common practice).  Qw = 0 (no rate cost).

  NMPC-Full : ℓ = above + ||ω-ω*||²_Qw
              q*, ω*, T* from full differential flatness (contribution).

Both share the 17-D parameter layout:
  p_param = [p*(3), v*(3), q*(4), ω*(3), T*(1), ω_cmd*(3)]
"""

import os
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_model_rate import f_system_model
from utils.casadi_utils import (
    quat_error_casadi  as quaternion_error,
    quat_log_casadi    as log_quat,
)
from config.experiment_config import T_MAX, T_MIN, T_S, N_PREDICTION, T_PREDICTION

W_MAX_GATE = 10.0   # [rad/s]

# ──────────────────────────────────────────────────────────────────────────────
#  Shared cost weight values
# ──────────────────────────────────────────────────────────────────────────────
_Qp     = [50.0, 50.0, 60.0]
_Qv     = [6.0,  6.0,  6.0]
_Qq     = [15.0, 15.0, 15.0]   # NMPC-Full: full flatness-derived q*(t)
_Qq_att = [1.0,  1.0,  1.0]    # NMPC-Att: yaw-only q* — relaxed to not fight tilt
_Qw     = [0.5,  0.5,  0.5]
_R      = [0.3,  1.0,  1.0,  1.0]
_ZERO3  = [0.0, 0.0, 0.0]


def _create_gate_ocp(
    model_name: str,
    export_dir:  str,
    x0:          np.ndarray,
    N:           int,
    T_h:         float,
    Qp_vals:     list,
    Qv_vals:     list,
    Qq_vals:     list,
    Qw_vals:     list,
    R_vals:      list,
) -> AcadosOcp:
    ocp   = AcadosOcp()
    model, _, _, _ = f_system_model()

    model.name                = model_name
    ocp.model                 = model
    ocp.code_export_directory = export_dir

    np_param = model.p.size()[0]   # 17
    ocp.solver_options.N_horizon = N

    Qp_mat = np.diag(Qp_vals)
    Qv_mat = np.diag(Qv_vals)
    Qq_mat = np.diag(Qq_vals)
    Qw_mat = np.diag(Qw_vals)
    R_mat  = np.diag(R_vals)

    ocp.parameter_values   = np.zeros(np_param)
    ocp.cost.cost_type     = 'EXTERNAL'
    ocp.cost.cost_type_e   = 'EXTERNAL'

    p_ref     = model.p[0:3]
    v_ref     = model.p[3:6]
    q_ref     = model.p[6:10]
    omega_ref = model.p[10:13]
    u_ff      = model.p[13:17]

    ep    = p_ref - model.x[0:3]
    ev    = v_ref - model.x[3:6]
    q_err = quaternion_error(model.x[6:10], q_ref)
    log_q = log_quat(q_err)
    ew    = omega_ref - model.x[10:13]
    eu    = model.u - u_ff

    ocp.model.cost_expr_ext_cost = (
        ep.T    @ Qp_mat @ ep
        + ev.T  @ Qv_mat @ ev
        + log_q.T @ Qq_mat @ log_q
        + ew.T  @ Qw_mat @ ew
        + eu.T  @ R_mat  @ eu
    )
    ocp.model.cost_expr_ext_cost_e = (
        ep.T    @ Qp_mat @ ep
        + ev.T  @ Qv_mat @ ev
        + log_q.T @ Qq_mat @ log_q
        + ew.T  @ Qw_mat @ ew
    )

    ocp.constraints.lbu   = np.array([T_MIN, -W_MAX_GATE, -W_MAX_GATE, -W_MAX_GATE])
    ocp.constraints.ubu   = np.array([T_MAX,  W_MAX_GATE,  W_MAX_GATE,  W_MAX_GATE])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0    = x0

    ocp.solver_options.qp_solver       = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tol             = 1e-4
    ocp.solver_options.tf              = T_h

    return ocp


def _build_solver(ocp: AcadosOcp, solver_json: str, rebuild: bool) -> AcadosOcpSolver:
    so_path = os.path.join(ocp.code_export_directory,
                           f'libacados_ocp_solver_{ocp.model.name}.so')
    if rebuild or not os.path.exists(so_path):
        print(f'[{ocp.model.name}] Generating and compiling...')
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    else:
        print(f'[{ocp.model.name}] Using cached solver ({so_path})')
    return AcadosOcpSolver.create_cython_solver(solver_json)


def build_gate_solver_att(
    x0:      np.ndarray,
    N:       int   = N_PREDICTION,
    T_h:     float = T_PREDICTION,
    rebuild: bool  = False,
) -> tuple:
    """Qp + Qv + Qq (yaw-derived attitude).  Qw = 0 — no angular-rate cost."""
    ocp = _create_gate_ocp(
        model_name='Drone_gate_att',
        export_dir='c_generated_code_gate_att',
        x0=x0, N=N, T_h=T_h,
        Qp_vals=_Qp, Qv_vals=_Qv,
        Qq_vals=_Qq_att, Qw_vals=_ZERO3, R_vals=_R,
    )
    solver = _build_solver(ocp, 'acados_ocp_Drone_gate_att.json', rebuild)
    _, f_system, _, _ = f_system_model()
    return solver, ocp, f_system


def build_gate_solver_full(
    x0:      np.ndarray,
    N:       int   = N_PREDICTION,
    T_h:     float = T_PREDICTION,
    rebuild: bool  = False,
) -> tuple:
    """Full differential flatness — Qp + Qv + Qq + Qw + R all active."""
    ocp = _create_gate_ocp(
        model_name='Drone_gate_full',
        export_dir='c_generated_code_gate_full',
        x0=x0, N=N, T_h=T_h,
        Qp_vals=_Qp, Qv_vals=_Qv,
        Qq_vals=_Qq, Qw_vals=_Qw, R_vals=_R,
    )
    solver = _build_solver(ocp, 'acados_ocp_Drone_gate_full.json', rebuild)
    _, f_system, _, _ = f_system_model()
    return solver, ocp, f_system


# Legacy alias
build_gate_solver = build_gate_solver_full
