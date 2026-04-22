"""
NMPC with embedded degree-2 HOCBF for separation safety (evader drone).

State   z ∈ ℝ¹⁹ = [x_e(13), p_p(3), v_p(3)]
Input   u ∈ ℝ⁴  = [T, wx_cmd, wy_cmd, wz_cmd]

Stage constraints (con_h_expr, at every k = 0…N-1)  — 3 HOCBF + soft:
  Separation degree-2 HOCBF (main contribution):
    h₁ = ψ₀(z)                          ≥ 0   [d ≥ d_min]
    h₂ = ψ₁(z)                          ≥ 0   [velocity condition]
    h₃ = λ(z)·T + Φ(z) + γ₂·ψ₁         ≥ 0   [rate condition, affine in T]

  Workspace containment via box state constraints on altitude:
    z_e ≥ Z_FLOOR  (lbx)
    z_e ≤ Z_CEIL   (ubx)

Terminal constraints (k = N): h₁, h₂ ≥ 0

Stage cost: ℓ(z, u) = -β·ψ₀(z) + u^T R u
Terminal:   V_f(z)  = -β_N·ψ₀(z)

Note: The spherical workspace HOCBF (φ₀, φ₁, μ·T+Ψ+κ₂φ₁) has a structural
conflict with gravity when μ<0 (evader above sphere centre), causing T→0 and
free-fall. Altitude containment is instead enforced via hard state box
constraints on z_e ∈ [Z_FLOOR, Z_CEIL] which acados handles robustly.
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat

from models.quadrotor_wargame_model import wargame_evader_model
from config.experiment_config import (
    T_MAX, T_MIN, W_MAX,
    T_S, N_PREDICTION, T_PREDICTION,
    MASS_MUJOCO as MASS, G,
)

# ── Separation CBF defaults ────────────────────────────────────────────────
D_MIN    = 0.5    # [m]  minimum safe separation
GAMMA1   = 1.0    # γ₁ — ψ₁ = ψ̇₀ + γ₁ψ₀
GAMMA2   = 1.0    # γ₂ — rate condition

# ── Workspace altitude box defaults ───────────────────────────────────────
Z_FLOOR  = 0.3    # [m]  minimum altitude (floor constraint on z_e)
Z_CEIL   = 4.0    # [m]  maximum altitude (ceiling constraint on z_e)

# ── Evasion cost defaults ──────────────────────────────────────────────────
BETA     = 20.0   # weight on -ψ₀ in stage cost
BETA_N   = 40.0   # terminal weight on -ψ₀
R_T      = 0.5    # thrust regularization
R_W      = 0.5    # angular-rate regularization (reduced to allow aggressive evasion)


# ──────────────────────────────────────────────────────────────────────────────
#  OCP builder
# ──────────────────────────────────────────────────────────────────────────────

def create_evader_ocp(
    z0:       np.ndarray,
    N:        int   = N_PREDICTION,
    T_h:      float = T_PREDICTION,
    d_min:    float = D_MIN,
    gamma1:   float = GAMMA1,
    gamma2:   float = GAMMA2,
    z_floor:  float = Z_FLOOR,
    z_ceil:   float = Z_CEIL,
    beta:     float = BETA,
    beta_N:   float = BETA_N,
) -> AcadosOcp:
    """Build the acados OCP for the evader with embedded degree-2 HOCBF.

    Parameters
    ----------
    z0      : initial augmented state (19,)
    N       : prediction horizon steps
    T_h     : prediction horizon length [s]
    d_min   : minimum safe separation [m]
    gamma1  : HOCBF decay γ₁
    gamma2  : HOCBF decay γ₂
    z_floor : minimum altitude [m]
    z_ceil  : maximum altitude [m]
    beta    : evasion weight (stage)
    beta_N  : evasion weight (terminal)

    Returns
    -------
    ocp : AcadosOcp
    """
    ocp = AcadosOcp()

    model, _, cbf = wargame_evader_model(d_min=d_min, gamma1=gamma1, gamma2=gamma2)
    ocp.model = model

    nz = model.x.size()[0]   # 19
    nu = model.u.size()[0]   # 4

    # ── Horizon ───────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N

    # ── Cost: EXTERNAL (purely symbolic) ─────────────────────────────────
    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    psi0_expr = cbf['psi0_expr']
    R_mat     = np.diag([R_T, R_W, R_W, R_W])

    # Stage: ℓ = -β·ψ₀ + u^T R u
    ocp.model.cost_expr_ext_cost   = (
        -beta * psi0_expr
        + model.u.T @ R_mat @ model.u
    )
    # Terminal: V_f = -β_N·ψ₀
    ocp.model.cost_expr_ext_cost_e = -beta_N * psi0_expr

    # ── Separation HOCBF stage constraints: h(z, u) ≥ 0 ──────────────────
    psi0_e = cbf['psi0_expr']
    psi1_e = cbf['psi1_expr']
    sep_e  = cbf['sep_cond_expr']

    ocp.model.con_h_expr   = vertcat(psi0_e, psi1_e, sep_e)   # stage k=0..N-1
    ocp.model.con_h_expr_e = vertcat(psi0_e, psi1_e)          # terminal k=N

    INF = 1e8
    # ── Soft stage HOCBF constraints (slack on lower bound) ──────────────
    W_slack  = 5e3   # quadratic penalty weight on violation
    w_slack  = 1e2   # linear penalty weight on violation
    ocp.constraints.lh    = np.zeros(3)
    ocp.constraints.uh    = np.full(3, INF)
    ocp.constraints.Zl    = W_slack * np.ones(3)
    ocp.constraints.Zu    = np.zeros(3)
    ocp.constraints.zl    = w_slack * np.ones(3)
    ocp.constraints.zu    = np.zeros(3)
    ocp.constraints.idxsl = np.arange(3)

    # ── Soft terminal HOCBF constraints ───────────────────────────────────
    ocp.constraints.lh_e    = np.zeros(2)
    ocp.constraints.uh_e    = np.full(2, INF)
    ocp.constraints.Zl_e    = W_slack * np.ones(2)
    ocp.constraints.Zu_e    = np.zeros(2)
    ocp.constraints.zl_e    = w_slack * np.ones(2)
    ocp.constraints.zu_e    = np.zeros(2)
    ocp.constraints.idxsl_e = np.arange(2)

    # ── Input box constraints ─────────────────────────────────────────────
    ocp.constraints.lbu    = np.array([T_MIN, -W_MAX, -W_MAX, -W_MAX])
    ocp.constraints.ubu    = np.array([T_MAX,  W_MAX,  W_MAX,  W_MAX])
    ocp.constraints.idxbu  = np.array([0, 1, 2, 3])

    # ── Altitude box constraints (state): z_e ∈ [Z_FLOOR, Z_CEIL] ────────
    # State index 2 = z_e in augmented state z = [p_e(3), v_e(3), q_e(4), ω_e(3), ...]
    ocp.constraints.lbx    = np.array([z_floor])
    ocp.constraints.ubx    = np.array([z_ceil])
    ocp.constraints.idxbx  = np.array([2])

    # ── Initial state ─────────────────────────────────────────────────────
    ocp.constraints.x0 = z0

    # ── Solver options ────────────────────────────────────────────────────
    # PARTIAL_CONDENSING_HPIPM is more numerically robust with state box
    # constraints and near-degenerate constraint Jacobians (λ≈0).
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N      = N // 5   # condense every 5 stages
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.nlp_solver_type       = 'SQP_RTI'
    ocp.solver_options.tol                   = 1e-3
    ocp.solver_options.tf                    = T_h
    ocp.solver_options.regularize_method     = 'CONVEXIFY'  # handles indefinite Hessian

    return ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Solver factory
# ──────────────────────────────────────────────────────────────────────────────

def build_evader_solver(
    z0:     np.ndarray,
    N:      int   = N_PREDICTION,
    T_h:    float = T_PREDICTION,
    rebuild: bool = False,
    **cbf_kwargs,
) -> tuple:
    """Code-generate, compile and return the acados evader OCP solver.

    If the compiled .so already exists and rebuild=False, skip regeneration
    and just load the cached Cython solver (fast path).

    Returns
    -------
    solver : AcadosOcpSolver
    ocp    : AcadosOcp
    """
    import os
    ocp    = create_evader_ocp(z0, N=N, T_h=T_h, **cbf_kwargs)
    solver_json  = f'acados_ocp_{ocp.model.name}.json'
    so_path      = os.path.join(ocp.code_export_directory,
                                f'libacados_ocp_solver_{ocp.model.name}.so')

    if rebuild or not os.path.exists(so_path):
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    else:
        print(f'[evader] using cached solver ({so_path})')

    solver = AcadosOcpSolver.create_cython_solver(solver_json)
    return solver, ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Warm-start helper
# ──────────────────────────────────────────────────────────────────────────────

def warm_start_hover(solver: AcadosOcpSolver, z0: np.ndarray, N: int = N_PREDICTION):
    """Initialise solver trajectory with a hover warm start.

    Sets all horizon states to z0 (current state) and all inputs to
    approximate hover thrust.  This avoids the cold-start NaN issue
    that arises when the zero warm start violates constraint Jacobians.
    """
    u_hover = np.array([MASS * G, 0.0, 0.0, 0.0])
    for k in range(N + 1):
        solver.set(k, 'x', z0)
    for k in range(N):
        solver.set(k, 'u', u_hover)


# ──────────────────────────────────────────────────────────────────────────────
#  Online solve step
# ──────────────────────────────────────────────────────────────────────────────

def solve_evader(
    solver:  AcadosOcpSolver,
    z_meas:  np.ndarray,
) -> np.ndarray:
    """Run one SQP-RTI step and return the first control action u*(0).

    Parameters
    ----------
    solver  : compiled acados solver
    z_meas  : current augmented state [p_e, v_e, q_e, ω_e, p_p, v_p] (19,)

    Returns
    -------
    u_opt   : optimal control input (4,)  [T, wx_cmd, wy_cmd, wz_cmd]
    """
    solver.set(0, 'lbx', z_meas)
    solver.set(0, 'ubx', z_meas)

    status = solver.solve()
    if status not in (0, 2):   # 0=success, 2=max-iter (acceptable at RTI)
        # Reset warm start to prevent cascade: MINSTEP → NaN → MINSTEP...
        warm_start_hover(solver, z_meas, N_PREDICTION)

    return solver.get(0, 'u')
