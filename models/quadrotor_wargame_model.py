"""
Augmented quadrotor model for the pursuit-evasion war game.

Evader augmented state  z ∈ ℝ¹⁹:
    z = [p_e(3), v_e(3), q_e(4), ω_e(3),   ← evader (13)
         p_p(3), v_p(3)]                     ← pursuer constant-velocity prediction (6)

Input  u ∈ ℝ⁴  = [T, wx_cmd, wy_cmd, wz_cmd]  (evader body-rate control)

Pursuer dynamics (constant-velocity predictor embedded in state):
    ṗ_p = v_p
    v̇_p = 0

HOCBF auxiliary functions (computed symbolically for constraints):
    ψ₀ = ||p_e - p_p||² - d_min²
    ψ₁ = 2(p_e-p_p)ᵀ(v_e-v_p) + γ₁·ψ₀
    λ  = 2/m · (p_e-p_p)ᵀ R(q_e) e₃          ← coefficient of T in HOCBF condition
    Φ  = 2||v_e-v_p||² - 2(p_e-p_p)ᵀ g e₃    ← T-free terms (pursuer a_p ≈ 0)
"""

import numpy as np
from acados_template import AcadosModel
from casadi import MX, vertcat, Function, norm_2, substitute, jacobian

from utils.casadi_utils import quat_to_rot_casadi as QuatToRot, quat_kinematics_casadi as quat_p
from config.experiment_config import MASS_MUJOCO as MASS, G, TAU_RC


def wargame_evader_model(
    d_min:   float = 0.5,
    gamma1:  float = 1.0,
    gamma2:  float = 1.0,
    r_ws:    float = 4.0,
    p_c:     tuple = (0.0, 0.0, 1.5),
    kappa1:  float = 1.0,
    kappa2:  float = 1.0,
):
    """Build augmented CasADi / acados model for the evader with HOCBF.

    Parameters
    ----------
    d_min   : minimum safe separation distance [m]
    gamma1  : HOCBF decay rate γ₁ for separation ψ₀ → ψ₁
    gamma2  : HOCBF decay rate γ₂ for separation rate condition
    r_ws    : workspace sphere radius [m]
    p_c     : workspace sphere centre (x, y, z) [m]
    kappa1  : HOCBF decay rate κ₁ for workspace φ₀ → φ₁
    kappa2  : HOCBF decay rate κ₂ for workspace rate condition

    Returns
    -------
    model     : AcadosModel  (state z ∈ R^19, input u ∈ R^4)
    f_aug     : casadi.Function(z, u) → ż
    cbf_funcs : dict of symbolic CasADi functions / expressions
    """
    model_name = 'Drone_wargame_evader'

    m      = MASS
    g      = G
    tau_rc = TAU_RC
    e3     = MX([0.0, 0.0, 1.0])

    # ── Evader states ────────────────────────────────────────────────────
    p1e = MX.sym('p1e'); p2e = MX.sym('p2e'); p3e = MX.sym('p3e')
    v1e = MX.sym('v1e'); v2e = MX.sym('v2e'); v3e = MX.sym('v3e')
    q0e = MX.sym('q0e'); q1e = MX.sym('q1e')
    q2e = MX.sym('q2e'); q3e = MX.sym('q3e')
    w1e = MX.sym('w1e'); w2e = MX.sym('w2e'); w3e = MX.sym('w3e')

    # ── Pursuer predicted states (constant-velocity model) ───────────────
    p1p = MX.sym('p1p'); p2p = MX.sym('p2p'); p3p = MX.sym('p3p')
    v1p = MX.sym('v1p'); v2p = MX.sym('v2p'); v3p = MX.sym('v3p')

    # ── Augmented state z ∈ R^19 ─────────────────────────────────────────
    z = vertcat(p1e, p2e, p3e,
                v1e, v2e, v3e,
                q0e, q1e, q2e, q3e,
                w1e, w2e, w3e,
                p1p, p2p, p3p,
                v1p, v2p, v3p)

    # ── State derivative (implicit, for acados) ──────────────────────────
    z_dot = MX.sym('z_dot', 19, 1)

    # ── Evader controls ───────────────────────────────────────────────────
    Tt      = MX.sym('Tt')
    w1_cmd  = MX.sym('w1_cmd')
    w2_cmd  = MX.sym('w2_cmd')
    w3_cmd  = MX.sym('w3_cmd')
    u = vertcat(Tt, w1_cmd, w2_cmd, w3_cmd)

    # ── Useful vectors ────────────────────────────────────────────────────
    p_e   = vertcat(p1e, p2e, p3e)
    v_e   = vertcat(v1e, v2e, v3e)
    q_e   = vertcat(q0e, q1e, q2e, q3e)
    w_e   = vertcat(w1e, w2e, w3e)
    w_cmd = vertcat(w1_cmd, w2_cmd, w3_cmd)
    p_p   = vertcat(p1p, p2p, p3p)
    v_p   = vertcat(v1p, v2p, v3p)

    R_e   = QuatToRot(q_e)          # rotation matrix of evader

    # ── Evader dynamics ───────────────────────────────────────────────────
    p_e_dot = v_e
    v_e_dot = -e3 * g + (R_e @ vertcat(0.0, 0.0, Tt)) / m
    q_e_dot = quat_p(q_e, w_e)
    w_e_dot = (w_cmd - w_e) / tau_rc

    # ── Pursuer constant-velocity dynamics ────────────────────────────────
    p_p_dot = v_p
    v_p_dot = MX.zeros(3, 1)

    # ── Augmented explicit dynamics ───────────────────────────────────────
    f_expl = vertcat(p_e_dot, v_e_dot, q_e_dot, w_e_dot,
                     p_p_dot, v_p_dot)

    # ── HOCBF auxiliary functions ─────────────────────────────────────────
    delta_p   = p_e - p_p                             # separation vector
    delta_v   = v_e - v_p                             # relative velocity

    psi0 = delta_p.T @ delta_p - d_min**2             # ψ₀ = ||p_e-p_p||²-d²
    psi1 = 2.0 * delta_p.T @ delta_v + gamma1 * psi0  # ψ₁ = ψ̇₀ + γ₁ψ₀

    # λ = 2/m (p_e-p_p)ᵀ R(q_e) e₃  (coefficient of T in HOCBF condition)
    lam  = (2.0 / m) * (delta_p.T @ (R_e @ e3))

    # Φ = 2||v_e-v_p||² - 2(p_e-p_p)ᵀ g e₃  (T-free part; a_p ≈ 0)
    Phi  = 2.0 * delta_v.T @ delta_v - 2.0 * delta_p.T @ (g * e3)

    # Full HOCBF rate condition: λ·T + Φ + γ₂·ψ₁ ≥ 0
    hocbf_cond = lam * Tt + Phi + gamma2 * psi1

    # ── Workspace HOCBF ───────────────────────────────────────────────────
    p_c_sym = MX(list(p_c))                            # centre as constant
    delta_c = p_e - p_c_sym                            # p_e - p_c

    phi0 = r_ws**2 - delta_c.T @ delta_c              # φ₀ = r²-||p_e-p_c||²
    phi1 = -2.0 * delta_c.T @ v_e + kappa1 * phi0     # φ₁ = φ̇₀ + κ₁φ₀

    # μ = -2/m (p_e-p_c)ᵀ R(q_e)e₃  (coefficient of T — opposite sign to λ)
    mu   = (-2.0 / m) * (delta_c.T @ (R_e @ e3))
    # Ψ = -2||v_e||² + 2(p_e-p_c)ᵀ g e₃ (T-free part)
    Psi  = -2.0 * v_e.T @ v_e + 2.0 * delta_c.T @ (g * e3)

    # Workspace HOCBF rate condition: μ·T + Ψ + κ₂·φ₁ ≥ 0
    ws_hocbf_cond = mu * Tt + Psi + kappa2 * phi1

    # ── CasADi functions for external use ────────────────────────────────
    f_aug  = Function('f_aug',  [z, u], [f_expl], ['z', 'u'], ['z_dot'])

    cbf_funcs = {
        # ── callable functions ──
        'psi0':      Function('psi0',   [z],    [psi0]),
        'psi1':      Function('psi1',   [z],    [psi1]),
        'phi0':      Function('phi0',   [z],    [phi0]),
        'phi1':      Function('phi1',   [z],    [phi1]),
        'sep_cond':  Function('sep_cond',  [z, u], [hocbf_cond]),
        'ws_cond':   Function('ws_cond',   [z, u], [ws_hocbf_cond]),
        # ── symbolic expressions (embed in acados con_h_expr) ──
        'psi0_expr':     psi0,
        'psi1_expr':     psi1,
        'sep_cond_expr': hocbf_cond,
        'phi0_expr':     phi0,
        'phi1_expr':     phi1,
        'ws_cond_expr':  ws_hocbf_cond,
    }

    # ── Acados model ──────────────────────────────────────────────────────
    model = AcadosModel()
    model.name       = model_name
    model.x          = z
    model.xdot       = z_dot
    model.u          = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = z_dot - f_expl

    return model, f_aug, cbf_funcs
