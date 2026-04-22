"""
reference_conversion.py — Differential flatness map: PMM → full quadrotor reference.

Given PMM output (p, v, a) and jerk j = Δa/Δt, computes:
  T_des   : desired collective thrust      [N]
  q_des   : desired quaternion             [qw, qx, qy, qz]
  omega_des: desired body angular velocity  [rad/s]

Then builds the 17-D acados parameter vector used by nmpc_gate_tracker:
  p_param = [p(3), v(3), q(4), omega(3), T(1), w_cmd(3)]

Three reference modes
---------------------
  'pos'   — p*(t) + v*(t) from PMM directly.  Qq=0, Qω=0 in cost.
  'att'   — p*(t) + v*(t) + yaw-derived q*(t) [q_att].  Qω=0 in cost.
  'full'  — all components from differential flatness (q*, ω*, T*).
"""

import numpy as np
from scipy.interpolate import CubicSpline

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
G        = 9.81
MASS     = 1.08
E3       = np.array([0.0, 0.0, 1.0])
Q_HOVER  = np.array([1.0, 0.0, 0.0, 0.0])   # [qw, qx, qy, qz]
T_HOVER  = MASS * G


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_from_zb_yaw(z_b: np.ndarray, yaw: float) -> np.ndarray:
    """Build body rotation matrix R = [x_b | y_b | z_b] from desired z_b and yaw.

    Parameters
    ----------
    z_b : ndarray (3,)  — desired body z-axis (unit vector)
    yaw : float         — desired yaw angle [rad]

    Returns
    -------
    R : ndarray (3, 3)  — rotation matrix (columns = body axes in world frame)
    """
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b = y_b / (np.linalg.norm(y_b) + 1e-12)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / (np.linalg.norm(x_b) + 1e-12)
    return np.column_stack([x_b, y_b, z_b])


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion [qw, qx, qy, qz] (Shepperd method).

    Parameters
    ----------
    R : ndarray (3, 3)

    Returns
    -------
    q : ndarray (4,)  — [qw, qx, qy, qz]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz])
    return q / (np.linalg.norm(q) + 1e-12)


def vee(S: np.ndarray) -> np.ndarray:
    """Extract axial vector from skew-symmetric matrix (vee map).

    Parameters
    ----------
    S : ndarray (3, 3)  — skew-symmetric matrix

    Returns
    -------
    v : ndarray (3,)
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def yaw_quaternion(yaw: float) -> np.ndarray:
    """Pure yaw rotation around world z-axis: q = [cos(ψ/2), 0, 0, sin(ψ/2)]."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def ensure_quat_hemisphere(quats: np.ndarray) -> np.ndarray:
    """Flip quaternions so consecutive ones stay in the same hemisphere.

    Parameters
    ----------
    quats : ndarray (4, N)

    Returns
    -------
    quats : ndarray (4, N)  — hemisphere-corrected copy
    """
    q = quats.copy()
    for i in range(1, q.shape[1]):
        if np.dot(q[:, i], q[:, i - 1]) < 0.0:
            q[:, i] *= -1.0
    return q


# ══════════════════════════════════════════════════════════════════════════════
#  Flatness map (vectorised over the full PMM trajectory)
# ══════════════════════════════════════════════════════════════════════════════

def flat_map_trajectory(
    X_opt: np.ndarray,
    U_opt: np.ndarray,
    t_pmm: np.ndarray,
    mass:  float = MASS,
    g:     float = G,
) -> dict:
    """Apply differential flatness to a full PMM trajectory.

    Parameters
    ----------
    X_opt : ndarray (6, N+1)  — PMM states  [p(3), v(3)]
    U_opt : ndarray (3, N)    — PMM controls [ax, ay, az]
    t_pmm : ndarray (N+1,)    — time vector
    mass  : float             — drone mass [kg]
    g     : float             — gravity [m/s²]

    Returns
    -------
    ref : dict with keys
        'p'     : ndarray (3, N+1)
        'v'     : ndarray (3, N+1)
        'q'     : ndarray (4, N+1)   — [qw, qx, qy, qz]
        'omega' : ndarray (3, N+1)
        'T'     : ndarray (N+1,)
        'yaw'   : ndarray (N+1,)
        't'     : ndarray (N+1,)
    """
    N1   = X_opt.shape[1]      # N+1
    N    = U_opt.shape[1]      # N
    dt   = np.diff(t_pmm)      # (N,)

    p    = X_opt[0:3, :]       # (3, N+1)
    v    = X_opt[3:6, :]       # (3, N+1)

    # Acceleration at each node: use U_opt for k=0..N-1, repeat last for k=N
    a = np.hstack([U_opt, U_opt[:, -1:]])   # (3, N+1)

    # Jerk via finite differences (central for interior, forward/backward at ends)
    jerk = np.zeros((3, N1))
    for k in range(N1):
        if k == 0:
            jerk[:, k] = (U_opt[:, 1] - U_opt[:, 0]) / dt[0]   if N > 1 else np.zeros(3)
        elif k == N:
            jerk[:, k] = (U_opt[:, -1] - U_opt[:, -2]) / dt[-1] if N > 1 else np.zeros(3)
        else:
            dt_c = t_pmm[k + 1] - t_pmm[k - 1]
            jerk[:, k] = (U_opt[:, min(k, N-1)] - U_opt[:, k - 1]) / (dt_c / 2 + 1e-12)

    # Yaw from velocity direction; fallback to previous when |v_xy| small
    yaw = np.zeros(N1)
    for k in range(N1):
        vxy = np.linalg.norm(v[0:2, k])
        if vxy > 0.1:
            yaw[k] = np.arctan2(v[1, k], v[0, k])
        else:
            yaw[k] = yaw[k - 1] if k > 0 else 0.0

    # ── Desired specific force, thrust, body z-axis ───────────────────────
    f_des = a + g * E3[:, None]   # (3, N+1)
    f_mag = np.linalg.norm(f_des, axis=0) + 1e-9   # (N+1,)
    T_des = mass * f_mag          # (N+1,)
    z_b   = f_des / f_mag         # (3, N+1)  unit vectors

    # ── Desired rotation matrices and quaternions ─────────────────────────
    R_all = np.zeros((3, 3, N1))
    q_all = np.zeros((4, N1))
    for k in range(N1):
        R_all[:, :, k] = rotation_matrix_from_zb_yaw(z_b[:, k], yaw[k])
        q_all[:, k]    = rotation_to_quaternion(R_all[:, :, k])

    q_all = ensure_quat_hemisphere(q_all)

    # ── Yaw-only quaternions (NMPC-Att reference) ─────────────────────────
    # Encodes only heading from velocity direction; tilt = identity.
    # This represents the common-practice incomplete attitude reference.
    q_att_all = np.zeros((4, N1))
    for k in range(N1):
        q_att_all[:, k] = yaw_quaternion(yaw[k])

    # ── Desired angular velocity via numerical diff of R ──────────────────
    omega_all = np.zeros((3, N1))
    for k in range(N1):
        if k < N1 - 1:
            dt_k   = t_pmm[k + 1] - t_pmm[k]
            R_dot  = (R_all[:, :, k + 1] - R_all[:, :, k]) / (dt_k + 1e-12)
        else:
            dt_k   = t_pmm[-1] - t_pmm[-2]
            R_dot  = (R_all[:, :, -1] - R_all[:, :, -2]) / (dt_k + 1e-12)
        Omega_skew    = R_all[:, :, k].T @ R_dot
        omega_all[:, k] = vee(Omega_skew)

    # ── Smooth omega to suppress numerical-diff spikes ───────────────────
    kernel = np.ones(5) / 5.0
    for i in range(3):
        omega_all[i, :] = np.convolve(omega_all[i, :], kernel, mode='same')

    return {
        'p':     p,
        'v':     v,
        'q':     q_all,       # full flatness quaternion
        'q_att': q_att_all,   # yaw-only quaternion (NMPC-Att)
        'omega': omega_all,
        'T':     T_des,
        'yaw':   yaw,
        't':     t_pmm,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Interpolation to NMPC timestep
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_reference(
    ref:   dict,
    T_s:   float,
    T_sim: float,
) -> dict:
    """Interpolate flatness reference to NMPC control timestep T_s.

    Parameters
    ----------
    ref   : dict from flat_map_trajectory
    T_s   : float   — NMPC sampling time [s]
    T_sim : float   — total simulation time [s]

    Returns
    -------
    ref_interp : dict  — same keys, arrays re-sampled at 1/T_s Hz
    """
    t_src  = ref['t']
    t_nmpc = np.arange(0.0, T_sim + T_s, T_s)

    # Clamp to PMM time range
    t_nmpc = np.clip(t_nmpc, t_src[0], t_src[-1])

    out = {'t': t_nmpc}
    for key in ('p', 'v', 'omega'):
        arr = ref[key]   # (3, N_pmm)
        interp = np.zeros((3, len(t_nmpc)))
        for i in range(3):
            cs = CubicSpline(t_src, arr[i, :])
            interp[i, :] = cs(t_nmpc)
        out[key] = interp

    # Quaternion: interpolate each component then re-normalise
    for q_key in ('q', 'q_att'):
        q_interp = np.zeros((4, len(t_nmpc)))
        for i in range(4):
            cs = CubicSpline(t_src, ref[q_key][i, :])
            q_interp[i, :] = cs(t_nmpc)
        norms = np.linalg.norm(q_interp, axis=0) + 1e-12
        out[q_key] = q_interp / norms

    # Thrust
    cs_T = CubicSpline(t_src, ref['T'])
    out['T'] = np.clip(cs_T(t_nmpc), 0.0, None)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  17-D acados parameter vector builders
# ══════════════════════════════════════════════════════════════════════════════
# Layout: [p(3), v(3), q(4), omega(3), T(1), w_cmd(3)]  →  17 values

def build_param_att(p_ref: np.ndarray, v_ref: np.ndarray,
                    q_att: np.ndarray) -> np.ndarray:
    """NMPC-Att: p*(t) + v*(t) + yaw-derived q*(t).
    No control feedforward: T*=0, ω*=0 (Qω=0 in solver)."""
    param = np.zeros(17)
    param[0:3]  = p_ref
    param[3:6]  = v_ref
    param[6:10] = q_att / (np.linalg.norm(q_att) + 1e-12)
    # param[10:17] all zero: no angular-rate reference, no thrust feedforward
    return param


from config.experiment_config import T_MAX as _T_MAX   # actuator limit shared w/ NMPC
_W_MAX = 9.0    # clip omega reference (matches W_MAX_GATE in nmpc_gate_tracker) [rad/s]


def build_param_full(p_ref: np.ndarray, v_ref: np.ndarray,
                     q_ref: np.ndarray, omega_ref: np.ndarray,
                     T_ref: float) -> np.ndarray:
    """Full differential flatness reference (clipped to actuator limits)."""
    param = np.zeros(17)
    param[0:3]   = p_ref
    param[3:6]   = v_ref
    param[6:10]  = q_ref / (np.linalg.norm(q_ref) + 1e-12)
    param[10:13] = np.clip(omega_ref, -_W_MAX, _W_MAX)
    param[13]    = float(np.clip(T_ref, 0.0, _T_MAX))
    return param


# ══════════════════════════════════════════════════════════════════════════════
#  High-level: load PMM files and build all references
# ══════════════════════════════════════════════════════════════════════════════

def load_and_convert(
    xref_path: str,
    uref_path: str,
    tref_path: str,
    T_s:       float,
    T_sim:     float,
    mass:      float = MASS,
    g:         float = G,
) -> tuple:
    """Load PMM .npy files, apply flatness, and return interpolated references.

    Returns
    -------
    ref_flat : dict   — flatness output at NMPC timestep
    gate_cfg : None   — placeholder (load separately if needed)
    """
    X_opt = np.load(xref_path)   # (6, N+1)
    U_opt = np.load(uref_path)   # (3, N)
    t_pmm = np.load(tref_path)   # (N+1,)

    ref_raw   = flat_map_trajectory(X_opt, U_opt, t_pmm, mass=mass, g=g)
    ref_interp = interpolate_reference(ref_raw, T_s=T_s, T_sim=T_sim)

    return ref_interp


# ══════════════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    base = os.path.join(os.path.dirname(__file__), '..')
    ref  = load_and_convert(
        xref_path = os.path.join(base, 'xref_optimo_3D_PMM.npy'),
        uref_path = os.path.join(base, 'uref_optimo_3D_PMM.npy'),
        tref_path = os.path.join(base, 'tref_optimo_3D_PMM.npy'),
        T_s       = 0.01,
        T_sim     = 9.0,
    )
    N = ref['t'].shape[0]
    print(f"Reference interpolated: {N} steps  |  T_sim = {ref['t'][-1]:.2f} s")
    print(f"  p  shape : {ref['p'].shape}")
    print(f"  q  shape : {ref['q'].shape}  |  q norms: {np.linalg.norm(ref['q'], axis=0).min():.4f} – {np.linalg.norm(ref['q'], axis=0).max():.4f}")
    print(f"  T  range : {ref['T'].min():.2f} – {ref['T'].max():.2f} N")
    print(f"  omega max: {np.abs(ref['omega']).max():.3f} rad/s")
    print("reference_conversion.py: OK")
