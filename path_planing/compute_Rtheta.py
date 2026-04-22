"""
compute_Rtheta.py
=================
Computa la función de radio de túnel R(θ) a partir de la trayectoria PMM optimizada.

Formulación:
    θ(k)  = longitud de arco acumulada a lo largo del path PMM   [m]
    R(θ)  = R_max - Σ_i (R_max - r_i) * exp(-0.5 * ((θ - θ_i)/σ)²)
          = radio libre disponible en cada punto del path

Propiedades:
    - R(θ_i) ≈ gate_radius  en cada gate
    - R(θ) → R_max         lejos de todos los gates
    - R(θ) ≥ gate_radius   en todo momento (clipped)
    - C∞ diferenciable     → gradiente CBF bien definido

Salida: tunnel_radius.mat  (mismo directorio)
    theta_arc    [N+1]  longitud de arco acumulada [m]
    R_theta      [N+1]  radio de túnel en cada punto [m]
    theta_gates  [n_g]  θ en cada gate [m]
    R_max        scalar
    sigma        scalar
    gate_radius  scalar
"""

import numpy as np
from scipy.io import loadmat, savemat

# ── Parámetros de diseño ──────────────────────────────────────────────────────
R_MAX   = 1.1    # radio libre entre gates [m]  (+10% sobre gate_radius=1.0)
SIGMA   = 1.2    # anchura de la campana Gaussiana en arc-length [m]

ROOT    = '/home/bryansgue/dev/ros2/Tunel-MPCC'
IN_MAT  = f'{ROOT}/path_planing/path_PMM_results.mat'
OUT_MAT = f'{ROOT}/path_planing/tunnel_radius.mat'

# ── Cargar trayectoria ────────────────────────────────────────────────────────
data = loadmat(IN_MAT)

px  = data['px'].flatten()       # [N+1]
py  = data['py'].flatten()
pz  = data['pz'].flatten()

gate_radius   = float(data['gate_radius'])
safety_margin = float(data['safety_margin'])
k_gate        = data['k_gate'].flatten().astype(int)   # [n_gates]
n_gates       = len(k_gate)

print(f"Puntos de trayectoria : {len(px)}")
print(f"Gates                 : {n_gates}  (k_gate = {k_gate})")
print(f"gate_radius           : {gate_radius} m")
print(f"R_max                 : {R_MAX} m")
print(f"σ                     : {SIGMA} m")

# ── Longitud de arco acumulada θ(k) ──────────────────────────────────────────
dx = np.diff(px);  dy = np.diff(py);  dz = np.diff(pz)
ds = np.sqrt(dx**2 + dy**2 + dz**2)          # longitud de cada segmento [N]
theta_arc = np.concatenate([[0.0], np.cumsum(ds)])   # [N+1], θ(0)=0

total_len = theta_arc[-1]
print(f"\nLongitud total de arco: {total_len:.3f} m")

# ── θ en cada gate ────────────────────────────────────────────────────────────
theta_gates = theta_arc[k_gate]               # [n_gates]
print("\nθ en cada gate [m]:")
for i, (kg, tg) in enumerate(zip(k_gate, theta_gates)):
    print(f"  G{i}: k={kg:3d}  θ={tg:.3f} m")

# ── Función R(θ) — suma de Gaussianas ─────────────────────────────────────────
def R_of_theta(theta_eval, theta_gates, r_gate, R_max, sigma):
    """
    R(θ) = R_max - Σ_i (R_max - r_gate) * exp(-0.5*((θ-θ_i)/σ)²)
    Clipped to [r_gate, R_max]
    """
    R = np.full_like(theta_eval, R_max, dtype=float)
    for th_i in theta_gates:
        R -= (R_max - r_gate) * np.exp(-0.5 * ((theta_eval - th_i) / sigma)**2)
    return np.clip(R, r_gate, R_max)

R_theta = R_of_theta(theta_arc, theta_gates, gate_radius, R_MAX, SIGMA)

print(f"\nR(θ) min : {R_theta.min():.4f} m  (en gates)")
print(f"R(θ) max : {R_theta.max():.4f} m  (entre gates)")
print(f"R en G0  : {R_theta[k_gate[0]]:.4f} m")
print(f"R en G4  : {R_theta[k_gate[4]]:.4f} m")

# ── También versión densa para CasADi (oversampled) ──────────────────────────
# Evalúa en 2000 puntos uniformes en θ — útil para lookup table en acados
N_dense = 2000
theta_dense = np.linspace(0.0, total_len, N_dense)
R_dense     = R_of_theta(theta_dense, theta_gates, gate_radius, R_MAX, SIGMA)

# ── Guardar ───────────────────────────────────────────────────────────────────
savemat(OUT_MAT, {
    # Path arc-length grid (matches trajectory points)
    'theta_arc'   : theta_arc,       # [N+1]  una entrada por punto del path
    'R_theta'     : R_theta,         # [N+1]  radio en cada punto
    # Gate info
    'theta_gates' : theta_gates,     # [n_gates]
    'gate_radius' : gate_radius,
    'R_max'       : R_MAX,
    'sigma'       : SIGMA,
    'n_gates'     : float(n_gates),
    # Dense lookup for controller (CasADi interpolant)
    'theta_dense' : theta_dense,     # [2000]
    'R_dense'     : R_dense,         # [2000]
    # Path length reference
    'theta_total' : total_len,
})

print(f"\nGuardado: {OUT_MAT}")
print("Variables: theta_arc, R_theta, theta_gates, theta_dense, R_dense")
