"""
Path Planning 3D - Point Mass Model (PMM)
Optimizado para:
- Tiempo mínimo
- Aprovechar la inercia (mantener dirección)
- Trayectorias suaves sin cambios abruptos
- No desacelerar innecesariamente

Estados: [x, y, z, vx, vy, vz] (6 estados)
Controles: [ax, ay, az] (3 controles - aceleración)

Dinámica PMM:
  ṗ = v
  v̇ = a
"""

import os
import sys
import argparse

import numpy as np
import casadi as ca
from scipy.io import savemat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")           # headless — saves to file without display

# Fix: override mpl_toolkits BEFORE pyplot loads it (system/pip conflict)
import sys as _sys
for _k in list(_sys.modules.keys()):
    if "mpl_toolkits" in _k:
        del _sys.modules[_k]
import mpl_toolkits
mpl_toolkits.__path__ = [
    p for p in [
        "/home/bryansgue/.local/lib/python3.10/site-packages/mpl_toolkits",
    ] if os.path.isdir(p)
] or mpl_toolkits.__path__
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================
g_val = 9.81   # gravedad [m/s²]

# =============================================================================
# CONFIGURACIÓN DEL PROBLEMA
# =============================================================================
n_states = 6     # [x, y, z, vx, vy, vz]
n_controls = 3   # [ax, ay, az]

# ══════════════════════════════════════════════════════════════════════════════
# CLI — optional --gates / --suffix for multi-circuit runs
# ══════════════════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(description='PMM time-optimal path planner')
_parser.add_argument('--gates',  default='gates.npz',
                     help='Gate config file in path_planing/ (default: gates.npz)')
_parser.add_argument('--suffix', default='',
                     help='Suffix for output .npy files, e.g. "_sprint"')
_args = _parser.parse_args()

_script_dir = os.path.dirname(os.path.abspath(__file__))
_gate_file  = _args.gates if os.path.isabs(_args.gates) \
              else os.path.join(_script_dir, _args.gates)
_out_suffix = _args.suffix   # e.g. "" | "_sprint" | "_helix"

# ══════════════════════════════════════════════════════════════════════════════
# GATE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
_gate_cfg    = np.load(_gate_file)
gate_positions = _gate_cfg['gate_positions'].astype(float)
gate_normals   = _gate_cfg['gate_normals'].astype(float)

# Close the loop: append first gate as last so PMM optimizes a full lap
gate_positions = np.vstack([gate_positions, gate_positions[0]])
gate_normals   = np.vstack([gate_normals,   gate_normals[0]])

n_gates = len(gate_positions)
N_segments = n_gates - 1
gate_radius = 0.4      # 6-inch drone: gate de 0.8m de diámetro
safety_margin = 0.25   # margen fuerte: ref PMM pasa por zona central (≤15 cm del centro)

print("=" * 60)
print("PATH PLANNING 3D - POINT MASS MODEL (PMM)")
print("Optimizado para inercia y suavidad")
print("=" * 60)
for i, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    print(f"Gate {i}: pos={pos}, n={np.round(n, 3)}")

# =============================================================================
# DISCRETIZACIÓN
# =============================================================================
N_per_segment = 20  # Pasos por segmento
N = N_per_segment * N_segments

# Índices donde debe estar exactamente en cada gate
k_gate = [i * N_per_segment for i in range(n_gates)]
delta_k = N_per_segment // 2

# Estimación inicial del tiempo
total_dist = sum(np.linalg.norm(gate_positions[i+1] - gate_positions[i]) for i in range(N_segments))
v_avg_est = 7.0    # estimación inicial
T_est = total_dist / v_avg_est

print(f"\nN total = {N}")
print(f"Índices de gates: {k_gate}")
print(f"Distancia total: {total_dist:.1f}m")
print(f"Tiempo estimado: {T_est:.1f}s")

# =============================================================================
# VARIABLES DE OPTIMIZACIÓN
# =============================================================================
X = ca.MX.sym('X', n_states, N + 1)
U = ca.MX.sym('U', n_controls, N)
T = ca.MX.sym('T')  # Tiempo total

dt = T / N  # dt variable

# =============================================================================
# FUNCIÓN DE COSTO - OPTIMIZADA PARA INERCIA Y SUAVIDAD
# =============================================================================
# Pesos - BALANCE ÓPTIMO
w_T = 500.0         # Tiempo — peso dominante para circuito rápido
w_a = 0.0005        # Aceleración — casi libre
w_da = 0.0          # Jerk — desactivado
w_dda = 0.0         # Snap — desactivado
w_v_change = 0.0    # Sin penalización por cambio de dirección
w_center = 0.5      # Centrado de gate

cost = w_T * T

# Costo de controles y suavidad
for k in range(N):
    a_k = U[:, k]
    v_k = X[3:6, k]
    
    # Penalizar aceleración (bajo)
    cost += w_a * ca.sumsqr(a_k) * dt
    
    # Penalizar jerk (cambio de aceleración) - SUAVIDAD
    if k > 0:
        da = (U[:, k] - U[:, k-1]) / dt
        cost += w_da * ca.sumsqr(da) * dt
    
    # Penalizar cambios bruscos de dirección de velocidad (INERCIA)
    if k > 0:
        v_prev = X[3:6, k-1]
        v_curr = X[3:6, k]
        dv = v_curr - v_prev
        cost += w_v_change * ca.sumsqr(dv) * dt

# Costo de pasar cerca del centro (bajo para flexibilidad)
for i in range(1, n_gates):
    k_i = k_gate[i]
    p_gate = X[0:3, k_i]
    gpos = gate_positions[i]
    cost += w_center * ca.sumsqr(p_gate - gpos)

# =============================================================================
# RESTRICCIONES
# =============================================================================
g = []
lbg = []
ubg = []

# --- 1. Dinámica PMM (Multiple Shooting con RK4) ---
def pmm_dynamics(x, u):
    """Dinámica Point Mass Model: ṗ = v, v̇ = a"""
    p = x[0:3]
    v = x[3:6]
    a = u
    return ca.vertcat(v, a)

for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    
    # RK4
    k1 = pmm_dynamics(xk, uk)
    k2 = pmm_dynamics(xk + dt/2 * k1, uk)
    k3 = pmm_dynamics(xk + dt/2 * k2, uk)
    k4 = pmm_dynamics(xk + dt * k3, uk)
    x_next = xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    g.append(X[:, k+1] - x_next)
    lbg.extend([0] * n_states)
    ubg.extend([0] * n_states)

# --- 2. Condiciones iniciales ---
x0_pos = gate_positions[0]
# Velocidad inicial cero (hover antes de empezar)
x0_vel = np.array([0, 0, 0])

g.append(X[0:3, 0] - x0_pos)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

g.append(X[3:6, 0] - x0_vel)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

# --- 3. Condiciones finales ---
xf_pos = gate_positions[-1]

g.append(X[0:3, N] - xf_pos)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

# --- 4. Restricciones de cruce por gates ---
# max_perp_dist: tolerancia perpendicular al plano del gate (relajada para convergencia)
max_perp_dist = 0.05       # ±5 cm — ref PMM cruza en el plano del gate
max_r = gate_radius - safety_margin   # 0.32 m radio útil

for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]

    p_i = X[0:3, k_i]
    d   = p_i - gpos

    # Distancia perpendicular al plano del gate (drone debe estar ≈ en el plano)
    dist_perp = ca.dot(d, gnorm)
    g.append(dist_perp)
    lbg.append(-max_perp_dist)
    ubg.append(max_perp_dist)

    # Distancia radial dentro del gate
    d_plane = d - dist_perp * gnorm
    g.append(ca.sumsqr(d_plane))
    lbg.append(0)
    ubg.append(max_r**2)

# --- 5. Dirección de cruce: velocidad en el gate debe alinearse con la normal ---
# (reemplaza el enfoque position-before/after que tenía bugs en gates extremos)
v_cross_min = 0.5   # m/s mínimo en la dirección del gate
for i in range(1, n_gates):
    k_i    = k_gate[i]
    gnorm  = gate_normals[i]
    v_i    = X[3:6, k_i]
    g.append(ca.dot(v_i, gnorm))
    lbg.append(v_cross_min)
    ubg.append(1000)

# --- 7. Monotonicidad (no volver atrás) - OPCIONAL, puede causar infeasibilidad ---
# Comentado para mejor convergencia - las restricciones de cruce ya garantizan dirección correcta
# for i in range(1, n_gates - 1):
#     k_i = k_gate[i]
#     k_next = k_gate[i + 1]
#     gpos = gate_positions[i]
#     gnorm = gate_normals[i]
#     
#     k_mid = (k_i + k_next) // 2
#     p_mid = X[0:3, k_mid]
#     sd_mid = ca.dot(p_mid - gpos, gnorm)
#     g.append(sd_mid)
#     lbg.append(0.05)
#     ubg.append(1000)

# =============================================================================
# FORMULAR NLP
# =============================================================================
opt_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1),
    T
)

n_vars = opt_vars.shape[0]
g_vec = ca.vertcat(*g)

nlp = {'f': cost, 'x': opt_vars, 'g': g_vec}

opts = {
    'ipopt.max_iter': 5000,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 20,
    'ipopt.print_level': 3,
    'print_time': 1,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.mu_init': 1.0,        # start with larger barrier — less likely to get stuck
    'ipopt.mu_min': 1e-9,
    'ipopt.bound_push': 1e-4,
    'ipopt.bound_frac': 1e-4,
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.hessian_approximation': 'limited-memory',  # L-BFGS — better for flat landscapes
}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# =============================================================================
# LÍMITES DE VARIABLES
# =============================================================================
lbx = np.full(n_vars, -np.inf)
ubx = np.full(n_vars, np.inf)

# Límites de estados
v_max = 14.0           # Velocidad máxima [m/s]  — kinematic cap (per-axis)
a_max = 23.0           # Aceleración máxima [m/s²] — 78% of (T_max/m - g) ≈ 29.5 with T_max=42.4N (4g), m=1.08 kg
z_min, z_max = 0.3, 5.5   # límites del espacio de vuelo — amplio para loops

for k in range(N + 1):
    idx = k * n_states
    
    # Posición z
    lbx[idx + 2] = z_min
    ubx[idx + 2] = z_max
    
    # Velocidades
    for j in range(3, 6):
        lbx[idx + j] = -v_max
        ubx[idx + j] = v_max

# Límites de controles (aceleración)
ctrl_offset = n_states * (N + 1)
for k in range(N):
    idx = ctrl_offset + k * n_controls
    lbx[idx:idx+3] = -a_max
    ubx[idx:idx+3] = a_max

# Tiempo
lbx[-1] = 1.0
ubx[-1] = 60.0


print(f"\nVariables: {n_vars}")
print(f"Restricciones: {len(lbg)}")

# =============================================================================
# HELPER: build warm-start vector for a given T_init and speed fraction
# =============================================================================
def build_x0(T_init_val, speed_frac=0.75):
    x0 = np.zeros(n_vars)
    for k in range(N + 1):
        seg = min(k // N_per_segment, N_segments - 1)
        t_local = (k - seg * N_per_segment) / N_per_segment
        pos_start = gate_positions[seg]
        pos_end   = gate_positions[seg + 1]
        pos = (1 - t_local) * pos_start + t_local * pos_end
        vel_dir = pos_end - pos_start
        vel_mag_dir = np.linalg.norm(vel_dir)
        near_gate = abs(t_local) < 0.15 or abs(t_local - 1.0) < 0.15
        spd = v_max * speed_frac * (0.6 if near_gate else 1.0)
        vel = (vel_dir / vel_mag_dir) * spd if vel_mag_dir > 0 else np.zeros(3)
        idx = k * n_states
        x0[idx:idx + 3] = pos
        x0[idx + 3:idx + 6] = vel
    x0[-1] = T_init_val
    return x0

# =============================================================================
# MULTI-START: try several T_init seeds, keep best feasible
# =============================================================================
T_seeds = [
    total_dist / (0.90 * v_max),   # very aggressive
    total_dist / (0.70 * v_max),   # moderate
    total_dist / (0.55 * v_max),   # conservative
    total_dist / (0.40 * v_max),   # very conservative
]
speed_fracs = [0.85, 0.75, 0.60, 0.45]

print("\n" + "=" * 60)
print("MULTI-START — trying 4 T_init seeds")
print("=" * 60)

best_sol  = None
best_Topt = np.inf

for seed_idx, (T_seed, spd_frac) in enumerate(zip(T_seeds, speed_fracs)):
    print(f"\n--- Seed {seed_idx+1}/4  T_init={T_seed:.2f}s  speed_frac={spd_frac:.2f} ---")
    x0_try = build_x0(T_seed, spd_frac)
    try:
        sol_try = solver(x0=x0_try, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        T_try = float(sol_try['x'].full().flatten()[-1])
        stats = solver.stats()
        feasible = stats['success'] or stats.get('return_status', '') in (
            'Solve_Succeeded', 'Solved_To_Acceptable_Level')
        print(f"    T_opt = {T_try:.3f} s  |  feasible={feasible}")
        if feasible and T_try < best_Topt:
            best_Topt = T_try
            best_sol  = sol_try
    except Exception as exc:
        print(f"    FAILED: {exc}")

# Phase-2 warm start from best multi-start result using tighter gate tolerance
if best_sol is not None:
    print(f"\n--- Phase-2 warm start from best multi-start (T={best_Topt:.2f}s) ---")
    sol_phase2 = solver(x0=best_sol['x'], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    T_p2 = float(sol_phase2['x'].full().flatten()[-1])
    stats2 = solver.stats()
    feasible2 = stats2['success'] or stats2.get('return_status', '') in (
        'Solve_Succeeded', 'Solved_To_Acceptable_Level')
    print(f"    Phase-2 T_opt = {T_p2:.3f} s  |  feasible={feasible2}")
    if feasible2 and T_p2 < best_Topt:
        best_Topt = T_p2
        best_sol  = sol_phase2

if best_sol is None:
    raise RuntimeError("All seeds failed — check gate geometry or relax constraints")

print(f"\n{'='*60}")
print(f"BEST T_opt = {best_Topt:.3f} s")
print(f"{'='*60}")

# =============================================================================
# EXTRAER RESULTADOS
# =============================================================================
sol = best_sol['x'].full().flatten()

X_opt = sol[:n_states * (N + 1)].reshape((N + 1, n_states)).T
U_opt = sol[n_states * (N + 1):n_states * (N + 1) + n_controls * N].reshape((N, n_controls)).T
T_opt = sol[-1]

dt_opt = T_opt / N
t_traj = np.linspace(0, T_opt, N + 1)

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)
print(f"Tiempo óptimo: {T_opt:.2f} s")

# =============================================================================
# VERIFICACIÓN DE CRUCE
# =============================================================================
print("\n" + "-" * 60)
print("VERIFICACIÓN DE CRUCE POR GATES")
print("-" * 60)

all_ok = True
for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]

    p_i = X_opt[0:3, k_i]
    d = p_i - gpos
    dist_perp = np.dot(d, gnorm)
    d_plane = d - dist_perp * gnorm
    dist_radial = np.linalg.norm(d_plane)

    is_terminal = (i == n_gates - 1)   # last gate = gate 0, trajectory ends here
    k_before = max(0, k_i - delta_k)

    if is_terminal:
        # trajectory endpoint IS the gate — check only radial error
        ok = dist_radial < gate_radius
        print(f"\nGate {i} [terminal=Gate 0]:")
        print(f"  En gate (t={t_traj[k_i]:.2f}s): r={dist_radial:.4f}m, perp={dist_perp:.4f}m")
        print(f"  Antes: signed_dist={np.dot(X_opt[0:3, k_before] - gpos, gnorm):.4f}m")
        print(f"  {'[✓ CRUCE CORRECTO]' if ok else '[✗ ERROR]'}")
    else:
        k_after = k_i + delta_k
        sd_before = np.dot(X_opt[0:3, k_before] - gpos, gnorm)
        sd_after  = np.dot(X_opt[0:3, k_after]  - gpos, gnorm)
        ok = (sd_before < 0) and (sd_after > 0) and (dist_radial < gate_radius)
        print(f"\nGate {i}:")
        print(f"  En gate (t={t_traj[k_i]:.2f}s): r={dist_radial:.4f}m, perp={dist_perp:.4f}m")
        print(f"  Antes: signed_dist={sd_before:.4f}m {'(<0 ✓)' if sd_before < 0 else '(ERROR)'}")
        print(f"  Después: signed_dist={sd_after:.4f}m {'(>0 ✓)' if sd_after > 0 else '(ERROR)'}")
        print(f"  {'[✓ CRUCE CORRECTO]' if ok else '[✗ ERROR]'}")

    if not ok:
        all_ok = False

if all_ok:
    print("\n" + "=" * 60)
    print("RESULTADO FINAL: TODOS LOS GATES CRUZADOS ✓")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("ADVERTENCIA: ALGUNOS GATES NO SE CRUZARON CORRECTAMENTE")
    print("=" * 60)

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
np.save(os.path.join(_script_dir, f'xref_optimo_3D_PMM{_out_suffix}.npy'), X_opt)
np.save(os.path.join(_script_dir, f'uref_optimo_3D_PMM{_out_suffix}.npy'), U_opt)
np.save(os.path.join(_script_dir, f'tref_optimo_3D_PMM{_out_suffix}.npy'), t_traj)

# ── Export .mat for MATLAB visualization ─────────────────────────────────────
vel_mag_export = np.linalg.norm(X_opt[3:6, :], axis=0)          # [N+1]
acc_mag_export = np.zeros(N + 1)
acc_mag_export[:N] = np.linalg.norm(U_opt, axis=0)              # [N]

# gates.npz is the authoritative gate definition — do not overwrite it here.

savemat(os.path.join(_script_dir, 'path_PMM_results.mat'), {
    # Trajectory
    'px':          X_opt[0, :],    # [N+1]
    'py':          X_opt[1, :],    # [N+1]
    'pz':          X_opt[2, :],    # [N+1]
    'vx':          X_opt[3, :],
    'vy':          X_opt[4, :],
    'vz':          X_opt[5, :],
    'vel_mag':     vel_mag_export,
    # Controls
    'ax':          U_opt[0, :],    # [N]
    'ay':          U_opt[1, :],
    'az':          U_opt[2, :],
    'acc_mag':     acc_mag_export[:N],
    # Time
    't_traj':      t_traj,         # [N+1]
    'T_opt':       float(T_opt),
    'dt_opt':      float(dt_opt),
    # Gates
    'gate_positions': gate_positions,   # [n_gates × 3]
    'gate_normals':   gate_normals,     # [n_gates × 3]
    'gate_radius':    float(gate_radius),
    'safety_margin':  float(safety_margin),
    'k_gate':         np.array(k_gate, dtype=float),
    'n_gates':        float(n_gates),
})

print(f"\nArchivos guardados:")
print(f"  xref_optimo_3D_PMM.npy: {X_opt.shape}")
print(f"  uref_optimo_3D_PMM.npy: {U_opt.shape}")
print(f"  path_PMM_results.mat   (MATLAB)")

# =============================================================================
# MÉTRICAS DE SUAVIDAD
# =============================================================================
vel_magnitudes = np.linalg.norm(X_opt[3:6, :], axis=0)
acc_magnitudes = np.linalg.norm(U_opt, axis=0)

# Calcular jerk (derivada de aceleración)
jerk = np.diff(U_opt, axis=1) / dt_opt
jerk_magnitudes = np.linalg.norm(jerk, axis=0)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"Gates: {n_gates}")
print(f"Tiempo óptimo: {T_opt:.2f}s")
print(f"Distancia total: {total_dist:.1f}m")
print(f"Vel promedio: {total_dist/T_opt:.2f} m/s")
print(f"Vel máxima: {vel_magnitudes.max():.2f} m/s")
print(f"Vel mínima: {vel_magnitudes.min():.2f} m/s")
print(f"Acc máxima: {acc_magnitudes.max():.2f} m/s²")
print(f"Jerk máximo: {jerk_magnitudes.max():.2f} m/s³")
print(f"Jerk promedio: {jerk_magnitudes.mean():.2f} m/s³")
print("=" * 60)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig = plt.figure(figsize=(16, 12))

x_min, x_max = X_opt[0, :].min() - gate_radius, X_opt[0, :].max() + gate_radius
y_min, y_max = X_opt[1, :].min() - gate_radius, X_opt[1, :].max() + gate_radius
z_min_plot, z_max = X_opt[2, :].min() - gate_radius, X_opt[2, :].max() + gate_radius

ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# --- Trajectory ---
ax1.plot(X_opt[0, :], X_opt[1, :], X_opt[2, :],
         'b-', linewidth=2, label='Trayectoria PMM', zorder=3)

# --- Speed colormap along trajectory ---
vel_norm = vel_magnitudes / vel_magnitudes.max()
for k in range(len(vel_norm) - 1):
    c = plt.cm.plasma(vel_norm[k])
    ax1.plot(X_opt[0, k:k+2], X_opt[1, k:k+2], X_opt[2, k:k+2],
             color=c, linewidth=2.5, alpha=0.85)

# --- Gates (circles + normal arrows) ---
theta_circle = np.linspace(0, 2 * np.pi, 60)
for idx, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    # Build two orthogonal vectors in the gate plane
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    perp1 = np.cross(n, ref)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(n, perp1)

    circle_pts = np.array([
        pos + gate_radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
        for t in theta_circle
    ])
    color_gate = 'limegreen' if idx > 0 else 'orange'
    ax1.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2],
             color=color_gate, linewidth=2, zorder=4)
    ax1.scatter(*pos, color='red', s=60, marker='o', zorder=5)
    ax1.quiver(pos[0], pos[1], pos[2],
               n[0] * 0.6, n[1] * 0.6, n[2] * 0.6,
               color='darkred', arrow_length_ratio=0.35, linewidth=1.2)
    ax1.text(pos[0], pos[1], pos[2] + 0.3, f'G{idx}', fontsize=7,
             color='black', ha='center')

ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
ax1.set_zlim([z_min_plot, z_max])
ax1.set_xlabel('X [m]', fontsize=9)
ax1.set_ylabel('Y [m]', fontsize=9)
ax1.set_zlabel('Z [m]', fontsize=9)
ax1.set_title('Trayectoria 3D — PMM (color = velocidad)', fontsize=10)
ax1.view_init(elev=25, azim=-55)
ax1.grid(True, alpha=0.3)

# Velocidad
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_traj, X_opt[3, :], 'r-', label='vx')
ax2.plot(t_traj, X_opt[4, :], 'g-', label='vy')
ax2.plot(t_traj, X_opt[5, :], 'b-', label='vz')
ax2.plot(t_traj, vel_magnitudes, 'k--', linewidth=2, label='|v|')
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Velocidad [m/s]')
ax2.set_title('Perfil de Velocidad')
ax2.legend()
ax2.grid(True)

# Aceleración
ax3 = fig.add_subplot(2, 2, 3)
t_ctrl = np.linspace(0, T_opt, N)
ax3.plot(t_ctrl, U_opt[0, :], 'r-', label='ax')
ax3.plot(t_ctrl, U_opt[1, :], 'g-', label='ay')
ax3.plot(t_ctrl, U_opt[2, :], 'b-', label='az')
ax3.plot(t_ctrl, acc_magnitudes, 'k--', linewidth=2, label='|a|')
ax3.set_xlabel('Tiempo [s]')
ax3.set_ylabel('Aceleración [m/s²]')
ax3.set_title('Perfil de Aceleración (Control)')
ax3.legend()
ax3.grid(True)

# Jerk (suavidad)
ax4 = fig.add_subplot(2, 2, 4)
t_jerk = np.linspace(0, T_opt, N-1)
ax4.plot(t_jerk, jerk_magnitudes, 'm-', linewidth=1.5)
ax4.set_xlabel('Tiempo [s]')
ax4.set_ylabel('Jerk [m/s³]')
ax4.set_title('Perfil de Jerk (Suavidad)')
ax4.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, 'path_3D_PMM.png'), dpi=150, bbox_inches='tight')
print("✓ Saved: path_3D_PMM.png")

# --- Extra: standalone 3D top-quality figure ---
fig3d, ax3d = plt.subplots(1, 1, figsize=(10, 7),
                            subplot_kw={'projection': '3d'})

for k in range(len(vel_norm) - 1):
    c = plt.cm.plasma(vel_norm[k])
    ax3d.plot(X_opt[0, k:k+2], X_opt[1, k:k+2], X_opt[2, k:k+2],
              color=c, linewidth=3, alpha=0.9)

for idx, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    perp1 = np.cross(n, ref); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(n, perp1)
    circle_pts = np.array([
        pos + gate_radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
        for t in theta_circle
    ])
    color_gate = 'limegreen' if idx > 0 else 'orange'
    ax3d.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2],
              color=color_gate, linewidth=2.5)
    ax3d.scatter(*pos, color='red', s=80, zorder=5)
    ax3d.text(pos[0], pos[1], pos[2] + 0.35, f'G{idx}', fontsize=8,
              color='black', ha='center', fontweight='bold')

sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(0, vel_magnitudes.max()))
sm.set_array([])
fig3d.colorbar(sm, ax=ax3d, shrink=0.5, label='Velocidad [m/s]', pad=0.1)

ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
ax3d.set_title('PMM Path Planning — Vista 3D con velocidad', fontsize=12)
ax3d.view_init(elev=30, azim=-60)
ax3d.grid(True, alpha=0.3)

fig3d.tight_layout()
fig3d.savefig(os.path.join(_script_dir, 'path_3D_PMM_detail.png'), dpi=200, bbox_inches='tight')
print("✓ Saved: path_3D_PMM_detail.png")

# --- Projections: XY, XZ, YZ ---
fig_proj, axes_proj = plt.subplots(1, 3, figsize=(16, 5))

proj_configs = [
    ('XY', 0, 1, 'X [m]', 'Y [m]'),
    ('XZ', 0, 2, 'X [m]', 'Z [m]'),
    ('YZ', 1, 2, 'Y [m]', 'Z [m]'),
]

for ax_p, (label, xi, yi, xl, yl) in zip(axes_proj, proj_configs):
    # trajectory colored by speed
    for k in range(len(vel_norm) - 1):
        c = plt.cm.plasma(vel_norm[k])
        ax_p.plot(X_opt[xi, k:k+2], X_opt[yi, k:k+2], color=c, linewidth=2.5, alpha=0.9)

    # gates projected as ellipses
    for idx, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
        ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
        perp1 = np.cross(n, ref); perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(n, perp1)
        circle_pts = np.array([
            pos + gate_radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
            for t in theta_circle
        ])
        color_gate = 'limegreen' if idx > 0 else 'orange'
        ax_p.plot(circle_pts[:, xi], circle_pts[:, yi], color=color_gate,
                  linewidth=1.8, alpha=0.85)
        ax_p.scatter(pos[xi], pos[yi], color='red', s=50, zorder=5)
        ax_p.annotate(f'G{idx}', (pos[xi], pos[yi]),
                      textcoords='offset points', xytext=(4, 4),
                      fontsize=8, fontweight='bold', color='black')

    ax_p.set_xlabel(xl, fontsize=11)
    ax_p.set_ylabel(yl, fontsize=11)
    ax_p.set_title(f'Proyección {label}', fontsize=12)
    ax_p.set_aspect('equal')
    ax_p.grid(True, alpha=0.3)

sm2 = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, vel_magnitudes.max()))
sm2.set_array([])
fig_proj.colorbar(sm2, ax=axes_proj, shrink=0.7, label='Velocidad [m/s]', pad=0.02)
fig_proj.suptitle(f'PMM Path Planning — Proyecciones  (T_opt={T_opt:.2f} s, v_max={vel_magnitudes.max():.1f} m/s)',
                  fontsize=13, fontweight='bold')
fig_proj.tight_layout()
fig_proj.savefig(os.path.join(_script_dir, 'path_PMM_projections_new.png'), dpi=200, bbox_inches='tight')
print("✓ Saved: path_PMM_projections_new.png")
