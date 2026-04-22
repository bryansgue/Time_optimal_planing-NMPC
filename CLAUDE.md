# CLAUDE.md — NMPC_baseline

## Project purpose

IEEE Access paper: **"Point Mass Planning and NMPC Tracking for
Time-Optimal Agile Quadrotor Gate Navigation"**
Authors: Bryan S. Guevara, Luis F. Simancas, Tiago Nascimento
(LASER/UFPB + INAUT/UNSJ)
LaTeX source: `ACCESS_latex/access.tex` (ieeeaccess.cls)

---

## Pipeline

```
path_planing/path_time_3D_PMM.py
    → path_planing/xref/uref/tref_optimo_3D_PMM.npy  +  path_planing/gate_config.npz

path_planing/reference_conversion.py          (flatness bridge)
    → 17-D NMPC parameter vector [p(3), v(3), q(4), ω(3), T(1), ω_cmd(3)]

ocp/nmpc_gate_tracker.py                      (builds acados OCP)
    → c_generated_code_gate_att/  (NMPC-Att solver)
    → c_generated_code_gate_full/ (NMPC-Full solver)

experiments/mil_gate_experiment.py            (MuJoCo sim loop)
    → experiments/results/gate_experiment_results.npy

experiments/plot_gate_results.py
    → ACCESS_latex/figs/fig1–fig8.pdf
```

---

## Quadrotor model (body-rate)

State:  `x = [p(3), v(3), q(4), ω(3)]` ∈ R¹³  (Hamilton quaternion, q_w first)
Control: `u = [T, ωx_cmd, ωy_cmd, ωz_cmd]`
Rate model: `ω̇ = (ω_cmd − ω) / τ_rc`,  τ_rc = 0.03 s
Thrust: `v̇ = −g e₃ + (T/m) R(q) e₃`

Key parameters:
- m = 1.08 kg, g = 9.81 m/s², T_max = 29.4 N
- W_MAX_GATE = 10.0 rad/s
- N = 50, T_h = 0.5 s, f_c = 100 Hz, ERK4 with 10 sub-steps

---

## Flatness bridge (`reference_conversion.py`)

- `ω*` via numerical forward finite difference of R (NOT Faessler 2018 analytical formula)
- 5-point causal MA filter + clip ±9 rad/s
- Yaw: `ψ = atan2(v_y, v_x)`, fallback to previous when ‖v_xy‖ < 0.1 m/s
- Hemisphere correction: flip q_k if q_k·q_{k-1} < 0
- Quaternion: Shepperd method
- `q_att` key added: yaw-only quaternion `[cos(ψ/2), 0, 0, sin(ψ/2)]` for NMPC-Att

---

## Two NMPC configurations (ablation study)

**NMPC-Att** — common-practice baseline:
- Reference: p*(t) + v*(t) from PMM + yaw-only q*_att = [cos(ψ/2), 0, 0, sin(ψ/2)]
- No reference control actions: T*=0, ω*=0
- Cost: Qp + Qv + Qq active; **Qω=0** (no angular-rate penalty without ω* reference)
- Solver: `Drone_gate_att` → `c_generated_code_gate_att/`
- Builder: `build_gate_solver_att()`
- Param builder: `build_param_att(p, v, q_att)` — param[13]=0 (T*=0)

**NMPC-Full** — complete flatness inversion (contribution):
- Reference: p*(t) + v*(t) + full q*(t) + ω*(t) + T*(t)
- All cost terms active including Qω=diag(0.5,0.5,0.5)
- Solver: `Drone_gate_full` → `c_generated_code_gate_full/`
- Builder: `build_gate_solver_full()`
- Param builder: `build_param_full(p, v, q, omega, T)`

Cost weights shared: Qp=diag(35,35,40), Qv=diag(6,6,6), Qq=diag(15,15,15), R=diag(0.3,1,1,1)

---

## Paper writing conventions

- Macros: `\nmpc{}`, `\pmm{}`, `\mujoco{}`
- Figures in `ACCESS_latex/figs/` as PDF
- `[H]` for figure, `[t]` for standalone tables
- Compile: `pdflatex → bibtex → pdflatex → pdflatex` from `ACCESS_latex/`
- 22 refs in `references.bib`

---

## Stack

- Python 3.10, acados 0.5.3, CasADi 3.7.2
- MuJoCo Python bindings (`import mujoco`)
- Ubuntu 22.04, pdflatex (TeX Live 2022)

---

---

# ══════════════════════════════════════════════════════
# SESSION STATE — 2026-04-18 (updated, end of day)
# ══════════════════════════════════════════════════════

## Paper status

- 9 pages, Results section consolidated around 2 circuits (fig-8 + loop).
- Tabla IV (`tab:results`) en columna simple (\footnotesize, 6 cols):
  4 filas = circuito × controlador (Att/Full).
- Figura 1 (`fig:pmm_circuits`): `figure*` 3-D double-panel con PMM
  reference (viridis por velocidad) + NMPC-Att (azul) + NMPC-Full (verde)
  superpuestos en la misma vista 3-D (sin proyecciones ortográficas).
  Generada por `path_planing/_preview_two_circuits.py`.
- Figura de errores de posición: `fig_pos_error_combined.pdf` en un solo
  eje, 4 curvas (fig-8 sólidas, loop punteadas).
- Figura omega-tracking eliminada.
- Todos los floats en Resultados usan `[!htbp]` (no más "todo al final").

## Key config changes this session

**T_MAX subido a 5g:** `config/experiment_config.py:72`
  `T_MAX = 53.0 N` (antes `3*G = 29.43 N` ≈ 2.78g real para m=1.08 kg).
  Actualizado también en `tab:params_left` de `access.tex`.

**PMM re-ejecutado con nueva envolvente** (`path_time_3D_PMM.py`):
  - `v_max = 14.0 m/s`  (antes 10.0)
  - `a_max = 30.0 m/s²` (antes 14.0, ~80% de T/m-g=39.3)
  - Fig-8:   T_opt **4.89 s** (antes 7.71), v_peak 17.82 m/s, a_peak 47.3 m/s²
  - Loop:    T_opt **4.44 s**,              v_peak 13.25 m/s, a_peak 44.6 m/s²
  - Archivos regenerados: `xref/uref/tref_optimo_3D_PMM{,_loop}.npy`.

## ⚠ Pending antes de lanzar SiL

1. **Re-compilar solvers acados** (att + full): los `c_generated_code_gate_{att,full}/`
   fueron generados con T_MAX=29.4 N. El parámetro está hard-codeado en
   `ocp/nmpc_gate_tracker.py:98` (`ocp.constraints.ubu`). Re-ejecutar con
   `rebuild=True` o borrar los json y c_generated_code.
2. **Re-generar la figura preview 3-D** si el planner cambió los paths:
   `python3 path_planing/_preview_two_circuits.py`.
3. **Los `*.mat` que usa la preview** (`path_PMM_results_fig8.mat`,
   `path_PMM_results_loop.mat`) pueden necesitar regeneración vía
   `export_circuits_mat.py` si la estructura cambió.

---

## Next: Fase 6 — SiL experiment in real MuJoCo (ROS 2)

**Script creado:** `experiments/sil_gate_experiment.py`

Diferencias vs MiL:
- Estado: `muj.get_state()` desde `/quadrotor/odom` (no RK4 interno).
- Comando: `muj.send_cmd(T, ωx, ωy, ωz)` → `/quadrotor/trpy_cmd`.
- Reset: `SimControl.reset()` recarga la escena (re-siembra la
  perturbación de fuerza externa random que MuJoCo aplica internamente).
- CIRCUIT={fig8|loop}, N_TRIALS={1..50}.
- Salida: `experiments/results/sil_gate_results_{circuit}.npy` (mismo
  esquema que MiL → `plot_gate_results.py` funciona).

**Flujo de ejecución:**
```
# Terminal 1
ros2 launch <mujoco_quadrotor_sim>

# Terminal 2
source ~/uav_ws/install/setup.bash
cd ~/dev/ros2/NMPC_baseline
CIRCUIT=fig8  python3 experiments/sil_gate_experiment.py
CIRCUIT=loop  python3 experiments/sil_gate_experiment.py
```

**Pasos siguientes en la próxima sesión:**
1. Re-build solvers con T_MAX=53 N (ver "Pending" arriba).
2. Sanity check: `CIRCUIT=fig8 N_TRIALS=1 python3 sil_gate_experiment.py`.
   Verificar: cruza todos los gates, RMSE razonable, timing < 10 ms.
3. Si (2) sale bien → `N_TRIALS=50` por circuito × controlador (≈200 corridas).
4. Ajustar `plot_gate_results.py` para leer `sil_gate_results_*.npy` y
   regenerar todas las figuras con la data SiL.
5. Reescribir Tabla IV con media ± IC95% + p-value Welch (Att vs Full
   por métrica), incorporando robustez a disturbio externo.
6. Actualizar Abstract/Conclusión para reflejar: "evaluado bajo
   disturbio random de fuerza externa en MuJoCo vía ROS 2 (SiL)".

**Fuera de alcance (confirmado con el usuario):**
- Vuelo real — no disponible.
- Baseline externo (Lee geometric / MPCC) — no disponible.

---

## Key file locations (quick reference)

| Propósito | Ruta |
|-----------|------|
| Config global (T_S, T_MAX, MASS, etc.) | `config/experiment_config.py` |
| NMPC solver builder (att/full) | `ocp/nmpc_gate_tracker.py` |
| Flatness bridge PMM→NMPC | `path_planing/reference_conversion.py` |
| PMM planner | `path_planing/path_time_3D_PMM.py` |
| Generador gates fig-8 | `path_planing/path_fig8.py` |
| Generador gates loop | `path_planing/path_loop.py` |
| MiL (RK4 interno, statistical N=10) | `experiments/mil_gate_experiment.py` |
| **SiL (ROS 2 + MuJoCo real)** | `experiments/sil_gate_experiment.py` |
| Plotting results | `experiments/plot_gate_results.py` |
| Preview 3-D PMM circuits | `path_planing/_preview_two_circuits.py` |
| ROS interface | `ros2_interface/{mujoco_interface,reset_sim}.py` |
| Plantilla comparable (SiL nominal) | `nmpc_mujoco_node.py` |
| Paper LaTeX | `ACCESS_latex/access.tex` |
