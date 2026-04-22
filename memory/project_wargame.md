---
name: Pursuit-Evasion NMPC-CBF War Game Paper
description: RAL paper + code — adversarial quadrotor pursuit-evasion using NMPC with two embedded degree-2 HOCBFs (separation + workspace). Active development.
type: project
---

## Idea central

Two-drone "war game": pursuer (missile) vs. evader. Decoupled architecture — cada dron corre su propio NMPC de forma independiente.

**Evader:** NMPC + 2× degree-2 HOCBF embebidos como hard constraints en el OCP:
1. Separación mínima: ψ₀ = ||p_e - p_p||² - d_min² ≥ 0
2. Workspace containment: φ₀ = r_ws² - ||p_e - p_c||² ≥ 0

**Pursuer:** NMPC estándar (sin CBF) que trata x_e como referencia móvil.

**Why:** el escenario "cornering" (pursuer empuja al evader contra la pared del workspace) hace que λ > 0 y μ < 0 simultáneamente → un CBF-QP post-hoc se vuelve infeasible. El NMPC los resuelve joint. Este es el aporte estructural real.

---

## Estado del código

### Archivos creados/modificados

| Archivo | Estado | Descripción |
|---|---|---|
| `models/quadrotor_wargame_model.py` | ✅ listo | Modelo aumentado z∈ℝ¹⁹ = [x_e(13), p_p(3), v_p(3)]; ψ₀,ψ₁,λ,Φ + φ₀,φ₁,μ,Ψ simbólicos en CasADi |
| `ocp/nmpc_evader_cbf.py` | ✅ listo | OCP evader: 6 con_h_expr (3 sep + 3 ws), costo -β·ψ₀, acados SQP-RTI |
| `ocp/nmpc_pursuer.py` | ✅ listo | OCP pursuer estándar, costo de tracking x_e |
| `wargame_mil_sim.py` | ✅ listo | Simulación MiL con RK4, escenarios free/cornering, baseline CBF-QP |
| `wargame_mujoco_node.py` | ⚠️ esqueleto | Loop MuJoCo — falta adaptar scene.xml y offsets qpos/qvel |
| `RAL/paper.tex` | ✅ 8 páginas | Intro + Tabla comp. + Sección II (preliminares completos) + Sección III (HOCBF + workspace) + Sección IV (impl.) + placeholders V,VI |

### Parámetros clave (config/experiment_config.py)
- m = 1.08 kg, g = 9.81, τ_rc = 0.03 s
- T ∈ [0, 3g] N, ω_cmd ∈ [-3, 3] rad/s
- Δt = 0.01 s (100 Hz), N = 50 steps (T_H = 0.5 s)

### CBF parámetros (defaults en ocp/nmpc_evader_cbf.py)
- d_min = 0.5 m, γ₁ = γ₂ = 1.0
- r_ws = 4.0 m, p_c = (0, 0, 1.5), κ₁ = κ₂ = 1.0
- β = 5.0, β_N = 10.0, R_T = 0.5, R_ω = 5.0

---

## Matemática del paper (Secciones II-IV escritas)

**Grado relativo (resultado clave):**
- h_sep = ||p_e - p_p||² - d_min² → grado relativo 2 desde T (no 4)
- h_ws  = r_ws² - ||p_e - p_c||² → grado relativo 2 desde T (mismo análisis)
- Grado 4 solo aplica para ω_cmd — T proporciona el camino más corto

**Augmented state:** z = [x_e(13); p_p(3); v_p(3)] ∈ ℝ¹⁹
- Pursuer dynamics embebidos: ṗ_p = v_p, v̇_p = 0 (constant velocity)
- ψ₀ y φ₀ son funciones puras de z → no necesitan parámetros externos en acados

**OCP constraints (con_h_expr, cada stage k):**
```
h = [ψ₀; ψ₁; λ·T + Φ + γ₂·ψ₁;   ← separación
     φ₀; φ₁; μ·T + Ψ + κ₂·φ₁]   ← workspace
≥ 0
```

---

## Plan de trabajo — próximas sesiones

### Paso 1 — Validación MiL (PRIORITARIO)
```bash
cd /home/bryansgue/dev/ros2/NMPC_baseline
python wargame_mil_sim.py --scenario free
python wargame_mil_sim.py --scenario cornering
```
- Verificar que los solvers de acados compilan correctamente
- Confirmar ψ₀ ≥ 0 y φ₀ ≥ 0 durante toda la simulación
- Ajustar γ₁, γ₂, κ₁, κ₂ si hay infeasibilidad

**Posibles errores al compilar:**
- El model name 'Drone_wargame_evader' puede chocar con archivos generados previos en c_generated_code/ → borrar y regenerar
- `f_system_model()` retorna el mismo model_name 'Drone_ode_rate_ctrl' para ambos drones → el pursuer solver debe usar un model_name diferente en `nmpc_pursuer.py` (cambiar a 'Drone_ode_rate_ctrl_pursuer')

### Paso 2 — Tuning de ganancias CBF
- Objetivo: ψ₀_min > 0 sin que el evader quede paralizado
- Empezar con γ₁ = γ₂ = 0.5 si hay chattering
- Si HOCBF infeasible en cornering: reducir d_min o γ₂

### Paso 3 — Escena MuJoCo con 2 drones
- Necesita `mujoco_model/scene.xml` con dos drones (body names, actuators, contacts)
- Adaptar `extract_state_from_mujoco()` en `wargame_mujoco_node.py` a los offsets reales del XML
- Verificar que contact events son detectables

### Paso 4 — Experimentos SiL (paper resultados)
Tres escenarios:
1. **free**: pursuer chases from 3m, evader has no workspace pressure
2. **cornering**: pursuer pushes evader to r_ws boundary (el aporte clave)
3. **speed_advantage**: pursuer 20% faster — test recursive feasibility condition

Métricas a reportar:
- Violation rate sep. (%) para NMPC+HOCBF vs. CBF-QP vs. NMPC sin CBF
- Violation rate ws. (%)
- Min observed separation [m]
- Mean/max solve time [ms]
- Contact events in MuJoCo (colisiones reales)

### Paso 5 — Escribir Sección V (Results) y VI (Conclusion)
- Placeholder ya existe en paper.tex
- Llenar con números reales del Paso 4
- Tabla II: comparativa cuantitativa de los 3 métodos × 3 escenarios

---

## Paper — estado actual (RAL/paper.tex)

- **8 páginas compiladas**, sin errores LaTeX
- Secciones I–IV completas con toda la matemática
- Secciones V–VI: placeholders con TODO
- El DQ-MPCC paper original (anterior contenido de paper.tex) es recuperable con `git checkout <hash> -- RAL/paper.tex` (commit e566068 o anterior)

**Target venue:** IEEE RA-L + ICRA 2026

**Why:** NMPC-CBF está completamente dominado por cooperative settings. Nadie tiene evader-side HOCBF en NMPC + workspace constraint + cornering analysis para quadrotors con body-rate control. Es el gap más limpio encontrado en el estado del arte (2022-2025).

---

## Nota importante sobre el papel del DQ-MPCC

El archivo RAL/paper.tex fue sobreescrito con el nuevo paper de pursuit-evasion. El DQ-MPCC paper original puede recuperarse de git. Si se necesita volver a ese paper, usar `git show e566068:RAL/paper.tex`.
