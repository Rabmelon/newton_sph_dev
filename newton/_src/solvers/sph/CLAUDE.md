# SPH Solver — Folder Guide

Scope: guidance for Claude Code sessions editing files **inside**
`newton/_src/solvers/sph/`. This extends (and where noted, corrects)
the project-level SPH section in `/AGENTS.md`; it does **not** replace
it. Read both.

---

## 1. Purpose

Smoothed Particle Hydrodynamics solver for granular / geotechnical
flows on top of Warp + Newton's `Model` / `State` infrastructure.
Targets Drucker-Prager elastic-plastic and μ(I) rheology. Not
differentiable.

## 2. Public API & import boundary

- `__init__.py` re-exports **only** `SolverSPH`
  (`__all__ = ["SolverSPH"]`).
- End-user / example / test path: `from newton.solvers import SolverSPH`.
- Direct imports from `newton._src.solvers.sph.*` outside this folder
  are forbidden, with one documented exception: unit tests in
  `newton/tests/test_sph.py` import the internal helpers
  `dp_return_mapping`, `drucker_prager_params`,
  `mat33_double_contraction` from `sph_constitutive`. Do not widen the
  exception set.

## 3. File-by-file map

| File | Responsibility | Primary symbols |
|---|---|---|
| `__init__.py` | Re-export `SolverSPH` | `SolverSPH` |
| `solver_sph.py` | Orchestrator. `Config` dataclass, neighbor build, substep dispatch, CFL, boundary dispatch, gravity + ground-plane caching, attribute registration, dummy-particle builder helper, geostatic init, CFL-dt compute | `SolverSPH`, `SolverSPH.Config`, `SolverSPH.register_custom_attributes`, `SolverSPH.add_dummy_particles`, `SolverSPH.initialize_geostatic_stress`, `SolverSPH.compute_cfl_dt`, `SolverSPH.notify_model_changed` |
| `sph_model.py` | Thin wrapper. Holds `particle_volume = m / ρ₀` | `SPHModel` |
| `sph_kernels.py` | Smoothing kernel `@wp.func`s and every per-particle `@wp.kernel` (density w/ Shepard, velocity gradient, strain rate, stress force, artificial viscosity, XSPH, integrators) | `wendland_c2_3d`, `wendland_c2_grad_3d`, `compute_density_kernel`, `compute_velocity_gradient_kernel`, `compute_strain_rate_kernel`, `compute_stress_force_kernel`, `compute_artificial_viscosity_kernel`, `make_xsph_correction_kernel`, `integrate_symplectic_euler_kernel`, `half_step_position_kernel`, `integrate_verlet_final_kernel` |
| `sph_constitutive.py` | Stress update kernels (DP + μ(I)), DP helpers, geostatic init kernel | `update_stress_dp_kernel`, `update_stress_mui_kernel`, `initialize_geostatic_stress_kernel`, `drucker_prager_params`, `dp_return_mapping`, `deviatoric_stress`, `hooke_stress_increment`, `mat33_double_contraction` |
| `sph_boundary.py` | Penalty ground-plane kernel + Coulomb friction with damping regularisation. The actual per-plane launch loop is in `SolverSPH._apply_boundary_forces` (uses cached `_ground_planes`), **not** in this file. | `ground_plane_penalty_kernel` |
| `sph_dummy_boundary.py` | Particle-type constants, virtual velocity/stress `@wp.func`s, CPU layered particle generator | `SPH_FLUID`, `SPH_DUMMY_NOSLIP`, `SPH_DUMMY_FREESLIP`, `compute_virtual_velocity`, `compute_virtual_stress`, `generate_dummy_particles`, `add_dummy_particles_to_builder` |

## 4. Invariants — MUST hold after any change

1. **Non-differentiable.** Every module sets
   `wp.set_module_options({"enable_backward": False})` at import time.
   Any new module here must do the same.
2. **Integration-center guard.** Every `@wp.kernel` that iterates over
   particles as integration centers must self-guard:
   ```python
   if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
       return
   if particle_type[i] != SPH_FLUID:
       return
   ```
   Dummies are never integrated; they contribute only as neighbors.
3. **Dummy neighbor substitution.** When a neighbor `j` satisfies
   `particle_type[j] != SPH_FLUID`, substitute:
   - velocity → `compute_virtual_velocity(...)`
   - stress  → `compute_virtual_stress(...)`
   - density → `reference_density`
   Do not skip dummies as neighbors — that re-introduces the kernel
   truncation error these boundaries exist to fix.
4. **`state_in → state_out` write pattern.** Stress-update kernels
   read `state_in.sph.stress`, write `state_out.sph.stress`. Force
   kernels in the same substep then consume `state_out.sph.stress`.
5. **Stress / pressure conventions.** Cauchy stress is `wp.mat33`,
   compression negative. `P = -tr(σ) / 3`.
6. **Z-up.** Geostatic init (`z_i = pos[i][2]`) and ground plane
   normal extraction (rotate `wp.vec3(0, 0, 1)` by shape quaternion)
   both assume Z-up. Any coordinate-system change must update both.
7. **Warp annotation style.** Prefer bracket form
   (`wp.array[wp.vec3]`, `wp.array[wp.mat33]`). Several kernels in
   this package still use `wp.array(dtype=...)` — notably in
   `sph_constitutive.py`, `sph_boundary.py`, and `sph_model.py`. Do
   not propagate that style to new code; converge toward bracket
   form when editing an existing kernel.
8. **Naming.** `@wp.func` smoothing kernels: `<name>_<dim>d` /
   `<name>_grad_<dim>d`. Every `@wp.kernel`: `_kernel` suffix.
9. **No GPU↔CPU sync inside `step()`.** Gravity is cached as
   `self._gravity_vec: wp.vec3` in `__init__` and refreshed only by
   `notify_model_changed`. Ground planes are cached as
   `self._ground_planes: list[tuple[wp.vec3, float]]` by
   `_extract_ground_planes()` and invalidated only on
   `SolverNotifyFlags.SHAPE_PROPERTIES`. Do not call `.numpy()` on
   Warp arrays in the hot path.
10. **NaN / Inf velocity guard.** Present in
    `integrate_symplectic_euler_kernel` and
    `integrate_verlet_final_kernel`; zeros any particle whose new
    velocity is non-finite. Do not remove — a silent NaN corrupts
    every subsequent step via the neighbor loop.
11. **Fluid contiguity.** Fluids should occupy the prefix of the
    particle array. `SolverSPH.__init__` checks this once and caches
    `self._fluid_count`. On violation it emits a `RuntimeWarning`
    (`"Fluid particles are not contiguous at the front of the
    particle array. Falling back to launching kernels with
    dim=particle_count."`) and launches every kernel over the full
    array. Correctness preserved, performance degraded. Do not
    silence the warning.

## 5. Dual-namespace schema

Per-particle **material** fields registered on `Model.sph.*`
(via `SolverSPH.register_custom_attributes(builder)`):

| Name | dtype | Default | Unit | Notes |
|---|---|---|---|---|
| `young_modulus` | `wp.float32` | `1.0e6` | Pa | — |
| `poisson_ratio` | `wp.float32` | `0.3` | — | — |
| `friction` | `wp.float32` | `0.5` | rad | Internal friction angle |
| `cohesion` | `wp.float32` | `0.0` | Pa | — |
| `viscosity` | `wp.float32` | `0.0` | Pa·s | μ(I) only |
| `particle_type` | `wp.int32` | `0` | — | 0 fluid / 1 dummy no-slip / 2 dummy free-slip |
| `wall_normal` | `wp.vec3` | `wp.vec3(0.0)` | unit | Required for every dummy; never read for fluids |

Per-particle **evolving** fields on `State.sph.*`:

| Name | dtype | Unit |
|---|---|---|
| `density` | `wp.float32` | kg/m³ |
| `pressure` | `wp.float32` | Pa (`P = -tr(σ)/3`) |
| `stress` | `wp.mat33` | Pa (Cauchy, compression negative) |
| `strain_rate` | `wp.mat33` | 1/s |
| `velocity_gradient` | `wp.mat33` | 1/s |
| `plastic_strain` | `wp.float32` | — |

All evolving fields default to the zero value of their dtype
(`0.0`, `wp.mat33(0.0)`), registered in
`SolverSPH.register_custom_attributes`.

Adding a field = register it in
`SolverSPH.register_custom_attributes`, then thread it through the
relevant kernels via the `state_in` / `state_out` pattern.

## 6. Canonical kernel pattern

```python
@wp.kernel
def <op>_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    ...,
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    h: float,
    support_radius: float,
    reference_density: float,
    # output
    out: wp.array[...],
):
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    xi = pos[i]
    acc = <zero>

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
            r_vec = xi - pos[j]
            r = wp.length(r_vec)
            if r < support_radius and r > _EPSILON:
                # dummy substitution if needed:
                # if particle_type[j] != SPH_FLUID:
                #     v_j = compute_virtual_velocity(...); rho_j = reference_density
                #     sigma_j = compute_virtual_stress(...)
                # accumulate using wendland_c2_3d / wendland_c2_grad_3d
                ...

    out[i] = acc  # or out[i] += acc if summing with prior contribution
```

`_EPSILON = 1.0e-8` and `_PI` live as `wp.constant`s at the top of
`sph_kernels.py`. Reuse them; do not redefine.

**Exception — density kernel self-inclusion.** `compute_density_kernel`
intentionally omits the `j != i` guard: particle `i` must receive its
own mass contribution `m_i W(0, h)` in the density sum. Every other
hash-grid kernel in this package uses `j != i`. If you copy this
pattern to build another density-like summation, preserve the
exception; if you copy it for anything else, keep `j != i`.

## 7. Time integration

Dispatched by exact string in `SolverSPH.step()`:

```python
if self._config.integration_scheme == "position_verlet":
    self._step_position_verlet(state_in, state_out, dt)
else:
    self._step_symplectic_euler(state_in, state_out, dt)
```

Unknown strings raise `ValueError` — validated at `Config.__post_init__`
and re-checked in each `_step_*` via `else: raise`.

- **Symplectic Euler** (1st order, default-ish): density → velocity
  gradient → strain rate → stress update → stress force →
  artificial viscosity → integrate → optional XSPH.
- **Position-based Verlet** (2nd order — Zhang et al. 2024,
  Computers and Geotechnics 167:106052): half-step position via
  `half_step_position_kernel` into scratch buffer
  `self._pos_mid: wp.array[wp.vec3]` → neighbor rebuild at midpoint
  → density & stress at midpoint → `integrate_verlet_final_kernel`
  updates velocity then position from the midpoint configuration.
  The `density_prev_source` for Shepard is explicitly
  `state_in.sph.density` (the pre-step density), not mid-step.
- Velocity is clamped to `model.particle_max_velocity` inside both
  integration kernels.
- NaN/Inf velocity → zeroed (see invariant 10).

## 8. Density — Shepard-corrected

`compute_density_kernel` (`sph_kernels.py`) implements

```
ρ_i = Σ_j m_j W_ij  /  Σ_j (m_j / ρ_j^prev) W_ij
```

- Dummy neighbors use `reference_density` in the Shepard denominator
  because their density is never computed.
- On the very first step, `ρ_j^prev = 0` for fluid neighbors →
  Shepard denominator is below `_EPSILON` → falls back automatically
  to raw summation `ρ_i = Σ_j m_j W_ij`. Dummy neighbors are still
  added via `reference_density` (separate `elif` branch in the
  kernel), so fluid particles near walls get a partially
  Shepard-corrected density even on step 1. No first-step special
  case is needed at the Python level.
- The solver keeps a scratch `self._density_prev` copied from
  `state_in.sph.density` at the top of each substep.

> **Note (supersedes `/AGENTS.md`)**: The project-level AGENTS.md
> currently says "Density is computed by direct summation; there is
> no continuity equation, Shepard filter, or δ-SPH term." The
> Shepard filter was added in commit `ebb4fbe9` and is now the
> default density path. δ-SPH is still absent. If you edit density,
> preserve Shepard + its dummy-substitution branch.

## 9. Boundary treatments

Selected by `Config.boundary_type ∈ {"penalty", "dummy"}`.

### Penalty — `sph_boundary.py`

- `ground_plane_penalty_kernel` uses cached plane normals/offsets
  from `self._ground_planes`.
- Normal force (only when signed distance `d = n·x + offset < 0`):
  `f_n = max(0, ke |d| − kd v_n) n`.
- Coulomb friction with damping regularisation:
  `|f_t| = min(μ |f_n|, kd |v_t|)`, applied opposite `v_t`, gated by
  `|v_t| > _EPSILON` to avoid stick-slip chatter.
- The per-plane launch loop lives in
  `SolverSPH._apply_boundary_forces`, iterating `self._ground_planes`
  (cache built at `__init__` by `_extract_ground_planes`, refreshed
  only on `SolverNotifyFlags.SHAPE_PROPERTIES`).
### Dummy — `sph_dummy_boundary.py`

- Built at setup time via
  `SolverSPH.add_dummy_particles(builder, bounds_lo, bounds_hi, h,
  dx, reference_density, slip_type)` with
  `slip_type ∈ {"noslip", "freeslip"}`.
- Layer thickness `2h`, layer count `max(1, ceil(2 h / dx))`.
- Virtual **velocity** uses `Config.dummy_beta` (default `1.7`):
  - no-slip: `v_vir = (1 − β) v_fluid + β v_wall`
  - free-slip: mirror the normal component of `v_fluid − v_wall`
    with factor `β − 1`.
- Virtual **stress** uses an axis-aligned gravity-projection
  correction `diag(ρ₀ g_x Δx, ρ₀ g_y Δy, ρ₀ g_z Δz)` added to the
  fluid's own σ, independent of `dummy_beta`. Named "isotropic K₀ = 1"
  in the source/AGENTS for brevity, but the correction is only
  isotropic when gravity points along all three axes equally — under
  standard Z-up gravity `(0, 0, -g)`, only `σ_zz` gets a depth term;
  `σ_xx, σ_yy` inherit the fluid's stress unchanged. Free-slip flips
  the sign of the off-diagonal components (zero shear traction across
  the wall).
- `wall_normal` must be a unit outward vector per dummy; set by
  `add_dummy_particles_to_builder`. Fluid particles keep
  `wp.vec3(0.0)` — never read from a fluid.

## 10. Constitutive routes

Selected by `Config.simulation_method ∈ {"dp", "mui"}`.

### `"dp"` → `update_stress_dp_kernel`

- Elastic moduli: `K = E / (3(1 − 2ν))`, `G = E / (2(1 + ν))`.
- Hooke trial increment: `dσ = (2G D_dev + K tr(D) I) dt`.
- **Jaumann objective rate** (commit `ebb4fbe9`):
  `W = 0.5 (L − Lᵀ)`, `σ_trial = σ_prev + dσ + (W σ − σ W) dt`.
  Required for frame-indifference under large rotation — do not
  substitute a co-rotational or Truesdell rate without validating
  against the existing large-deformation tests.
- Drucker-Prager return mapping (`dp_return_mapping`) with three
  regimes: elastic / tension apex / smooth plastic.
- Plastic strain accumulated from the plastic stress jump.

### `"mui"` → `update_stress_mui_kernel`

- Tait EOS pressure, clamped ≥ 0.
- Effective viscosity `η = (c + P tan φ) / max(D_eq, ε)`.
- Viscosity cap: per-particle `viscosity[i]` if > 0, else
  `η_max = ρ₀ c_s h` (artificial-viscosity scale).
- `σ = 2 η D_dev − P I`; DP return mapping applied as a final safety.

### Geostatic init

`initialize_geostatic_stress_kernel`:
`σ_zz = −ρ₀ g (z_max − z_i)`,
`σ_xx = σ_yy = K₀ σ_zz`, `K₀ = 1 − sin φ`.
Z-up only. Skips non-fluid particles.

## 10b. Body coupling — `sph_body_coupling.py`

Two-way coupling between SPH fluid particles and rigid bodies driven
by an external MBD solver (e.g. MuJoCo, Featherstone). Enabled by
`Config.body_coupling_enabled` (default `False`).

### Config

| Field | Default | Unit | Meaning |
|---|---|---|---|
| `body_coupling_enabled` | `False` | — | Master switch. |
| `body_coupling_stiffness` | `2.0e4` | N/m | Normal penalty stiffness `k_n`. Calibrated to dry-sand bearing capacity (~10 kPa) at ~5 mm overlap. |
| `body_coupling_damping` | `5.0e1` | N·s/m | Normal damping `c_n` (resists approach only, `max(0, -v·n)`). |
| `body_coupling_friction` | `0.5` | — | Coulomb friction coefficient `μ`. |

> **2026-05-14 — Phase 2c partial MVP success (4/5).** The sphere-drop
> MVP at `newton/examples/multiphysics/example_sph_twoway_sphere_drop.py`
> passes 4 of 5 `test_final()` criteria after Fix A (wrench reset
> semantics) and Fix B (MBD inside SPH substep loop): no-NaN,
> penetration, settling-by-bounce-apex, no-leak. The remaining FAIL is
> criterion 5 (terminal-z within ±5 cm of analytic crater estimate):
> sphere reaches z ~= -1.2 m vs predicted -0.04 m because the default
> 10 cm sand bed (2× sphere radius) is too thin to arrest a 0.30 m drop;
> after 3 bounces the sphere slips through a vertical channel opened
> by laterally-displaced particles. The coupling code path is
> end-to-end functional; SPH baseline (`test_sph`, 14/14) unaffected.
> See "Known issues" subsection below for the criterion-2 artefact
> caveat and remaining tuning followups.

### Architecture

SDF + penalty per primitive shape (sphere / capsule along local +Z /
axis-aligned box). For each fluid particle inside a body's SDF:

```
F_n = (k_n d + c_n max(0, -v_rel·n)) n
F_t = -min(μ |F_n|, m_p |v_t| / dt) v_t / |v_t|   (chatter-regularised Coulomb)
```

Reaction `-F` is applied at the contact point and atomic-added into a
per-body `wp.spatial_vector` accumulator (`Model.body_count`-sized).
Layout matches Newton's `body_f`: `wp.spatial_vector(force_world,
torque_world)` — `wp.spatial_top` = linear, `wp.spatial_bottom` =
angular. The caller integrates the accumulator into `State.body_f`
before the MBD step (see `newton/examples/mpm/example_mpm_twoway_coupling.py`
for the wire-up pattern; the SPH example lives elsewhere — Phase 2c).

The accumulator is exposed via ``SolverSPH.collect_body_wrench(state) ->
wp.array[wp.spatial_vector] | None`` (matching MPM's
``collect_collider_impulses(state)`` signature). **Reset is automatic:**
``_apply_body_forces`` zeros the internal accumulator at the start of
each call, and the result is stored on ``state_out._sph_body_wrench``
at the end of each step path. The caller reads the wrench from the
output state after each ``step()`` returns::

    for _ in range(substeps):
        # Apply previous substep's sand wrench to body_f
        wrench = solver.collect_body_wrench(state_0)
        if wrench is not None:
            compute_body_forces(wrench, body_q, body_com, state_0.body_f)
        # MBD step
        mbd_solver.step(state_0, state_1, ...)
        # SPH step (auto-resets and re-accumulates wrench internally)
        sph_solver.step(state_0, state_1, ...)
        state_0, state_1 = state_1, state_0

The glue kernel ``compute_body_forces`` (in the example) converts the
wrench to body forces — see
``newton/examples/mpm/example_mpm_twoway_coupling.py`` for the reference
pattern.

### Collider table

`build_body_collider(model, device)` walks `model.shape_*` once at
solver init, keeping only shapes with `ShapeFlags.COLLIDE_PARTICLES`,
`shape_body >= 0`, and primitive type ∈ `{SPHERE, CAPSULE, BOX}`.
Mesh / plane / ellipsoid / hfield are warn-skipped (mesh-SDF is a
Phase 2 feature). Rebuilt on
`SolverNotifyFlags.SHAPE_PROPERTIES`.

Shape parameter packing (from `model.shape_scale`):
- Sphere: `(r, _, _)` — radius only.
- Capsule: `(r, half_height, _)` — axis along local **+Z**.
- Box: `(hx, hy, hz)` — half-extents.

### Risks: mitigation status

1. **Penalty CFL** — checked at init by `_check_penalty_cfl`; warns
   if `k_n · dt_cfl² / m_p_min > 4` (explicit-Euler 1-DOF stability
   bound).
2. **NaN cascade** — kernel skips particles with non-positive or
   non-finite density (`not (rho > 0.0)`) and clamps penetration to
   `pen ≥ 0`. Stress is **not** zeroed here (out of scope; that's the
   stress-update kernel's responsibility).
3. **Body-pose freshness** — body pose is **frozen** during a single
   `SolverSPH.step()` call (the kernel reads `body_q` once at launch).
   The caller is responsible for refreshing the pose between substeps;
   the sphere-drop example does this by co-stepping MBD at `sim_dt`
   inside the SPH substep loop (see Fix B in
   `example_sph_twoway_sphere_drop.py:step`). Without this, a body
   moving at O(2.5 m/s) across an outer frame_dt = 20 ms presents a
   step-function ~5 cm = ~10·dx intrusion to the first substep, which
   the penalty kernel resolves as a slab-impulse → sand explodes.
4. **First-contact elastic bounce** — **not** mitigated in the
   solver. The penalty coupling ``F = k_n·pen + c_n·max(0,-v_n)`` is
   purely elastic: the spring stores energy that returns to the body
   on separation (damping only resists approach). The body always
   bounces off the granular material. A plastic coupling scheme
   (e.g. Akinci-style kernel interpolation where the DP return
   mapping dissipates energy) is needed for true settling. MVP callers
   should expect bounce dynamics; the current sphere-drop example logs
   these bounces in telemetry.
5. **Z-up assumption** — `_init_body_coupling` raises `ValueError` if
   `model.up_axis != Axis.Z` (the SPH solver's geostatic init and
   ground-plane extraction both assume Z-up).

### Hooks in `SolverSPH.step()`

After `_apply_boundary_forces`, before `_integrate`, both step paths
launch `_apply_body_forces(...)` which zeros the internal accumulator,
launches the coupling kernel, and then stores the result on
``state_out._sph_body_wrench``. The Verlet path passes ``pos =
self._pos_mid`` and ``vel = state_in.particle_qd`` and reads density
from ``state_out.sph.density`` (the midpoint density, computed at step
3 of ``_step_position_verlet``).

### Known issues (2026-05-14)

**A. Wrench reset semantics — FIXED (2026-05-14, refactored 2026-05-17).**
   ``_apply_body_forces`` zeros ``_body_f_sand`` at the start of each call
   and the result is stored on ``state_out._sph_body_wrench`` at the end
   of each step path.  ``collect_body_wrench(state)`` reads from state
   (matching MPM's ``collect_collider_impulses(state)``).  The explicit
   ``reset_wrench_accumulator()`` method has been removed.

**B. Body pose frozen across substep loop — FIXED (2026-05-14) in the
   sphere-drop example via option (b)**: MBD runs at `sim_dt` inside
   the SPH substep loop, so the body pose advances by at most
   `v · sim_dt` (~5 μm at 2.5 m/s and `sim_dt = 40 μs`) between SPH
   evaluations. The slab-impulse pathology is gone: max total KE
   dropped from ~5870 J (Cycle 0, `k_n=1e5`) → 22 J after Fix B;
   particle leak from 23 → 0. Note: `SolverSemiImplicit` co-stepping
   is cheap because there's a single free body; for articulated
   robots with MuJoCo, option (a) — `body_q` interpolation across
   substeps — may be the better lever and remains TBD.

**C. Terminal-z FAIL — fundamental limitation of penalty coupling.**
   `test_final` criterion 5 (terminal sphere z near analytic crater
   estimate) cannot pass with the current coupling design. The
   penalty interface ``F = k_n·pen + c_n·max(0,-v_n)`` is inherently
   **elastic**: the spring term stores energy that returns to the
   sphere on separation (damping is off when ``v_n > 0``, i.e., during
   separation). The sphere always bounces off the sand regardless of
   drop height or damping coefficient. Tuning attempts:

   - Deeper sand bed (``--sand-bed-bottom < 0``): adds material below,
     but coupling impulse propagates through force chains and pushes
     bottom-layer particles through the ground plane — massive leaks.
   - Increased damping ``c_n`` > 200: destabilises per-particle
     explicit integration (``c_n·v·dt / m_p`` ≫ 1 for μgram particles).
   - Pure damping ``k_n=0``: no spring, but the SPH stress model
     provides the restoring stiffness independently — the sand bed
     still pushes back through SPH pressure, so bounce persists AND
     the lack of spring restraint makes per-particle dashpot unstable.
   - Lowered ``drop_height``: reduces KE but bounce persists; sphere
     launched at multiple m/s every time.

   The coupling end-to-end plumbing is correct; the physics limitation
   is that real granular energy dissipation (plastic strain, grain
   rearrangement) happens inside the SPH constitutive model, not at
   the coupling boundary. Fixing criterion 5 requires a **plastic
   coupling contact** law (e.g. Akinci-style interpolation through
   the SPH smoothing kernel so the DP return mapping handles energy
   dissipation) or a fundamentally different coupling scheme. Out of
   scope for the MVP.

**D. `test_final` criterion 2 (settling) passes by bounce-apex artefact.**
   The sphere bounces to ±5 m/s after each impact. At each bounce apex,
   ``|v_z| < 0.05 m/s`` momentarily, but the sphere is mid-flight not
   settled. `any(row['t'] >= 0.5 and abs(row['sphere_vz']) < 0.05)` 
   catches the apex and returns True. The **honest** 4/5 PASS tally
   (with criterion 2 artificially passing) overstates stability; the
   true physics pass count is 3/5 (no-NaN, penetration, no-leak).
   A correct check would window-min ``|v_z|`` over the last K frames
   or require mean ``|v_z|`` < threshold across an observation window.

**Status of validation (2026-05-14, after tuning session)**: `test_final`
3/5 honest PASS (no-NaN, penetration within 0.40 s below sand top, no
particle leak below ground plane). Criterion 2 (settling) passes by
bounce-apex artefact (not true settling). Criterion 5 (terminal-z vs
analytic crater estimate) cannot pass with the penalty coupling design
because the coupling is inherently elastic — the spring stores and
returns energy while damping is off during separation. See issues C-D
above. Run with::

    PYTHONPATH=/path/to/worktree UV_PROJECT_ENVIRONMENT=.../.venv \\
    uv run --no-sync --extra dev python \\
    newton/examples/multiphysics/example_sph_twoway_sphere_drop.py \\
    --viewer null --test

The example now supports ``--sand-bed-bottom`` (default 0.0) to extend
the sand bed below z=0 for deeper energy-absorbing columns; use with
caution — deep beds leak particles during impact unless
``--penalty-stiffness`` is increased to ≥ 5e6.

Court archive `.claude/.court/20260513-sph-mbd-coupling-quadruped/`
retains the Phase 2d diagnostic chain; the 2026-05-14 tuning verdict
reproduces from current worktree state.

## 11. Dead / silently-ignored knobs

None at present. **Rule**: do not add silently-ignored knobs. Either
wire a new option through to a kernel before exposing it, or do not
expose it.

## 12. Deltas vs. `/AGENTS.md`

| AGENTS.md claim | Status |
|---|---|
| Virtual stress: "gravity-based hydrostatic (no `dummy_beta`), with off-diagonal sign flip for free-slip" | **Imprecise wording**. The correction is axis-aligned `diag(ρ₀ g · Δx)`, not isotropic. Under Z-up gravity only `σ_zz` picks up a depth term (see §9 dummy). The sign-flip claim itself is correct. |

Everything else in the `/AGENTS.md` "SPH solver" section matches the
current code. When updating either document, keep this delta table
in sync (or delete rows as they are fixed upstream).

## 13. Verification

Single test file: `newton/tests/test_sph.py` (class `TestSPH`).

Run everything:

```bash
uv run --extra dev -m newton.tests -k test_sph
```

Targeted tests (copy the exact name of the method you want):

```bash
uv run --extra dev -m newton.tests -k test_density_uniform_grid
uv run --extra dev -m newton.tests -k test_drucker_prager_return_mapping
uv run --extra dev -m newton.tests -k test_sand_cube_on_plane
uv run --extra dev -m newton.tests -k test_geostatic_initialization
uv run --extra dev -m newton.tests -k test_sand_cube_dummy_boundary
uv run --extra dev -m newton.tests -k test_sand_cube_position_verlet
```

Interactive examples:

```bash
uv run -m newton.examples sph_granular
uv run -m newton.examples sph_grain_rendering
```

Lint before committing:

```bash
uvx pre-commit run -a
```

## 14. Gotchas

- **No `wp.synchronize()` before `.numpy()`.** `.numpy()` on a Warp
  array is already a synchronous device-to-host copy. The extra call
  is dead overhead. Project-wide rule (see `/AGENTS.md`).
- **`Config` is nested.** Instantiate as `SolverSPH.Config(...)`, not
  any module-level alias.
- **Private derived attributes.** `_h`, `_support_radius`,
  `_hash_grid`, `_accel`, `_gravity_vec`, `_ground_planes`,
  `_density_prev`, `_pos_mid`, `_cfl_warned`, `_dt_cfl_static`,
  `_fluid_count`. Set in `__init__`, refreshed only via
  `notify_model_changed` when flagged. Don't expose, don't mutate
  from outside the solver.
- **CFL is acoustic-only.** Velocity-aware estimate via
  `solver.compute_cfl_dt(state, courant_number=0.3)`. The static
  `_dt_cfl_static = 0.3 · h / c_s` is the velocity-free threshold
  used only for the one-shot over-CFL `RuntimeWarning`.
- **Tests use `unittest`, not pytest.** Project-wide.
- **SPDX header year.** Use the year the file was first created; do
  not introduce year ranges or bump on edit.
