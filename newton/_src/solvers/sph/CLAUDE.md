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
| `sph_kernels.py` | Smoothing kernel `@wp.func`s and every per-particle `@wp.kernel` (density w/ Shepard, velocity gradient, strain rate, stress force, artificial viscosity, XSPH, integrators) | `wendland_c2_3d`, `wendland_c2_grad_3d`, `cubic_spline_3d`, `cubic_spline_grad_3d`, `compute_density_kernel`, `compute_velocity_gradient_kernel`, `compute_strain_rate_kernel`, `compute_stress_force_kernel`, `compute_artificial_viscosity_kernel`, `xsph_correction_kernel`, `integrate_symplectic_euler_kernel`, `half_step_position_kernel`, `integrate_verlet_final_kernel` |
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
| `dilatancy` | `wp.float32` | `0.0` | rad | **dead** (passed to DP kernel signature, never used inside) |
| `viscosity` | `wp.float32` | `0.0` | Pa·s | μ(I) only |
| `yield_pressure` | `wp.float32` | `1.0e12` | Pa | **dead** |
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
> default density path. If you edit density, preserve Shepard + its
> dummy-substitution branch.

### Optional δ-SPH-equivalent smoothing pass

`make_smooth_density_kernel(has_dummies)` (`sph_kernels.py`) provides
an optional Marrone-style post-summation density smoothing pass, gated
on `Config.density_smoothing_delta ∈ [0, 1]` (default `0.0` → skipped).
When active, it computes a Shepard-normalized smoothed estimate
`ρ̃_i = Σ_j m_j W_ij / Σ_j (m_j / ρ_j) W_ij` and blends
`ρ_out = (1 − δ) · ρ_in + δ · ρ̃`. Dummy neighbors substitute
`ρ_j ← reference_density` (same branch as the main density kernel).
The smoothing kernel self-includes `j == i` per §6 density exception.
The pass runs once per substep in both Symplectic Euler and
Position-Verlet paths (`solver_sph.py:_smooth_density`); on Verlet it
operates on the midpoint density. Canonical Antuono / Marrone δ-SPH
based on `dρ/dt + δ-term` is **not** applicable here because Newton
recomputes density via fresh Shepard summation; this pass is the
formulation-equivalent stabilizer.

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

## 11. Dead / silently-ignored knobs

Do not trust these — they compile and are accepted silently.

| Knob | Declared in | Claimed purpose | Reality |
|---|---|---|---|
| `Config.kernel_type` | `solver_sph.py` | Pick kernel | Ignored. Only `wendland_c2_3d` is called from any `@wp.kernel`. |
| `Config.restitution` | `solver_sph.py` | Inelastic collision restitution | Never read after construction. |
| `Config.viscous_damping` | `solver_sph.py` | Linear velocity damping | Never read after construction. |
| `Model.sph.dilatancy` | `register_custom_attributes` | Non-associative DP flow rule | Passed into `update_stress_dp_kernel` signature, not used inside. |
| `Model.sph.yield_pressure` | `register_custom_attributes` | Yield-pressure cap | Never read. |
| `cubic_spline_3d` / `cubic_spline_grad_3d` | `sph_kernels.py` | Alternate kernel | Defined, never called. |

**Rule**: do not add another silently-ignored knob. Either wire it
before exposing it, or do not expose it. If you wire one of the rows
above, delete that row from this table.

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
