# Newton Development Guidelines

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`. Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Breaking changes require a deprecation first. Do not remove or rename public API symbols without deprecating them in a prior release.
- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with bracket syntax (`wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[Any]`), not the parenthesized form (`wp.array(dtype=...)`). Use `wp.array[X]` for 1-D arrays, not `wp.array1d[X]`.
- Follow Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
- Run `docs/generate_api.py` when adding public API symbols.
- Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.
- Create a feature branch before committing — never commit directly to `main`. Use `<username>/feature-desc`.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.
- Verify regression tests fail without the fix before committing.
- Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.
- In SPDX copyright lines, use the year the file was first created. Do not create date ranges or update the year when modifying a file.

Run `uvx pre-commit run -a` to lint/format before committing. Use `uv` for all commands; fall back to `venv`/`conda` if unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Tests

Always use `unittest`, not pytest.

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

### Testing guidelines

- Never call `wp.synchronize()` or `wp.synchronize_device()` right before `.numpy()` on a Warp array. This is redundant as `.numpy()` performs a synchronous device-to-host copy that completes all outstanding work.

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- If a change modifies user-facing behavior, append an entry at the end of the correct category (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`) in `CHANGELOG.md`'s `[Unreleased]` section. Use imperative present tense ("Add X") and avoid internal implementation details.
- For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: "Deprecate `Model.geo_meshes` in favor of `Model.shapes`".

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` — runs after the example completes to verify simulation state is valid.
  - Optionally implement `test_post_step()` — runs after every `step()` for per-step validation.
- Register in `README.md` with `python -m newton.examples <name>` command and a 320x320 jpg screenshot.

## SPH solver

- Solver lives at `newton/_src/solvers/sph/`; only `SolverSPH` is re-exported from its `__init__.py`. Keep the package internal — user code must import via `newton.solvers.SolverSPH`.
- Dual-namespace model: per-particle **material** parameters on `Model.sph.*` (`young_modulus`, `poisson_ratio`, `friction`, `cohesion`, `dilatancy` [dead], `viscosity`, `yield_pressure` [dead], `particle_type`, `wall_normal`); per-particle **evolving** fields on `State.sph.*` (`density`, `pressure`, `plastic_strain`, `stress`, `strain_rate`, `velocity_gradient`). New attributes must be registered in `SolverSPH.register_custom_attributes()`.
- `particle_type`: `0` fluid, `1` dummy no-slip, `2` dummy free-slip. Dummies are skipped as integration centers in every Warp kernel (each kernel self-guards on `particle_type`). Fluids should be contiguous at the front of the particle array so kernels can launch over the fluid subset only; if not, the runtime emits a `RuntimeWarning` and falls back to full-array launches — correctness is preserved, performance degrades.
- Cauchy stress is a `wp.mat33` in `State.sph.stress`; pressure is `P = -tr(σ)/3` in `State.sph.pressure`. Stress-update kernels follow the `state_in → state_out` write pattern.
- Kernel function naming: `<name>_<dim>d` (e.g. `wendland_c2_3d`); gradients `<name>_grad_<dim>d` taking `r_vec: wp.vec3`. Warp kernels are suffixed `_kernel` and follow `wp.tid() → bounds check → hash_grid_query → neighbor loop → accumulate`.
- All SPH modules set `wp.set_module_options({"enable_backward": False})` — the solver is not differentiable. New modules in this package must match.
- Config lives in the nested dataclass `SolverSPH.Config`. String-enum options: `simulation_method ∈ {"dp", "mui"}`, `boundary_type ∈ {"penalty", "dummy"}`. Derived attrs (`_h`, `_support_radius`, `_hash_grid`, `_accel`) are private.
- Neighbor search is `wp.HashGrid` (3D) with `support_radius = support_radius_factor · h`, rebuilt every step. Density is computed by Shepard-corrected direct summation; there is no continuity equation or δ-SPH term.
- Time integration: `integration_scheme ∈ {"symplectic_euler", "position_verlet"}`. Symplectic Euler is 1st order; Position-based Verlet is 2nd order per Zhang et al. (2024) Computers and Geotechnics 167:106052 — forces are evaluated at the midpoint configuration. Velocity is updated before position and clamped to `model.particle_max_velocity`. A NaN/Inf guard zeroes corrupted velocities — do not remove it.
- CFL is acoustic-only: `dt = C · h / (c_s + ‖v‖_max)`, default Courant 0.3. Use `solver.compute_cfl_dt(state, courant_number)` for the velocity-aware estimate.
- Gravity is cached as a `wp.vec3` at init (from `Model.gravity`) to avoid GPU↔CPU sync — preserve this pattern when adding body forces.
- Geostatic init (`initialize_geostatic_stress`) assumes Z-up with K₀ = 1 - sin φ. Document any coordinate-system change.
- Dummy boundaries: generated via `SolverSPH.add_dummy_particles()` (CPU numpy), spacing `dx`, layer count `ceil(2h/dx)`. Virtual **velocity** extrapolation uses `dummy_beta` (default 1.7); virtual **stress** extrapolation is gravity-based hydrostatic (no `dummy_beta`), with off-diagonal sign flip for free-slip. Each dummy must carry a unit `wall_normal`.
- Known dead / unwired knobs — `Config`: `kernel_type`, `restitution`, `viscous_damping`. `Model.sph`: `dilatancy`, `yield_pressure`. Kernel funcs: `cubic_spline_3d`, `cubic_spline_grad_3d`. Either wire them before exposing a new one, or prune — do not add more silently-ignored options.
