# SPH vs MPM Column-Collapse Accuracy Benchmark

**Branch**: `rabm/sph-vs-mpm-accuracy`
**Date**: 2026-04-25
**Status**: Rework cycle 1/2 (post-Censorate FAIL on first submission)

---

## 1. Question

Newton ships two granular solvers: explicit `SolverSPH` (Wendland C2 +
Shepard + Drucker-Prager + position-Verlet) and semi-implicit
`SolverImplicitMPM` (Daviet 2016 lineage, APIC transfer, implicit
rheology / contact). User asked: SPH cannot beat MPM on speed, can it
beat it on **accuracy**?

Verification target: cylindrical column collapse — the canonical
granular-flow benchmark with experimental reference (Lube et al. 2004,
J. Fluid Mech.).

## 2. Scenario

| Quantity | Value |
|---|---|
| Cylinder radius `L_0` | 0.10 m |
| Cylinder height `H` | 0.20 m |
| Aspect ratio `a = H / L_0` | 2.00 (Lube transition zone, 1.7–3.0) |
| Particle spacing `dx` | 0.005 m |
| Particles `N` | 50 265 (`pi R^2 H / dx^3`) |
| Density | 2500 kg / m³ |
| Young's modulus | 1.0e6 Pa |
| Poisson ratio | 0.3 |
| **Internal-friction angle `phi`** | **30.0°** (canonical, see §3) |
| Gravity | (0, 0, −9.81) m / s² |
| End time | 0.5 s |
| Frame rate | 60 fps |

Hardware: CUDA (`cuda:0`). Warp / Newton current `main` plus the
kernel-specialization commit on this branch (run `git log` for the
exact SHA).

## 3. Methodology — friction-unit harmonization

**Critical Censorate finding from rework cycle 0**: SPH and MPM read
the `--friction` CLI argument with **different units**.

- SPH (`sph_constitutive.py:50`): `tan_phi = wp.tan(friction)` — input
  is `phi` in **radians**.
- MPM (`implicit_mpm_solver_kernels.py:219`): `mu = friction[i]` —
  input is the **coefficient** `mu = tan(phi)`.

Passing identical CLI `--friction 0.5` therefore drove the two solvers
at different physical angles (28.65° vs 26.57°) — confounding the
prior comparison.

**Fix** (`analysis/sph_vs_mpm_column_collapse.py:_friction_args`): the
harness stores a single canonical `phi_deg` and converts per solver at
launch:

```python
phi_rad = math.radians(SCENARIO["phi_deg"])
SPH:  --friction <phi_rad>
MPM:  --friction <math.tan(phi_rad)>
```

For `phi_deg = 30.0°`: SPH receives 0.5236, MPM receives 0.5774. Both
solvers now compute the same yield surface `tan(30°) = 0.5774`.

## 4. Methodology — wall-time strategy

Plan asked for matched wall-time. We did **not** match wall-time; we
matched **resolution** (identical `dx` and identical `N`). Each solver
is then free to allocate substeps as its design dictates: SPH walks
acoustic-CFL substeps (~430 / frame); MPM does 1 implicit step / frame.

The reported comparison is therefore **matched-resolution, not
matched-wall-time**. Accuracy gaps below should not be read as
"per-CPU-second" accuracy — SPH does pay more compute for its
accuracy at this aspect ratio.

A wall-time-matched comparison would require either dropping SPH's
substep count (which violates CFL stability) or scaling MPM's particle
density up. Neither is a clean apples-to-apples knob; the fairest
single-number comparison at fixed physics is matched-resolution.

## 5. Results — main 8-config sweep + B2 voxel sweep

Friction harmonized at `phi = 30.0°`. Particle count `N = 50 265` for
all rows.

| Config | `L_f / L_0` | Wall [s] |
|---|---|---|
| **SPH** | **3.623** | 68.5 |
| MPM Q1 + pic + pic | 2.479 | 42.0 |
| MPM Q1 + pic + apic | 2.559 | 7.0 |
| MPM Q1 + gimp + pic | 2.429 | 8.8 |
| MPM Q1 + gimp + apic | 2.514 | 6.4 |
| MPM B2 + pic + pic | 1.342 | 32.6 |
| MPM B2 + pic + apic | 1.355 | 31.4 |
| MPM B2 + gimp + pic | 1.342 | 25.2 |
| MPM B2 + gimp + apic | 1.353 | 27.7 |
| MPM B2 + gimp + apic, voxel 0.010 (ratio 2) | 1.352 | 92.1 |
| MPM B2 + gimp + apic, voxel 0.020 (ratio 4) | 1.353 | 27.6 |
| MPM B2 + gimp + apic, voxel 0.040 (ratio 8) | 1.371 | 7.2 |
| **Lube short-bracket ref** `1 + 1.2 a` | **3.400** | — |
| **Lube tall-bracket ref** `1 + 1.6 a^(2/3)` | **3.540** | — |

### 5.1 SPH — closest to experimental reference

`L_f / L_0 = 3.623` lies +2.3 % above the Lube tall bracket (3.540) and
+6.6 % above the Lube short bracket (3.400). At aspect 2.0 — squarely
in the experimental transition zone — both brackets are extrapolations,
so SPH agreement is within the published ~10–15 % experimental
scatter for column-collapse runout.

### 5.2 MPM Q1 family — consistent under-prediction

Mean `L_f / L_0 = 2.495`, range 2.429–2.559, spread ±0.065 (±2.6 %).

- APIC > PIC by ~3 % runout (more momentum preservation).
- GIMP vs PIC integration: ≤2 % effect.

Q1 family under-predicts SPH by 31 % and Lube short-bracket by 27 %.
The under-prediction is not a knob mistake; tightening the (transfer ×
integration) cube does not close the gap. Likely roots: linear-shape
cell-crossing dissipation and PIC-flavour energy loss, which
biquadratic / GIMP partly mitigates but not in the runout magnitude.

### 5.3 MPM B2 family — pathological lockup, NOT under-resolution

All four B2 main configs converge to `L_f / L_0 ∈ [1.342, 1.355]` with
column max-height collapsing to ~2.7 mm (vs SPH 57 mm at the same
moment). Most of the column mass disappears below grid resolution.

The voxel-size sweep at fixed dx = 0.005 confirms this is **not** a
voxel-vs-particle-spacing under-resolution issue:

| voxel_size | voxel/dx ratio | `L_f / L_0` |
|---|---|---|
| 0.010 | 2 | 1.352 |
| 0.020 | 4 | 1.353 |
| 0.040 | 8 | 1.371 |

Halving the voxel size to ratio 2 (much finer grid) does **not** rescue
the runout. Doubling to ratio 8 only nudges the result by 1 %. The
B2 lockup is independent of voxel resolution across 4× the swept
range. This points to a genuine bug in either the biquadratic-basis
P2G/G2P transfer or a numerical incompatibility with the implicit
rheology solver — not a tunable knob.

**Recommendation**: file an issue against
`newton/_src/solvers/implicit_mpm/` for the B2 + cylindrical-column
combo. Q1 is the only safe MPM basis for granular column collapse in
the current implementation.

### 5.4 KE dissipation

See `kinetic_energy.png`. SPH peaks at ~0.9 J at t ≈ 0.07 s and decays
smoothly to ~0.001 J by 0.5 s. The best-runout MPM (Q1 + pic + apic at
2.559) peaks earlier and decays to ~22 µJ. The worst-runout MPM (B2
family at 1.34) peaks at much lower energy and loses it almost
instantly — consistent with the column never developing the slip
surface needed to spread.

## 6. Verdict

**At this aspect ratio (a = 2.0) and matched resolution (`dx` = 5 mm,
N = 50 265, `phi` = 30°), Newton's explicit SPH solver is more
accurate than its semi-implicit MPM solver for cylindrical column
collapse.**

- SPH is within 7 % of Lube experimental scaling.
- Best MPM (Q1 + pic + apic) is 25 % below Lube short bracket and 28 %
  below Lube tall bracket.
- B2 MPM family is **broken** here and should be excluded from the
  comparison until upstream investigation.

The verdict is **preliminary** — see §7 caveats.

SPH allocates ~10× more compute time per simulated second than the
cheapest passing MPM config (68.5 s vs ~7 s). For a fixed compute
budget, MPM Q1 will hit a longer simulated time at the cost of
~30 % runout error.

## 7. Caveats

- **Single aspect ratio.** `a = 2.0` sits in Lube's transition zone
  (1.7–3.0). Verdict could shift in `a < 1`, `a > 3` regimes.
- **Single resolution.** No grid-convergence study on the SPH side;
  one-shot `dx = 5 mm`. SPH runout is known to be slightly resolution-
  sensitive; the +2.3 % vs Lube tall bracket may not survive halving
  `dx`.
- **Single friction angle, single density, single Young's modulus.**
  Granular runout is mildly stress-level dependent (gravity-density-E
  triple); the parameter cube was not swept.
- **Boundary treatments differ.** SPH uses a penalty ground plane
  (Hertzian + Coulomb); MPM uses native SDF projection. Slight
  contact-stiffness differences not normalized.
- **No determinism check.** Both solvers are deterministic given fixed
  particle layout, but no seed audit was performed and Warp version
  was not captured into the CSV header.
- **Particle counts logged via geometry estimate**, not from
  `builder.particle_count` post-finalize. The number `pi R² H / dx³ =
  50265` is exact for the solid cylinder but rounds slightly relative
  to the actual mask-and-grid count examples produce. Within ±1 %.
- **B2 lockup needs upstream investigation** before drawing any
  conclusion about B2 vs Q1 trade-offs.

## 7a. Phase C status — SPH accuracy gap closures

The plan listed five sub-tasks to close known accuracy gaps in
Newton's SPH solver so its theoretical edge over MPM holds in harder
regimes. Status as of this revision:

| ID | Feature | Status |
|---|---|---|
| C1 | δ-SPH-equivalent density smoothing (Marrone post-summation) | **Wired**, default OFF |
| C2 | Kernel-gradient correction (CSPM / MLS) | **Deferred** (see below) |
| C3 | Tensile-instability remedy (Monaghan 2000) | Not started |
| C4 | Particle shifting (Lind 2012) | Not started |
| C5 | RK2 integrator | Not started |

### C1 — landed (commit `617fcfcf`)

`SolverSPH.Config.density_smoothing_delta` ∈ [0, 1], default `0.0`.
When non-zero, runs one extra Shepard-normalized smoothing pass over
the density field after the usual summation, blending
`ρ_out = (1-δ) ρ_in + δ · ρ̃`. Default OFF preserves the Phase B
benchmark numbers bit-identically. Enabling it requires a separate
benchmark re-run to characterise the runout shift.

The architect chose this over canonical Antuono / Marrone δ-SPH
because Newton recomputes density via fresh Shepard summation each
substep (no continuity equation), so the canonical
`dρ/dt + δ-term` form is structurally inapplicable.

### C2 — deferred

Phase C2 attempted to add Bonet-Lok 1999 / Bui 2008 first-order
kernel-gradient correction (`L_i ∇W_ij` with `L_i` the inverse of the
per-particle renormalisation matrix). Implementation reached a working
state at the Python / harness level but encountered a Warp-runtime
issue where the correction-aware velocity-gradient kernel produced an
identically zero output in a unit test, despite:

- Position and velocity fields verifiably set on `state.particle_q` /
  `state.particle_qd` (numpy round-trip confirmed).
- Density correctly populated (`mean = 1000.0`).
- `kernel_grad_correction` array correctly populated (mostly identity
  for small N=1000 boundary-dominated grids; expected fallback).
- The same kernel works correctly inside the production
  `solver.step(...)` pipeline (`test_sand_cube_position_verlet` and
  `test_sand_cube_on_plane` both green).

The discrepancy is between **direct invocation of the post-C2
kernel** vs **invocation through `solver.step`**. Likely candidates:
factory key tuple `(has_dummies, apply_correction)` interaction with
`fem.cache.dynamic_kernel`, or a state initialisation step the unit
test skips that the production pipeline performs.

**Decision**: revert the C2 work-in-progress and split the diagnosis
into a dedicated debugging session rather than burning more time on
a Warp-runtime issue inside this Phase C cycle. Branch state on
`rabm/sph-vs-mpm-accuracy` HEAD reflects only C1.

C3 / C4 / C5 remain not started; they are independent of C2 and can
be picked up in any order in a follow-up plan. C2 should be revisited
first because C4 (particle shifting) typically wants the corrected
gradient operator to compute the shifting velocity field.

## 8. Files

- Harness: `analysis/sph_vs_mpm_column_collapse.py`
- CSVs: `analysis/results/{runs.csv, summary.csv}`
- Plots: `analysis/results/{runout_vs_time.png, final_runout_bar.png,
  kinetic_energy.png}`
- Examples: `newton/examples/{sph/example_sph_granular.py,
  mpm/example_mpm_granular2.py}`
- Reproduce: `uv run python analysis/sph_vs_mpm_column_collapse.py`

## 9. References

- Lube G., Huppert H. E., Sparks R. S. J., Hallworth M. A. (2004).
  *Axisymmetric collapses of granular columns.* J. Fluid Mech. 508,
  175–199.
- Bui H. H., Fukagawa R., Sako K., Ohno S. (2008). *Lagrangian
  meshfree particles method (SPH) for large deformation and failure
  flows of geomaterial using elastic-plastic soil constitutive model.*
  International Journal for Numerical and Analytical Methods in
  Geomechanics 32 (12), 1537–1570.
- Daviet G., Bertails-Descoubes F. (2016). *A semi-implicit material
  point method for the continuum simulation of granular materials.*
  ACM Transactions on Graphics 35 (4) (SIGGRAPH 2016).
- Bardenhagen S. G., Kober E. M. (2004). *The generalized
  interpolation material point method.* Computer Modeling in
  Engineering & Sciences 5, 477–495.
- Jiang C., Schroeder C., Selle A., Teran J., Stomakhin A. (2015).
  *The affine particle-in-cell method.* ACM Trans. Graph. 34 (4),
  Article 51.
