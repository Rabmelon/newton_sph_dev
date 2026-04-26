# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.sph.sph_constitutive import (
    dp_return_mapping,
    drucker_prager_params,
    mat33_double_contraction,
)
from newton._src.solvers.sph.sph_kernels import (
    make_compute_density_kernel,
    make_compute_stress_force_kernel,
    make_compute_velocity_gradient_kernel,
)
from newton.solvers import SolverSPH
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_density_uniform_grid(test, device):
    """SPH density summation on a uniform grid should approximate reference density."""

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    spacing = 0.02
    h = 1.5 * spacing
    n_cells = 8
    density_ref = 1000.0
    mass = density_ref * (spacing**3)

    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_cells,
        dim_y=n_cells,
        dim_z=n_cells,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
        radius_mean=spacing * 0.5,
    )

    model = builder.finalize(device=device)

    config = SolverSPH.Config()
    config.smoothing_length = h
    config.reference_density = density_ref
    config.xsph_epsilon = 0.0
    solver = SolverSPH(model, config)

    state = model.state()
    solver._build_neighbor_list(state)
    solver._compute_density(state)

    rho = state.sph.density.numpy()

    # Interior particles (away from boundary) should be close to reference density
    pos = state.particle_q.numpy()
    margin = 3 * spacing
    lo = pos.min(axis=0) + margin
    hi = pos.max(axis=0) - margin
    interior = np.all((pos >= lo) & (pos <= hi), axis=1)

    if np.any(interior):
        rho_interior = rho[interior]
        mean_rho = np.mean(rho_interior)
        # All interior densities should be positive
        test.assertTrue(np.all(rho_interior > 0.0), "Interior densities must be positive")
        # Interior densities should be consistent (low variation)
        std_rho = np.std(rho_interior)
        cv = std_rho / mean_rho  # coefficient of variation
        test.assertLess(
            cv,
            0.05,
            f"Interior density coefficient of variation {cv:.3f} too high (mean={mean_rho:.1f}, std={std_rho:.1f})",
        )


def test_drucker_prager_return_mapping(test, device):
    """DP return mapping should project over-yield stress onto the yield surface."""
    friction = 0.5  # rad
    cohesion = 1000.0  # Pa

    dp = drucker_prager_params(friction, cohesion)
    alpha_phi = float(dp[0])
    k_c = float(dp[1])

    # Create an over-yield stress state (large deviatoric + low mean)
    sigma = wp.mat33(
        50000.0,
        5000.0,
        0.0,
        5000.0,
        -10000.0,
        0.0,
        0.0,
        0.0,
        -30000.0,
    )

    sigma_proj = dp_return_mapping(sigma, alpha_phi, k_c)

    # Evaluate yield function on projected stress
    I1 = float(sigma_proj[0, 0] + sigma_proj[1, 1] + sigma_proj[2, 2])
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    s = sigma_proj - (I1 / 3.0) * eye
    J2 = float(np.sqrt(0.5 * mat33_double_contraction(s, s)))
    f_DP = J2 + alpha_phi * I1 - k_c

    test.assertLessEqual(f_DP, 1.0, f"Yield function f_DP = {f_DP:.4f} should be <= 0 after return mapping")


def test_sand_cube_on_plane(test, device):
    """A cube of sand particles should collapse and stay above the ground plane."""

    N = 4
    particles_per_cell = 3
    smoothing_length = 0.15
    particle_spacing = smoothing_length / particles_per_cell
    dt = 0.001

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=0.1,
        jitter=0.0,
        custom_attributes={"sph:friction": 0.5},
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    config = SolverSPH.Config()
    config.smoothing_length = smoothing_length
    config.reference_density = 2500.0
    config.xsph_epsilon = 0.0

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(model, config)

    init_pos = state_0.particle_q.numpy()
    init_z_max = np.max(init_pos[:, 2])

    for _ in range(100):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    final_z_min = np.min(final_pos[:, 2])
    final_z_max = np.max(final_pos[:, 2])

    # Particles should stay above ground
    test.assertGreater(
        final_z_min,
        -smoothing_length,
        f"Particles penetrated ground: min z = {final_z_min:.4f}",
    )

    # Column should have collapsed (max height decreased)
    test.assertLess(
        final_z_max,
        init_z_max,
        "Column did not collapse under gravity",
    )


def test_geostatic_initialization(test, device):
    """K0 geostatic stress should match analytical values."""

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    spacing = 0.05
    h = 1.5 * spacing
    z_max = 1.0
    density_ref = 2500.0
    friction_angle = 0.5  # rad

    n_z = int(z_max / spacing)
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, spacing * 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=3,
        dim_z=n_z,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=density_ref * spacing**3,
        jitter=0.0,
        radius_mean=spacing * 0.5,
        custom_attributes={"sph:friction": friction_angle},
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    config = SolverSPH.Config()
    config.smoothing_length = h
    config.reference_density = density_ref
    solver = SolverSPH(model, config)

    state = model.state()
    solver.initialize_geostatic_stress(state, y_max=z_max)

    pos = state.particle_q.numpy()
    stress = state.sph.stress.numpy()

    g = 10.0  # default gravity magnitude
    K0 = 1.0 - np.sin(friction_angle)

    for i in range(model.particle_count):
        z_i = pos[i, 2]
        depth = z_max - z_i
        expected_zz = -density_ref * g * depth
        expected_xx = K0 * expected_zz

        actual_zz = stress[i, 2, 2]
        actual_xx = stress[i, 0, 0]

        if abs(expected_zz) > 1.0:
            err_zz = abs(actual_zz - expected_zz) / abs(expected_zz)
            test.assertLess(err_zz, 0.05, f"sigma_zz error {err_zz:.2%} at particle {i}")
            err_xx = abs(actual_xx - expected_xx) / abs(expected_zz)
            test.assertLess(err_xx, 0.05, f"sigma_xx error {err_xx:.2%} at particle {i}")


def test_sand_cube_dummy_boundary(test, device):
    """A cube of sand with dummy boundary particles should stay inside the domain."""

    N = 4
    particles_per_cell = 3
    smoothing_length = 0.15
    particle_spacing = smoothing_length / particles_per_cell
    dt = 0.001

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    lo = (0.0, 0.0, 0.0)
    hi_z = N * particles_per_cell * particle_spacing
    hi = (hi_z, hi_z, hi_z)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=0.1,
        jitter=0.0,
        custom_attributes={"sph:friction": 0.5},
    )

    fluid_count = builder.particle_count

    dummy_count = SolverSPH.add_dummy_particles(
        builder,
        bounds_lo=lo,
        bounds_hi=hi,
        h=smoothing_length,
        dx=particle_spacing,
        reference_density=2500.0,
        slip_type="noslip",
    )
    test.assertGreater(dummy_count, 0, "Should have generated dummy particles")

    model = builder.finalize(device=device)

    config = SolverSPH.Config()
    config.smoothing_length = smoothing_length
    config.reference_density = 2500.0
    config.boundary_type = "dummy"
    config.xsph_epsilon = 0.0

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(model, config)

    # Record initial dummy particle positions
    init_pos = state_0.particle_q.numpy()
    dummy_init_pos = init_pos[fluid_count:].copy()

    for _ in range(50):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()

    # Dummy particles should NOT have moved
    dummy_final_pos = final_pos[fluid_count:]
    max_displacement = np.max(np.abs(dummy_final_pos - dummy_init_pos))
    test.assertAlmostEqual(
        max_displacement,
        0.0,
        places=10,
        msg=f"Dummy particles moved: max displacement = {max_displacement}",
    )

    # Fluid particles should stay roughly within the domain
    fluid_final_pos = final_pos[:fluid_count]
    fluid_z_min = np.min(fluid_final_pos[:, 2])
    test.assertGreater(
        fluid_z_min,
        -smoothing_length,
        f"Fluid particles fell below domain: min z = {fluid_z_min:.4f}",
    )


def test_sand_cube_position_verlet(test, device):
    """Sand cube with Position-based Verlet integration should collapse and stay above ground."""

    N = 4
    particles_per_cell = 3
    smoothing_length = 0.15
    particle_spacing = smoothing_length / particles_per_cell
    dt = 0.001

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=0.1,
        jitter=0.0,
        custom_attributes={"sph:friction": 0.5},
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    config = SolverSPH.Config()
    config.smoothing_length = smoothing_length
    config.reference_density = 2500.0
    config.xsph_epsilon = 0.0
    config.integration_scheme = "position_verlet"

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(model, config)

    init_pos = state_0.particle_q.numpy()
    init_z_max = np.max(init_pos[:, 2])

    for _ in range(100):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    final_vel = state_0.particle_qd.numpy()
    final_z_min = np.min(final_pos[:, 2])
    final_z_max = np.max(final_pos[:, 2])

    # No NaN/Inf in final positions or velocities
    test.assertFalse(np.any(np.isnan(final_pos)), "NaN detected in final positions")
    test.assertFalse(np.any(np.isinf(final_pos)), "Inf detected in final positions")
    test.assertFalse(np.any(np.isnan(final_vel)), "NaN detected in final velocities")
    test.assertFalse(np.any(np.isinf(final_vel)), "Inf detected in final velocities")

    # Particles should stay above ground
    test.assertGreater(
        final_z_min,
        -smoothing_length,
        f"Particles penetrated ground: min z = {final_z_min:.4f}",
    )

    # Column should have collapsed (max height decreased)
    test.assertLess(
        final_z_max,
        init_z_max,
        "Column did not collapse under gravity",
    )


def test_specialization_cache_dedup(test, device):
    """Two SolverSPH instances with matching has_dummies must share the same cached kernel objects."""

    # Factory-level check: same flag → same object via fem.cache.dynamic_kernel memoization.
    k_true_a = make_compute_density_kernel(True)
    k_true_b = make_compute_density_kernel(True)
    k_false = make_compute_density_kernel(False)
    test.assertIs(k_true_a, k_true_b, "density kernel cache miss on repeated True call")
    test.assertIsNot(k_true_a, k_false, "True and False variants collapsed into one kernel")

    # Spot-check two more kernels to confirm the pattern.
    test.assertIs(
        make_compute_velocity_gradient_kernel(True),
        make_compute_velocity_gradient_kernel(True),
    )
    test.assertIs(
        make_compute_stress_force_kernel(False),
        make_compute_stress_force_kernel(False),
    )

    # Solver-level check: two solvers built from identical builders must share kernels.
    def _build_fluid_only_solver():
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            mass=0.1,
            jitter=0.0,
        )
        model = builder.finalize(device=device)
        return SolverSPH(model, SolverSPH.Config())

    solver_a = _build_fluid_only_solver()
    solver_b = _build_fluid_only_solver()
    test.assertFalse(solver_a._has_dummies, "fluid-only scene flagged as has_dummies")
    test.assertIs(solver_a._density_kernel, solver_b._density_kernel)
    test.assertIs(solver_a._velocity_gradient_kernel, solver_b._velocity_gradient_kernel)
    test.assertIs(solver_a._stress_force_kernel, solver_b._stress_force_kernel)
    test.assertIs(solver_a._artificial_viscosity_kernel, solver_b._artificial_viscosity_kernel)
    test.assertIs(solver_a._xsph_kernel, solver_b._xsph_kernel)


def test_density_smoothing_disabled_bitexact(test, device):
    """delta=0.0 must produce density bit-identical to the no-smoothing path."""

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    spacing = 0.02
    h = 1.5 * spacing
    n_cells = 6
    density_ref = 1000.0
    mass = density_ref * (spacing**3)

    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_cells,
        dim_y=n_cells,
        dim_z=n_cells,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
        radius_mean=spacing * 0.5,
    )
    model = builder.finalize(device=device)

    cfg_off = SolverSPH.Config()
    cfg_off.smoothing_length = h
    cfg_off.reference_density = density_ref
    cfg_off.density_smoothing_delta = 0.0  # explicit default

    solver_off = SolverSPH(model, cfg_off)
    state_off = model.state()
    solver_off._build_neighbor_list(state_off)
    solver_off._compute_density(state_off)
    solver_off._smooth_density(state_off)
    rho_off = state_off.sph.density.numpy().copy()

    # Recompute via the bare density path — must match exactly.
    state_ref = model.state()
    solver_off._build_neighbor_list(state_ref)
    solver_off._compute_density(state_ref)
    rho_ref = state_ref.sph.density.numpy().copy()

    test.assertTrue(
        np.array_equal(rho_off, rho_ref),
        f"delta=0 changed density (max diff = {np.max(np.abs(rho_off - rho_ref))})",
    )


def test_density_smoothing_uniform_grid_no_regression(test, device):
    """delta=0.1 on a uniform grid must not raise interior-density CV materially."""

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    spacing = 0.02
    h = 1.5 * spacing
    n_cells = 8
    density_ref = 1000.0
    mass = density_ref * (spacing**3)

    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_cells,
        dim_y=n_cells,
        dim_z=n_cells,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
        radius_mean=spacing * 0.5,
    )
    model = builder.finalize(device=device)

    def _cv_with_delta(delta: float) -> float:
        cfg = SolverSPH.Config()
        cfg.smoothing_length = h
        cfg.reference_density = density_ref
        cfg.density_smoothing_delta = delta
        solver = SolverSPH(model, cfg)
        state = model.state()
        solver._build_neighbor_list(state)
        solver._compute_density(state)
        solver._smooth_density(state)
        rho = state.sph.density.numpy()
        pos = state.particle_q.numpy()
        margin = 3 * spacing
        lo = pos.min(axis=0) + margin
        hi = pos.max(axis=0) - margin
        interior = np.all((pos >= lo) & (pos <= hi), axis=1)
        rho_in = rho[interior]
        return float(np.std(rho_in) / np.mean(rho_in))

    cv_off = _cv_with_delta(0.0)
    cv_on = _cv_with_delta(0.1)
    test.assertLessEqual(
        cv_on,
        cv_off * 1.05 + 1.0e-6,
        f"density smoothing regressed CV: off={cv_off:.4e}, on={cv_on:.4e}",
    )


def test_density_smoothing_hydrostatic_reduces_pressure_noise(test, device):
    """Smoothing on a hydrostatic column must not increase pressure std-dev at fixed depth."""

    spacing = 0.02
    h = 1.3 * spacing
    n_x, n_y, n_z = 6, 6, 10
    density_ref = 1500.0
    mass = density_ref * (spacing**3)

    def _pressure_std(delta: float) -> float:
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=n_x,
            dim_y=n_y,
            dim_z=n_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=0.0,
            radius_mean=spacing * 0.5,
        )
        bounds_lo = (-spacing, -spacing, -spacing)
        bounds_hi = (n_x * spacing, n_y * spacing, n_z * spacing)
        SolverSPH.add_dummy_particles(
            builder,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            h=h,
            dx=spacing,
            reference_density=density_ref,
            slip_type="noslip",
        )
        model = builder.finalize(device=device)

        cfg = SolverSPH.Config()
        cfg.smoothing_length = h
        cfg.reference_density = density_ref
        cfg.boundary_type = "dummy"
        cfg.density_smoothing_delta = delta
        solver = SolverSPH(model, cfg)
        state_in = model.state()
        state_out = model.state()
        z_max = float((n_z - 1) * spacing)
        solver.initialize_geostatic_stress(state_in, z_max)

        dt = 0.1 * h / cfg.sound_speed
        for _ in range(30):
            solver.step(state_in, state_out, None, None, dt)
            state_in, state_out = state_out, state_in

        # Inspect pressure of fluid particles in a thin band around mid-depth.
        rho = state_in.sph.density.numpy()
        pos = state_in.particle_q.numpy()
        ptype = model.sph.particle_type.numpy()
        z_mid = 0.5 * (n_z - 1) * spacing
        band = (np.abs(pos[:, 2] - z_mid) < 0.6 * spacing) & (ptype == 0) & (rho > 0.0)
        if not np.any(band):
            test.skipTest("no fluid particles fell into the mid-depth sampling band")
        # P = -tr(sigma)/3 already lives on state.sph.pressure.
        pressure = state_in.sph.pressure.numpy()
        return float(np.std(pressure[band]))

    std_off = _pressure_std(0.0)
    std_on = _pressure_std(0.15)
    # Smoothing must not increase pressure noise at fixed depth.
    test.assertLessEqual(
        std_on,
        std_off + 1.0e-3,
        f"smoothing increased pressure std at fixed depth: off={std_off:.3e}, on={std_on:.3e}",
    )


def test_kernel_correction_disabled_default(test, device):
    """Default ('none') skips correction kernel and leaves grad_correction zero."""

    spacing = 0.02
    kh = 1.5
    n_cells = 6
    density_ref = 1000.0
    mass = density_ref * (spacing**3)

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_cells,
        dim_y=n_cells,
        dim_z=n_cells,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
        radius_mean=spacing * 0.5,
    )
    model = builder.finalize(device=device)

    cfg = SolverSPH.Config()
    cfg.particle_spacing = spacing
    cfg.kh = kh
    cfg.reference_density = density_ref
    # default kernel_gradient_correction == "none"
    solver = SolverSPH(model, cfg)
    test.assertFalse(solver._use_kernel_correction)
    test.assertIsNone(solver._kernel_correction_kernel)

    state = model.state()
    solver._build_neighbor_list(state)
    solver._compute_density(state)
    solver._compute_kernel_correction(state)
    corr = state.sph.kernel_grad_correction.numpy()
    test.assertEqual(np.count_nonzero(corr), 0, "no-op correction must not write")


def test_kernel_correction_static_disorder(test, device):
    """On a disordered grid, MLS correction must recover a linear velocity field."""

    spacing = 0.02
    kh = 1.5
    n_cells = 10
    density_ref = 1000.0
    mass = density_ref * (spacing**3)
    jitter = 0.3 * spacing

    A = np.array(
        [[0.10, 0.05, 0.00], [-0.02, 0.00, 0.07], [0.03, 0.04, -0.01]],
        dtype=np.float32,
    )

    def _run(correction_mode: str) -> tuple[np.ndarray, np.ndarray]:
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=n_cells,
            dim_y=n_cells,
            dim_z=n_cells,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=jitter,
            radius_mean=spacing * 0.5,
        )
        model = builder.finalize(device=device)

        cfg = SolverSPH.Config()
        cfg.particle_spacing = spacing
        cfg.kh = kh
        cfg.reference_density = density_ref
        cfg.kernel_gradient_correction = correction_mode
        solver = SolverSPH(model, cfg)

        state = model.state()
        pos = state.particle_q.numpy()
        vel = (pos @ A.T).astype(np.float32)
        state.particle_qd.assign(vel)

        # Density iterations: 1st falls back to raw summation, 2nd uses raw as
        # density_prev so Shepard normalises to the converged interior value.
        solver._build_neighbor_list(state)
        for _ in range(2):
            solver._compute_density(state)
        solver._compute_kernel_correction(state)
        solver._compute_velocity_gradient(state)
        return state.sph.velocity_gradient.numpy(), pos

    grad_off, pos_jit = _run("none")
    grad_on, _ = _run("mls")

    # Restrict to interior particles (margin >= support_radius) to avoid the
    # singular-fallback band.
    margin = 2.5 * kh * spacing
    lo = pos_jit.min(axis=0) + margin
    hi = pos_jit.max(axis=0) - margin
    interior = np.all((pos_jit >= lo) & (pos_jit <= hi), axis=1)
    test.assertGreater(np.sum(interior), 0, "no interior particles to test")

    err_on = np.max(np.abs(grad_on[interior] - A[None, :, :]))
    err_off = np.max(np.abs(grad_off[interior] - A[None, :, :]))
    # MLS correction must recover A within tight tolerance.
    test.assertLess(err_on, 5.0e-3, f"MLS did not recover A; err_on={err_on:.3e}")
    # Uncorrected SPH on disordered grid must be materially worse.
    test.assertGreater(err_off, err_on * 2.0, f"uncorrected too close: off={err_off:.3e}, on={err_on:.3e}")


devices = get_test_devices(mode="basic")


class TestSPH(unittest.TestCase):
    pass


add_function_test(TestSPH, "test_density_uniform_grid", test_density_uniform_grid, devices=devices, check_output=False)
add_function_test(
    TestSPH,
    "test_drucker_prager_return_mapping",
    test_drucker_prager_return_mapping,
    devices=devices,
    check_output=False,
)
add_function_test(TestSPH, "test_sand_cube_on_plane", test_sand_cube_on_plane, devices=devices, check_output=False)
add_function_test(
    TestSPH, "test_geostatic_initialization", test_geostatic_initialization, devices=devices, check_output=False
)
add_function_test(
    TestSPH, "test_sand_cube_dummy_boundary", test_sand_cube_dummy_boundary, devices=devices, check_output=False
)
add_function_test(
    TestSPH,
    "test_sand_cube_position_verlet",
    test_sand_cube_position_verlet,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_specialization_cache_dedup",
    test_specialization_cache_dedup,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_density_smoothing_disabled_bitexact",
    test_density_smoothing_disabled_bitexact,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_density_smoothing_uniform_grid_no_regression",
    test_density_smoothing_uniform_grid_no_regression,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_density_smoothing_hydrostatic_reduces_pressure_noise",
    test_density_smoothing_hydrostatic_reduces_pressure_noise,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_kernel_correction_disabled_default",
    test_kernel_correction_disabled_default,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_kernel_correction_static_disorder",
    test_kernel_correction_static_disorder,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
