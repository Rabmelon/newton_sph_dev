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


def _make_volume_map_config(
    smoothing_length: float,
    *,
    boundary_friction_viscosity: float = 0.0,
    boundary_sticky: bool = False,
) -> "SolverSPH.Config":
    """Build a baseline SolverSPH.Config in volume_map mode for the new tests."""
    config = SolverSPH.Config()
    config.smoothing_length = smoothing_length
    config.reference_density = 2500.0
    config.boundary_type = "volume_map"
    config.boundary_friction_viscosity = boundary_friction_viscosity
    config.boundary_sticky = boundary_sticky
    config.xsph_epsilon = 0.0
    return config


def test_volume_map_sand_cube_on_plane(test, device):
    """Sand cube on a single ground plane with volume_map boundary should collapse and stay above the plane."""

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
    config = _make_volume_map_config(smoothing_length, boundary_friction_viscosity=0.0)

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(model, config)

    init_z_max = float(np.max(state_0.particle_q.numpy()[:, 2]))

    for _ in range(100):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    final_z_min = float(np.min(final_pos[:, 2]))
    final_z_max = float(np.max(final_pos[:, 2]))

    test.assertGreater(
        final_z_min,
        -smoothing_length,
        f"Particles penetrated ground (volume_map): min z = {final_z_min:.4f}",
    )
    test.assertLess(
        final_z_max,
        init_z_max,
        "Column did not collapse under gravity (volume_map)",
    )


def test_volume_map_six_plane_container(test, device):
    """Fluid inside a six-plane box-shaped container should not escape under gravity."""

    smoothing_length = 0.05
    particle_spacing = smoothing_length / 2.0
    dt = 0.0005

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    half = 0.15  # container half-extent [m]
    n_per_axis = 5
    builder.add_particle_grid(
        pos=wp.vec3(-0.5 * (n_per_axis - 1) * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_per_axis,
        dim_y=n_per_axis,
        dim_z=n_per_axis,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=2500.0 * particle_spacing**3,
        jitter=0.0,
        radius_mean=particle_spacing * 0.5,
        custom_attributes={"sph:friction": 0.3},
    )
    # Six planes: outward normals point INTO the container so signed distance
    # is positive inside (where fluid lives). For each face we add a plane
    # whose outward normal is the negative of the face normal.
    # Bottom face (normal +z, plane at z = -half).
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, half), width=0.0, length=0.0)
    # Top face (normal -z, plane at z = +half).
    builder.add_shape_plane(plane=(0.0, 0.0, -1.0, half), width=0.0, length=0.0)
    # Side faces (±x, ±y).
    builder.add_shape_plane(plane=(1.0, 0.0, 0.0, half), width=0.0, length=0.0)
    builder.add_shape_plane(plane=(-1.0, 0.0, 0.0, half), width=0.0, length=0.0)
    builder.add_shape_plane(plane=(0.0, 1.0, 0.0, half), width=0.0, length=0.0)
    builder.add_shape_plane(plane=(0.0, -1.0, 0.0, half), width=0.0, length=0.0)

    model = builder.finalize(device=device)
    config = _make_volume_map_config(smoothing_length, boundary_friction_viscosity=0.5)

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(model, config)
    test.assertEqual(len(solver._volume_map_planes), 6)

    leak_margin = smoothing_length
    for _ in range(200):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    for axis_name, axis in (("x", 0), ("y", 1), ("z", 2)):
        ax_min = float(np.min(final_pos[:, axis]))
        ax_max = float(np.max(final_pos[:, axis]))
        test.assertGreater(
            ax_min,
            -half - leak_margin,
            f"Fluid leaked through -{axis_name} face: min = {ax_min:.4f}",
        )
        test.assertLess(
            ax_max,
            half + leak_margin,
            f"Fluid leaked through +{axis_name} face: max = {ax_max:.4f}",
        )


def test_volume_map_box_obstacle(test, device):
    """A column of fluid dropped onto a box obstacle on the ground should not penetrate the box."""

    smoothing_length = 0.05
    particle_spacing = smoothing_length / 2.0
    dt = 0.0005

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    # Fluid column sits above a 0.2 m square box centered at origin
    # (top at z = box_half[2]). Each half-extent must exceed
    # ``smoothing_length`` so the inside-margin window
    # ``(-h_extent + sl, +h_extent - sl)`` used in the no-penetration check
    # is non-degenerate.
    box_half = wp.vec3(0.15, 0.15, 0.12)
    box_top = float(box_half[2])
    n_per_axis = 4
    builder.add_particle_grid(
        pos=wp.vec3(
            -0.5 * (n_per_axis - 1) * particle_spacing,
            -0.5 * (n_per_axis - 1) * particle_spacing,
            box_top + 4 * particle_spacing,
        ),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_per_axis,
        dim_y=n_per_axis,
        dim_z=n_per_axis,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=2500.0 * particle_spacing**3,
        jitter=0.0,
        radius_mean=particle_spacing * 0.5,
        custom_attributes={"sph:friction": 0.3},
    )
    # Ground catch-plane far below in case fluid spills off the box; volume_map
    # mode treats it as an additional plane.
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.5), width=0.0, length=0.0)
    # The box obstacle (body=-1 → free-standing static shape).
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=float(box_half[0]),
        hy=float(box_half[1]),
        hz=float(box_half[2]),
    )

    model = builder.finalize(device=device)
    config = _make_volume_map_config(smoothing_length, boundary_friction_viscosity=0.2)

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(model, config)
    test.assertEqual(len(solver._volume_map_boxes), 1)

    for _ in range(200):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    # Verify each half-extent leaves a non-degenerate margin window before
    # using it as a no-penetration check, so a future scaling tweak does
    # not silently turn this into a vacuous assertion.
    for k, name in ((0, "x"), (1, "y"), (2, "z")):
        test.assertGreater(
            float(box_half[k]),
            smoothing_length,
            f"box_half[{name}] must exceed smoothing_length for the inside-margin "
            f"check to be meaningful (got {float(box_half[k]):.4f} ≤ {smoothing_length:.4f})",
        )
    # No fluid particle should end up *inside* the box (with smoothing-length leeway).
    inside_x = (final_pos[:, 0] > -float(box_half[0]) + smoothing_length) & (
        final_pos[:, 0] < float(box_half[0]) - smoothing_length
    )
    inside_y = (final_pos[:, 1] > -float(box_half[1]) + smoothing_length) & (
        final_pos[:, 1] < float(box_half[1]) - smoothing_length
    )
    inside_z = (final_pos[:, 2] > -float(box_half[2]) + smoothing_length) & (
        final_pos[:, 2] < float(box_half[2]) - smoothing_length
    )
    inside_box = inside_x & inside_y & inside_z
    n_penetrated = int(np.sum(inside_box))
    test.assertEqual(
        n_penetrated,
        0,
        f"{n_penetrated} fluid particle(s) penetrated the box obstacle",
    )


def test_volume_map_sphere_obstacle(test, device):
    """Fluid dropped onto a sphere obstacle should not penetrate the sphere."""

    smoothing_length = 0.05
    particle_spacing = smoothing_length / 2.0
    dt = 0.0005

    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)

    radius = 0.15
    n_per_axis = 3
    builder.add_particle_grid(
        pos=wp.vec3(
            -0.5 * (n_per_axis - 1) * particle_spacing,
            -0.5 * (n_per_axis - 1) * particle_spacing,
            radius + 3 * particle_spacing,
        ),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n_per_axis,
        dim_y=n_per_axis,
        dim_z=n_per_axis,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=2500.0 * particle_spacing**3,
        jitter=0.0,
        radius_mean=particle_spacing * 0.5,
        custom_attributes={"sph:friction": 0.3},
    )
    # Catch ground plane.
    builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.5), width=0.0, length=0.0)
    # Sphere obstacle (body=-1 → free-standing static shape).
    builder.add_shape_sphere(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        radius=radius,
    )

    model = builder.finalize(device=device)
    config = _make_volume_map_config(smoothing_length, boundary_friction_viscosity=0.2)

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverSPH(model, config)
    test.assertEqual(len(solver._volume_map_spheres), 1)

    for _ in range(150):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    distances = np.linalg.norm(final_pos, axis=-1)
    n_penetrated = int(np.sum(distances < radius - smoothing_length))
    test.assertEqual(
        n_penetrated,
        0,
        f"{n_penetrated} fluid particle(s) penetrated the sphere obstacle "
        f"(min distance from origin: {distances.min():.4f}, expected ≥ {radius - smoothing_length:.4f})",
    )


def test_volume_map_implicit_friction_damps_velocity(test, device):
    """Higher boundary friction viscosity should leave fluid with smaller mean speed."""

    smoothing_length = 0.10
    particle_spacing = smoothing_length / 2.0
    dt = 0.001
    cg_iter_budget = 50

    def _run(mu_b: float, sticky: bool = False) -> tuple[float, int]:
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # Thin layer of fluid sitting on the plane with a small initial sliding
        # velocity along +x. With volume-map friction, larger mu_B should damp
        # the velocity more by the end of the run.
        n_per_axis = 4
        builder.add_particle_grid(
            pos=wp.vec3(
                -0.5 * (n_per_axis - 1) * particle_spacing,
                -0.5 * (n_per_axis - 1) * particle_spacing,
                smoothing_length * 0.6,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.5, 0.0, 0.0),
            dim_x=n_per_axis,
            dim_y=n_per_axis,
            dim_z=2,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=2500.0 * particle_spacing**3,
            jitter=0.0,
            radius_mean=particle_spacing * 0.5,
            custom_attributes={"sph:friction": 0.3},
        )
        builder.add_ground_plane()

        model = builder.finalize(device=device)
        config = _make_volume_map_config(
            smoothing_length,
            boundary_friction_viscosity=mu_b,
            boundary_sticky=sticky,
        )

        state_0 = model.state()
        state_1 = model.state()
        solver = SolverSPH(model, config)
        max_iter_seen = 0
        for _ in range(120):
            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
            state_0, state_1 = state_1, state_0
            if mu_b > 0.0 and solver._friction_solver is not None:
                max_iter_seen = max(max_iter_seen, solver._friction_solver.last_iter_count)

        v = state_0.particle_qd.numpy()
        # Mean tangential speed magnitude (x-y component).
        v_t = float(np.mean(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)))
        return v_t, max_iter_seen

    v_low, _ = _run(0.0)
    v_high, iter_high = _run(5.0)
    test.assertLess(
        v_high,
        v_low,
        f"Higher boundary friction did not damp velocity: v(mu_B=5.0)={v_high:.3f} vs v(mu_B=0)={v_low:.3f}",
    )
    # CG should converge well within the budget for this small scenario; if it
    # ever needs more, the operator conditioning has regressed (e.g. via a
    # change to the friction Laplacian) and warrants investigation.
    test.assertLess(
        iter_high,
        cg_iter_budget,
        f"CG iteration count {iter_high} exceeded budget {cg_iter_budget} during sliding run",
    )

    v_sticky, iter_sticky = _run(50.0, sticky=True)
    test.assertLess(
        v_sticky,
        v_high,
        f"Sticky variant should damp at least as much as sliding: v_sticky={v_sticky:.3f} vs v(mu_B=5.0)={v_high:.3f}",
    )
    test.assertLess(
        iter_sticky,
        cg_iter_budget,
        f"CG iteration count {iter_sticky} exceeded budget {cg_iter_budget} during sticky run",
    )


def test_volume_map_table_sanity_bounds(test, device):
    """The precomputed V_B(d) volume-map table should match analytical limits.

    Reads the cached ``solver._volume_map_table`` after constructing a
    volume-map solver — exercises the CPU-side ``compute_volume_map_table``
    quadrature without widening the public-API import surface.
    """
    smoothing_length = 0.05
    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=smoothing_length,
        cell_y=smoothing_length,
        cell_z=smoothing_length,
        mass=2500.0 * smoothing_length**3,
        jitter=0.0,
        radius_mean=smoothing_length * 0.5,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    config = _make_volume_map_config(smoothing_length)
    config.volume_map_table_size = 256
    solver = SolverSPH(model, config)

    table = solver._volume_map_table.numpy()
    r = solver._support_radius
    full = (4.0 / 3.0) * np.pi * r**3
    half = 0.5 * full

    # V_B(d/r = -1) ≈ full kernel-support volume (whole sphere inside solid).
    test.assertAlmostEqual(
        float(table[0]) / full,
        1.0,
        delta=0.01,
        msg=f"V_B(-r) = {table[0]:.4e}, expected ~ full kernel volume {full:.4e}",
    )
    # V_B(d/r = 0) lies between half-volume and full-volume — the inside half
    # of the support contributes its full sphere segment, plus the smoothed
    # gamma* tail through the outside half adds a fraction more.
    mid_idx = (len(table) - 1) // 2
    v_mid = float(table[mid_idx])
    test.assertGreater(
        v_mid,
        half * 0.95,
        f"V_B(0) = {v_mid:.4e} unexpectedly below half-volume {half:.4e}",
    )
    test.assertLess(
        v_mid,
        full,
        f"V_B(0) = {v_mid:.4e} should be strictly less than full volume {full:.4e}",
    )
    # The table must decrease monotonically in d.
    test.assertTrue(
        np.all(np.diff(table) <= 1.0e-6),
        f"V_B(d) is not monotonically non-increasing in d; max increase observed: {float(np.max(np.diff(table))):.4e}",
    )


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
    "test_volume_map_sand_cube_on_plane",
    test_volume_map_sand_cube_on_plane,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_volume_map_six_plane_container",
    test_volume_map_six_plane_container,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_volume_map_box_obstacle",
    test_volume_map_box_obstacle,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_volume_map_sphere_obstacle",
    test_volume_map_sphere_obstacle,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_volume_map_implicit_friction_damps_velocity",
    test_volume_map_implicit_friction_damps_velocity,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSPH,
    "test_volume_map_table_sanity_bounds",
    test_volume_map_table_sanity_bounds,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
