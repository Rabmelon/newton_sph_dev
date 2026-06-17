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


def test_penalty_friction_reduces_runout(test, device):
    """Coulomb friction on the ground plane should reduce lateral runout.

    A thin slab of particles with lateral velocity falls onto the ground.
    With friction (mu=1.0), lateral motion is damped; without (mu=0.0),
    particles slide freely.  Internal (DP) friction is set to zero so
    boundary friction is the dominant dissipation mechanism.
    """

    particles_per_cell = 3
    particle_spacing = 0.05
    kh = 1.3
    sound_speed = 50.0
    h = kh * particle_spacing
    # CFL-stable time step
    dt = 0.3 * h / sound_speed
    n_steps = 400

    density_ref = 2500.0
    mass = density_ref * particle_spacing**3

    # Thin slab: wide in XY, 2 layers in Z, starting just above ground
    nx = 6 * particles_per_cell
    ny = 6 * particles_per_cell
    nz = 2 * particles_per_cell
    lateral_vel = wp.vec3(1.0, 0.0, 0.0)

    def _run_with_mu(mu_value: float) -> float:
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        builder.add_particle_grid(
            pos=wp.vec3(0.5 * particle_spacing, 0.5 * particle_spacing, 0.5 * particle_spacing),
            rot=wp.quat_identity(),
            vel=lateral_vel,
            dim_x=nx,
            dim_y=ny,
            dim_z=nz,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=mass,
            jitter=0.0,
            custom_attributes={"sph:friction": 0.0},
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu_value))

        model = builder.finalize(device=device)

        config = SolverSPH.Config()
        config.particle_spacing = particle_spacing
        config.kh = kh
        config.sound_speed = sound_speed
        config.reference_density = density_ref
        config.xsph_epsilon = 0.0

        state_0 = model.state()
        state_1 = model.state()
        solver = SolverSPH(model, config)

        for _ in range(n_steps):
            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
            state_0, state_1 = state_1, state_0

        pos = state_0.particle_q.numpy()
        # Mean X displacement measures lateral runout
        return float(np.mean(pos[:, 0]))

    runout_no_friction = _run_with_mu(0.0)
    runout_full_friction = _run_with_mu(1.0)

    test.assertLess(
        runout_full_friction,
        runout_no_friction,
        f"Friction should reduce lateral runout: mu=1.0 mean_x={runout_full_friction:.4f} "
        f">= mu=0.0 mean_x={runout_no_friction:.4f}",
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
    "test_penalty_friction_reduces_runout",
    test_penalty_friction_reduces_runout,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
