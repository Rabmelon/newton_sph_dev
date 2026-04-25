# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH smoothing kernel functions and computation kernels.

This module provides:
- SPH smoothing kernel functions (Wendland C2, cubic spline) and their gradients.
- Warp kernels for density computation, velocity gradient, strain rate,
  stress divergence forces, artificial viscosity, XSPH correction, and
  time integration.

All neighbor queries use :class:`wp.HashGrid`.

References:
    - tiSPHi: https://github.com/Rabmelon/tiSPHi
"""

import warp as wp
import warp.fem as fem

from ...geometry import ParticleFlags
from .sph_dummy_boundary import (
    SPH_FLUID,
    compute_virtual_stress,
    compute_virtual_velocity,
)

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = wp.constant(3.14159265358979323846)
_EPSILON = wp.constant(1.0e-8)

# ---------------------------------------------------------------------------
# Wendland C2 kernel (3D)
# ---------------------------------------------------------------------------


@wp.func
def wendland_c2_3d(r: float, h: float) -> float:
    """Wendland C2 smoothing kernel in 3-D.

    .. math::
        W(r, h) = \\frac{21}{2 \\pi h^3} \\left(1 - \\frac{q}{2}\\right)^4 (1 + 2q),
        \\quad q = r / h,\\; q \\le 2

    Args:
        r: Distance between two particles [m].
        h: Smoothing length [m].

    Returns:
        Kernel value W(r, h).
    """
    q = r / h
    if q >= 2.0:
        return 0.0
    alpha = 21.0 / (2.0 * _PI * h * h * h)
    t = 1.0 - 0.5 * q
    return alpha * t * t * t * t * (1.0 + 2.0 * q)


@wp.func
def wendland_c2_grad_3d(r_vec: wp.vec3, r: float, h: float) -> wp.vec3:
    """Gradient of the Wendland C2 kernel in 3-D.

    Args:
        r_vec: Vector from particle j to particle i (x_i - x_j).
        r: |r_vec|, distance between particles.
        h: Smoothing length.

    Returns:
        Gradient vector nabla_i W(r, h).
    """
    q = r / h
    if q >= 2.0 or r < _EPSILON:
        return wp.vec3(0.0)
    alpha = 21.0 / (2.0 * _PI * h * h * h)
    t = 1.0 - 0.5 * q
    # dW/dr = alpha * [ -5q * (1 - q/2)^3 ] / h
    dWdr = alpha * (-5.0 * q * t * t * t) / h
    return dWdr * (r_vec / r)


# ---------------------------------------------------------------------------
# Cubic spline kernel (3D)
# ---------------------------------------------------------------------------


@wp.func
def cubic_spline_3d(r: float, h: float) -> float:
    """Cubic spline smoothing kernel in 3-D.

    Args:
        r: Distance between two particles [m].
        h: Smoothing length [m].

    Returns:
        Kernel value W(r, h).
    """
    q = r / h
    alpha = 1.0 / (_PI * h * h * h)
    if q >= 2.0:
        return 0.0
    elif q >= 1.0:
        t = 2.0 - q
        return alpha * (t * t * t) / 6.0
    else:
        return alpha * (2.0 / 3.0 - q * q + 0.5 * q * q * q)


@wp.func
def cubic_spline_grad_3d(r_vec: wp.vec3, r: float, h: float) -> wp.vec3:
    """Gradient of the cubic spline kernel in 3-D.

    Args:
        r_vec: Vector from particle j to particle i (x_i - x_j).
        r: |r_vec|, distance between particles.
        h: Smoothing length.

    Returns:
        Gradient vector nabla_i W(r, h).
    """
    q = r / h
    if q >= 2.0 or r < _EPSILON:
        return wp.vec3(0.0)
    alpha = 1.0 / (_PI * h * h * h)
    if q >= 1.0:
        t = 2.0 - q
        dWdr = alpha * (-0.5 * t * t) / h
    else:
        dWdr = alpha * (-2.0 * q + 1.5 * q * q) / h
    return dWdr * (r_vec / r)


# ---------------------------------------------------------------------------
# Density summation kernel
# ---------------------------------------------------------------------------


@wp.kernel
def compute_density_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    density_prev: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    h: float,
    support_radius: float,
    reference_density: float,
    # output
    density: wp.array[float],
):
    """Compute SPH density via Shepard-corrected direct summation.

    .. math::

        \\rho_i = \\frac{\\sum_j m_j \\, W_{ij}}{\\sum_j (m_j / \\rho_j^{\\mathrm{prev}}) \\, W_{ij}}

    On the first step (when ``density_prev`` is zero), the raw summation is
    used without correction. Dummy particles use ``reference_density`` in the
    Shepard denominator since their density is never computed.

    Dummy particles (``particle_type != 0``) are skipped as center particles
    but contribute as neighbors.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    xi = pos[i]
    rho = float(0.0)
    shepard_sum = float(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius:
                W = wendland_c2_3d(r, h)
                rho += mass[j] * W
                rho_j_prev = density_prev[j]
                if rho_j_prev > _EPSILON:
                    shepard_sum += (mass[j] / rho_j_prev) * W
                elif particle_type[j] != SPH_FLUID and reference_density > _EPSILON:
                    # Dummy particles: use reference density for Shepard denominator
                    shepard_sum += (mass[j] / reference_density) * W

    # Apply Shepard correction when previous density is available
    if shepard_sum > _EPSILON:
        density[i] = rho / shepard_sum
    else:
        density[i] = rho


# ---------------------------------------------------------------------------
# Velocity gradient kernel
# ---------------------------------------------------------------------------


@wp.kernel
def compute_velocity_gradient_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    mass: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    wall_normal: wp.array[wp.vec3],
    h: float,
    support_radius: float,
    dummy_beta: float,
    reference_density: float,
    # output
    velocity_gradient: wp.array[wp.mat33],
):
    """Compute velocity gradient tensor L via SPH.

    .. math:: L_i = \\sum_j \\frac{m_j}{\\rho_j} (\\mathbf{v}_j - \\mathbf{v}_i) \\otimes \\nabla W_{ij}

    For dummy neighbors, virtual velocity and reference density are used.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        velocity_gradient[i] = wp.mat33(0.0)
        return
    if particle_type[i] != SPH_FLUID:
        velocity_gradient[i] = wp.mat33(0.0)
        return

    xi = pos[i]
    vi = vel[i]
    L = wp.mat33(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius and r > _EPSILON:
                # Effective velocity and density for neighbor j
                vj = vel[j]
                rho_j = density[j]
                if particle_type[j] != SPH_FLUID:
                    vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                    rho_j = reference_density
                if rho_j > _EPSILON:
                    grad_w = wendland_c2_grad_3d(r_vec, r, h)
                    v_diff = vj - vi
                    L += (mass[j] / rho_j) * wp.outer(v_diff, grad_w)

    velocity_gradient[i] = L


# ---------------------------------------------------------------------------
# Strain rate from velocity gradient
# ---------------------------------------------------------------------------


@wp.kernel
def compute_strain_rate_kernel(
    velocity_gradient: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    # output
    strain_rate: wp.array[wp.mat33],
):
    """Compute symmetric strain rate tensor D = 0.5*(L + L^T)."""
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        strain_rate[i] = wp.mat33(0.0)
        return
    L = velocity_gradient[i]
    strain_rate[i] = 0.5 * (L + wp.transpose(L))


# ---------------------------------------------------------------------------
# Stress divergence force kernel
# ---------------------------------------------------------------------------


@wp.kernel
def compute_stress_force_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    density: wp.array[float],
    stress: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    h: float,
    support_radius: float,
    reference_density: float,
    gravity: wp.vec3,
    # output
    accel: wp.array[wp.vec3],
):
    """Compute acceleration from stress tensor divergence.

    .. math::
        \\mathbf{a}_i = \\sum_j m_j
        \\left(\\frac{\\boldsymbol{\\sigma}_i}{\\rho_i^2}
        + \\frac{\\boldsymbol{\\sigma}_j}{\\rho_j^2}\\right) \\cdot \\nabla W_{ij}

    For dummy neighbors, virtual stress and reference density are used.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    xi = pos[i]
    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    sigma_i = stress[i]
    a = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius and r > _EPSILON:
                # Effective stress and density for neighbor j
                sigma_j = stress[j]
                rho_j = density[j]
                if particle_type[j] != SPH_FLUID:
                    sigma_j = compute_virtual_stress(
                        reference_density,
                        gravity,
                        xi,
                        xj,
                        sigma_i,
                        particle_type[j],
                    )
                    rho_j = reference_density
                if rho_j > _EPSILON:
                    grad_w = wendland_c2_grad_3d(r_vec, r, h)
                    combined = sigma_i / (rho_i * rho_i) + sigma_j / (rho_j * rho_j)
                    a += mass[j] * (combined @ grad_w)

    accel[i] = accel[i] + a


# ---------------------------------------------------------------------------
# Artificial viscosity kernel (Monaghan)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_artificial_viscosity_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    mass: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    wall_normal: wp.array[wp.vec3],
    h: float,
    support_radius: float,
    alpha_visc: float,
    sound_speed: float,
    dummy_beta: float,
    reference_density: float,
    # output
    accel: wp.array[wp.vec3],
):
    """Monaghan-type artificial viscosity.

    .. math::
        \\Pi_{ij} = \\frac{-\\alpha \\, c_s \\, \\mu_{ij}}{\\bar{\\rho}_{ij}},
        \\quad \\mu_{ij} = \\frac{h \\, \\mathbf{v}_{ij} \\cdot \\mathbf{x}_{ij}}
        {|\\mathbf{x}_{ij}|^2 + \\epsilon h^2}

    For dummy neighbors, virtual velocity and reference density are used.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    xi = pos[i]
    vi = vel[i]
    rho_i = density[i]
    a = wp.vec3(0.0)
    eta_sq = 0.01 * h * h

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius and r > _EPSILON:
                vj = vel[j]
                rho_j = density[j]
                if particle_type[j] != SPH_FLUID:
                    vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                    rho_j = reference_density
                v_ij = vi - vj
                vx = wp.dot(v_ij, r_vec)
                if vx < 0.0:
                    rho_avg = 0.5 * (rho_i + rho_j)
                    if rho_avg > _EPSILON:
                        mu_ij = h * vx / (r * r + eta_sq)
                        Pi_ij = -alpha_visc * sound_speed * mu_ij / rho_avg
                        grad_w = wendland_c2_grad_3d(r_vec, r, h)
                        a -= mass[j] * Pi_ij * grad_w

    accel[i] = accel[i] + a


# ---------------------------------------------------------------------------
# XSPH velocity correction kernel
# ---------------------------------------------------------------------------


@wp.kernel
def xsph_correction_kernel(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    mass: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    wall_normal: wp.array[wp.vec3],
    h: float,
    support_radius: float,
    epsilon: float,
    dummy_beta: float,
    reference_density: float,
    # output (in-place)
    vel_out: wp.array[wp.vec3],
):
    """XSPH velocity correction for stability.

    .. math::
        \\mathbf{v}_i^{\\text{corr}} = \\mathbf{v}_i
        + \\varepsilon \\sum_j \\frac{m_j}{\\bar{\\rho}_{ij}}
        (\\mathbf{v}_j - \\mathbf{v}_i) W_{ij}

    For dummy neighbors, virtual velocity and reference density are used.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    xi = pos[i]
    vi = vel[i]
    rho_i = density[i]
    correction = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius:
                vj = vel[j]
                rho_j = density[j]
                if particle_type[j] != SPH_FLUID:
                    vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                    rho_j = reference_density
                rho_avg = 0.5 * (rho_i + rho_j)
                if rho_avg > _EPSILON:
                    w = wendland_c2_3d(r, h)
                    correction += (mass[j] / rho_avg) * (vj - vi) * w

    vel_out[i] = vi + epsilon * correction


# ---------------------------------------------------------------------------
# Symplectic Euler integration kernel
# ---------------------------------------------------------------------------


@wp.kernel
def integrate_symplectic_euler_kernel(
    pos_in: wp.array[wp.vec3],
    vel_in: wp.array[wp.vec3],
    accel: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    # output
    pos_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
):
    """Symplectic (semi-implicit) Euler time integration.

    .. math::
        \\mathbf{v}^{n+1} = \\mathbf{v}^n + (\\mathbf{a} + \\mathbf{g}) \\Delta t, \\quad
        \\mathbf{x}^{n+1} = \\mathbf{x}^n + \\mathbf{v}^{n+1} \\Delta t

    Dummy particles (``particle_type != 0``) are kept static.
    """
    i = wp.tid()
    x0 = pos_in[i]

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        pos_out[i] = x0
        vel_out[i] = vel_in[i]
        return

    v0 = vel_in[i]
    a = accel[i]

    world_idx = particle_world[i]
    g = gravity[wp.max(world_idx, 0)]

    # v1 = v0 + (a + g) * dt
    v1 = v0 + (a + g) * dt

    # Guard against NaN/Inf from upstream stress or density errors.  A NaN
    # velocity would silently propagate and corrupt the entire simulation;
    # zeroing the affected particle keeps it recoverable.
    if wp.isnan(v1[0]) or wp.isnan(v1[1]) or wp.isnan(v1[2]) or wp.isinf(v1[0]) or wp.isinf(v1[1]) or wp.isinf(v1[2]):
        v1 = wp.vec3(0.0)

    # enforce velocity limit
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag

    # x1 = x0 + v1 * dt
    x1 = x0 + v1 * dt

    pos_out[i] = x1
    vel_out[i] = v1


# ---------------------------------------------------------------------------
# Position-based Verlet integration kernels
# ---------------------------------------------------------------------------


@wp.kernel
def half_step_position_kernel(
    pos_in: wp.array[wp.vec3],
    vel_in: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    half_dt: float,
    # output
    pos_mid: wp.array[wp.vec3],
):
    """Advance position by half time step for Verlet midpoint.

    .. math::
        \\mathbf{x}^{n+1/2} = \\mathbf{x}^n + \\frac{\\Delta t}{2} \\mathbf{v}^n

    Dummy particles (``particle_type != 0``) are kept static.
    """
    i = wp.tid()
    x0 = pos_in[i]

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        pos_mid[i] = x0
        return

    pos_mid[i] = x0 + vel_in[i] * half_dt


@wp.kernel
def integrate_verlet_final_kernel(
    pos_mid: wp.array[wp.vec3],
    vel_in: wp.array[wp.vec3],
    accel: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    # output
    pos_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
):
    """Verlet final step: full-step velocity and position from midpoint.

    .. math::
        \\mathbf{v}^{n+1} = \\mathbf{v}^n + (\\mathbf{a}^{n+1/2} + \\mathbf{g}) \\Delta t, \\quad
        \\mathbf{x}^{n+1} = \\mathbf{x}^{n+1/2} + \\frac{\\Delta t}{2} \\mathbf{v}^{n+1}

    Dummy particles (``particle_type != 0``) are kept static.

    Reference:
        Zhang et al. (2024) Computers and Geotechnics 167:106052.
    """
    i = wp.tid()
    x_mid = pos_mid[i]

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        pos_out[i] = x_mid
        vel_out[i] = vel_in[i]
        return

    v0 = vel_in[i]
    a = accel[i]

    world_idx = particle_world[i]
    g = gravity[wp.max(world_idx, 0)]

    v1 = v0 + (a + g) * dt

    # Guard against NaN/Inf from upstream stress or density errors.
    if wp.isnan(v1[0]) or wp.isnan(v1[1]) or wp.isnan(v1[2]) or wp.isinf(v1[0]) or wp.isinf(v1[1]) or wp.isinf(v1[2]):
        v1 = wp.vec3(0.0)

    # enforce velocity limit
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag

    pos_out[i] = x_mid + v1 * (dt * 0.5)
    vel_out[i] = v1


# ---------------------------------------------------------------------------
# Specialized kernel factories
#
# Each factory returns a Warp kernel specialized on ``has_dummies``. When the
# model contains only fluid particles, the neighbor-side dummy-substitution
# branch is eliminated at compile time via ``wp.static``. The integration-
# center ``particle_type[i]`` guard is kept unconditional per invariant 2 in
# sph/CLAUDE.md.
#
# Factories are cached by ``fem.cache.dynamic_kernel`` keyed on the boolean
# ``has_dummies`` — two solver instances with the same flag share the same
# compiled kernel object.
#
# ``kernel_options["fast_math"]`` is currently False (Phase A). Phase B will
# flip it to True after Phase A validation.
# ---------------------------------------------------------------------------


_SPECIALIZED_KERNEL_OPTIONS_PHASE_A = {"fast_math": True, "enable_backward": False}


def make_compute_density_kernel(has_dummies: bool):
    """Factory for a Shepard-corrected density kernel specialized on ``has_dummies``."""

    @fem.cache.dynamic_kernel(
        suffix=has_dummies,
        kernel_options=_SPECIALIZED_KERNEL_OPTIONS_PHASE_A,
    )
    def compute_density_kernel_impl(
        grid: wp.uint64,
        pos: wp.array[wp.vec3],
        mass: wp.array[float],
        density_prev: wp.array[float],
        particle_flags: wp.array[wp.int32],
        particle_type: wp.array[wp.int32],
        h: float,
        support_radius: float,
        reference_density: float,
        density: wp.array[float],
    ):
        i = wp.tid()
        if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
            return
        if particle_type[i] != SPH_FLUID:
            return

        xi = pos[i]
        rho = float(0.0)
        shepard_sum = float(0.0)

        query = wp.hash_grid_query(grid, xi, support_radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if (particle_flags[j] & ParticleFlags.ACTIVE) != 0:
                xj = pos[j]
                r_vec = xi - xj
                r = wp.length(r_vec)
                if r < support_radius:
                    W = wendland_c2_3d(r, h)
                    rho += mass[j] * W
                    rho_j_prev = density_prev[j]
                    if rho_j_prev > _EPSILON:
                        shepard_sum += (mass[j] / rho_j_prev) * W
                    elif wp.static(has_dummies):
                        if particle_type[j] != SPH_FLUID and reference_density > _EPSILON:
                            shepard_sum += (mass[j] / reference_density) * W

        if shepard_sum > _EPSILON:
            density[i] = rho / shepard_sum
        else:
            density[i] = rho

    return compute_density_kernel_impl


def make_compute_velocity_gradient_kernel(has_dummies: bool):
    """Factory for a velocity-gradient kernel specialized on ``has_dummies``."""

    @fem.cache.dynamic_kernel(
        suffix=has_dummies,
        kernel_options=_SPECIALIZED_KERNEL_OPTIONS_PHASE_A,
    )
    def compute_velocity_gradient_kernel_impl(
        grid: wp.uint64,
        pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        mass: wp.array[float],
        density: wp.array[float],
        particle_flags: wp.array[wp.int32],
        particle_type: wp.array[wp.int32],
        wall_normal: wp.array[wp.vec3],
        h: float,
        support_radius: float,
        dummy_beta: float,
        reference_density: float,
        velocity_gradient: wp.array[wp.mat33],
    ):
        i = wp.tid()
        if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
            velocity_gradient[i] = wp.mat33(0.0)
            return
        if particle_type[i] != SPH_FLUID:
            velocity_gradient[i] = wp.mat33(0.0)
            return

        xi = pos[i]
        vi = vel[i]
        L = wp.mat33(0.0)

        query = wp.hash_grid_query(grid, xi, support_radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
                xj = pos[j]
                r_vec = xi - xj
                r = wp.length(r_vec)
                if r < support_radius and r > _EPSILON:
                    vj = vel[j]
                    rho_j = density[j]
                    if wp.static(has_dummies):
                        if particle_type[j] != SPH_FLUID:
                            vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                            rho_j = reference_density
                    if rho_j > _EPSILON:
                        grad_w = wendland_c2_grad_3d(r_vec, r, h)
                        v_diff = vj - vi
                        L += (mass[j] / rho_j) * wp.outer(v_diff, grad_w)

        velocity_gradient[i] = L

    return compute_velocity_gradient_kernel_impl


def make_compute_stress_force_kernel(has_dummies: bool):
    """Factory for a stress-divergence force kernel specialized on ``has_dummies``."""

    @fem.cache.dynamic_kernel(
        suffix=has_dummies,
        kernel_options=_SPECIALIZED_KERNEL_OPTIONS_PHASE_A,
    )
    def compute_stress_force_kernel_impl(
        grid: wp.uint64,
        pos: wp.array[wp.vec3],
        mass: wp.array[float],
        density: wp.array[float],
        stress: wp.array[wp.mat33],
        particle_flags: wp.array[wp.int32],
        particle_type: wp.array[wp.int32],
        h: float,
        support_radius: float,
        reference_density: float,
        gravity: wp.vec3,
        accel: wp.array[wp.vec3],
    ):
        i = wp.tid()
        if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
            return
        if particle_type[i] != SPH_FLUID:
            return

        xi = pos[i]
        rho_i = density[i]
        if rho_i < _EPSILON:
            return

        sigma_i = stress[i]
        a = wp.vec3(0.0)

        query = wp.hash_grid_query(grid, xi, support_radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
                xj = pos[j]
                r_vec = xi - xj
                r = wp.length(r_vec)
                if r < support_radius and r > _EPSILON:
                    sigma_j = stress[j]
                    rho_j = density[j]
                    if wp.static(has_dummies):
                        if particle_type[j] != SPH_FLUID:
                            sigma_j = compute_virtual_stress(
                                reference_density,
                                gravity,
                                xi,
                                xj,
                                sigma_i,
                                particle_type[j],
                            )
                            rho_j = reference_density
                    if rho_j > _EPSILON:
                        grad_w = wendland_c2_grad_3d(r_vec, r, h)
                        combined = sigma_i / (rho_i * rho_i) + sigma_j / (rho_j * rho_j)
                        a += mass[j] * (combined @ grad_w)

        accel[i] = accel[i] + a

    return compute_stress_force_kernel_impl


def make_compute_artificial_viscosity_kernel(has_dummies: bool):
    """Factory for a Monaghan artificial-viscosity kernel specialized on ``has_dummies``."""

    @fem.cache.dynamic_kernel(
        suffix=has_dummies,
        kernel_options=_SPECIALIZED_KERNEL_OPTIONS_PHASE_A,
    )
    def compute_artificial_viscosity_kernel_impl(
        grid: wp.uint64,
        pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        mass: wp.array[float],
        density: wp.array[float],
        particle_flags: wp.array[wp.int32],
        particle_type: wp.array[wp.int32],
        wall_normal: wp.array[wp.vec3],
        h: float,
        support_radius: float,
        alpha_visc: float,
        sound_speed: float,
        dummy_beta: float,
        reference_density: float,
        accel: wp.array[wp.vec3],
    ):
        i = wp.tid()
        if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
            return
        if particle_type[i] != SPH_FLUID:
            return

        xi = pos[i]
        vi = vel[i]
        rho_i = density[i]
        a = wp.vec3(0.0)
        eta_sq = 0.01 * h * h

        query = wp.hash_grid_query(grid, xi, support_radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
                xj = pos[j]
                r_vec = xi - xj
                r = wp.length(r_vec)
                if r < support_radius and r > _EPSILON:
                    vj = vel[j]
                    rho_j = density[j]
                    if wp.static(has_dummies):
                        if particle_type[j] != SPH_FLUID:
                            vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                            rho_j = reference_density
                    v_ij = vi - vj
                    vx = wp.dot(v_ij, r_vec)
                    if vx < 0.0:
                        rho_avg = 0.5 * (rho_i + rho_j)
                        if rho_avg > _EPSILON:
                            mu_ij = h * vx / (r * r + eta_sq)
                            Pi_ij = -alpha_visc * sound_speed * mu_ij / rho_avg
                            grad_w = wendland_c2_grad_3d(r_vec, r, h)
                            a -= mass[j] * Pi_ij * grad_w

        accel[i] = accel[i] + a

    return compute_artificial_viscosity_kernel_impl


def make_xsph_correction_kernel(has_dummies: bool):
    """Factory for an XSPH velocity-correction kernel specialized on ``has_dummies``."""

    @fem.cache.dynamic_kernel(
        suffix=has_dummies,
        kernel_options=_SPECIALIZED_KERNEL_OPTIONS_PHASE_A,
    )
    def xsph_correction_kernel_impl(
        grid: wp.uint64,
        pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        mass: wp.array[float],
        density: wp.array[float],
        particle_flags: wp.array[wp.int32],
        particle_type: wp.array[wp.int32],
        wall_normal: wp.array[wp.vec3],
        h: float,
        support_radius: float,
        epsilon: float,
        dummy_beta: float,
        reference_density: float,
        vel_out: wp.array[wp.vec3],
    ):
        i = wp.tid()
        if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
            return
        if particle_type[i] != SPH_FLUID:
            return

        xi = pos[i]
        vi = vel[i]
        rho_i = density[i]
        correction = wp.vec3(0.0)

        query = wp.hash_grid_query(grid, xi, support_radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if (particle_flags[j] & ParticleFlags.ACTIVE) != 0 and j != i:
                xj = pos[j]
                r_vec = xi - xj
                r = wp.length(r_vec)
                if r < support_radius:
                    vj = vel[j]
                    rho_j = density[j]
                    if wp.static(has_dummies):
                        if particle_type[j] != SPH_FLUID:
                            vj = compute_virtual_velocity(vi, vel[j], dummy_beta, wall_normal[j], particle_type[j])
                            rho_j = reference_density
                    rho_avg = 0.5 * (rho_i + rho_j)
                    if rho_avg > _EPSILON:
                        w = wendland_c2_3d(r, h)
                        correction += (mass[j] / rho_avg) * (vj - vi) * w

        vel_out[i] = vi + epsilon * correction

    return xsph_correction_kernel_impl
