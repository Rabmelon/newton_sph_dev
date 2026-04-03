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

from ...geometry import ParticleFlags

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
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    support_radius: float,
    # output
    density: wp.array(dtype=float),
):
    """Compute SPH density via direct summation.

    .. math:: \\rho_i = \\sum_j m_j \\, W(|\\mathbf{x}_i - \\mathbf{x}_j|, h)
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    xi = pos[i]
    rho = float(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) != 0:
            xj = pos[j]
            r_vec = xi - xj
            r = wp.length(r_vec)
            if r < support_radius:
                rho += mass[j] * wendland_c2_3d(r, h)

    density[i] = rho


# ---------------------------------------------------------------------------
# Velocity gradient kernel
# ---------------------------------------------------------------------------


@wp.kernel
def compute_velocity_gradient_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    support_radius: float,
    # output
    velocity_gradient: wp.array(dtype=wp.mat33),
):
    """Compute velocity gradient tensor L via SPH.

    .. math:: L_i = \\sum_j \\frac{m_j}{\\rho_j} (\\mathbf{v}_j - \\mathbf{v}_i) \\otimes \\nabla W_{ij}
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
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
                rho_j = density[j]
                if rho_j > _EPSILON:
                    grad_w = wendland_c2_grad_3d(r_vec, r, h)
                    v_diff = vel[j] - vi
                    # L += (m_j / rho_j) * outer(v_diff, grad_w)
                    L += (mass[j] / rho_j) * wp.outer(v_diff, grad_w)

    velocity_gradient[i] = L


# ---------------------------------------------------------------------------
# Strain rate from velocity gradient
# ---------------------------------------------------------------------------


@wp.kernel
def compute_strain_rate_kernel(
    velocity_gradient: wp.array(dtype=wp.mat33),
    particle_flags: wp.array(dtype=wp.int32),
    # output
    strain_rate: wp.array(dtype=wp.mat33),
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
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    density: wp.array(dtype=float),
    stress: wp.array(dtype=wp.mat33),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    support_radius: float,
    # output
    accel: wp.array(dtype=wp.vec3),
):
    """Compute acceleration from stress tensor divergence.

    .. math::
        \\mathbf{a}_i = \\sum_j m_j
        \\left(\\frac{\\boldsymbol{\\sigma}_i}{\\rho_i^2}
        + \\frac{\\boldsymbol{\\sigma}_j}{\\rho_j^2}\\right) \\cdot \\nabla W_{ij}
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
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
                rho_j = density[j]
                if rho_j > _EPSILON:
                    grad_w = wendland_c2_grad_3d(r_vec, r, h)
                    sigma_j = stress[j]
                    # a += m_j * (sigma_i/rho_i^2 + sigma_j/rho_j^2) . grad_w
                    combined = sigma_i / (rho_i * rho_i) + sigma_j / (rho_j * rho_j)
                    a += mass[j] * (combined @ grad_w)

    accel[i] = accel[i] + a


# ---------------------------------------------------------------------------
# Artificial viscosity kernel (Monaghan)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_artificial_viscosity_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    support_radius: float,
    alpha_visc: float,
    sound_speed: float,
    # output
    accel: wp.array(dtype=wp.vec3),
):
    """Monaghan-type artificial viscosity.

    .. math::
        \\Pi_{ij} = \\frac{-\\alpha \\, c_s \\, \\mu_{ij}}{\\bar{\\rho}_{ij}},
        \\quad \\mu_{ij} = \\frac{h \\, \\mathbf{v}_{ij} \\cdot \\mathbf{x}_{ij}}
        {|\\mathbf{x}_{ij}|^2 + \\epsilon h^2}
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
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
                v_ij = vi - vel[j]
                vx = wp.dot(v_ij, r_vec)
                if vx < 0.0:
                    rho_j = density[j]
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
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    support_radius: float,
    epsilon: float,
    # output (in-place)
    vel_out: wp.array(dtype=wp.vec3),
):
    """XSPH velocity correction for stability.

    .. math::
        \\mathbf{v}_i^{\\text{corr}} = \\mathbf{v}_i
        + \\varepsilon \\sum_j \\frac{m_j}{\\bar{\\rho}_{ij}}
        (\\mathbf{v}_j - \\mathbf{v}_i) W_{ij}
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
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
                rho_avg = 0.5 * (rho_i + density[j])
                if rho_avg > _EPSILON:
                    w = wendland_c2_3d(r, h)
                    correction += (mass[j] / rho_avg) * (vel[j] - vi) * w

    vel_out[i] = vi + epsilon * correction


# ---------------------------------------------------------------------------
# Symplectic Euler integration kernel
# ---------------------------------------------------------------------------


@wp.kernel
def integrate_symplectic_euler_kernel(
    pos_in: wp.array(dtype=wp.vec3),
    vel_in: wp.array(dtype=wp.vec3),
    accel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    # output
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """Symplectic (semi-implicit) Euler time integration.

    .. math::
        \\mathbf{v}^{n+1} = \\mathbf{v}^n + (\\mathbf{a} + \\mathbf{g}) \\Delta t, \\quad
        \\mathbf{x}^{n+1} = \\mathbf{x}^n + \\mathbf{v}^{n+1} \\Delta t
    """
    i = wp.tid()
    x0 = pos_in[i]

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        pos_out[i] = x0
        vel_out[i] = vel_in[i]
        return

    v0 = vel_in[i]
    a = accel[i]

    world_idx = particle_world[i]
    g = gravity[wp.max(world_idx, 0)]

    # v1 = v0 + (a + g) * dt
    v1 = v0 + (a + g) * dt

    # enforce velocity limit
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag

    # x1 = x0 + v1 * dt
    x1 = x0 + v1 * dt

    pos_out[i] = x1
    vel_out[i] = v1


# ---------------------------------------------------------------------------
# Zero acceleration kernel
# ---------------------------------------------------------------------------


@wp.kernel
def zero_accel_kernel(
    accel: wp.array(dtype=wp.vec3),
):
    """Zero out the acceleration array."""
    i = wp.tid()
    accel[i] = wp.vec3(0.0)
