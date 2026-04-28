# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Volume-map boundary force kernels for SPH.

Implements the boundary pressure force of Bender et al. 2020 (IEEE TVCG)
Eq. 15 specialised to plane / box / sphere shapes with analytical SDF.
The friction-Laplacian contribution (Eqs. 16-19) lives in
:mod:`sph_implicit_viscosity` as part of the matrix-free PCG operator.

Three kernels — one per shape kind — each launched once per shape from
the solver's dispatch loop. The unified pattern:

1. Guard ``ACTIVE`` + ``SPH_FLUID``.
2. Transform world position to the shape's local frame, evaluate
   analytical SDF and outward unit normal.
3. Early-exit if signed distance ≥ support radius.
4. Sample precomputed V_B(d) volume-map table.
5. Accumulate Bender 2020 Eq. 15 boundary pressure force into the
   acceleration array.

Numerical fix vs. literal Bender 2020: when a fluid particle has
penetrated the boundary (``d < 0``), the literal kernel gradient flips
direction and the formula yields an *inward* force. Bender's implicit
pressure solver prevents this from arising; Newton's explicit WCSPH
allows momentary penetration, so we always evaluate ∇W with
``r_vec = |d| · n_outward`` so the force stays outward (away from solid).
The boundary pressure mirror is also clamped to ``max(p_i, 0)`` to stop
tensile stress states from pulling fluid into walls.

References:
    Bender, J., Kugelstadt, T., Weiler, M., & Koschier, D. (2020).
    Volume Maps: An Implicit Boundary Representation for SPH.
    IEEE Transactions on Visualization and Computer Graphics.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import sdf_box, sdf_box_grad, sdf_sphere, sdf_sphere_grad
from .sph_dummy_boundary import SPH_FLUID
from .sph_kernels import wendland_c2_grad_3d
from .sph_volume_map import frisvad_tangent_basis, sample_volume_map

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)
# 2 (d + 2) prefactor for the SPH Laplacian in d = 3 dimensions
# (Bender 2020 Eq. 19; Weiler et al. 2018).
_LAP_PREFACTOR = wp.constant(10.0)
# Number of boundary surface points contributed per fluid particle —
# Bender 2020 Sec. 4.2 specifies the closest point x* plus two tangent
# samples at r/2.
_N_SURFACE_POINTS = 3
# Volume share per surface point: V_j = V_B / 3.
_INV_N_SURFACE_POINTS = wp.constant(1.0 / 3.0)


@wp.func
def _eval_volume_map_pressure_accel(
    d: float,
    n_world: wp.vec3,
    h: float,
    support_radius: float,
    p_i: float,
    rho_i: float,
    reference_density: float,
    volume_map_table: wp.array[float],
) -> wp.vec3:
    """Bender 2020 Eq. 15 boundary pressure acceleration (per unit mass).

    Computed once per (particle, boundary-shape) pair from the unified
    inputs that every shape-specific kernel below produces. Wrapped as
    ``@wp.func`` so the three shape kernels share identical force-side
    logic and only differ in how they extract ``d`` and ``n_world``.

    Args:
        d: Signed distance from the particle to the boundary surface
            [m]. Positive outside the solid (Bender 2020 Eq. 10).
        n_world: Outward unit normal at the closest surface point in
            world frame.
        h: Smoothing length [m].
        support_radius: SPH support radius [m] (= 2 h for cubic / Wendland).
        p_i: Particle pressure [Pa].
        rho_i: Particle density [kg/m³].
        reference_density: Reference density rho₀ [kg/m³].
        volume_map_table: Precomputed V_B(d) table from
            :func:`compute_volume_map_table`.

    Returns:
        Acceleration contribution [m/s²] from this boundary, to be
        added to the particle acceleration array.
    """
    # Outward-direction kernel gradient regardless of penetration sign.
    abs_d = wp.abs(d)
    r = wp.max(abs_d, _EPSILON)
    r_vec = r * n_world
    grad_w = wendland_c2_grad_3d(r_vec, r, h)

    v_b = sample_volume_map(d, support_radius, volume_map_table)

    # Clamp BOTH the particle-side pressure (used directly in the symmetric
    # form) and the boundary mirror to a non-negative value. Either alone
    # leaves a leak: a negative p_i (DP tension transient before return
    # mapping) flips the coefficient sign and yields an inward force; a
    # negative p_b would do the same via the mirror term. Clamping both
    # terms keeps the boundary force monotonically outward.
    p_clamped = wp.max(p_i, 0.0)
    rho_b = reference_density

    coeff = v_b * reference_density * (p_clamped / (rho_i * rho_i) + p_clamped / (rho_b * rho_b))
    return -coeff * grad_w


@wp.kernel
def volume_map_pressure_plane_kernel(
    pos: wp.array[wp.vec3],
    pressure: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_offset: float,
    h: float,
    support_radius: float,
    reference_density: float,
    volume_map_table: wp.array[float],
    # output
    accel: wp.array[wp.vec3],
):
    """Apply volume-map boundary pressure for an infinite plane shape.

    Plane is described by world-frame outward normal and offset such that
    the signed distance is ``d = n · x + offset`` (matches the existing
    ``ground_plane_penalty_kernel`` convention for cache compatibility).

    Args:
        pos: Particle positions [m], shape [particle_count, 3].
        pressure: Particle pressure [Pa].
        density: Particle density [kg/m³].
        particle_flags: Particle activity flags.
        particle_type: Particle type (fluid / dummy variants).
        plane_normal: Outward unit normal of the plane (away from solid).
        plane_offset: Plane offset such that ``n·x + offset = signed distance``.
        h: Smoothing length [m].
        support_radius: SPH support radius [m].
        reference_density: Reference density rho₀ [kg/m³].
        volume_map_table: Precomputed V_B(d) values.
        accel: Acceleration array (accumulated in-place) [m/s²].
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x = pos[i]
    d = wp.dot(plane_normal, x) + plane_offset

    if d >= support_radius:
        return

    a = _eval_volume_map_pressure_accel(
        d,
        plane_normal,
        h,
        support_radius,
        pressure[i],
        rho_i,
        reference_density,
        volume_map_table,
    )
    accel[i] = accel[i] + a


@wp.kernel
def volume_map_pressure_box_kernel(
    pos: wp.array[wp.vec3],
    pressure: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    box_transform: wp.transform,
    box_half_extents: wp.vec3,
    h: float,
    support_radius: float,
    reference_density: float,
    volume_map_table: wp.array[float],
    # output
    accel: wp.array[wp.vec3],
):
    """Apply volume-map boundary pressure for a box obstacle (fluid outside).

    Box surface is described by ``shape_transform`` and half-extents
    ``(hx, hy, hz)``. SDF and outward normal are evaluated in the box's
    local frame via the geometry module's analytical helpers, then the
    normal is rotated back to world frame for the kernel-gradient step.

    Conventions:
        - Box is treated as a SOLID OBSTACLE: ``d > 0`` outside the box,
          ``d < 0`` inside. Fluid is expected on the outside.
        - For a CONTAINER (fluid inside a box), build six planes instead.

    Args:
        pos: Particle positions [m].
        pressure: Particle pressure [Pa].
        density: Particle density [kg/m³].
        particle_flags: Particle activity flags.
        particle_type: Particle type.
        box_transform: World-frame pose of the box (3 pos + 4 quat).
        box_half_extents: Local-frame half-extents (hx, hy, hz) [m].
        h: Smoothing length [m].
        support_radius: SPH support radius [m].
        reference_density: Reference density rho₀ [kg/m³].
        volume_map_table: Precomputed V_B(d) values.
        accel: Acceleration array (accumulated in-place) [m/s²].
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x_world = pos[i]
    inv_tf = wp.transform_inverse(box_transform)
    x_local = wp.transform_point(inv_tf, x_world)

    hx = box_half_extents[0]
    hy = box_half_extents[1]
    hz = box_half_extents[2]
    d = sdf_box(x_local, hx, hy, hz)
    if d >= support_radius:
        return

    n_local = sdf_box_grad(x_local, hx, hy, hz)
    n_world = wp.transform_vector(box_transform, n_local)
    # Defensive normalisation — sdf_box_grad axis-projection branches can
    # leak non-unit length under the half-extent tie-break logic.
    n_len = wp.length(n_world)
    if n_len < _EPSILON:
        return
    n_world = n_world / n_len

    a = _eval_volume_map_pressure_accel(
        d,
        n_world,
        h,
        support_radius,
        pressure[i],
        rho_i,
        reference_density,
        volume_map_table,
    )
    accel[i] = accel[i] + a


# ---------------------------------------------------------------------------
# Boundary friction Laplacian (Bender 2020 Eqs. 16, 18, 19)
# ---------------------------------------------------------------------------


@wp.func
def _surface_point_laplacian_contribution(
    x_i: wp.vec3,
    x_s: wp.vec3,
    v_in: wp.vec3,
    h: float,
    support_radius: float,
    eta_sq: float,
    v_b_share: float,
) -> wp.vec3:
    """Per-surface-point contribution to the SPH friction Laplacian.

    Implements one term of Bender 2020 Eq. 19 with the static-boundary
    surface velocity (v_j = 0) already substituted into v_ij = v_in - 0,
    and the tangent projection (Eq. 18) or sticky-variant choice already
    applied by the caller in ``v_in``.

    Returns the contribution

        Δ = V_j · (v_in · x_ij) / (||x_ij||² + ε²_h) · ∇W(x_ij, h)

    The 2(d+2) prefactor and any μ_B / rho_i scaling are applied by the
    outer kernel. Returns zero if the surface point is outside support
    or coincides with the fluid particle (degenerate r_ij ≈ 0).
    """
    x_ij = x_i - x_s
    r_ij = wp.length(x_ij)
    if r_ij < _EPSILON:
        return wp.vec3(0.0)
    if r_ij >= support_radius:
        return wp.vec3(0.0)
    grad_w = wendland_c2_grad_3d(x_ij, r_ij, h)
    coef = wp.dot(v_in, x_ij) / (r_ij * r_ij + eta_sq)
    return v_b_share * coef * grad_w


@wp.func
def _project_tangent(v: wp.vec3, n: wp.vec3, sticky: int) -> wp.vec3:
    """Apply Bender 2020 Eq. 18 projection (sliding) or pass-through (sticky)."""
    if sticky != 0:
        return v
    return v - wp.dot(v, n) * n


@wp.func
def _eval_friction_laplacian_apply(
    x_i: wp.vec3,
    n_world: wp.vec3,
    d: float,
    v_input: wp.vec3,
    rho_i: float,
    h: float,
    support_radius: float,
    dt: float,
    boundary_viscosity: float,
    sticky: int,
    volume_map_table: wp.array[float],
) -> wp.vec3:
    """Evaluate the friction-Laplacian apply contribution, ``-dt μ_B/rho_i · L_B v``.

    Builds the three Bender 2020 surface points (closest point plus two
    tangent samples at r/2) on a Frisvad orthonormal frame around
    ``n_world``, then accumulates each one's Laplacian contribution and
    returns the per-particle apply increment. The caller adds the result
    to its CG output array ``y[i]``.

    Returns zero when the boundary is too far away (``d ≥ support_radius``)
    or when the volume map yields no boundary mass.
    """
    if d >= support_radius:
        return wp.vec3(0.0)

    v_b = sample_volume_map(d, support_radius, volume_map_table)
    if v_b < _EPSILON:
        return wp.vec3(0.0)

    # Three surface points: x*, x* + r/2 t1, x* + r/2 t2.
    x_star = x_i - d * n_world
    t1, t2 = frisvad_tangent_basis(n_world)
    half_r = 0.5 * support_radius
    x_s0 = x_star
    x_s1 = x_star + half_r * t1
    x_s2 = x_star + half_r * t2

    v_t = _project_tangent(v_input, n_world, sticky)

    eta_sq = 0.01 * h * h
    v_b_share = v_b * _INV_N_SURFACE_POINTS

    lap = wp.vec3(0.0)
    lap = lap + _surface_point_laplacian_contribution(x_i, x_s0, v_t, h, support_radius, eta_sq, v_b_share)
    lap = lap + _surface_point_laplacian_contribution(x_i, x_s1, v_t, h, support_radius, eta_sq, v_b_share)
    lap = lap + _surface_point_laplacian_contribution(x_i, x_s2, v_t, h, support_radius, eta_sq, v_b_share)
    lap = _LAP_PREFACTOR * lap

    return -dt * (boundary_viscosity / rho_i) * lap


@wp.kernel
def friction_laplacian_plane_kernel(
    pos: wp.array[wp.vec3],
    v_input: wp.array[wp.vec3],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_offset: float,
    h: float,
    support_radius: float,
    dt: float,
    boundary_viscosity: float,
    sticky: int,
    volume_map_table: wp.array[float],
    # in/out
    y: wp.array[wp.vec3],
):
    """Apply ``y[i] -= dt μ_B/rho_i · L_B[v_input]_i`` for an infinite plane shape.

    Used inside the matrix-free CG iteration in
    :class:`sph_implicit_viscosity.ImplicitFrictionSolver` after the
    identity-init step. The negative sign comes from the implicit
    operator ``A = I - dt diag(μ_B/rho_i) L_B``.

    Args:
        pos: Fluid particle positions [m].
        v_input: CG input vector (search direction or current iterate),
            same shape as positions.
        density: Particle density [kg/m³].
        particle_flags: Particle activity flags.
        particle_type: Particle type.
        plane_normal: Outward unit normal of the plane.
        plane_offset: Plane offset (signed-distance constant ``c`` in
            ``d = n·x + c``).
        h: Smoothing length [m].
        support_radius: SPH support radius [m].
        dt: Time step [s].
        boundary_viscosity: Surface viscosity μ_B [kg/(m·s)].
        sticky: 0 → tangent projection (Eq. 18); ≠0 → sticky variant.
        volume_map_table: Precomputed V_B(d) values.
        y: CG output vector ``A · v_input`` (accumulated in-place).
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x_i = pos[i]
    d = wp.dot(plane_normal, x_i) + plane_offset

    inc = _eval_friction_laplacian_apply(
        x_i,
        plane_normal,
        d,
        v_input[i],
        rho_i,
        h,
        support_radius,
        dt,
        boundary_viscosity,
        sticky,
        volume_map_table,
    )
    y[i] = y[i] + inc


@wp.kernel
def friction_laplacian_box_kernel(
    pos: wp.array[wp.vec3],
    v_input: wp.array[wp.vec3],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    box_transform: wp.transform,
    box_half_extents: wp.vec3,
    h: float,
    support_radius: float,
    dt: float,
    boundary_viscosity: float,
    sticky: int,
    volume_map_table: wp.array[float],
    # in/out
    y: wp.array[wp.vec3],
):
    """Apply boundary-friction Laplacian for a box obstacle.

    Args mirror :func:`friction_laplacian_plane_kernel`; SDF and outward
    normal are computed in the box's local frame and the normal rotated
    back to world. Same convention as the pressure kernel: box is a
    SOLID obstacle (``d > 0`` outside, ``d < 0`` inside).
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x_world = pos[i]
    inv_tf = wp.transform_inverse(box_transform)
    x_local = wp.transform_point(inv_tf, x_world)

    hx = box_half_extents[0]
    hy = box_half_extents[1]
    hz = box_half_extents[2]
    d = sdf_box(x_local, hx, hy, hz)
    if d >= support_radius:
        return

    n_local = sdf_box_grad(x_local, hx, hy, hz)
    n_world = wp.transform_vector(box_transform, n_local)
    n_len = wp.length(n_world)
    if n_len < _EPSILON:
        return
    n_world = n_world / n_len

    inc = _eval_friction_laplacian_apply(
        x_world,
        n_world,
        d,
        v_input[i],
        rho_i,
        h,
        support_radius,
        dt,
        boundary_viscosity,
        sticky,
        volume_map_table,
    )
    y[i] = y[i] + inc


@wp.kernel
def friction_laplacian_sphere_kernel(
    pos: wp.array[wp.vec3],
    v_input: wp.array[wp.vec3],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    sphere_transform: wp.transform,
    sphere_radius: float,
    h: float,
    support_radius: float,
    dt: float,
    boundary_viscosity: float,
    sticky: int,
    volume_map_table: wp.array[float],
    # in/out
    y: wp.array[wp.vec3],
):
    """Apply boundary-friction Laplacian for a sphere obstacle."""
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x_world = pos[i]
    inv_tf = wp.transform_inverse(sphere_transform)
    x_local = wp.transform_point(inv_tf, x_world)

    d = sdf_sphere(x_local, sphere_radius)
    if d >= support_radius:
        return

    n_local = sdf_sphere_grad(x_local, sphere_radius)
    n_world = wp.transform_vector(sphere_transform, n_local)
    n_len = wp.length(n_world)
    if n_len < _EPSILON:
        return
    n_world = n_world / n_len

    inc = _eval_friction_laplacian_apply(
        x_world,
        n_world,
        d,
        v_input[i],
        rho_i,
        h,
        support_radius,
        dt,
        boundary_viscosity,
        sticky,
        volume_map_table,
    )
    y[i] = y[i] + inc


@wp.kernel
def volume_map_pressure_sphere_kernel(
    pos: wp.array[wp.vec3],
    pressure: wp.array[float],
    density: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    sphere_transform: wp.transform,
    sphere_radius: float,
    h: float,
    support_radius: float,
    reference_density: float,
    volume_map_table: wp.array[float],
    # output
    accel: wp.array[wp.vec3],
):
    """Apply volume-map boundary pressure for a sphere obstacle (fluid outside).

    Conventions:
        - Sphere is treated as a SOLID OBSTACLE: ``d > 0`` outside,
          ``d < 0`` inside. Fluid lives outside.

    Args:
        pos: Particle positions [m].
        pressure: Particle pressure [Pa].
        density: Particle density [kg/m³].
        particle_flags: Particle activity flags.
        particle_type: Particle type.
        sphere_transform: World-frame pose of the sphere (3 pos + 4 quat).
        sphere_radius: Sphere radius [m].
        h: Smoothing length [m].
        support_radius: SPH support radius [m].
        reference_density: Reference density rho₀ [kg/m³].
        volume_map_table: Precomputed V_B(d) values.
        accel: Acceleration array (accumulated in-place) [m/s²].
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = density[i]
    if rho_i < _EPSILON:
        return

    x_world = pos[i]
    inv_tf = wp.transform_inverse(sphere_transform)
    x_local = wp.transform_point(inv_tf, x_world)

    d = sdf_sphere(x_local, sphere_radius)
    if d >= support_radius:
        return

    n_local = sdf_sphere_grad(x_local, sphere_radius)
    n_world = wp.transform_vector(sphere_transform, n_local)
    n_len = wp.length(n_world)
    if n_len < _EPSILON:
        return
    n_world = n_world / n_len

    a = _eval_volume_map_pressure_accel(
        d,
        n_world,
        h,
        support_radius,
        pressure[i],
        rho_i,
        reference_density,
        volume_map_table,
    )
    accel[i] = accel[i] + a
