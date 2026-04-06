# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Layered dummy particle boundary treatment for SPH.

Generates static dummy particles in layers outside an AABB domain box.
During SPH computation, dummy particles provide virtual (extrapolated)
velocity, stress, and pressure values to neighboring fluid particles,
ensuring full kernel support near walls.

The distance-based method uses layered dummy particles with velocity and
stress extrapolation, ported from tiSPHi ``SPHBoundaryDistanceBased``.

References:
    - tiSPHi ``sph_bdy.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    import newton

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Particle type constants
# ---------------------------------------------------------------------------

SPH_FLUID = wp.constant(0)
"""Fluid particle type constant."""
SPH_DUMMY_NOSLIP = wp.constant(1)
"""Dummy particle with no-slip wall condition."""
SPH_DUMMY_FREESLIP = wp.constant(2)
"""Dummy particle with free-slip wall condition."""

# Plain Python constants for use in numpy/host code
_SPH_DUMMY_NOSLIP_VAL = 1
_SPH_DUMMY_FREESLIP_VAL = 2

_EPSILON = wp.constant(1.0e-8)

# ---------------------------------------------------------------------------
# Virtual field computation helpers (@wp.func)
# ---------------------------------------------------------------------------


@wp.func
def compute_virtual_velocity(
    v_fluid: wp.vec3,
    v_wall: wp.vec3,
    beta: float,
    wall_normal: wp.vec3,
    particle_type_j: int,
) -> wp.vec3:
    """Compute virtual velocity for a dummy particle neighbor.

    Args:
        v_fluid: Velocity of the querying fluid particle.
        v_wall: Actual velocity of the dummy particle (0 for static walls).
        beta: Extrapolation factor (default 1.7).
        wall_normal: Outward wall normal of the dummy particle.
        particle_type_j: Particle type of the dummy (1=no-slip, 2=free-slip).

    Returns:
        Virtual velocity for the dummy particle.
    """
    if particle_type_j == SPH_DUMMY_NOSLIP:
        # No-slip: v_vir = (1 - beta) * v_fluid + beta * v_wall
        return (1.0 - beta) * v_fluid + beta * v_wall
    # Free-slip: mirror normal component
    v_diff = v_fluid - v_wall
    v_n = wp.dot(v_diff, wall_normal) * wall_normal
    m = beta - 1.0
    return v_wall + m * v_diff - 2.0 * m * v_n


@wp.func
def compute_virtual_stress(
    rho0: float,
    gravity: wp.vec3,
    x_fluid: wp.vec3,
    x_wall: wp.vec3,
    stress_fluid: wp.mat33,
    particle_type_j: int,
) -> wp.mat33:
    """Compute virtual stress for a dummy particle neighbor.

    Hydrostatic extrapolation from the fluid particle with isotropic K0=1.
    For static walls (a_wall = 0), the correction is based on gravity only.

    Args:
        rho0: Reference density [kg/m^3].
        gravity: Gravity vector [m/s^2].
        x_fluid: Position of the querying fluid particle.
        x_wall: Position of the dummy particle.
        stress_fluid: Stress tensor of the querying fluid particle.
        particle_type_j: Particle type (1=no-slip, 2=free-slip).

    Returns:
        Virtual stress tensor for the dummy particle.
    """
    dx = x_fluid - x_wall
    # Hydrostatic correction (isotropic K0=1)
    correction = wp.mat33(
        gravity[0] * dx[0],
        0.0,
        0.0,
        0.0,
        gravity[1] * dx[1],
        0.0,
        0.0,
        0.0,
        gravity[2] * dx[2],
    )
    result = rho0 * correction + stress_fluid
    if particle_type_j == SPH_DUMMY_FREESLIP:
        # Flip sign on off-diagonal (shear) components
        result = wp.mat33(
            result[0, 0],
            -result[0, 1],
            -result[0, 2],
            -result[1, 0],
            result[1, 1],
            -result[1, 2],
            -result[2, 0],
            -result[2, 1],
            result[2, 2],
        )
    return result


# ---------------------------------------------------------------------------
# Dummy particle generation (numpy, CPU-side)
# ---------------------------------------------------------------------------


def generate_dummy_particles(
    bounds_lo: tuple[float, float, float],
    bounds_hi: tuple[float, float, float],
    h: float,
    dx: float,
    slip_type: str = "noslip",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate layered dummy particle positions, normals, and types.

    Creates multiple layers of particles outside each face of an AABB domain
    box. The layer thickness is ``2 * h``, with ``ceil(2h / dx)`` layers
    spaced by ``dx``.  Edges and corners of the AABB are also covered by
    extending the in-plane ranges of higher-priority faces.

    Args:
        bounds_lo: Domain AABB lower corner [m].
        bounds_hi: Domain AABB upper corner [m].
        h: Smoothing length [m].
        dx: Particle spacing [m].
        slip_type: ``'noslip'`` or ``'freeslip'``.

    Returns:
        Tuple of (positions, normals, particle_types), each shape ``[M, 3]``
        or ``[M]`` for types.
    """
    lo = np.asarray(bounds_lo, dtype=np.float32)
    hi = np.asarray(bounds_hi, dtype=np.float32)
    thickness = 2.0 * h
    layer_count = max(1, int(np.ceil(thickness / dx)))

    ptype = _SPH_DUMMY_NOSLIP_VAL if slip_type == "noslip" else _SPH_DUMMY_FREESLIP_VAL

    # Interior ranges (domain faces only)
    xs = np.arange(lo[0], hi[0] + 0.5 * dx, dx, dtype=np.float32)
    ys = np.arange(lo[1], hi[1] + 0.5 * dx, dx, dtype=np.float32)
    zs = np.arange(lo[2], hi[2] + 0.5 * dx, dx, dtype=np.float32)

    # Extended ranges that include the dummy layer region, covering edges
    # and corners.  Priority: ±X covers all edges/corners it touches,
    # ±Y covers remaining Z-edges, ±Z uses interior ranges only.
    ys_ext = np.arange(lo[1] - thickness, hi[1] + thickness + 0.5 * dx, dx, dtype=np.float32)
    zs_ext = np.arange(lo[2] - thickness, hi[2] + thickness + 0.5 * dx, dx, dtype=np.float32)

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    def _add_plane(axis: int, sign: float, coord: float, u: np.ndarray, v: np.ndarray) -> None:
        uu, vv = np.meshgrid(u, v, indexing="ij")
        p = np.zeros((uu.size, 3), dtype=np.float32)
        p[:, axis] = coord
        other = [a for a in range(3) if a != axis]
        p[:, other[0]] = uu.ravel()
        p[:, other[1]] = vv.ravel()
        n = np.zeros_like(p)
        n[:, axis] = sign
        positions.append(p)
        normals.append(n)

    for layer in range(1, layer_count + 1):
        off = float(layer) * dx
        # ±X faces use extended y and z (covers X-edges and all 8 corners)
        _add_plane(0, 1.0, float(lo[0] - off), ys_ext, zs_ext)
        _add_plane(0, -1.0, float(hi[0] + off), ys_ext, zs_ext)
        # ±Y faces use interior x, extended z (covers Y-Z edges)
        _add_plane(1, 1.0, float(lo[1] - off), xs, zs_ext)
        _add_plane(1, -1.0, float(hi[1] + off), xs, zs_ext)
        # ±Z faces use interior x and y (edges/corners already covered)
        _add_plane(2, 1.0, float(lo[2] - off), xs, ys)
        # _add_plane(2, -1.0, float(hi[2] + off), xs, ys)

    if not positions:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
    pos_all = np.concatenate(positions, axis=0)
    norm_all = np.concatenate(normals, axis=0)
    types_all = np.full(pos_all.shape[0], ptype, dtype=np.int32)
    return pos_all, norm_all, types_all


def add_dummy_particles_to_builder(
    builder: newton.ModelBuilder,
    bounds_lo: tuple[float, float, float],
    bounds_hi: tuple[float, float, float],
    h: float,
    dx: float,
    reference_density: float,
    slip_type: str = "noslip",
) -> int:
    """Generate and add dummy boundary particles to a model builder.

    Must be called after :meth:`SolverSPH.register_custom_attributes` and
    before ``builder.finalize()``.

    Args:
        builder: Newton model builder.
        bounds_lo: Domain AABB lower corner [m].
        bounds_hi: Domain AABB upper corner [m].
        h: Smoothing length [m].
        dx: Particle spacing [m].
        reference_density: Reference density for mass computation [kg/m^3].
        slip_type: ``'noslip'`` or ``'freeslip'``.

    Returns:
        Number of dummy particles added.
    """
    pos_np, norm_np, types_np = generate_dummy_particles(bounds_lo, bounds_hi, h, dx, slip_type)
    count = pos_np.shape[0]
    if count == 0:
        return 0

    vol = dx**3
    mass_val = reference_density * vol

    pos_list = [tuple(float(v) for v in pos_np[i]) for i in range(count)]
    vel_list = [(0.0, 0.0, 0.0)] * count
    mass_list = [mass_val] * count
    radius_list = [dx / 2.0] * count

    builder.add_particles(
        pos=pos_list,
        vel=vel_list,
        mass=mass_list,
        radius=radius_list,
        custom_attributes={
            "sph:particle_type": types_np.tolist(),
            "sph:wall_normal": [tuple(float(v) for v in norm_np[i]) for i in range(count)],
        },
    )

    return count
