# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH boundary handling.

Provides the penalty-based boundary force kernel for ground planes. The
per-plane launch loop lives in :class:`SolverSPH` (using cached plane
data); this module exposes only the device-side kernel.

References:
    - tiSPHi ``enforce_boundary()``
    - Newton MPM ``rasterized_collisions.py``
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from .sph_dummy_boundary import SPH_FLUID

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)


@wp.kernel
def ground_plane_penalty_kernel(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_offset: float,
    ke: float,
    kd: float,
    mu_f: float,
    # output (accumulated)
    accel: wp.array[wp.vec3],
):
    """Apply penalty force for a ground plane boundary.

    The signed distance from the plane is d = dot(n, x) + offset.
    When d < 0 (penetrating), a restoring normal force and damping are applied:

        f_n = (ke * |d| - kd * v_n) * n

    Coulomb tangential friction opposes sliding with magnitude capped by
    mu * |f_n|, regularized by a damping-like term to prevent chatter at
    low sliding velocities.

    The force is divided by particle mass (handled externally) to produce
    acceleration.

    Args:
        pos: Particle positions.
        vel: Particle velocities.
        particle_flags: Particle activity flags.
        particle_type: Particle type tags; dummy particles are skipped.
        plane_normal: Outward normal of the ground plane.
        plane_offset: Plane offset (d in ax + by + cz + d = 0).
        ke: Penalty stiffness [N/m per unit mass -> m/s^2 per m penetration].
        kd: Penalty damping [N*s/m per unit mass].
        mu_f: Coulomb friction coefficient.
        accel: Acceleration array (accumulated in-place).
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    x = pos[i]
    v = vel[i]

    # Signed distance
    d = wp.dot(plane_normal, x) + plane_offset
    if d < 0.0:
        # Normal velocity component
        v_n = wp.dot(v, plane_normal)
        # Penalty force (per unit mass = acceleration)
        f_mag = ke * (-d) - kd * v_n
        if f_mag < 0.0:
            f_mag = 0.0
        accel[i] = accel[i] + f_mag * plane_normal

        # Coulomb tangential friction
        if mu_f > 0.0:
            v_t = v - wp.dot(v, plane_normal) * plane_normal
            v_t_norm = wp.length(v_t)
            if v_t_norm > _EPSILON:
                f_t_mag = mu_f * f_mag
                # Regularization: cap by damping-like term to avoid chatter
                f_t_mag = wp.min(f_t_mag, kd * v_t_norm)
                accel[i] = accel[i] - (f_t_mag / wp.max(v_t_norm, _EPSILON)) * v_t
