# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH boundary handling.

Provides penalty-based boundary forces for ground planes and other collider
shapes. The ground plane is detected from the model's shape list (shapes
with ``GEO_PLANE`` type).

References:
    - tiSPHi ``enforce_boundary()``
    - Newton MPM ``rasterized_collisions.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import GeoType, ParticleFlags

if TYPE_CHECKING:
    import newton

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)


@wp.kernel
def ground_plane_penalty_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    plane_normal: wp.vec3,
    plane_offset: float,
    ke: float,
    kd: float,
    mu_f: float,
    # output (accumulated)
    accel: wp.array(dtype=wp.vec3),
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


def apply_ground_plane_penalty(
    model: newton.Model,
    state: newton.State,
    accel: wp.array,
    ke: float,
    kd: float,
    mu_f: float = 0.0,
) -> None:
    """Apply ground-plane penalty forces for all plane shapes in the model.

    Iterates over model shapes and applies a penalty force for each shape
    whose geometry type is ``GEO_PLANE``.

    Args:
        model: Newton model with shapes.
        state: Current simulation state.
        accel: Acceleration array to accumulate into.
        ke: Penalty stiffness [m/s^2 per m penetration].
        kd: Penalty damping coefficient.
        mu_f: Coulomb friction coefficient for tangential friction.
    """
    n = model.particle_count
    if n == 0:
        return

    # Read shape data on CPU to iterate over planes
    shape_count = model.shape_count
    if shape_count == 0:
        return

    geo_types = model.shape_type.numpy()
    shape_transforms = model.shape_transform.numpy() if model.shape_count > 0 else None

    for s in range(shape_count):
        if geo_types[s] == int(GeoType.PLANE):
            # Ground plane: normal is the z-axis of the shape transform
            # For a default ground plane, the transform rotation encodes
            # the plane orientation. We extract the "up" vector.
            tf = shape_transforms[s]
            # Transform is stored as [tx, ty, tz, qx, qy, qz, qw]
            px, py, pz = float(tf[0]), float(tf[1]), float(tf[2])
            qx, qy, qz, qw = float(tf[3]), float(tf[4]), float(tf[5]), float(tf[6])

            # Compute the plane normal (rotate [0,0,1] by the quaternion)
            # Using quaternion rotation formula
            q = wp.quat(qx, qy, qz, qw)
            up = wp.vec3(0.0, 0.0, 1.0)
            normal = wp.quat_rotate(q, up)
            # Plane offset: d = -dot(n, p) where p is a point on the plane
            plane_offset = -(normal[0] * px + normal[1] * py + normal[2] * pz)

            wp.launch(
                ground_plane_penalty_kernel,
                dim=n,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.particle_flags,
                    normal,
                    plane_offset,
                    ke,
                    kd,
                    mu_f,
                ],
                outputs=[accel],
                device=model.device,
            )
