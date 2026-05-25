# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way coupling between SolverSPH and rigid bodies driven by an MBD solver.

Architecture: SDF + plastic indentation per primitive collision shape
(sphere / capsule along local Z / axis-aligned box). For each fluid particle
inside a body collider, the normal force depends on the sign of the relative
normal velocity ``v_n = (v_p - v_body_at_x) . n``:

    Approach (v_n < 0):
        F_n = (bearing_capacity * dx^2 + c_n * (-v_n)) * n
    Separation (v_n >= 0):
        F_n = 0     (no restoring spring — sand does not pull the body back)

    F_t = -min(mu * |F_n|, m_p * |v_t| / dt) * v_t / |v_t|   (Coulomb, regularised)

Each SPH particle acts as a quadrature point covering area ``dx^2``
(``dx`` = ``particle_spacing``). Summing the per-particle bearing force
``bearing_capacity * dx^2`` over the ``N`` particles inside the SDF
recovers the Terzaghi total ``bearing_capacity * A_projected`` because
``N * dx^2`` approximates the projected contact area. The asymmetric on/off
law dissipates impact kinetic energy: work done by the contact during
approach is **not** returned during separation, so a body dropped on the
granular bed settles instead of bouncing elastically.

Reaction wrench ``-F`` is applied at the contact point and atomic-added into a
per-body :class:`wp.spatial_vector` accumulator (layout ``(linear, angular)``
to match Newton's :attr:`State.body_f` convention). The caller integrates the
accumulator into :attr:`State.body_f` before the MBD step.

References:
    - Newton MPM ``compute_body_forces`` (``newton.examples.mpm.example_mpm_twoway_coupling``)
    - tiSPHi penalty boundary
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import warp as wp

from ...geometry import GeoType, ParticleFlags, ShapeFlags
from .sph_dummy_boundary import SPH_FLUID

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)
_PI = wp.constant(3.141592653589793)

# Collider shape-type tag values used by the Warp kernel. Kept as a small dense
# enum (not :class:`GeoType` directly) so the kernel branches stay tight and so
# unsupported primitives are filtered out at solver init.
_SHAPE_SPHERE = wp.constant(0)
_SHAPE_CAPSULE = wp.constant(1)
_SHAPE_BOX = wp.constant(2)


@dataclass
class BodyCollider:
    """Flat per-collider arrays describing each body's collision primitive.

    Built once by :meth:`SolverSPH._init_body_coupling` and rebuilt on
    ``SolverNotifyFlags.SHAPE_PROPERTIES``. Only sphere, capsule (local Z axis)
    and axis-aligned box primitives are supported; other shapes are
    warn-skipped at construction time.

    Attributes:
        body_id: Parent body index per collider entry, shape ``[count]``.
        shape_type: Tag in ``{0=sphere, 1=capsule, 2=box}``, shape ``[count]``.
        shape_params: Primitive dims [m] per entry. Sphere: ``(r, _, _)``;
            capsule: ``(r, half_height, _)``; box: ``(hx, hy, hz)``.
            Shape ``[count, 3]``.
        shape_xform: Shape-local transform relative to the parent body frame,
            shape ``[count]``.
        count: Number of body colliders.
    """

    body_id: wp.array
    shape_type: wp.array
    shape_params: wp.array
    shape_xform: wp.array
    count: int


@wp.func
def _sdf_sphere_3d(x_local: wp.vec3, params: wp.vec3):
    """Sphere SDF in shape-local frame. ``params.x`` is the radius [m]."""
    r = params[0]
    p = wp.length(x_local)
    if p > _EPSILON:
        n = x_local / p
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    return p - r, n


@wp.func
def _sdf_capsule_3d(x_local: wp.vec3, params: wp.vec3):
    """Capsule SDF with axis along local +Z.

    ``params.x`` is the radius and ``params.y`` is the half-height of the
    cylindrical segment (excluding the hemispherical caps).
    """
    r = params[0]
    hh = params[1]
    z = x_local[2]
    z_clamped = wp.clamp(z, -hh, hh)
    q = x_local - wp.vec3(0.0, 0.0, z_clamped)
    p = wp.length(q)
    if p > _EPSILON:
        n = q / p
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    return p - r, n


@wp.func
def _sdf_box_3d(x_local: wp.vec3, params: wp.vec3):
    """Axis-aligned box SDF. ``params`` holds the half-extents (hx, hy, hz) [m]."""
    hx = params[0]
    hy = params[1]
    hz = params[2]
    qx = wp.abs(x_local[0]) - hx
    qy = wp.abs(x_local[1]) - hy
    qz = wp.abs(x_local[2]) - hz
    # Exterior distance + interior distance contributions.
    ex = wp.max(qx, 0.0)
    ey = wp.max(qy, 0.0)
    ez = wp.max(qz, 0.0)
    outside = wp.length(wp.vec3(ex, ey, ez))
    inside = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
    d = outside + inside
    # Outward normal: gradient of |x| - h along each axis, then normalize.
    nx = wp.sign(x_local[0]) * wp.where(qx > qy and qx > qz, 1.0, 0.0)
    ny = wp.sign(x_local[1]) * wp.where(qy >= qx and qy > qz, 1.0, 0.0)
    nz = wp.sign(x_local[2]) * wp.where(qz >= qx and qz >= qy, 1.0, 0.0)
    n = wp.vec3(nx, ny, nz)
    n_len = wp.length(n)
    if n_len > _EPSILON:
        n = n / n_len
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    return d, n


@wp.func
def _body_sdf_3d(shape_type: int, params: wp.vec3, x_local: wp.vec3):
    """Dispatch SDF + outward unit normal in shape-local frame.

    Returns ``(d, n_local)`` where ``d`` is the signed distance (negative
    inside) and ``n_local`` is the outward unit normal in the shape's local
    frame.
    """
    if shape_type == _SHAPE_SPHERE:
        return _sdf_sphere_3d(x_local, params)
    if shape_type == _SHAPE_CAPSULE:
        return _sdf_capsule_3d(x_local, params)
    # Default: box.
    return _sdf_box_3d(x_local, params)


@wp.func
def _plastic_contact_area(shape_type: int, params: wp.vec3, pen: float) -> float:
    """Projected contact area [m^2] for a shape at penetration depth ``pen``.

    Sphere: spherical cap projected area, saturating at full cross-section.
    Capsule / box: characteristic cross-section (order-of-magnitude estimate).

    .. note::
        Not called by :func:`apply_body_coupling_force_kernel` — that kernel
        uses the per-particle quadrature area ``dx^2`` so that the sum over
        all penetrating particles recovers the projected contact area
        automatically. Kept as a utility for CFL estimation and future
        kernels that need an explicit per-shape contact area.
    """
    if shape_type == _SHAPE_SPHERE:
        r = params[0]
        a_sq = wp.where(pen < r, 2.0 * r * pen - pen * pen, r * r)
        a_sq = wp.max(a_sq, 0.0)
        return _PI * a_sq
    # Non-sphere shapes: use full cross-section as a constant-area
    # approximation (capsule & box coupling is not validated yet).
    if shape_type == _SHAPE_CAPSULE:
        return _PI * params[0] * params[0]
    # Box
    return 4.0 * params[0] * params[1]


@wp.kernel
def apply_body_coupling_force_kernel(
    # particle inputs
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    particle_density: wp.array[float],
    particle_mass: wp.array[float],
    # body inputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # collider inputs
    collider_body_id: wp.array[wp.int32],
    collider_shape_type: wp.array[wp.int32],
    collider_shape_params: wp.array[wp.vec3],
    collider_shape_xform: wp.array[wp.transform],
    n_colliders: int,
    # coupling params
    c_n: float,
    mu: float,
    bearing_capacity: float,
    particle_spacing: float,
    inv_mass_scale_dt: float,
    # outputs (accumulated)
    accel: wp.array[wp.vec3],
    body_f_sand: wp.array[wp.spatial_vector],
):
    """Apply plastic coupling + Coulomb friction forces between SPH fluids and rigid bodies.

    For each fluid particle, query every collider, evaluate the analytic SDF
    in shape-local frame, and if penetrating, apply a plastic indentation
    force during approach (zero on separation) and a Coulomb tangential
    friction force. The equal-and-opposite wrench is atomic-added to
    ``body_f_sand`` at the body's COM, expressed as
    ``wp.spatial_vector(force_world, torque_world)``.

    The normal force is ``bearing_capacity * dx^2 + c_n * (-v_n)`` per
    particle while ``v_n < 0`` (body approaching the particle), and zero
    during separation. Summing the bearing term over the ``N`` penetrating
    particles recovers the Terzaghi total
    ``bearing_capacity * A_projected`` since ``N * dx^2`` approximates the
    projected contact area. Body pose is frozen during the SPH substep loop.

    Args:
        particle_q: Particle positions [m].
        particle_qd: Particle velocities [m/s].
        particle_flags: Particle activity flags.
        particle_type: Particle type tags (only fluids are coupled).
        particle_density: Per-particle density [kg/m^3]; particles with
            non-positive or non-finite density are skipped.
        particle_mass: Per-particle mass [kg].
        body_q: Body transforms (world).
        body_qd: Body twists (linear, angular) in world frame.
        body_com: Body COM offsets [m] in body-local frame.
        collider_body_id: Parent body index per collider entry.
        collider_shape_type: Collider primitive tag (0=sphere, 1=capsule, 2=box).
        collider_shape_params: Primitive dims [m] (see :class:`BodyCollider`).
        collider_shape_xform: Shape-local transform relative to parent body.
        n_colliders: Number of collider entries.
        c_n: Normal damping [N*s/m], active only during approach.
        mu: Coulomb friction coefficient.
        bearing_capacity: Plastic bearing capacity of the granular material
            [Pa]. The approach-phase normal force per particle is
            ``bearing_capacity * dx^2`` plus the damping term.
        particle_spacing: SPH particle spacing ``dx`` [m]; each particle is
            treated as a quadrature point covering area ``dx^2``.
        inv_mass_scale_dt: Tangential damping regularization scale (``1/dt``-style
            term) [1/s], used to cap friction at low slip velocities to avoid
            stick-slip chatter; see :meth:`SolverSPH._apply_body_forces`.
        accel: Particle acceleration accumulator [m/s^2].
        body_f_sand: Per-body sand-on-body wrench accumulator [N, N*m],
            layout ``(force_world, torque_world)``.
    """
    pid = wp.tid()
    if (particle_flags[pid] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[pid] != SPH_FLUID:
        return

    rho = particle_density[pid]
    # NaN / non-positive density guard: skip but never overwrite stress here.
    if not (rho > 0.0):
        return

    x_w = particle_q[pid]
    v_p = particle_qd[pid]
    m_p = particle_mass[pid]

    for ci in range(n_colliders):
        bid = collider_body_id[ci]
        # World <- body <- shape composition.
        x_wb = body_q[bid]
        x_bs = collider_shape_xform[ci]
        x_ws = wp.transform_multiply(x_wb, x_bs)

        # Particle in shape-local frame.
        x_s = wp.transform_point(wp.transform_inverse(x_ws), x_w)

        d, n_s = _body_sdf_3d(collider_shape_type[ci], collider_shape_params[ci], x_s)
        if d >= 0.0:
            continue  # outside this collider.

        # Outward normal -> world frame (vector rotation, ignore translation).
        n_w = wp.transform_vector(x_ws, n_s)

        # Body velocity at contact point: v_body + omega x (x - x_com_world).
        com_w = wp.transform_point(x_wb, body_com[bid])
        r_w = x_w - com_w
        v_b = wp.spatial_top(body_qd[bid])  # linear
        w_b = wp.spatial_bottom(body_qd[bid])  # angular
        v_body_at_x = v_b + wp.cross(w_b, r_w)

        v_rel = v_p - v_body_at_x
        v_n = wp.dot(v_rel, n_w)
        v_t = v_rel - v_n * n_w

        # Plastic coupling: no restoring force on separation.
        # Approach: bearing pressure * per-particle quadrature area + rate damping.
        # Each particle covers dx^2, so summing over N penetrating particles
        # yields bearing_capacity * N * dx^2 ~= bearing_capacity * A_projected
        # (Terzaghi). Separation: zero force — no energy returned to body.
        if v_n >= 0.0:
            continue
        particle_area = particle_spacing * particle_spacing
        f_n_mag = bearing_capacity * particle_area + c_n * (-v_n)
        if f_n_mag <= 0.0:
            continue
        f_n = f_n_mag * n_w

        # Coulomb friction, regularized by a damping-like cap to avoid chatter.
        f_t = wp.vec3(0.0)
        v_t_norm = wp.length(v_t)
        if mu > 0.0 and v_t_norm > _EPSILON:
            f_t_mag = wp.min(mu * f_n_mag, m_p * v_t_norm * inv_mass_scale_dt)
            f_t = -(f_t_mag / wp.max(v_t_norm, _EPSILON)) * v_t

        f_total = f_n + f_t

        # Particle accumulates acceleration, not force.
        wp.atomic_add(accel, pid, f_total / m_p)

        # Newton's 3rd law: equal-and-opposite force at contact point.
        f_on_body = -f_total
        torque_on_body = wp.cross(r_w, f_on_body)
        # Newton convention: body_f = (force_world, torque_world).
        wp.atomic_add(body_f_sand, bid, wp.spatial_vector(f_on_body, torque_on_body))


def build_body_collider(
    model,
    device,
) -> BodyCollider | None:
    """Extract per-body collider primitives from a Newton model.

    Walks ``model.shape_*`` arrays once on the CPU, keeps only shapes whose
    ``ShapeFlags.COLLIDE_PARTICLES`` bit is set and whose ``shape_body >= 0``
    and whose primitive type is in ``{SPHERE, CAPSULE, BOX}``. Other shape
    types (mesh, plane, ellipsoid, ...) emit a ``RuntimeWarning`` and are
    skipped -- mesh-SDF coupling is a Phase 2 feature.

    Returns:
        A :class:`BodyCollider` with arrays allocated on ``device``, or
        ``None`` if no usable colliders were found.
    """
    if model.shape_count == 0 or model.body_count == 0:
        return None

    shape_type_np = model.shape_type.numpy()
    shape_body_np = model.shape_body.numpy()
    shape_flags_np = model.shape_flags.numpy()
    shape_transform_np = model.shape_transform.numpy()
    shape_scale_np = model.shape_scale.numpy()

    supported = {int(GeoType.SPHERE), int(GeoType.CAPSULE), int(GeoType.BOX)}
    body_ids: list[int] = []
    shape_tags: list[int] = []
    params: list[tuple[float, float, float]] = []
    xforms: list[tuple[float, float, float, float, float, float, float]] = []

    n_skipped = 0
    for s in range(model.shape_count):
        if (int(shape_flags_np[s]) & int(ShapeFlags.COLLIDE_PARTICLES)) == 0:
            continue
        bid = int(shape_body_np[s])
        if bid < 0:
            continue
        gt = int(shape_type_np[s])
        if gt not in supported:
            n_skipped += 1
            continue
        scale = shape_scale_np[s]
        if gt == int(GeoType.SPHERE):
            tag = int(_SHAPE_SPHERE)
            p = (float(scale[0]), 0.0, 0.0)
        elif gt == int(GeoType.CAPSULE):
            tag = int(_SHAPE_CAPSULE)
            p = (float(scale[0]), float(scale[1]), 0.0)
        else:  # BOX
            tag = int(_SHAPE_BOX)
            p = (float(scale[0]), float(scale[1]), float(scale[2]))
        tf = shape_transform_np[s]
        xf = (
            float(tf[0]),
            float(tf[1]),
            float(tf[2]),
            float(tf[3]),
            float(tf[4]),
            float(tf[5]),
            float(tf[6]),
        )
        body_ids.append(bid)
        shape_tags.append(tag)
        params.append(p)
        xforms.append(xf)

    if n_skipped > 0:
        warnings.warn(
            f"SPH-body coupling: skipping {n_skipped} non-primitive collider shape(s); "
            "only sphere, capsule (local Z) and box are currently supported.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not body_ids:
        return None

    count = len(body_ids)
    body_id_arr = wp.array(body_ids, dtype=wp.int32, device=device)
    shape_type_arr = wp.array(shape_tags, dtype=wp.int32, device=device)
    shape_params_arr = wp.array(params, dtype=wp.vec3, device=device)
    shape_xform_arr = wp.array(xforms, dtype=wp.transform, device=device)
    return BodyCollider(
        body_id=body_id_arr,
        shape_type=shape_type_arr,
        shape_params=shape_params_arr,
        shape_xform=shape_xform_arr,
        count=count,
    )
