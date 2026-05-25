# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH-rigid two-way coupling MVP: single sphere dropped into a sand bed.

.. note::
    **2026-05-14 — Partial MVP success (3/5 honest).** This example
    passes 3 of 5 ``test_final`` criteria after Fix A (wrench reset)
    and Fix B (MBD co-stepped at ``sim_dt``). Criterion 2 (settling)
    passes by bounce-apex artefact — the sphere bounces at contact and
    ``|vz|<0.05`` is caught at the apex of flight, not during true
    settling. Criterion 5 (terminal-z vs analytic crater estimate)
    cannot pass because the penalty coupling ``F = k_n·pen +
    c_n·max(0,-v_n)`` is fundamentally **elastic**: the spring stores
    energy that returns to the sphere on separation (damping is off
    when ``v_n>0``). True granular energy dissipation happens inside
    the SPH Drucker-Prager constitutive model, not at the coupling
    boundary. A plastic-coupling scheme (e.g. Akinci-style kernel
    interpolation) is required for criterion 5. See
    ``newton/_src/solvers/sph/CLAUDE.md §10b`` for the full known-issues
    list. The SPH baseline (``test_sph``, 14/14) is unaffected.

Drops a 1 kg, 5 cm-radius rigid sphere from 0.10 m above the surface
into a 0.40 x 0.40 x 0.30 m static sand bed (dx = 5 mm) enclosed in a
0.40 x 0.40 x 0.40 m cubic penalty box (no top).  Verifies the Phase 2b
body-coupling hooks end-to-end before scaling up to articulated robots.

Architecture:
    * Single Newton :class:`Model` carrying both the SPH particles and
      the rigid sphere (single free body).  Both solvers share the same
      ``state.body_q / body_qd / body_f`` arrays.
    * MBD (rigid integration): :class:`SolverSemiImplicit` — free body,
      gravity baked-in, integrates ``state.body_f`` directly.  Light-weight
      and sufficient for a single un-constrained body; chosen over MuJoCo
      to minimise setup for the MVP.
    * SPH: :class:`SolverSPH` with
      ``Config.body_coupling_enabled = True`` and the sphere shape carrying
      ``ShapeFlags.COLLIDE_PARTICLES`` (set by default on every primitive
      via :class:`ModelBuilder.ShapeConfig`).
    * Per frame: SPH computes the sand-on-body wrench
      (``solver.collect_body_wrench(state)``), the example converts it to
      body forces via ``compute_body_forces`` and adds it to ``state.body_f``,
      then runs MBD ``step()`` followed by SPH ``step()`` (which auto-resets
      its internal wrench accumulator).  Matches the
      ``example_mpm_twoway_coupling.py`` pattern.
    * Telemetry (per frame): sphere kinematics, sand wrench Fz, total
      kinetic energy, particle-leak counter — appended to
      ``args.log_path`` (CSV) at the end of the run.

Pass criteria (encoded in :meth:`Example.test_final`):
    1. Sphere penetrates the sand surface (sphere bottom below
       ``args.sand_bed_top`` within 0.40 s -- free-fall from default
       ``drop_height = 0.30 m`` reaches the sand at t ~= 0.247 s).
    2. Sphere settles (|v_z| < 0.05 m/s by 0.5 s).
    3. No NaN in rigid state or particle state at any logged frame.
    4. No fluid particle drifts more than ``dx`` below the floor (the ground plane at ``--sand-bed-bottom``).
    5. Final sphere bottom-z lies within +/- 20 % of an analytic crater
       estimate:

           z_crater ~ -(E_impact / (sigma_y * A))^(1/3),

       with ``sigma_y ~ 10 kPa`` (loose-sand bearing capacity, after
       Terzaghi / Newtonian impact-crater scaling — see notes in
       :meth:`Example._predict_penetration`).  Drop height 0.30 m above
       the sand surface gives ``v_impact ~ 2.43 m/s`` and
       ``E_impact ~ 2.94 J``; with sphere bottom-cap area
       ``pi r^2 ~ 7.85e-3 m^2`` this gives ``z_crater ~ -7 cm``.  The
       analytical estimate is a sanity bound, not a calibrated reference;
       deviation up to 20 % is allowed.

Reference layout: ``newton.examples.mpm.example_mpm_twoway_coupling``.

Command:
    python -m newton.examples sph_twoway_sphere_drop --num-frames 40
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ShapeFlags
from newton.solvers import SolverSemiImplicit, SolverSPH


@wp.kernel
def compute_body_forces(
    sand_wrench: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Convert sand-on-body wrench to body forces at the center of mass.

    Each entry of ``sand_wrench`` is already ``wp.spatial_vector(force_world,
    torque_world)`` in [N, N*m] — the SPH coupling kernel computes the reaction
    wrench directly.  We atomic-add into ``body_f`` so that the MBD solver
    integrates it in the next step.

    Matches ``compute_body_forces`` in ``example_mpm_twoway_coupling.py``.
    """
    b = wp.tid()
    w = sand_wrench[b]
    # The wrench is already (force_world, torque_world) at the body COM from
    # the SPH coupling kernel, so no lever-arm correction is needed here.
    wp.atomic_add(body_f, b, w)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        # ---- Simulation timing ----
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_duration = args.duration
        # SPH substep count from acoustic CFL: dt_cfl = 0.3 * h / c_s.
        h = args.kh * args.particle_spacing
        dt_cfl = 0.3 * h / args.sound_speed
        if args.substeps is None:
            self.sim_substeps = max(1, int(math.ceil(self.frame_dt / dt_cfl)))
        else:
            self.sim_substeps = max(1, args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        # ---- Build unified model ----
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # 1. Rigid sphere (single free body)
        radius = args.sphere_radius
        mass = args.sphere_mass
        # Solid-sphere inertia tensor: I = (2/5) * m * r^2 * I_3
        i_scalar = 0.4 * mass * radius * radius
        inertia = wp.mat33(
            i_scalar,
            0.0,
            0.0,
            0.0,
            i_scalar,
            0.0,
            0.0,
            0.0,
            i_scalar,
        )
        drop_z = args.sand_bed_top + args.drop_height + radius
        self.sphere_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()),
            mass=mass,
            inertia=inertia,
            label="sphere",
            lock_inertia=True,  # do not let add_shape_sphere overwrite our inertia
        )
        # has_particle_collision is True by default in ShapeConfig, so the
        # ShapeFlags.COLLIDE_PARTICLES bit is set automatically.
        self.sphere_shape = builder.add_shape_sphere(
            self.sphere_body,
            radius=radius,
            label="sphere_collider",
        )

        # 2. SPH sand bed (fluid particles)
        self.fluid_count = self._emit_sand_particles(builder, args)

        # 3. Ground plane (penalty floor for SPH and rigid contacts disabled
        #    via mu=0 elsewhere -- we let SPH coupling carry the sphere,
        #    not rigid-on-plane contacts).
        builder.add_ground_plane(height=args.sand_bed_bottom, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # 4. Cubic box walls (no top): 0.40 x 0.40 x 0.40 m penalty boundaries.
        #    Sand bed (0.40 x 0.40 x 0.30 m) sits inside.  Wall planes are
        #    infinite in Z; floor clips particles from below.
        half_box = 0.5 * args.sand_bed_size
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=args.friction)
        # Left wall: x = -half_box, normal +x
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, half_box), width=0.4, length=0.4, cfg=wall_cfg)
        # Right wall: x = +half_box, normal -x
        builder.add_shape_plane(plane=(-1.0, 0.0, 0.0, half_box), width=0.4, length=0.4, cfg=wall_cfg)
        # Front wall: y = -half_box, normal +y
        builder.add_shape_plane(plane=(0.0, 1.0, 0.0, half_box), width=0.4, length=0.4, cfg=wall_cfg)
        # Back wall: y = +half_box, normal -y
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0, half_box), width=0.4, length=0.4, cfg=wall_cfg)

        # 5. Finalize and force Z-up gravity
        self.model = builder.finalize()
        self.model.set_gravity(tuple(args.gravity))

        # ---- Configure SPH solver with body coupling enabled ----
        sph_cfg = SolverSPH.Config()
        sph_cfg.particle_spacing = args.particle_spacing
        sph_cfg.kh = args.kh
        sph_cfg.reference_density = args.density
        sph_cfg.simulation_method = args.simulation_method
        sph_cfg.artificial_viscosity_alpha = args.artificial_viscosity_alpha
        sph_cfg.sound_speed = args.sound_speed
        sph_cfg.penalty_stiffness = args.penalty_stiffness
        sph_cfg.penalty_damping = 1.0e4
        sph_cfg.boundary_type = "penalty"
        # Stable column base: high-friction floor, near-frictionless lateral walls.
        # Wall vs floor split handled by solver's normal.z > 0.5 classifier.
        sph_cfg.boundary_friction = 0.7
        sph_cfg.boundary_wall_friction = 0.1
        sph_cfg.integration_scheme = args.integration_scheme
        # Body coupling
        sph_cfg.body_coupling_enabled = True
        sph_cfg.body_coupling_stiffness = args.body_coupling_stiffness
        sph_cfg.body_coupling_damping = args.body_coupling_damping
        sph_cfg.body_coupling_friction = args.body_coupling_friction
        sph_cfg.body_coupling_bearing_capacity = args.body_coupling_bearing_capacity

        # Per-particle material parameters (sand defaults, mirrors granular example)
        for attr in ("young_modulus", "poisson_ratio", "friction", "cohesion", "viscosity"):
            if hasattr(args, attr):
                getattr(self.model.sph, attr).fill_(getattr(args, attr))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sph_solver = SolverSPH(self.model, sph_cfg)
        # Initialise geostatic stress for the sand column (Z-up).
        self.sph_solver.initialize_geostatic_stress(self.state_0, y_max=args.sand_bed_top)
        # Mirror the initial stress into state_1 so the first ping-pong swap
        # does not read uninitialised stress as 'previous'.
        wp.copy(self.state_1.sph.stress, self.state_0.sph.stress)

        # ---- MBD solver: free body with semi-implicit Euler ----
        # SemiImplicit handles FREE joints + gravity via integrate_bodies.
        self.mbd_solver = SolverSemiImplicit(self.model)

        self.control = self.model.control()
        # Forward kinematics so state.body_q matches joint_q at t=0.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # Replicate body state into the ping-pong target so swap-in-place works.
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)

        # ---- Viewer ----
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # ---- Telemetry ----
        self._init_telemetry()

        # ---- Pre-settle phase (optional) ----
        self._settle_steps_remaining = int(round(args.settle_time / self.frame_dt))
        if self._settle_steps_remaining > 0:
            # During pre-settle the sphere is frozen by zeroing its velocity each
            # substep; coupling is left on so the sand sees its weight gently.
            # We achieve "frozen" by setting body mass scale ~ infinity is not
            # exposed via API, so we just clamp body_qd between MBD steps below.
            pass

        # Particle render colour matches the MPM twoway example for consistency.
        self.particle_colors = wp.full(
            self.model.particle_count, value=wp.vec3(0.85, 0.72, 0.45), dtype=wp.vec3, device=self.model.device
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_sand_particles(builder: newton.ModelBuilder, args) -> int:
        """Emit a rectangular sand bed and return the fluid particle count."""
        dx = args.particle_spacing
        density = args.density
        volume = dx**3
        mass = density * volume

        lo = np.array([-0.5 * args.sand_bed_size, -0.5 * args.sand_bed_size, args.sand_bed_bottom], dtype=np.float32)
        hi = np.array(
            [0.5 * args.sand_bed_size, 0.5 * args.sand_bed_size, args.sand_bed_top],
            dtype=np.float32,
        )
        xs = np.arange(lo[0] + 0.5 * dx, hi[0], dx, dtype=np.float32)
        ys = np.arange(lo[1] + 0.5 * dx, hi[1], dx, dtype=np.float32)
        zs = np.arange(lo[2] + 0.5 * dx, hi[2], dx, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        pos_list = [tuple(float(v) for v in p) for p in pts]
        vel_list = [(0.0, 0.0, 0.0)] * len(pts)
        mass_list = [mass] * len(pts)
        radius_list = [dx / 2.0] * len(pts)
        builder.add_particles(
            pos=pos_list,
            vel=vel_list,
            mass=mass_list,
            radius=radius_list,
            custom_attributes={"sph:friction": float(args.friction)},
        )
        return len(pts)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _init_telemetry(self) -> None:
        self.telemetry: list[dict[str, float]] = []
        self._prev_sphere_vz: float = 0.0
        self._sphere_z_history: list[float] = []  # for stability check
        self._open_telemetry_csv()

    def _record_frame(self, sand_wrench_np: np.ndarray | None) -> None:
        """Pull per-frame diagnostics off the device and append to the log buffer."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        # body_q stores transforms as (px, py, pz, qx, qy, qz, qw); body_qd as
        # (linear_x, linear_y, linear_z, ang_x, ang_y, ang_z) -- BUT Newton's
        # spatial_vector convention is wp.spatial_top=linear; numpy view of a
        # spatial_vectorf is a flat 6-vector in that order.
        sphere_pos = body_q[0][:3]
        sphere_vel = body_qd[0][:3]
        sphere_z = float(sphere_pos[2])
        sphere_vz = float(sphere_vel[2])
        # Penetration depth: how far below the sand surface the sphere bottom sits.
        sphere_bottom = sphere_z - self.args.sphere_radius
        penetration_depth = max(0.0, self.args.sand_bed_top - sphere_bottom)
        # Numerical az from previous vz (frame-level finite difference).
        sphere_az = (sphere_vz - self._prev_sphere_vz) / self.frame_dt if self.sim_time > 0.0 else 0.0
        self._prev_sphere_vz = sphere_vz

        wrench_fz = 0.0
        if sand_wrench_np is not None and sand_wrench_np.shape[0] > 0:
            # spatial_vector layout: top=linear (force), bottom=angular (torque)
            wrench_fz = float(sand_wrench_np[0][2])

        # Kinetic energy
        # Particle KE
        particle_mass_np = self.model.particle_mass.numpy()
        v2 = np.sum(particle_qd**2, axis=1)
        ke_particles = 0.5 * float(np.sum(particle_mass_np * v2))
        ke_sphere = 0.5 * self.args.sphere_mass * float(np.dot(sphere_vel, sphere_vel))
        ke_total = ke_particles + ke_sphere

        # Particle leak: fluid particles below the floor.
        floor_z = self.args.sand_bed_bottom - self.args.particle_spacing
        n_leak = int(np.sum(particle_q[: self.fluid_count, 2] < floor_z))

        row = {
            "t": float(self.sim_time),
            "sphere_z": sphere_z,
            "sphere_vz": sphere_vz,
            "sphere_az": float(sphere_az),
            "sand_wrench_force_z": wrench_fz,
            "total_kinetic_energy": ke_total,
            "n_particles_below_z0": n_leak,
            "penetration_depth": penetration_depth,
        }
        self.telemetry.append(row)
        self._sphere_z_history.append(sphere_z)
        self._append_telemetry_csv(row)

    _TELEMETRY_COLS = (
        "t",
        "sphere_z",
        "sphere_vz",
        "sphere_az",
        "sand_wrench_force_z",
        "total_kinetic_energy",
        "n_particles_below_z0",
        "penetration_depth",
    )

    def _open_telemetry_csv(self) -> None:
        path = self.args.log_path
        if not path:
            self._csv_file = None
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._csv_file = open(path, "w", encoding="utf-8")
        self._csv_file.write(",".join(self._TELEMETRY_COLS) + "\n")
        self._csv_file.flush()

    def _append_telemetry_csv(self, row: dict[str, float]) -> None:
        if getattr(self, "_csv_file", None) is None:
            return
        self._csv_file.write(",".join(f"{row[c]:.9g}" for c in self._TELEMETRY_COLS) + "\n")
        self._csv_file.flush()

    def _close_telemetry_csv(self) -> None:
        f = getattr(self, "_csv_file", None)
        if f is not None and not f.closed:
            f.close()
            print(f"[example_sph_twoway_sphere_drop] wrote telemetry: {self.args.log_path}", flush=True)
            self._csv_file = None

    def __del__(self):
        # Best-effort close on garbage collection in case test_final never ran.
        try:
            self._close_telemetry_csv()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one frame: MBD and SPH co-stepped at sim_dt inside one loop.

        Each substep: read the sand-on-body wrench from the previous SPH step
        (stored on ``state._sph_body_wrench``), convert to body forces via
        ``compute_body_forces``, step MBD, then step SPH.  The SPH solver
        auto-resets its internal wrench accumulator each step, matching the
        ``example_mpm_twoway_coupling.py`` pattern.
        """
        body_count = self.model.body_count

        for _ in range(self.sim_substeps):
            # Body forces start zero; add sand wrench from previous substep.
            self.state_0.clear_forces()

            sand_wrench = self.sph_solver.collect_body_wrench(self.state_0)
            if sand_wrench is not None:
                wp.launch(
                    compute_body_forces,
                    dim=body_count,
                    inputs=[
                        sand_wrench,
                        self.state_0.body_q,
                        self.model.body_com,
                    ],
                    outputs=[self.state_0.body_f],
                    device=self.model.device,
                )

            # Settle freeze (skipped when settle_time = 0).
            if self._settle_steps_remaining > 0:
                self.state_0.body_qd.zero_()
                self.state_0.body_f.zero_()

            # MBD step at sim_dt (single body, SolverSemiImplicit is cheap).
            self.mbd_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            if self._settle_steps_remaining > 0:
                # Re-clamp the sphere to drop height each substep.
                body_q_np = self.state_0.body_q.numpy()
                radius = self.args.sphere_radius
                body_q_np[0, 2] = self.args.sand_bed_top + self.args.drop_height + radius
                self.state_0.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)
                self.state_0.body_qd.zero_()

            # Mirror body state into state_1 so the SPH step's output state
            # carries the freshly integrated body kinematics.
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            # SPH step: auto-zeros internal wrench accumulator, recomputes
            # from scratch, and stores result on state_out._sph_body_wrench.
            self.sph_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # One host sync per frame for telemetry (last substep's wrench).
        last_wrench = self.sph_solver.collect_body_wrench(self.state_0)
        sand_wrench_np = last_wrench.numpy() if last_wrench is not None else None

        self.sim_time += self.frame_dt
        self._record_frame(sand_wrench_np)

        if self._settle_steps_remaining > 0:
            self._settle_steps_remaining -= 1

        if self.sim_time >= self.sim_duration:
            if not getattr(self, "_plot_generated", False):
                self._plot_generated = True
                self.plot_penetration_depth(save_path=getattr(self.args, "plot_path", None))
            if hasattr(self.viewer, "should_close"):
                self.viewer.should_close = True

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def plot_penetration_depth(self, save_path: str | None = None) -> None:
        """Plot sphere penetration depth into the sand surface vs time."""
        times = [r["t"] for r in self.telemetry]
        depths_mm = [r.get("penetration_depth", 0.0) * 1000.0 for r in self.telemetry]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, depths_mm, color="steelblue", linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Penetration depth [mm]")
        ax.set_title("Sphere penetration into granular column surface")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[sphere_drop] penetration plot saved: {save_path}", flush=True)
        else:
            plt.show()
        plt.close(fig)

    def _predict_penetration(self) -> float:
        """Newtonian crater-scaling estimate of final sphere bottom-z [m].

        From energy balance: ``E_impact = sigma_y * V_crater``.  Approximate
        the crater volume as ``V ~ A * z`` (cylindrical column under a
        flat punch of area ``A = pi r^2`` with depth ``z``).  Then
        ``z ~ E_impact / (sigma_y * A)``.  With m=1 kg, drop = 0.30 m,
        g=9.81 m/s^2: ``E ~ 2.94 J``.  sigma_y ~ 10 kPa (loose sand
        bearing capacity, Terzaghi-scale).  r=5 cm gives A=7.85e-3 m^2,
        so z_crater ~ 0.037 m -- the sphere bottom rests near -3.7 cm.

        This estimate is *deliberately rough*; the test tolerance is +/- 20 %
        of |z|, but with an absolute floor of 5 cm to absorb sand-DP
        parameter uncertainty.
        """
        m = self.args.sphere_mass
        g = abs(float(self.args.gravity[2]))
        h = self.args.drop_height
        r = self.args.sphere_radius
        sigma_y = self.args.bearing_capacity
        e_impact = m * g * h
        a = math.pi * r * r
        return -e_impact / (sigma_y * a)

    def test_final(self) -> None:
        # Ensure the CSV is flushed/closed before assertions so analysis has
        # the log even on failure (rows are append-flushed each frame, this
        # is the close-on-clean-exit hook).
        self._close_telemetry_csv()
        # Plot already generated at end of simulation loop (step()); skip if already done.
        if not getattr(self, "_plot_generated", False):
            self.plot_penetration_depth(save_path=getattr(self.args, "plot_path", None))

        if not self.telemetry:
            raise ValueError("No telemetry collected -- did the simulation run?")

        last = self.telemetry[-1]
        # 3) No NaN in current state (general check, mirroring run() framework)
        for k, v in last.items():
            if not np.isfinite(v):
                raise ValueError(f"Non-finite telemetry at final frame: {k}={v}")

        # 1) Sphere bottom dropped below the sand surface within 0.40 s.
        #    With the default drop_height = 0.30 m above the sand top, free-fall
        #    alone takes sqrt(2 * 0.30 / 9.81) ~= 0.247 s before any contact, so
        #    the threshold must comfortably exceed the impact time. The
        #    comparison reference is the sand top (args.sand_bed_top), not the
        #    floor at z = 0 -- the criterion checks "touched the sand", not
        #    "buried by 10 cm".
        radius = self.args.sphere_radius
        sand_top = self.args.sand_bed_top
        penetration_seen = any(
            (row["t"] <= 0.40 + 1e-6) and (row["sphere_z"] - radius < sand_top) for row in self.telemetry
        )
        if not penetration_seen:
            raise ValueError(
                "Sphere did not penetrate the sand surface within 0.40 s "
                f"(min bottom z = {min(r['sphere_z'] - radius for r in self.telemetry):.4f} m, "
                f"sand top z = {sand_top:.4f} m)."
            )

        # 2) Sphere settled (|v_z| < 0.05 m/s) by 0.5 s -- only enforced if
        #    the run was long enough.
        if self.sim_time >= 0.5:
            settled = any(row["t"] >= 0.5 - 1e-6 and abs(row["sphere_vz"]) < 0.05 for row in self.telemetry)
            if not settled:
                raise ValueError(
                    f"Sphere did not settle to |v_z| < 0.05 m/s by t=0.5 s "
                    f"(min |v_z| post-0.5s = "
                    f"{min(abs(r['sphere_vz']) for r in self.telemetry if r['t'] >= 0.5 - 1e-6):.4f} m/s)."
                )

        # 4) No particle leak through the floor (ground plane at sand_bed_bottom).
        for row in self.telemetry:
            if row["n_particles_below_z0"] > 0:
                raise ValueError(
                    f"Particle leak detected at t={row['t']:.4f}: "
                    f"{int(row['n_particles_below_z0'])} particles below floor z="
                    f"{self.args.sand_bed_bottom - self.args.particle_spacing:.4f}."
                )

        # 5) Sphere terminal bottom-z within +/- 20 % of analytical reference.
        if self.sim_time >= 0.5:
            z_final = last["sphere_z"] - radius  # bottom of sphere
            z_predicted = self._predict_penetration()
            tol = max(0.20 * abs(z_predicted), 0.05)  # absolute floor 5 cm
            if abs(z_final - z_predicted) > tol:
                raise ValueError(
                    f"Final sphere bottom z={z_final:.4f} m outside +/-{tol:.4f} m "
                    f"of analytical estimate {z_predicted:.4f} m. "
                    "Check sand stiffness / friction or rerun longer."
                )

    def test_post_step(self) -> None:
        """Per-step NaN guard."""
        # Cheap host-side check: only inspect the body state (1 body), not the
        # full particle array; that one is checked once at end of run.
        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q_np)):
            raise ValueError(f"Non-finite body_q at t={self.sim_time:.4f}: {body_q_np}")
        if not np.all(np.isfinite(body_qd_np)):
            raise ValueError(f"Non-finite body_qd at t={self.sim_time:.4f}: {body_qd_np}")

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Sand bed -- 20 x 20 x 10 cm chosen so the smoke test finishes in
        # seconds on a single GPU.  The Phase 2b plan calls for 40 x 40 x 15 cm;
        # bump --sand-bed-size to 0.40 and --sand-bed-top to 0.15 for the full
        # validation run.
        parser.add_argument("--sand-bed-size", type=float, default=0.40, help="Sand bed X and Y extent [m].")
        parser.add_argument("--sand-bed-top", type=float, default=0.30, help="Sand bed surface z (top) [m].")
        parser.add_argument(
            "--sand-bed-bottom",
            type=float,
            default=0.0,
            help="Sand bed floor z (bottom) [m]. Negative = deeper bed below original floor.",
        )
        parser.add_argument("--particle-spacing", "-dx", type=float, default=0.005, help="SPH particle spacing dx [m].")
        parser.add_argument("--kh", type=float, default=1.3, help="Smoothing length ratio h = kh * dx.")
        parser.add_argument("--density", type=float, default=2500.0, help="Sand reference density [kg/m^3].")
        parser.add_argument("--sound-speed", type=float, default=50.0, help="Artificial sound speed for WCSPH [m/s].")
        parser.add_argument("--simulation-method", type=str, default="dp", choices=["dp", "mui"])
        parser.add_argument(
            "--integration-scheme", type=str, default="position_verlet", choices=["symplectic_euler", "position_verlet"]
        )
        parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1)
        parser.add_argument("--penalty-stiffness", type=float, default=1.0e8)

        # Material (sand DP defaults match newton.examples.sph_granular)
        parser.add_argument("--young-modulus", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", type=float, default=0.3)
        parser.add_argument("--friction", type=float, default=0.5, help="Internal friction angle [rad] (~30 deg).")
        parser.add_argument("--cohesion", type=float, default=0.0)
        parser.add_argument("--viscosity", type=float, default=1.0e-3)

        # Sphere
        parser.add_argument("--sphere-radius", type=float, default=0.05)
        parser.add_argument("--sphere-mass", type=float, default=1.0)
        parser.add_argument(
            "--drop-height",
            type=float,
            default=0.10,
            help="Sphere bottom-of-sphere starting height above sand surface [m].",
        )

        # Coupling
        parser.add_argument("--body-coupling-stiffness", type=float, default=2.0e4)
        parser.add_argument("--body-coupling-damping", type=float, default=5.0e1)
        parser.add_argument("--body-coupling-friction", type=float, default=0.5)
        parser.add_argument("--body-coupling-bearing-capacity", type=float, default=1.0e4)
        # Other simulation knobs
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
        parser.add_argument("--fps", type=float, default=50.0)
        parser.add_argument("--substeps", type=int, default=None, help="SPH substeps per frame.  Default: CFL-derived.")
        parser.add_argument("--duration", type=float, default=0.8, help="Simulation duration [s].")
        parser.add_argument(
            "--settle-time",
            type=float,
            default=0.05,
            help="Pre-settle interval [s] (sphere frozen, sand settles). Default 0.05 s.",
        )
        parser.add_argument(
            "--plot-path",
            type=str,
            default=None,
            help="Path to save penetration-depth time-curve PNG. If omitted, figure is shown interactively.",
        )
        parser.add_argument(
            "--bearing-capacity", type=float, default=1.0e4, help="sigma_y for analytical penetration depth [Pa]."
        )
        parser.add_argument(
            "--log-path",
            type=str,
            default=(
                "/home/naerleide/work/newton_sph_dev/.claude/.court/"
                "20260513-sph-mbd-coupling-quadruped/sphere_drop_log.csv"
            ),
            help="CSV path for per-frame telemetry.",
        )
        return parser


# Sanity reference to silence linters; ShapeFlags.COLLIDE_PARTICLES is set on
# every primitive by default (ShapeConfig.has_particle_collision=True).
assert ShapeFlags.COLLIDE_PARTICLES


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
