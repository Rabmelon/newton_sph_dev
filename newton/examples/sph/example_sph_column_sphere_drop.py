# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH sphere free-drop into a confined granular column.

Coordinate system: Z-up (gravity along -Z).

Box domain:    x=0.2m (+-0.10m), y=0.2m (+-0.10m), z=0.15m (floor at z=0, open top)
Granular fill: x=[-0.10,0.10], y=[-0.10,0.10], z=[0,0.10]

A rigid sphere is released above the column surface; per-frame telemetry
records the sphere's penetration depth ``max(0, column_top - sphere_bottom)``
and writes a time-curve plot at the end of the run.

The granular density is 1510 kg/m^3 (vs the static column's 1500) to balance
the body-coupling reaction wrench against the sphere's added load.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSemiImplicit, SolverSPH


@wp.kernel
def compute_body_forces(
    sand_wrench: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    b = wp.tid()
    w = sand_wrench[b]
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

        self.particle_spacing = 0.005
        kh = 1.3
        sound_speed = 50.0
        h = kh * self.particle_spacing
        dt_cfl = 0.3 * h / sound_speed
        self.sim_substeps = max(1, int(math.ceil(self.frame_dt / dt_cfl)))
        self.sim_dt = self.frame_dt / self.sim_substeps

        self._column_top = 0.10
        sphere_radius = args.sphere_radius
        drop_z = self._column_top + args.drop_height + sphere_radius

        # ---- Build model ----
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # 1. Rigid sphere added BEFORE particles to preserve fluid contiguity.
        sphere_density = 700.0
        sphere_mass = sphere_density * (4.0 / 3.0) * math.pi * sphere_radius**3
        i_scalar = 0.4 * sphere_mass * sphere_radius * sphere_radius
        inertia = wp.mat33(
            i_scalar, 0.0, 0.0,
            0.0, i_scalar, 0.0,
            0.0, 0.0, i_scalar,
        )
        self.sphere_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()),
            mass=sphere_mass,
            inertia=inertia,
            label="sphere",
            lock_inertia=True,
        )
        builder.add_shape_sphere(
            self.sphere_body,
            radius=sphere_radius,
            label="sphere_collider",
        )

        # 2. Emit granular particles
        self.fluid_count = self._emit_column_particles(builder, self.particle_spacing)

        # 3. 5 penalty planes (4 walls + floor, no top)
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.4, length=0.4)
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, 0.10), width=0.4, length=0.4)
        builder.add_shape_plane(plane=(-1.0, 0.0, 0.0, 0.10), width=0.4, length=0.4)
        builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.10), width=0.4, length=0.4)
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0, 0.10), width=0.4, length=0.4)

        self.model = builder.finalize()
        # MBD's particle-particle contact (model.particle_ke) must not run on SPH particles;
        # SPH manages inter-particle forces via its own hash grid.
        self.model.particle_grid = None
        self.model.set_gravity((0.0, 0.0, -9.81))

        # ---- SPH solver configuration ----
        cfg = SolverSPH.Config()
        cfg.particle_spacing = self.particle_spacing
        cfg.kh = kh
        cfg.reference_density = 1510.0
        cfg.simulation_method = "dp"
        cfg.integration_scheme = "position_verlet"
        cfg.boundary_type = "penalty"
        cfg.sound_speed = sound_speed
        cfg.artificial_viscosity_alpha = 0.1
        cfg.penalty_stiffness = 1.0e6
        cfg.penalty_damping = 1.0e3
        cfg.boundary_friction = 0.7
        cfg.boundary_wall_friction = 0.1
        cfg.body_coupling_enabled = True
        cfg.body_coupling_stiffness = 2.0e4
        cfg.body_coupling_damping = 5.0e1
        cfg.body_coupling_friction = 0.5
        cfg.body_coupling_bearing_capacity = 1.0e4

        self.model.sph.young_modulus.fill_(1.0e6)
        self.model.sph.poisson_ratio.fill_(0.3)
        self.model.sph.friction.fill_(0.6)
        self.model.sph.cohesion.fill_(0.0)
        self.model.sph.viscosity.fill_(0.0)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sph_solver = SolverSPH(self.model, cfg)
        self.sph_solver.initialize_geostatic_stress(self.state_0, y_max=self._column_top)
        wp.copy(self.state_1.sph.stress, self.state_0.sph.stress)

        # ---- MBD solver: free body integration ----
        self.mbd_solver = SolverSemiImplicit(self.model)
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.particle_colors = wp.full(
            shape=self.model.particle_count,
            value=wp.vec3(0.5, 0.0, 0.5),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self._telemetry: list[dict[str, float]] = []
        self._plot_generated = False

    @staticmethod
    def _emit_column_particles(builder: newton.ModelBuilder, dx: float) -> int:
        density = 1510.0
        volume = dx**3
        mass = density * volume

        lo = np.array([-0.10, -0.10, 0.0], dtype=np.float32)
        hi = np.array([0.10, 0.10, 0.10], dtype=np.float32)
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
        )
        return len(pts)

    def simulate(self) -> None:
        if self.sim_time >= self.sim_duration:
            return

        body_count = self.model.body_count

        for _ in range(self.sim_substeps):
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

            # Single-swap: both MBD and SPH write to state_1; swap once at the end.
            # This preserves the SPH stress chain (sph.stress never resets to geostatic).
            self.mbd_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.sph_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        v = self.state_0.particle_qd.numpy()
        max_speed = float(np.max(np.linalg.norm(v, axis=1)))
        body_q_np = self.state_0.body_q.numpy()
        sphere_z = float(body_q_np[0][2])
        pen = max(0.0, self._column_top - (sphere_z - self.args.sphere_radius))
        print(
            f"t={self.sim_time:.4f}s  v_max={max_speed:.4f} m/s  pen={pen * 1000:.2f} mm",
            flush=True,
        )
        self._telemetry.append({"t": self.sim_time, "pen": pen})

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        if self.sim_time >= self.sim_duration:
            if not self._plot_generated:
                self._plot_generated = True
                self._plot_penetration()
            if hasattr(self.viewer, "should_close"):
                self.viewer.should_close = True

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        s = -self.state_0.sph.stress.numpy()[:, 2, 2]
        lo, hi = np.percentile(s, [5, 95])
        s_range = max(hi - lo, 1.0)
        t = np.clip((s - lo) / (s_range + 1e-9), 0.0, 1.0).astype(np.float32)
        colors_np = np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)
        self.particle_colors.assign(colors_np)

        self.viewer.log_points(
            name="/sph/column",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )

        self.viewer.end_frame()

    def _plot_penetration(self) -> None:
        import matplotlib.pyplot as plt

        ts = [r["t"] for r in self._telemetry]
        pens = [r["pen"] * 1000 for r in self._telemetry]
        fig, ax = plt.subplots()
        ax.plot(ts, pens)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Penetration depth [mm]")
        ax.set_title("Sphere penetration into granular column")
        if self.args.plot_path:
            fig.savefig(self.args.plot_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy()
        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        margin = 2.0 * self.particle_spacing

        assert np.all(q[:, 2] > -margin), "Particles leaked below floor"
        assert np.all(np.abs(q[:, 0]) < 0.10 + margin), "Particles leaked past x walls"
        assert np.all(np.abs(q[:, 1]) < 0.10 + margin), "Particles leaked past y walls"

        assert np.all(np.isfinite(body_q_np)), "Sphere body_q contains NaN/Inf"
        assert np.all(np.isfinite(body_qd_np)), "Sphere body_qd contains NaN/Inf"

        sphere_z = float(body_q_np[0][2])
        assert sphere_z > -margin, "Sphere fell through floor"

        max_pen = max((r["pen"] for r in self._telemetry), default=0.0)
        assert max_pen > 0.0, "Sphere never penetrated the granular surface"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--duration", type=float, default=0.5, help="Simulation duration [s]")
        parser.add_argument("--sphere-radius", type=float, default=0.0125)
        parser.add_argument("--drop-height", type=float, default=0.05)
        parser.add_argument("--plot-path", type=str, default=None)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
