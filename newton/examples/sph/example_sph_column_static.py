# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
SPH static granular column stability test.

Coordinate system: Z-up (gravity along -Z).

Box domain:    x=0.2m (+-0.10m), y=0.2m (+-0.10m), z=0.15m (floor at z=0, open top)
Granular fill: x=[-0.10,0.10], y=[-0.10,0.10], z=[0,0.10]  (fills box laterally, H=0.1m)

The column is fully confined on 4 lateral faces; H/W = 0.1/0.2 = 0.5 (well below the
angle-of-repose limit), so cohesion is not required for static stability.

Penalty boundaries: 5 faces (4 walls + floor)
  - Walls (x+/-, y+/-): boundary_wall_friction=0.1 (near-frictionless)
  - Floor (z=0):        boundary_friction=0.7 (high friction)
Note: per-face friction split is handled by the solver's normal-direction
classifier (normal.z > 0.5 => floor, else wall).

Purpose: verify the confined column does NOT move -- a static-stability smoke test.
Particle coloring: vertical stress sigma_zz (compression positive = warm colour).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSPH


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
        # Acoustic CFL: dt_cfl = 0.3 * h / c_s
        dt_cfl = 0.3 * h / sound_speed
        self.sim_substeps = max(1, int(math.ceil(self.frame_dt / dt_cfl)))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # ---- Build model ----
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # 1. Emit granular particles
        self.fluid_count = self._emit_column_particles(builder, self.particle_spacing)

        # 2. Add 5 penalty planes (4 walls + floor, no top).
        #    Plane equation: a*x + b*y + c*z + d = 0, (a,b,c) = inward normal.
        #    Per-face friction (wall=0.1, floor=0.7) is controlled by
        #    Config.boundary_wall_friction / Config.boundary_friction via the
        #    solver's normal-direction classifier (normal.z > 0.5 => floor).
        #    ShapeConfig.mu is NOT read by the SPH penalty kernel.
        # Floor z = 0, inward normal +z
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.4, length=0.4)
        # x = -0.10, inward normal +x  ->  x + 0.10 = 0
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, 0.10), width=0.4, length=0.4)
        # x = +0.10, inward normal -x  ->  -x + 0.10 = 0
        builder.add_shape_plane(plane=(-1.0, 0.0, 0.0, 0.10), width=0.4, length=0.4)
        # y = -0.10, inward normal +y  ->  y + 0.10 = 0
        builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.10), width=0.4, length=0.4)
        # y = +0.10, inward normal -y  ->  -y + 0.10 = 0
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0, 0.10), width=0.4, length=0.4)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        # ---- SPH solver configuration ----
        cfg = SolverSPH.Config()
        cfg.particle_spacing = self.particle_spacing
        cfg.kh = kh
        cfg.simulation_method = "dp"
        cfg.integration_scheme = "position_verlet"
        cfg.boundary_type = "penalty"
        cfg.sound_speed = sound_speed
        cfg.artificial_viscosity_alpha = 0.1
        cfg.penalty_stiffness = 1.0e6
        cfg.penalty_damping = 1.0e3
        cfg.boundary_friction = 0.7  # floor
        cfg.boundary_wall_friction = 0.1  # lateral walls

        # Per-particle material parameters
        self.model.sph.young_modulus.fill_(1.0e6)
        self.model.sph.poisson_ratio.fill_(0.3)
        self.model.sph.friction.fill_(0.6)  # internal DP friction angle ~34 deg
        self.model.sph.cohesion.fill_(0.0)
        self.model.sph.viscosity.fill_(0.0)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverSPH(self.model, cfg)

        # Geostatic stress init: column top at z = 0.10 m.
        self.solver.initialize_geostatic_stress(self.state_0, y_max=0.10)
        # Mirror into state_1 so first ping-pong swap doesn't read uninitialised stress.
        wp.copy(self.state_1.sph.stress, self.state_0.sph.stress)

        self._substep_count = 0
        self._substep_phys_time = 0.0
        # Cache immutable per-particle masses for KE computation (avoid per-substep GPU sync).
        self._masses_np = self.model.particle_mass.numpy()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Particle colour buffer (vertical-stress colormap, written each render).
        self.particle_colors = wp.full(
            shape=self.model.particle_count,
            value=wp.vec3(0.5, 0.0, 0.5),
            dtype=wp.vec3,
            device=self.model.device,
        )

    # ------------------------------------------------------------------
    # Particle emission
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_column_particles(builder: newton.ModelBuilder, dx: float) -> int:
        """Emit a rectangular granular column on a staggered grid.

        Fills x in [-0.10, 0.10], y in [-0.10, 0.10], z in [0, 0.10] with
        a 0.5*dx inset so particles do not start in contact with walls.
        """
        density = 1500.0
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

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        if self.sim_time >= self.sim_duration:
            return
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._substep_count += 1
            self._substep_phys_time += self.sim_dt
        v = self.state_0.particle_qd.numpy()
        ke = float(0.5 * np.einsum("i,ij->", self._masses_np, v * v))
        print(
            f"substep={self._substep_count:6d}  t={self._substep_phys_time:.6f}s  KE={ke:.6e} J",
            flush=True,
        )

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        if self.sim_time >= self.sim_duration and hasattr(self.viewer, "should_close"):
            self.viewer.should_close = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Colour by vertical stress sigma_zz: compression (negative) -> warm colour.
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

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy()
        v = self.state_0.particle_qd.numpy()

        # 1. Domain containment: no particles leaked out of the box.
        margin = 2.0 * self.particle_spacing
        assert np.all(q[:, 2] > -margin), "Particles leaked below floor"
        assert np.all(np.abs(q[:, 0]) < 0.10 + margin), "Particles leaked past x walls"
        assert np.all(np.abs(q[:, 1]) < 0.10 + margin), "Particles leaked past y walls"

        # 2. Velocity settled: confined column should be nearly at rest.
        speeds = np.linalg.norm(v, axis=1)
        max_speed = float(np.max(speeds))
        assert max_speed < 0.2, f"Max particle speed {max_speed:.3f} m/s; column not at rest"

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--duration", type=float, default=0.2, help="Simulation duration [s]")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
