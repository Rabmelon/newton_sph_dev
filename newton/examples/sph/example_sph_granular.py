# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH granular column collapse example.

Simulates a column of granular particles collapsing under gravity using
the SPH solver with a Drucker-Prager elastic-plastic constitutive model.

Usage::

    python -m newton.examples sph_granular
    python -m newton.examples sph_granular --simulation-method mui
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSPH


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        builder = newton.ModelBuilder()

        SolverSPH.register_custom_attributes(builder)

        Example.emit_particles(builder, args)

        if args.collider == "cube":
            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=wp.transform(wp.vec3(0.75, 0.0, 0.8), wp.quat_identity()),
                hx=0.5,
                hy=2.0,
                hz=0.8,
            )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        config = SolverSPH.Config()
        config.smoothing_length = args.smoothing_length
        config.reference_density = args.density
        config.simulation_method = args.simulation_method
        config.artificial_viscosity_alpha = args.artificial_viscosity_alpha
        config.sound_speed = args.sound_speed
        config.penalty_stiffness = args.penalty_stiffness

        # Set per-particle material attributes
        for attr in ("young_modulus", "poisson_ratio", "friction", "cohesion", "dilatancy", "viscosity"):
            key = attr.replace("-", "_")
            if hasattr(args, key):
                getattr(self.model.sph, attr).fill_(getattr(args, key))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverSPH(self.model, config)

        # Initialize geostatic stress
        emit_hi = args.emit_hi
        self.solver.initialize_geostatic_stress(self.state_0, y_max=emit_hi[2])

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.viewer.show_particles = True
        self.show_stress = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        h = self.solver.smoothing_length
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -h,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_stress = imgui.checkbox("Show Stress", self.show_stress)

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        density = args.density
        smoothing_length = args.smoothing_length

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / smoothing_length),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene
        parser.add_argument("--collider", default="none", choices=["cube", "none"], type=str)
        parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
        parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 2.0])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)

        # SPH
        parser.add_argument("--smoothing-length", "-dx", type=float, default=0.1)
        parser.add_argument("--simulation-method", type=str, default="dp", choices=["dp", "mui"])
        parser.add_argument("--sound-speed", type=float, default=50.0)
        parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1)
        parser.add_argument("--penalty-stiffness", type=float, default=1.0e6)

        # Material
        parser.add_argument("--density", type=float, default=2500.0)
        parser.add_argument("--young-modulus", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", type=float, default=0.3)
        parser.add_argument("--friction", type=float, default=0.5)
        parser.add_argument("--cohesion", type=float, default=0.0)
        parser.add_argument("--dilatancy", type=float, default=0.0)
        parser.add_argument("--viscosity", type=float, default=0.0)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
