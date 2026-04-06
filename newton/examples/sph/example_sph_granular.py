# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH granular cylindrical column collapse example.

Simulates a cylindrical column of granular particles collapsing under
gravity using the SPH solver with a Drucker-Prager elastic-plastic
constitutive model.  Particles are initialized on a regular grid
(voxelization) filtered by the cylinder geometry.

Usage::

    python -m newton.examples sph_granular
    python -m newton.examples sph_granular --simulation-method mui
    python -m newton.examples sph_granular --boundary-type penalty
"""

import numpy as np

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
        self.fluid_count = builder.particle_count

        if args.boundary_type == "dummy":
            # Add layered dummy particles around the computational domain
            SolverSPH.add_dummy_particles(
                builder,
                bounds_lo=tuple(args.domain_lo),
                bounds_hi=tuple(args.domain_hi),
                h=args.kh * args.particle_spacing,
                dx=args.particle_spacing,
                reference_density=args.density,
                slip_type=args.dummy_slip_type,
            )
        else:
            # Penalty boundary: use ground plane
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        config = SolverSPH.Config()
        config.particle_spacing = args.particle_spacing
        config.kh = args.kh
        config.reference_density = args.density
        config.simulation_method = args.simulation_method
        config.artificial_viscosity_alpha = args.artificial_viscosity_alpha
        config.sound_speed = args.sound_speed
        config.penalty_stiffness = args.penalty_stiffness
        config.boundary_type = args.boundary_type

        # Set per-particle material attributes
        for attr in ("young_modulus", "poisson_ratio", "friction", "cohesion", "dilatancy", "viscosity"):
            key = attr.replace("-", "_")
            if hasattr(args, key):
                getattr(self.model.sph, attr).fill_(getattr(args, key))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverSPH(self.model, config)

        # Initialize geostatic stress (Z-up: y_max is the column height)
        self.solver.initialize_geostatic_stress(self.state_0, y_max=args.cylinder_height)

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
        fluid_indices = list(range(self.fluid_count))
        newton.examples.test_particle_state(
            self.state_0,
            "all fluid particles are above the ground",
            lambda q, qd: q[2] > -h,
            indices=fluid_indices,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_stress = imgui.checkbox("Show Stress", self.show_stress)

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """Emit particles in a cylindrical column via voxelization.

        Creates a regular 3D grid over the cylinder bounding box and
        keeps only particles whose XY distance from the axis is within
        the cylinder radius.
        """
        density = args.density
        radius = args.cylinder_radius
        height = args.cylinder_height

        dx = args.particle_spacing
        vol = dx**3
        mass = density * vol

        # Regular grid covering the cylinder bounding box
        base = np.array(args.cylinder_base, dtype=np.float32)

        xs = np.arange(-radius + 0.5 * dx, radius, dx, dtype=np.float32)
        ys = np.arange(-radius + 0.5 * dx, radius, dx, dtype=np.float32)
        zs = np.arange(0.5 * dx, height, dx, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Keep only particles inside the cylinder (x² + y² ≤ r²)
        mask = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= radius**2
        pts = pts[mask] + base

        pos_list = [tuple(float(v) for v in p) for p in pts]
        vel_list = [(0.0, 0.0, 0.0)] * len(pts)
        mass_list = [mass] * len(pts)
        radius_list = [dx / 2.0] * len(pts)

        builder.add_particles(
            pos=pos_list,
            vel=vel_list,
            mass=mass_list,
            radius=radius_list,
            custom_attributes={"sph:friction": args.friction},
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene — cylindrical column geometry
        parser.add_argument("--cylinder-radius", type=float, default=0.1)
        parser.add_argument("--cylinder-height", type=float, default=0.2)
        parser.add_argument("--cylinder-base", type=float, nargs=3, default=[0.0, 0.0, 0.0])
        parser.add_argument("--domain-lo", type=float, nargs=3, default=[-1.0, -1.0, 0.0])
        parser.add_argument("--domain-hi", type=float, nargs=3, default=[1.0, 1.0, 0.3])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)

        # SPH
        parser.add_argument("--particle-spacing", "-dx", type=float, default=0.005)
        parser.add_argument("--kh", type=float, default=1.3)
        parser.add_argument("--simulation-method", type=str, default="dp", choices=["dp", "mui"])
        parser.add_argument("--sound-speed", type=float, default=50.0)
        parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1)
        parser.add_argument("--penalty-stiffness", type=float, default=1.0e6)
        parser.add_argument("--boundary-type", type=str, default="dummy", choices=["penalty", "dummy"])
        parser.add_argument("--dummy-slip-type", type=str, default="noslip", choices=["noslip", "freeslip"])

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
