# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH granular cylindrical column collapse example.

Simulates a cylindrical column of granular particles collapsing under
gravity using the SPH solver with a Drucker-Prager elastic-plastic
constitutive model.  Particles are initialized on a regular grid
(voxelization) filtered by the cylinder geometry.

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
        self.sim_step = 0
        # Simulation duration (seconds)
        self.sim_duration = args.duration
        # How to behave when the simulation ends
        self.end_behavior = args.end_behavior
        # Compute CFL-stable substep count automatically when not overridden.
        # dt_CFL = 0.3 * h / c_s;  h = kh * dx.
        if args.substeps is None:
            h = args.kh * args.particle_spacing
            dt_cfl = 0.3 * h / args.sound_speed
            self.sim_substeps = max(1, int(np.ceil(self.frame_dt / dt_cfl)))
        else:
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
        # --smoothing-length overrides particle_spacing when provided (test convenience).
        if args.smoothing_length is not None:
            args.particle_spacing = args.smoothing_length
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
        # Skip stepping if we've already reached the requested duration
        if self.sim_time >= self.sim_duration:
            return
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.sim_step += self.sim_substeps

        # Per-step particle statistics (mirrors example_mpm_granular2.py for benchmarking)
        # - run-out distance: max sqrt(x^2 + y^2)
        # - max height: max z
        # - total kinetic energy: sum(0.5 * m * |v|^2)
        # Restricted to fluid particles so dummy boundary particles (if any) don't pollute stats.
        q_np = self.state_0.particle_q.numpy()[: self.fluid_count]
        qd_np = self.state_0.particle_qd.numpy()[: self.fluid_count]
        runout = float(np.max(np.sqrt(q_np[:, 0] ** 2 + q_np[:, 1] ** 2))) if q_np.size else 0.0
        max_h = float(np.max(q_np[:, 2])) if q_np.size else 0.0

        m_arr = getattr(self.model, "particle_mass", None)
        if m_arr is not None:
            m_np = m_arr.numpy().reshape(-1)[: self.fluid_count]
            if m_np.size == qd_np.shape[0]:
                ke = float(0.5 * np.sum(m_np * np.sum(qd_np * qd_np, axis=1)))
            else:
                ke = float(0.5 * np.sum(np.sum(qd_np * qd_np, axis=1)))
        else:
            ke = float(0.5 * np.sum(np.sum(qd_np * qd_np, axis=1)))

        # Print simulation progress to the command line
        print(
            f"sim step: {self.sim_step:6d}, time: {self.sim_time:.6f}s, "
            f"runout: {runout:.6f}m, max_h: {max_h:.6f}m, KE: {ke:.6f}J",
            flush=True,
        )

        # Check if we've reached or exceeded the target duration
        if self.sim_time >= self.sim_duration:
            if self.end_behavior == "exit":
                # Signal the examples runner to quit if it supports this pattern
                if hasattr(self.viewer, "should_close"):
                    self.viewer.should_close = True

        # Additionally stop the simulation once 0.1 seconds of simulated time is reached
        if self.sim_time >= self.sim_duration:
            raise SystemExit(f"Reached {self.sim_duration}s of simulated time, stopping simulation.")

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

        # Scene
        parser.add_argument("--cylinder-radius", type=float, default=0.1)
        parser.add_argument("--cylinder-height", type=float, default=0.2)
        parser.add_argument("--cylinder-base", type=float, nargs=3, default=[0.0, 0.0, 0.0])
        parser.add_argument("--domain-lo", type=float, nargs=3, default=[-1.0, -1.0, 0.0])
        parser.add_argument("--domain-hi", type=float, nargs=3, default=[1.0, 1.0, 0.3])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        # Substeps: auto-computed from CFL condition when not set.
        # dt_CFL = 0.3 * h / c_s (e.g. ≈ 0.039 ms for default dx=0.005, c_s=50).
        parser.add_argument("--substeps", type=int, default=None)
        # Simulation duration and end-of-run behavior
        parser.add_argument("--duration", type=float, default=0.5, help="Simulation duration in seconds")
        parser.add_argument(
            "--end-behavior",
            type=str,
            default="exit",
            choices=["pause", "exit"],
            help=(
                "What to do when the simulation reaches the specified duration: "
                "'pause' keeps the GUI open, 'exit' closes the simulation window."
            ),
        )

        # SPH
        parser.add_argument("--particle-spacing", "-dx", type=float, default=0.005)
        # --smoothing-length is a convenience alias for particle-spacing (same unit, [m]).
        parser.add_argument("--smoothing-length", type=float, default=None)
        parser.add_argument("--kh", type=float, default=1.3)
        parser.add_argument("--simulation-method", type=str, default="dp", choices=["dp", "mui"])
        parser.add_argument("--sound-speed", type=float, default=50.0)
        parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1)
        parser.add_argument("--boundary-type", type=str, default="penalty", choices=["penalty", "dummy"])
        parser.add_argument("--penalty-stiffness", type=float, default=1.0e6)
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
    import time

    # Measure total wall-clock time for initialization + simulation run
    _t_start = time.perf_counter()

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)

    try:
        newton.examples.run(example, args)
    finally:
        _t_end = time.perf_counter()
        print(f"Total wall time (compile + run): {_t_end - _t_start:.6f}s", flush=True)
