# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM granular cylindrical column collapse example.

This example mirrors the SPH granular column collapse setup
(`example_sph_granular.py`), but uses the implicit MPM solver.

A cylindrical column of granular material is initialized on a regular
Cartesian grid and filtered by the cylinder geometry. The resulting
particles are simulated with a Drucker–Prager type elastoplastic
constitutive model using the implicit MPM solver.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        # simulation timing
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.step_count = 0

        # viewer / model builder
        self.viewer = viewer
        builder = newton.ModelBuilder()

        # register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # emit particles in a cylindrical column similar to the SPH example
        Example.emit_particles(builder, args)

        # simple collision geometry: ground plane only by default
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # finalize model and set gravity
        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        # copy CLI args into MPM config options and per-particle attributes
        mpm_options = SolverImplicitMPM.Config()
        for key, value in vars(args).items():
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, value)
            if hasattr(self.model.mpm, key):
                getattr(self.model.mpm, key).fill_(value)

        # allocate states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # initialize solver
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        # connect to viewer
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.viewer.show_particles = True
        self.show_normals = False
        self.show_stress = False

        # attempt CUDA graph capture for performance (optional)
        self.capture()

    # --- simulation control -------------------------------------------------

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.step_count += self.sim_substeps

        # Print simulation progress to the command line
        print(f"sim step: {self.step_count}, time: {self.sim_time:.6f}s", flush=True)

        # Stop the simulation after 0.1 seconds of simulated time
        if self.sim_time >= 0.1:
            raise SystemExit("Reached 0.1s of simulated time, stopping simulation.")

        # Expose simulation info to the viewer so it can be shown in the
        # top-right performance panel (increments and time).
        if hasattr(self.viewer, "set_simulation_info"):
            self.viewer.set_simulation_info(time=self.sim_time, step=self.step_count)

    # --- testing ------------------------------------------------------------

    def test_final(self):
        voxel_size = self.solver.voxel_size

        # all particles should remain above the ground plane (z = 0)
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

        # ensure the column has not completely collapsed to a flat pile
        max_z = np.max(self.state_0.particle_q.numpy()[:, 2])
        # require some particles to remain sufficiently high
        min_expected_height = 0.4 * self.args.cylinder_height if hasattr(self, "args") else 0.04
        assert max_z > min_expected_height, "All particles have collapsed too much"

    # --- rendering ----------------------------------------------------------

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.show_normals:
            # visualize collider normals for debugging
            _impulses, pos, _cid = self.solver.collect_collider_impulses(self.state_0)
            normals = self.state_0.collider_normal_field.dof_values

            normal_vecs = 0.25 * self.solver.voxel_size * normals
            root = pos
            mid = pos + normal_vecs
            tip = mid + normal_vecs

            self.viewer.log_lines(
                "/normal_roots",
                starts=root,
                ends=mid,
                colors=wp.full(pos.shape[0], value=wp.vec3(0.8, 0.0, 0.0), dtype=wp.vec3),
            )
            self.viewer.log_lines(
                "/normal_tips",
                starts=mid,
                ends=tip,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.5, 0.3), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/normal_roots", None, None, None)
            self.viewer.log_lines("/normal_tips", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_normals = imgui.checkbox("Show Normals", self.show_normals)

    # --- particle emission --------------------------------------------------

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """Emit particles in a cylindrical column via voxelization.

        Matches the geometry used in the SPH granular example as closely as
        possible (same radius/height/base and particle spacing), but feeds
        the particles into the MPM solver.
        """

        density = args.density
        radius = args.cylinder_radius
        height = args.cylinder_height
        base = np.array(args.cylinder_base, dtype=np.float32)

        dx = args.particle_spacing
        vol = dx**3
        mass = density * vol

        # Regular grid covering the cylinder bounding box in local space
        xs = np.arange(-radius + 0.5 * dx, radius, dx, dtype=np.float32)
        ys = np.arange(-radius + 0.5 * dx, radius, dx, dtype=np.float32)
        zs = np.arange(0.5 * dx, height, dx, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Keep only particles inside the cylinder (x^2 + y^2 <= r^2)
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
        )

    # --- CLI parser ---------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration (mirrors SPH granular where possible)
        parser.add_argument("--cylinder-radius", type=float, default=0.1)
        parser.add_argument("--cylinder-height", type=float, default=0.2)
        parser.add_argument("--cylinder-base", type=float, nargs=3, default=[0.0, 0.0, 0.0])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=1)

        # Particle resolution
        parser.add_argument("--particle-spacing", "-dx", type=float, default=0.005)

        # Material parameters (subset of MPM options)
        parser.add_argument("--density", type=float, default=2500.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
        parser.add_argument("--friction", "-mu", type=float, default=0.5)
        parser.add_argument("--damping", type=float, default=0.0)
        parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e4)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
        parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
        parser.add_argument("--hardening", type=float, default=0.0)
        parser.add_argument("--dilatancy", type=float, default=0.0)
        parser.add_argument("--viscosity", type=float, default=0.0)

        # MPM grid and solver configuration (same as example_mpm_granular)
        parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
        parser.add_argument("--grid-padding", "-gp", type=int, default=0)
        parser.add_argument("--max-active-cell-count", "-mac", type=int, default=-1)
        parser.add_argument(
            "--solver",
            "-s",
            type=str,
            default="gauss-seidel",
            choices=["gauss-seidel", "jacobi", "cg", "cg+jacobi", "cg+gauss-seidel"],
        )
        parser.add_argument("--transfer-scheme", "-ts", type=str, default="apic", choices=["apic", "pic"])
        parser.add_argument("--integration-scheme", "-is", type=str, default="pic", choices=["pic", "gimp"])

        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="Q1")
        parser.add_argument("--velocity-basis", "-vb", type=str, default="Q1")

        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument("--voxel-size", type=float, default=0.02)

        return parser


if __name__ == "__main__":
    import time

    # Measure total wall-clock time for initialization + simulation run
    _t_start = time.perf_counter()

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Save args for use in test_final height criterion
    example.args = args

    try:
        newton.examples.run(example, args)
    finally:
        _t_end = time.perf_counter()
        print(f"Total wall time (compile + run): {_t_end - _t_start:.6f}s", flush=True)
