# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH granular column collapse with grain rendering.

Similar to :mod:`example_sph_granular` but uses sub-particle point rendering
for a more visually appealing output.

Usage::

    python -m newton.examples sph_grain_rendering
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSPH


class Example:
    def __init__(self, viewer, args):
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Auto-compute CFL-stable substeps: dt_CFL = 0.3 * h / c_s.
        # smoothing_length is used as h; default sound_speed = 50 m/s.
        _sound_speed = 50.0
        _dt_cfl = 0.3 * args.smoothing_length / _sound_speed
        self.sim_substeps = max(1, int(np.ceil(self.frame_dt / _dt_cfl)))
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        builder = newton.ModelBuilder()

        SolverSPH.register_custom_attributes(builder)

        Example.emit_particles(builder, args)
        builder.add_ground_plane()
        self.model = builder.finalize()

        config = SolverSPH.Config()
        config.smoothing_length = args.smoothing_length
        config.reference_density = 2500.0

        self.solver = SolverSPH(self.model, config)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver.initialize_geostatic_stress(self.state_0, y_max=2.0)

        # Setup grain rendering: scatter points around each particle
        n = self.model.particle_count
        ppp = int(args.points_per_particle)
        grain_radius = args.smoothing_length / (3 * ppp)

        # Generate random offsets around each particle
        rng = np.random.default_rng(42)
        offsets = rng.normal(0, args.smoothing_length * 0.3, size=(n, ppp, 3)).astype(np.float32)
        self._grain_offsets = wp.array(offsets, dtype=wp.vec3, device=self.model.device)
        self._grain_positions = wp.zeros((n, ppp), dtype=wp.vec3, device=self.model.device)
        self._grain_radii = wp.full(n * ppp, value=grain_radius, dtype=float, device=self.model.device)
        self._grain_colors = wp.full(n * ppp, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.model.device)
        self._ppp = ppp

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        # Update grain positions from particle positions + offsets
        n = self.model.particle_count
        wp.launch(
            _update_grains_kernel,
            dim=(n, self._ppp),
            inputs=[
                self.state_0.particle_q,
                self._grain_offsets,
            ],
            outputs=[self._grain_positions],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "grains",
            points=self._grain_positions.flatten(),
            radii=self._grain_radii,
            colors=self._grain_colors,
            hidden=False,
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        smoothing_length = args.smoothing_length
        particles_per_cell = 3
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, 0.5, 2.0])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / smoothing_length),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * 2500.0

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
        parser.add_argument("--smoothing-length", "-dx", type=float, default=0.1)
        parser.add_argument("--points-per-particle", "-ppp", type=float, default=8)
        return parser


@wp.kernel
def _update_grains_kernel(
    particle_pos: wp.array[wp.vec3],
    offsets: wp.array2d[wp.vec3],
    # output
    grain_pos: wp.array2d[wp.vec3],
):
    """Update grain positions: grain_pos[i,j] = particle_pos[i] + offsets[i,j]."""
    i, j = wp.tid()
    grain_pos[i, j] = particle_pos[i] + offsets[i, j]


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
