# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH sphere free-drop into a confined granular column.

Coordinate system: Z-up (gravity along -Z).

Box domain:    x=0.15m (+-0.075m), y=0.15m (+-0.075m), z=0.10m (floor at z=0, open top)
Granular fill: x=[-0.075,0.075], y=[-0.075,0.075], z=[0,0.05]

A rigid sphere is released above the column surface; per-frame telemetry
records the sphere's penetration depth ``max(0, column_top - sphere_bottom)``
and writes a time-curve plot at the end of the run.

The granular density is 1510 kg/m^3 (vs the static column's 1500) to balance
the body-coupling reaction wrench against the sphere's added load.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSemiImplicit, SolverSPH
from newton.viewer import ViewerNull

RHO_GRANULAR = 1510.0
"""Granular material density [kg/m^3] used for particle mass and geostatic init."""


def _predict_penetration_depth(mu_s: float, rho_s: float, rho_g: float, H: float, R: float, n_iter: int = 50) -> float:
    """Predicted sphere penetration depth into a granular bed [m].

    Iterative formula (50 steps from delta_0 = 0):

        delta_{n+1} = (0.14 / mu_s) * sqrt(rho_s / rho_g) * (2 * R)^(2/3) * (H + delta_n)^(1/3)

    Args:
        mu_s: Sphere-granular friction coefficient.
        rho_s: Sphere density [kg/m^3].
        rho_g: Granular density [kg/m^3].
        H: Drop height from granular surface to bottom of sphere [m].
        R: Sphere radius [m].
        n_iter: Number of fixed-point iterations (default 50).

    Returns:
        Predicted penetration depth delta [m].
    """
    delta = 0.0
    coef = (0.14 / mu_s) * math.sqrt(rho_s / rho_g) * (2.0 * R) ** (2.0 / 3.0)
    for _ in range(n_iter):
        delta = coef * (H + delta) ** (1.0 / 3.0)
    return delta


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

        self.particle_spacing = 0.002
        kh = 1.3
        sound_speed = 50.0
        h = kh * self.particle_spacing
        dt_cfl = 0.3 * h / sound_speed
        self.sim_substeps = max(1, int(math.ceil(self.frame_dt / dt_cfl)))
        self.sim_dt = self.frame_dt / self.sim_substeps

        self._column_top = 0.05
        sphere_radius = args.sphere_radius
        # Place sphere just above bed surface; free-fall kinetic energy encoded as initial velocity.
        _surface_gap = 5e-4
        drop_z = self._column_top + sphere_radius + _surface_gap
        self._sphere_start_z = drop_z

        # ---- Build model ----
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # 1. Rigid sphere added BEFORE particles to preserve fluid contiguity.
        sphere_mass = args.sphere_density * (4.0 / 3.0) * math.pi * sphere_radius**3
        i_scalar = 0.4 * sphere_mass * sphere_radius * sphere_radius
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
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.3, length=0.3)
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, 0.075), width=0.3, length=0.3)
        builder.add_shape_plane(plane=(-1.0, 0.0, 0.0, 0.075), width=0.3, length=0.3)
        builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.075), width=0.3, length=0.3)
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0, 0.075), width=0.3, length=0.3)

        self.model = builder.finalize()
        # MBD's particle-particle contact (model.particle_ke) must not run on SPH particles;
        # SPH manages inter-particle forces via its own hash grid.
        self.model.particle_grid = None
        self.model.set_gravity((0.0, 0.0, -9.81))

        # ---- SPH solver configuration ----
        cfg = SolverSPH.Config()
        cfg.particle_spacing = self.particle_spacing
        cfg.kh = kh
        cfg.reference_density = RHO_GRANULAR
        cfg.simulation_method = "dp"
        cfg.integration_scheme = "position_verlet"
        cfg.boundary_type = "penalty"
        cfg.sound_speed = sound_speed
        cfg.artificial_viscosity_alpha = args.artificial_viscosity_alpha
        cfg.penalty_stiffness = 1.0e6
        cfg.penalty_damping = 1.0e3
        cfg.boundary_friction = 0.7
        cfg.boundary_wall_friction = 0.1
        cfg.body_coupling_enabled = True
        cfg.body_coupling_damping = 1.0
        cfg.body_coupling_friction = args.sphere_friction

        self.model.sph.young_modulus.fill_(1.0e6)
        self.model.sph.poisson_ratio.fill_(0.3)
        self.model.sph.friction.fill_(args.material_friction)
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

        # Encode free-fall from drop_height as initial downward velocity (v = sqrt(2gH)).
        init_vz = -math.sqrt(2.0 * 9.81 * args.drop_height)
        body_qd_np = self.state_0.body_qd.numpy().copy()
        body_qd_np[0, 2] = init_vz  # spatial_top = linear vel; index 2 = v_z
        self.state_0.body_qd.assign(body_qd_np)
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
        density = RHO_GRANULAR
        volume = dx**3
        mass = density * volume

        lo = np.array([-0.075, -0.075, 0.0], dtype=np.float32)
        hi = np.array([0.075, 0.075, 0.05], dtype=np.float32)
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
        body_qd_np = self.state_0.body_qd.numpy()
        sphere_z = float(body_q_np[0][2])
        sphere_vz = float(body_qd_np[0][2])
        pen = max(0.0, self._column_top - (sphere_z - self.args.sphere_radius))
        print(
            f"t={self.sim_time:.4f}s  v_max={max_speed:.4f} m/s  pen={pen * 1000:.2f} mm  v_sphere_z={sphere_vz:.3f} m/s",
            flush=True,
        )
        self._telemetry.append({"t": self.sim_time, "pen": pen, "sphere_z": sphere_z, "sphere_vz": sphere_vz})

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

        # 1. No NaN/Inf in body state.
        assert np.all(np.isfinite(body_q_np)), "Sphere body_q contains NaN/Inf"
        assert np.all(np.isfinite(body_qd_np)), "Sphere body_qd contains NaN/Inf"

        # 2. Sphere did not punch through floor.
        sphere_z = float(body_q_np[0][2])
        assert sphere_z > -margin, "Sphere fell through floor"

        # 3. Penetration depth within 20% of analytical prediction.
        args = self.args
        delta_pred = _predict_penetration_depth(
            args.sphere_friction,
            args.sphere_density,
            RHO_GRANULAR,
            args.drop_height,
            args.sphere_radius,
        )
        sphere_bottom_z = sphere_z - args.sphere_radius
        delta_meas = max(0.0, self._column_top - sphere_bottom_z)
        rel_err = abs(delta_meas - delta_pred) / max(delta_pred, 1e-6)
        assert rel_err < 0.20, (
            f"Penetration depth mismatch: measured={delta_meas * 1e3:.2f} mm, "
            f"predicted={delta_pred * 1e3:.2f} mm, rel_err={rel_err:.2%}"
        )

        # 4. Sphere settled: window-min |v_z| over last 10 frames < 0.05 m/s.
        recent_vz = [abs(r["sphere_vz"]) for r in self._telemetry[-10:]]
        assert recent_vz and min(recent_vz) < 0.05, (
            f"Sphere not settled: min|v_z| over last 10 frames = {min(recent_vz) if recent_vz else float('nan'):.3f} m/s"
        )

        # 5. No particles leaked below floor.
        assert np.all(q[:, 2] > -margin), "Particles leaked below floor"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--duration", type=float, default=0.8, help="Simulation duration [s]")
        parser.add_argument("--sphere-radius", type=float, default=0.0125, help="Sphere radius [m]")
        parser.add_argument(
            "--sphere-density",
            type=float,
            default=2200.0,
            help="Sphere density [kg/m^3]",
        )
        parser.add_argument(
            "--sphere-friction",
            type=float,
            default=0.3,
            help="Sphere-granular Coulomb friction coefficient",
        )
        parser.add_argument(
            "--drop-height",
            type=float,
            default=0.1,
            help="Drop height from granular surface to bottom of sphere [m]",
        )
        parser.add_argument("--plot-path", type=str, default=None)
        parser.add_argument("--sweep", action="store_true", help="Run parameter sweep over drop heights, densities, and frictions")
        parser.add_argument("--material-friction", type=float, default=0.6, help="Granular material friction coefficient (tan phi) [default: 0.6]")
        parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1,
                            help="Monaghan artificial viscosity alpha coefficient [default: 0.1]")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if getattr(args, "sweep", False):
        import argparse as _ap

        BALL_DROP_HEIGHTS = [0.05, 0.1, 0.2]
        BALL_DENSITIES = [700, 2200]
        MU_S_VALUES = [0.3, 0.5]
        SPHERE_RADIUS = 0.0125

        wp.config.quiet = True
        combos = [
            (H, rho, mu)
            for H in BALL_DROP_HEIGHTS
            for rho in BALL_DENSITIES
            for mu in MU_S_VALUES
        ]
        rows = []
        for i, (H, rho_s, mu_s) in enumerate(combos, 1):
            print(f"\n[{i}/{len(combos)}] H={H}m  rho_s={rho_s}  mu_s={mu_s}", flush=True)
            sweep_args = _ap.Namespace(
                fps=60.0,
                duration=args.duration,
                sphere_radius=SPHERE_RADIUS,
                sphere_density=rho_s,
                sphere_friction=mu_s,
                material_friction=math.atan(mu_s),
                drop_height=H,
                plot_path=f"/tmp/sph_sweep_{i}.png",
                device=args.device,
                test=False,
                artificial_viscosity_alpha=getattr(args, "artificial_viscosity_alpha", 0.1),
            )
            sweep_viewer = ViewerNull(num_frames=10000)
            ex = Example(sweep_viewer, sweep_args)
            while ex.sim_time < ex.sim_duration:
                ex.step()

            delta_sim = ex._telemetry[-1]["pen"]
            delta_pred = _predict_penetration_depth(mu_s, rho_s, RHO_GRANULAR, H, SPHERE_RADIUS)
            err_pct = 100.0 * abs(delta_sim - delta_pred) / max(delta_pred, 1e-6)
            rows.append((H, rho_s, mu_s, delta_sim * 1e3, delta_pred * 1e3, err_pct))
            print(f"  sim={delta_sim * 1e3:.2f}mm  pred={delta_pred * 1e3:.2f}mm  err={err_pct:.1f}%")
            del ex

        hdr = f"{'H[m]':>6} {'rho_s':>6} {'mu_s':>5} {'sim[mm]':>9} {'pred[mm]':>9} {'err%':>7}"
        sep = "-" * len(hdr)
        print("\n" + "=" * len(hdr))
        print(hdr)
        print(sep)
        for H, rho_s, mu_s, sim_mm, pred_mm, err_pct in rows:
            print(f"{H:>6.2f} {rho_s:>6} {mu_s:>5.1f} {sim_mm:>9.2f} {pred_mm:>9.2f} {err_pct:>7.1f}%")
        print("=" * len(hdr))
    else:
        example = Example(viewer, args)
        newton.examples.run(example, args)
