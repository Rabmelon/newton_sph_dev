# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH sphere free-drop into a confined granular column — dummy-particle variant.

Same physical setup as ``example_sph_column_sphere_drop.py``, but the sphere is
discretized as dummy particles and domain walls use dummy boundary particles
instead of penalty planes. No MBD solver — sphere free-fall and reaction force
are integrated in Python.

Coordinate system: Z-up (gravity along -Z).

Comparison target: the analytical penetration-depth formula from the original
script (``_predict_penetration_depth``). If dummy boundaries equal or beat the
penalty body-coupling accuracy on this metric, the dummy approach is a viable
alternative.
"""

from __future__ import annotations

import argparse as _ap
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry import ParticleFlags
from newton._src.solvers.sph.sph_dummy_boundary import (
    SPH_FLUID,
    SPH_DUMMY_EMBEDDED,
    _SPH_DUMMY_EMBEDDED_VAL,
    _SPH_DUMMY_NOSLIP_VAL,
    compute_virtual_velocity,
)
from newton._src.solvers.sph.sph_kernels import wendland_c2_3d, wendland_c2_grad_3d
from newton.solvers import SolverSPH
from newton.viewer import ViewerNull

RHO_GRANULAR = 1510.0
"""Granular material density [kg/m^3] used for particle mass and geostatic init."""

_EPSILON = wp.constant(1.0e-8)
_RHO_GRANULAR_C = wp.constant(RHO_GRANULAR)


def _predict_penetration_depth(
    mu_s: float, rho_s: float, rho_g: float, H: float, R: float, n_iter: int = 50
) -> float:
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


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _kernel_interpolate_embedded_stress(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    stress: wp.array[wp.mat33],
    density: wp.array[float],
    mass: wp.array[float],
    particle_type: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    embedded_start: int,
    embedded_end: int,
    h: float,
    support_radius: float,
    out_stress: wp.array[wp.mat33],
):
    """Hu et al. (2021) CMAME: Shepard-interpolate fluid stress onto embedded dummies.

    For each embedded dummy *d*, computes the kernel-weighted average of
    neighbouring fluid stresses:

        σ_d = Σ_j V_j σ_j W(x_d − x_j) / Σ_j V_j W(x_d − x_j)

    where V_j = m_j / ρ_j is the particle volume.  No hydrostatic correction
    is added — the embedded boundary inherits the local stress field of the
    granular material.
    """
    local_idx = wp.tid()
    if local_idx >= (embedded_end - embedded_start):
        return
    d = embedded_start + local_idx

    xd = pos[d]
    sum_w = float(0.0)
    sum_stress = wp.mat33(0.0)

    query = wp.hash_grid_query(grid, xd, support_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:  # ParticleFlags.ACTIVE
            continue
        if particle_type[j] != SPH_FLUID:
            continue

        xj = pos[j]
        r_vec = xd - xj
        r = wp.length(r_vec)
        if r > support_radius or r < _EPSILON:
            continue

        w = wendland_c2_3d(r, h)
        rho_j = density[j]
        if rho_j > _EPSILON:
            v_j = mass[j] / rho_j
            sum_w += v_j * w
            sum_stress += v_j * stress[j] * w

    if sum_w > _EPSILON:
        out_stress[local_idx] = sum_stress / sum_w
    else:
        out_stress[local_idx] = wp.mat33(0.0)


@wp.kernel
def _kernel_compute_embedded_reaction(
    grid: wp.uint64,
    pos: wp.array[wp.vec3],
    rho: wp.array[float],
    stress: wp.array[wp.mat33],
    mass: wp.array[float],
    particle_type: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    embedded_start: int,
    embedded_end: int,
    embedded_stress: wp.array[wp.mat33],
    h: float,
    support_radius: float,
    sphere_force: wp.array[wp.vec3],
):
    """Net force on embedded dummy particles from fluid neighbours.

    Uses the **same** kernel-interpolated stress that the SPH stress-force
    kernel reads, so Newton's third law is satisfied exactly.  The force
    is the equal-and-opposite sum of the per-pair SPH stress forces.
    """
    i = wp.tid()
    if particle_type[i] != SPH_FLUID:
        return

    rho_i = rho[i]
    if rho_i <= 0.0:
        return

    xi = pos[i]
    sigma_i = stress[i]
    m_i = mass[i]

    query = wp.hash_grid_query(grid, xi, support_radius)
    j = int(0)
    f_local = wp.vec3(0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue
        if j < embedded_start or j >= embedded_end:
            continue

        xj = pos[j]
        r_vec = xi - xj
        r = wp.length(r_vec)
        if r > support_radius or r < _EPSILON:
            continue

        # Hu 2021: use pre-interpolated stress (same value as SPH solver reads)
        sigma_j = embedded_stress[j - embedded_start]
        rho_j = _RHO_GRANULAR_C
        m_j = mass[j]

        grad_w = wendland_c2_grad_3d(r_vec, r, h)
        combined = sigma_i / (rho_i * rho_i) + sigma_j / (rho_j * rho_j)
        f_ij = m_i * m_j * (combined @ grad_w)
        f_local += f_ij

    wp.atomic_add(sphere_force, 0, -f_local)


@wp.kernel
def _kernel_update_dummy_block(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    dummy_start: int,
    dummy_end: int,
    delta: wp.vec3,
    new_vel: wp.vec3,
):
    """Translate a contiguous block of dummy particles by ``delta`` and set velocity."""
    i = wp.tid()
    idx = dummy_start + i
    if idx >= dummy_end:
        return
    pos[idx] = pos[idx] + delta
    vel[idx] = new_vel


# ---------------------------------------------------------------------------
# Sphere dummy generation (CPU, numpy)
# ---------------------------------------------------------------------------


def _generate_sphere_dummy_particles(
    center: tuple[float, float, float],
    radius: float,
    dx: float,
    h: float,
    slip_type: str = "noslip",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate layered dummy particles on concentric spherical shells.

    Uses Fibonacci sphere sampling for uniform point distribution on each shell.
    Layer thickness is ``2h`` with ``ceil(2h / dx)`` layers, matching the AABB
    dummy boundary convention.

    Args:
        center: Sphere center [m].
        radius: Sphere radius [m].
        dx: Particle spacing [m].
        h: Smoothing length [m].
        slip_type: ``'noslip'`` or ``'freeslip'``.

    Returns:
        Tuple of (positions [N,3], normals [N,3], types [N]).
    """
    cx, cy, cz = center
    thickness = 2.0 * h
    layer_count = max(1, int(math.ceil(thickness / dx)))
    ptype = _SPH_DUMMY_NOSLIP_VAL if slip_type == "noslip" else _SPH_DUMMY_FREESLIP_VAL

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for layer in range(1, layer_count + 1):
        r_layer = radius + float(layer) * dx
        area = 4.0 * math.pi * r_layer * r_layer
        n_pts = max(1, int(math.ceil(area / (dx * dx))))

        layer_pos = np.zeros((n_pts, 3), dtype=np.float32)
        layer_norm = np.zeros((n_pts, 3), dtype=np.float32)

        for k in range(n_pts):
            # Fibonacci sphere: map k to unit sphere point
            y = 1.0 - (float(k) / float(n_pts - 1)) * 2.0 if n_pts > 1 else 0.0
            radius_xy = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * float(k)
            nx = math.cos(theta) * radius_xy
            nz = math.sin(theta) * radius_xy
            ny = y

            layer_pos[k, 0] = cx + nx * r_layer
            layer_pos[k, 1] = cy + ny * r_layer
            layer_pos[k, 2] = cz + nz * r_layer
            layer_norm[k, 0] = nx
            layer_norm[k, 1] = ny
            layer_norm[k, 2] = nz

        positions.append(layer_pos)
        normals.append(layer_norm)

    if not positions:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    pos_all = np.concatenate(positions, axis=0)
    norm_all = np.concatenate(normals, axis=0)
    types_all = np.full(pos_all.shape[0], ptype, dtype=np.int32)
    return pos_all, norm_all, types_all


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


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

        self._h = h
        self._support_radius = 2.0 * h  # kh=1.3 → factor=2.0 (default)

        self._column_top = 0.05
        sphere_radius = args.sphere_radius
        _surface_gap = 5e-4
        drop_z = self._column_top + sphere_radius + _surface_gap
        self._sphere_start_z = drop_z
        sphere_center = (0.0, 0.0, drop_z)

        # Physical sphere properties (for free-fall integration)
        self._sphere_mass = args.sphere_density * (4.0 / 3.0) * math.pi * sphere_radius**3
        self._sphere_radius = sphere_radius
        self._sphere_z = drop_z
        init_vz = -math.sqrt(2.0 * 9.81 * args.drop_height)
        self._sphere_vz = init_vz

        # ---- Build model ----
        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)

        # 1. Emit granular particles (must be first for fluid contiguity)
        self.fluid_count = self._emit_column_particles(builder, self.particle_spacing)

        # 2. Wall dummy boundaries (bottom + 4 sides, no top)
        domain_lo = (-0.075, -0.075, 0.0)
        domain_hi = (0.075, 0.075, 0.10)
        self.wall_dummy_count = SolverSPH.add_dummy_particles(
            builder,
            bounds_lo=domain_lo,
            bounds_hi=domain_hi,
            h=h,
            dx=self.particle_spacing,
            reference_density=RHO_GRANULAR,
            slip_type="noslip",
        )

        # 3. Sphere dummy particles (after walls to preserve fluid contiguity)
        self.sphere_dummy_start = self.fluid_count + self.wall_dummy_count
        sphere_pos_np, sphere_norm_np, _sphere_type_ignored = _generate_sphere_dummy_particles(
            sphere_center, sphere_radius, self.particle_spacing, h, slip_type="noslip"
        )
        sphere_type_np = np.full(sphere_pos_np.shape[0], _SPH_DUMMY_EMBEDDED_VAL, dtype=np.int32)
        self.sphere_dummy_count = sphere_pos_np.shape[0]
        self.sphere_dummy_end = self.sphere_dummy_start + self.sphere_dummy_count

        if self.sphere_dummy_count > 0:
            dummy_vol = self.particle_spacing**3
            dummy_mass_val = (
                self._sphere_mass / self.sphere_dummy_count
            )  # total dummy mass == sphere physical mass
            pos_list = [
                tuple(float(v) for v in sphere_pos_np[i]) for i in range(self.sphere_dummy_count)
            ]
            vel_list = [(0.0, 0.0, 0.0)] * self.sphere_dummy_count
            mass_list = [dummy_mass_val] * self.sphere_dummy_count
            radius_list = [self.particle_spacing / 2.0] * self.sphere_dummy_count
            builder.add_particles(
                pos=pos_list,
                vel=vel_list,
                mass=mass_list,
                radius=radius_list,
                custom_attributes={
                    "sph:particle_type": sphere_type_np.tolist(),
                    "sph:wall_normal": [
                        tuple(float(v) for v in sphere_norm_np[i])
                        for i in range(self.sphere_dummy_count)
                    ],
                },
            )

        self.model = builder.finalize()
        self.model.particle_grid = None
        self.model.set_gravity((0.0, 0.0, -9.81))

        if self.sphere_dummy_count > 0:
            m_np = self.model.particle_mass.numpy()
            m_dummy_total = float(
                m_np[self.sphere_dummy_start : self.sphere_dummy_end].sum()
            )
            print(
                f"[dummy-sphere] particles={self.sphere_dummy_count} "
                f"m_per_dummy={m_dummy_total / self.sphere_dummy_count:.6e} kg "
                f"m_total={m_dummy_total:.6f} kg "
                f"m_theoretical={self._sphere_mass:.6f} kg "
                f"ratio={m_dummy_total / self._sphere_mass:.4f}"
            )

        # ---- SPH solver configuration ----
        cfg = SolverSPH.Config()
        cfg.particle_spacing = self.particle_spacing
        cfg.kh = kh
        cfg.reference_density = RHO_GRANULAR
        cfg.simulation_method = "dp"
        cfg.integration_scheme = "symplectic_euler"
        cfg.boundary_type = "dummy"
        cfg.sound_speed = sound_speed
        cfg.artificial_viscosity_alpha = args.artificial_viscosity_alpha
        cfg.body_coupling_enabled = False

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

        # Store initial sphere dummy positions (relative to sphere center) for
        # fast update during simulation.
        if self.sphere_dummy_count > 0:
            q_np = self.state_0.particle_q.numpy().copy()
            self._sphere_dummy_init_pos = q_np[
                self.sphere_dummy_start : self.sphere_dummy_end
            ].copy()
        else:
            self._sphere_dummy_init_pos = np.zeros((0, 3), dtype=np.float32)

        # ---- Hu 2021 embedded-dummy scratch ----
        self._embedded_stress = wp.zeros(
            self.sphere_dummy_count, dtype=wp.mat33, device=self.model.device
        )
        self._sphere_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._reaction_grid = wp.HashGrid(128, 128, 128)

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
        """Emit the granular column particles (same as original)."""
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

    def _update_sphere_position(self, state: newton.State, delta_z: float, vz: float) -> None:
        """Translate sphere dummy particles vertically by ``delta_z`` and set velocity."""
        if self.sphere_dummy_count == 0:
            return
        n = self.sphere_dummy_count
        new_pos = self._sphere_dummy_init_pos + np.array(
            [0.0, 0.0, self._sphere_z - self._sphere_start_z], dtype=np.float32
        )
        q_np = state.particle_q.numpy()
        q_np[self.sphere_dummy_start : self.sphere_dummy_end] = new_pos
        state.particle_q.assign(q_np)

        qd_np = state.particle_qd.numpy()
        qd_np[self.sphere_dummy_start : self.sphere_dummy_end] = np.array(
            [0.0, 0.0, vz], dtype=np.float32
        )
        state.particle_qd.assign(qd_np)

    def _prepare_embedded_stress(self) -> None:
        """Hu 2021: Shepard-interpolate fluid stress onto sphere dummies.

        Writes the result both to the ``_embedded_stress`` scratch (for the
        reaction-force kernel) and to ``state_0.sph.stress`` (for the SPH
        stress-force kernel pass-through).
        """
        if self.sphere_dummy_count == 0:
            return

        self._reaction_grid.build(self.state_0.particle_q, self._support_radius)

        wp.launch(
            _kernel_interpolate_embedded_stress,
            dim=self.sphere_dummy_count,
            inputs=[
                self._reaction_grid.id,
                self.state_0.particle_q,
                self.state_0.sph.stress,
                self.state_0.sph.density,
                self.model.particle_mass,
                self.model.sph.particle_type,
                self.model.particle_flags,
                self.sphere_dummy_start,
                self.sphere_dummy_end,
                self._h,
                self._support_radius,
            ],
            outputs=[self._embedded_stress],
            device=self.model.device,
        )

        # Copy into state_0.sph.stress so the SPH stress-force kernel's
        # copy-through (stress_out[d] = stress_in[d]) picks up the
        # interpolated value.
        stress_np = self.state_0.sph.stress.numpy().copy()
        i_stress_np = self._embedded_stress.numpy()
        stress_np[self.sphere_dummy_start : self.sphere_dummy_end] = i_stress_np
        self.state_0.sph.stress.assign(stress_np)

    def _compute_reaction_force(self) -> float:
        """Total vertical SPH force on sphere (Hu 2021 interpolated stress)."""
        if self.sphere_dummy_count == 0:
            return 0.0

        self._sphere_force.zero_()
        wp.launch(
            _kernel_compute_embedded_reaction,
            dim=self.model.particle_count,
            inputs=[
                self._reaction_grid.id,
                self.state_0.particle_q,
                self.state_0.sph.density,
                self.state_0.sph.stress,
                self.model.particle_mass,
                self.model.sph.particle_type,
                self.model.particle_flags,
                self.sphere_dummy_start,
                self.sphere_dummy_end,
                self._embedded_stress,
                self._h,
                self._support_radius,
            ],
            outputs=[self._sphere_force],
            device=self.model.device,
        )
        f_np = self._sphere_force.numpy()
        return float(f_np[0, 2])

    def simulate(self) -> None:
        if self.sim_time >= self.sim_duration:
            return

        gz = -9.81

        for _ in range(self.sim_substeps):
            # 1. Hu 2021: Shepard-interpolate fluid stress onto sphere dummies
            self._prepare_embedded_stress()

            # 2. Reaction force from interpolated stress (same as SPH solver)
            f_z = self._compute_reaction_force()

            # 3. Integrate sphere free-fall motion
            self._sphere_vz += (gz + f_z / self._sphere_mass) * self.sim_dt
            self._sphere_z += self._sphere_vz * self.sim_dt

            # 4. Update sphere dummy positions and velocities
            self._update_sphere_position(
                self.state_0,
                self._sphere_z - self._sphere_start_z,
                self._sphere_vz,
            )

            # 5. SPH step (stress-force kernel reads embedded stress from state_0)
            self.sph_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)

            # 6. Swap
            self.state_0, self.state_1 = self.state_1, self.state_0

        v = self.state_0.particle_qd.numpy()
        fluid_v = v[: self.fluid_count]
        max_speed = float(np.max(np.linalg.norm(fluid_v, axis=1)))
        pen = max(0.0, self._column_top - (self._sphere_z - self._sphere_radius))
        print(
            f"t={self.sim_time:.4f}s  v_max={max_speed:.4f} m/s  pen={pen * 1000:.2f} mm  v_sphere_z={self._sphere_vz:.3f} m/s",
            flush=True,
        )
        self._telemetry.append(
            {"t": self.sim_time, "pen": pen, "sphere_z": self._sphere_z, "sphere_vz": self._sphere_vz}
        )

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

        # Highlight sphere dummies in green
        if self.sphere_dummy_count > 0:
            green = np.array([0.2, 0.8, 0.2], dtype=np.float32)
            colors_np[self.sphere_dummy_start : self.sphere_dummy_end] = green

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
        ax.set_title("Sphere penetration (dummy-particle boundaries)")
        if self.args.plot_path:
            fig.savefig(self.args.plot_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        margin = 2.0 * self.particle_spacing

        # 1. No NaN/Inf in particle state.
        assert np.all(np.isfinite(q)), "Particle positions contain NaN/Inf"
        assert np.all(np.isfinite(qd)), "Particle velocities contain NaN/Inf"

        # 2. Sphere did not punch through floor.
        assert self._sphere_z > -margin, f"Sphere fell through floor: z={self._sphere_z:.4f} m"

        # 3. Penetration depth within 20% of analytical prediction.
        args = self.args
        delta_pred = _predict_penetration_depth(
            args.sphere_friction,
            args.sphere_density,
            RHO_GRANULAR,
            args.drop_height,
            args.sphere_radius,
        )
        sphere_bottom_z = self._sphere_z - args.sphere_radius
        delta_meas = max(0.0, self._column_top - sphere_bottom_z)
        rel_err = abs(delta_meas - delta_pred) / max(delta_pred, 1e-6)
        assert rel_err < 0.20, (
            f"Penetration depth mismatch: measured={delta_meas * 1e3:.2f} mm, "
            f"predicted={delta_pred * 1e3:.2f} mm, rel_err={rel_err:.2%}"
        )

        # 4. Sphere settled: window-min |v_z| over last 10 frames < 0.05 m/s.
        recent_vz = [abs(r["sphere_vz"]) for r in self._telemetry[-10:]]
        assert recent_vz and min(recent_vz) < 0.05, (
            f"Sphere not settled: min|v_z| over last 10 frames = "
            f"{min(recent_vz) if recent_vz else float('nan'):.3f} m/s"
        )

        # 5. No fluid particles leaked below floor.
        fluid_q = q[: self.fluid_count]
        assert np.all(fluid_q[:, 2] > -margin), "Fluid particles leaked below floor"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument(
            "--duration", type=float, default=0.8, help="Simulation duration [s]"
        )
        parser.add_argument(
            "--sphere-radius", type=float, default=0.0125, help="Sphere radius [m]"
        )
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
            help="Sphere-granular Coulomb friction (for analytical prediction only) [default: 0.3]",
        )
        parser.add_argument(
            "--drop-height",
            type=float,
            default=0.1,
            help="Drop height from granular surface to bottom of sphere [m]",
        )
        parser.add_argument("--plot-path", type=str, default=None)
        parser.add_argument(
            "--sweep",
            action="store_true",
            help="Run parameter sweep over drop heights, densities, and frictions",
        )
        parser.add_argument(
            "--material-friction",
            type=float,
            default=0.5,
            help="Granular material friction angle [rad] [default: 0.5]",
        )
        parser.add_argument(
            "--artificial-viscosity-alpha",
            type=float,
            default=0.1,
            help="Monaghan artificial viscosity alpha coefficient [default: 0.1]",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if getattr(args, "sweep", False):
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
            print(
                f"\n[{i}/{len(combos)}] H={H}m  rho_s={rho_s}  mu_s={mu_s}", flush=True
            )
            sweep_args = _ap.Namespace(
                fps=60.0,
                duration=args.duration,
                sphere_radius=SPHERE_RADIUS,
                sphere_density=rho_s,
                sphere_friction=mu_s,
                material_friction=0.5,
                drop_height=H,
                plot_path=f"/tmp/sph_dummy_sweep_{i}.png",
                device=args.device,
                test=False,
                artificial_viscosity_alpha=getattr(
                    args, "artificial_viscosity_alpha", 0.1
                ),
            )
            sweep_viewer = ViewerNull(num_frames=10000)
            ex = Example(sweep_viewer, sweep_args)
            while ex.sim_time < ex.sim_duration:
                ex.step()

            delta_sim = ex._telemetry[-1]["pen"]
            delta_pred = _predict_penetration_depth(
                mu_s, rho_s, RHO_GRANULAR, H, SPHERE_RADIUS
            )
            err_pct = 100.0 * abs(delta_sim - delta_pred) / max(delta_pred, 1e-6)
            rows.append((H, rho_s, mu_s, delta_sim * 1e3, delta_pred * 1e3, err_pct))
            print(
                f"  sim={delta_sim * 1e3:.2f}mm  pred={delta_pred * 1e3:.2f}mm  err={err_pct:.1f}%"
            )
            del ex

        hdr = (
            f"{'H[m]':>6} {'rho_s':>6} {'mu_s':>5} {'sim[mm]':>9} {'pred[mm]':>9} {'err%':>7}"
        )
        sep = "-" * len(hdr)
        print("\n" + "=" * len(hdr))
        print(hdr)
        print(sep)
        for H, rho_s, mu_s, sim_mm, pred_mm, err_pct in rows:
            print(
                f"{H:>6.2f} {rho_s:>6} {mu_s:>5.1f} {sim_mm:>9.2f} {pred_mm:>9.2f} {err_pct:>7.1f}%"
            )
        print("=" * len(hdr))
    else:
        example = Example(viewer, args)
        newton.examples.run(example, args)
