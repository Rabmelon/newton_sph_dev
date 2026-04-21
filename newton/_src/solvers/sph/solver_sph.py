# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH solver for granular material large deformation simulation.

This solver implements Smoothed Particle Hydrodynamics (SPH) with two
constitutive models targeting granular/soil materials:

- **Drucker-Prager** (``dp``): elastic-plastic model with DP yield surface
  and return mapping.
- **mu(I) rheology** (``mui``): rate-dependent granular rheology using
  DP yield surface with effective viscosity from inertial number.

Call :meth:`register_custom_attributes` on your :class:`~newton.ModelBuilder`
before building the model to enable the SPH-specific per-particle material
parameters and state variables (e.g. ``sph:young_modulus``,
``sph:density``, ``sph:stress``).

References:
    - tiSPHi: https://github.com/Rabmelon/tiSPHi
    - Newton implicit MPM solver architecture
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from ...geometry import GeoType
from ..solver import SolverBase
from .sph_boundary import ground_plane_penalty_kernel
from .sph_constitutive import (
    initialize_geostatic_stress_kernel,
    update_stress_dp_kernel,
    update_stress_mui_kernel,
)
from .sph_dummy_boundary import add_dummy_particles_to_builder
from .sph_kernels import (
    compute_artificial_viscosity_kernel,
    compute_density_kernel,
    compute_strain_rate_kernel,
    compute_stress_force_kernel,
    compute_velocity_gradient_kernel,
    half_step_position_kernel,
    integrate_symplectic_euler_kernel,
    integrate_verlet_final_kernel,
    xsph_correction_kernel,
)
from .sph_model import SPHModel


class SolverSPH(SolverBase):
    """SPH solver for granular material simulation.

    Supports Drucker-Prager elastic-plastic and mu(I) rheology constitutive
    models. Uses :class:`wp.HashGrid` for neighbor search.

    Per-particle properties can be configured using custom attributes on the Model.
    See :meth:`SolverSPH.register_custom_attributes` for details.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverSPH`."""

        # --- SPH discretization ---
        particle_spacing: float = 0.005
        """Initial inter-particle distance dx [m]. Controls resolution."""
        kh: float = 1.3
        """Smoothing-length ratio: h = kh * particle_spacing."""
        kernel_type: str = "wendland_c2"
        """SPH kernel function: ``'wendland_c2'`` or ``'cubic_spline'``."""
        support_radius_factor: float = 2.0
        """Support radius = support_radius_factor * smoothing_length."""

        # --- Simulation method ---
        simulation_method: str = "dp"
        """Constitutive model: ``'dp'`` (Drucker-Prager) or ``'mui'`` (mu(I) rheology)."""

        # --- Equation of state ---
        sound_speed: float = 50.0
        """Artificial speed of sound c_s [m/s] for WCSPH pressure."""
        eos_exponent: float = 7.0
        """Tait EOS exponent gamma."""
        reference_density: float = 1500.0
        """Reference density rho_0 [kg/m^3]."""

        # --- Artificial viscosity ---
        artificial_viscosity_alpha: float = 0.1
        """Monaghan artificial viscosity coefficient alpha."""

        # --- Time integration ---
        integration_scheme: str = "position_verlet"
        """Time integration: ``'symplectic_euler'`` or ``'position_verlet'``."""

        # --- Neighbor search ---
        hash_grid_dim: int = 128
        """Dimension of :class:`wp.HashGrid` per axis."""

        # --- Boundary ---
        boundary_type: str = "penalty"
        """Boundary treatment: ``'penalty'`` (default) or ``'dummy'`` (layered dummy particles)."""
        penalty_stiffness: float = 1.0e6
        """SDF penalty boundary stiffness [N/m]."""
        penalty_damping: float = 1.0e3
        """SDF penalty boundary damping [N*s/m]."""
        boundary_friction: float = 0.5
        """Coulomb friction coefficient mu for penalty boundary surfaces."""
        restitution: float = 0.3
        """Domain boundary inelastic collision restitution coefficient."""
        dummy_beta: float = 1.7
        """Distance-based dummy boundary extrapolation factor."""

        # --- Corrections ---
        xsph_epsilon: float = 0.0
        """XSPH velocity smoothing factor (0 = off, 1 = full)."""

        # --- Granular damping ---
        viscous_damping: float = 0.0
        """Viscous damping factor: F_d = -eps * sqrt(E / (rho * h^2)) * v."""

        def __post_init__(self) -> None:
            supported_simulation_methods = {"dp", "mui"}
            if self.simulation_method not in supported_simulation_methods:
                raise ValueError(
                    f"Invalid simulation_method: {self.simulation_method}. "
                    f"Must be one of {supported_simulation_methods}."
                )
            supported_integration_schemes = {"symplectic_euler", "position_verlet"}
            if self.integration_scheme not in supported_integration_schemes:
                raise ValueError(
                    f"Invalid integration_scheme: {self.integration_scheme}. "
                    f"Must be one of {supported_integration_schemes}."
                )
            supported_boundary_types = {"penalty", "dummy"}
            if self.boundary_type not in supported_boundary_types:
                raise ValueError(
                    f"Invalid boundary_type: {self.boundary_type}. Must be one of {supported_boundary_types}."
                )

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register SPH-specific custom attributes in the ``'sph'`` namespace.

        This method registers per-particle material parameters and state
        variables for the SPH solver. Must be called before
        ``builder.finalize()``.

        Attributes registered on Model (per-particle):
            - ``sph:young_modulus``: Young's modulus [Pa]
            - ``sph:poisson_ratio``: Poisson's ratio
            - ``sph:friction``: Internal friction angle [rad]
            - ``sph:cohesion``: Cohesion [Pa]
            - ``sph:dilatancy``: Dilatancy angle [rad]
            - ``sph:viscosity``: Dynamic viscosity [Pa*s]
            - ``sph:yield_pressure``: Yield pressure cap [Pa]
            - ``sph:particle_type``: Particle type (0=fluid, 1=dummy no-slip, 2=dummy free-slip)
            - ``sph:wall_normal``: Outward wall normal for dummy particles

        Attributes registered on State (per-particle):
            - ``sph:density``: Current density [kg/m^3]
            - ``sph:pressure``: Current pressure [Pa]
            - ``sph:stress``: Cauchy stress tensor [Pa]
            - ``sph:strain_rate``: Strain rate tensor [1/s]
            - ``sph:velocity_gradient``: Velocity gradient tensor [1/s]
            - ``sph:plastic_strain``: Accumulated equivalent plastic strain
        """
        CA = newton.ModelBuilder.CustomAttribute
        AF = newton.Model.AttributeFrequency
        AA = newton.Model.AttributeAssignment
        ns = "sph"

        # Per-particle material parameters (on Model)
        for name, dtype, default in [
            ("young_modulus", wp.float32, 1.0e6),
            ("poisson_ratio", wp.float32, 0.3),
            ("friction", wp.float32, 0.5),
            ("cohesion", wp.float32, 0.0),
            ("dilatancy", wp.float32, 0.0),
            ("viscosity", wp.float32, 0.0),
            ("yield_pressure", wp.float32, 1.0e12),
            ("particle_type", wp.int32, 0),
            ("wall_normal", wp.vec3, wp.vec3(0.0)),
        ]:
            builder.add_custom_attribute(
                CA(
                    name=name,
                    frequency=AF.PARTICLE,
                    assignment=AA.MODEL,
                    dtype=dtype,
                    default=default,
                    namespace=ns,
                )
            )

        # Per-particle state variables (on State)
        for name, dtype, default in [
            ("density", wp.float32, 0.0),
            ("pressure", wp.float32, 0.0),
            ("stress", wp.mat33, wp.mat33(0.0)),
            ("strain_rate", wp.mat33, wp.mat33(0.0)),
            ("velocity_gradient", wp.mat33, wp.mat33(0.0)),
            ("plastic_strain", wp.float32, 0.0),
        ]:
            builder.add_custom_attribute(
                CA(
                    name=name,
                    frequency=AF.PARTICLE,
                    assignment=AA.STATE,
                    dtype=dtype,
                    default=default,
                    namespace=ns,
                )
            )

    @staticmethod
    def add_dummy_particles(
        builder: newton.ModelBuilder,
        bounds_lo: tuple[float, float, float],
        bounds_hi: tuple[float, float, float],
        h: float,
        dx: float,
        reference_density: float,
        slip_type: str = "noslip",
    ) -> int:
        """Generate and add layered dummy boundary particles.

        Creates multiple layers of static particles outside each face of an
        AABB domain box. Must be called after
        :meth:`register_custom_attributes` and before ``builder.finalize()``.

        Args:
            builder: Newton model builder.
            bounds_lo: Domain AABB lower corner [m].
            bounds_hi: Domain AABB upper corner [m].
            h: Smoothing length [m].
            dx: Particle spacing [m].
            reference_density: Reference density for mass computation [kg/m^3].
            slip_type: ``'noslip'`` or ``'freeslip'``.

        Returns:
            Number of dummy particles added.
        """
        return add_dummy_particles_to_builder(
            builder,
            bounds_lo,
            bounds_hi,
            h,
            dx,
            reference_density,
            slip_type,
        )

    def __init__(
        self,
        model: newton.Model,
        config: Config | None = None,
    ):
        super().__init__(model)

        if config is None:
            config = SolverSPH.Config()
        self._config = config

        self._sph_model = SPHModel(model, config.reference_density)

        # Derived parameters: smoothing length computed from spacing and ratio
        self._h = config.kh * config.particle_spacing
        self._support_radius = config.support_radius_factor * self._h

        # Hash grid for neighbor search
        dim = config.hash_grid_dim
        with wp.ScopedDevice(model.device):
            self._hash_grid = wp.HashGrid(dim, dim, dim)

        # Scratch arrays
        n = model.particle_count
        self._accel = wp.zeros(n, dtype=wp.vec3, device=model.device)
        self._density_prev = wp.zeros(n, dtype=float, device=model.device)

        # Midpoint position buffer for Position-based Verlet integration
        if config.integration_scheme == "position_verlet":
            self._pos_mid = wp.zeros(n, dtype=wp.vec3, device=model.device)

        # Cached references for dummy boundary kernel args
        self._particle_type = model.sph.particle_type
        self._wall_normal = model.sph.wall_normal

        # Static CFL limit (velocity-independent; used for one-time warning).
        self._dt_cfl_static = 0.3 * self._h / config.sound_speed
        self._cfl_warned = False
        self._gravity_vec = self._extract_gravity_vec()

        # Cache ground plane data for penalty boundary to avoid GPU→CPU sync
        # in the hot loop.
        if config.boundary_type == "penalty":
            self._ground_planes = self._extract_ground_planes()

        # Compute fluid particle count for optimized kernel launches.
        # Fluid particles must be contiguous at the front of the array
        # (guaranteed when add_dummy_particles is called after fluid emission).
        pt = self._particle_type.numpy()
        self._fluid_count = int(np.sum(pt == 0))
        if 0 < self._fluid_count < n:
            if not np.all(pt[: self._fluid_count] == 0):
                warnings.warn(
                    "Fluid particles are not contiguous at the front of the particle "
                    "array. Falling back to launching kernels with dim=particle_count.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fluid_count = n

    def _extract_gravity_vec(self) -> wp.vec3:
        """Extract gravity as a :class:`wp.vec3` from the model array."""
        g = self.model.gravity.numpy()
        return wp.vec3(float(g[0][0]), float(g[0][1]), float(g[0][2]))

    def _extract_ground_planes(self) -> list[tuple[wp.vec3, float]]:
        """Pre-compute ground plane normals and offsets from model shapes.

        Reads shape geometry arrays once so that the hot loop can launch
        penalty kernels without GPU→CPU synchronization.
        """
        model = self.model
        if model.shape_count == 0:
            return []
        geo_types = model.shape_type.numpy()
        shape_transforms = model.shape_transform.numpy()
        planes: list[tuple[wp.vec3, float]] = []
        for s in range(model.shape_count):
            if geo_types[s] == int(GeoType.PLANE):
                tf = shape_transforms[s]
                px, py, pz = float(tf[0]), float(tf[1]), float(tf[2])
                qx, qy, qz, qw = float(tf[3]), float(tf[4]), float(tf[5]), float(tf[6])
                q = wp.quat(qx, qy, qz, qw)
                normal = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
                offset = -(normal[0] * px + normal[1] * py + normal[2] * pz)
                planes.append((normal, float(offset)))
        return planes

    @property
    def config(self) -> Config:
        """Current solver configuration."""
        return self._config

    @property
    def smoothing_length(self) -> float:
        """SPH smoothing length h [m]."""
        return self._h

    def notify_model_changed(self, flags: int) -> None:
        # Deferred import avoids circular `newton.solvers → SolverSPH → newton.solvers`.
        from newton.solvers import SolverNotifyFlags  # noqa: PLC0415

        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._gravity_vec = self._extract_gravity_vec()
        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            if self._config.boundary_type == "penalty":
                self._ground_planes = self._extract_ground_planes()

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance the SPH simulation by one time step.

        Args:
            state_in: Input state (positions, velocities, forces).
            state_out: Output state (updated positions, velocities).
            control: Control input (unused, for API compatibility).
            contacts: Contact information (unused, for API compatibility).
            dt: Time step [s].
        """
        model = self.model
        n = model.particle_count
        if n == 0:
            return

        # CFL safety check — fire once using the static (v=0) limit to avoid a
        # GPU→CPU sync every step.  compute_cfl_dt() gives a tighter bound when
        # called explicitly by the user.
        if not self._cfl_warned and dt > self._dt_cfl_static * 2.0:
            warnings.warn(
                f"SPH dt={dt:.3e} s exceeds acoustic CFL limit "
                f"{self._dt_cfl_static:.3e} s (ratio={dt / self._dt_cfl_static:.1f}×). "  # noqa: RUF001
                "Reduce sim_dt or increase substeps to avoid particle explosion.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._cfl_warned = True

        with wp.ScopedDevice(model.device):
            if self._config.integration_scheme == "position_verlet":
                self._step_position_verlet(state_in, state_out, dt)
            else:
                self._step_symplectic_euler(state_in, state_out, dt)

    def _step_symplectic_euler(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Symplectic (semi-implicit) Euler time integration step."""
        # 1. Build neighbor list
        self._build_neighbor_list(state_in)

        # 2. Compute density (stored on state_in for use during force computation)
        self._compute_density(state_in)

        # 3. Zero acceleration (cudaMemset is cheaper than a kernel launch)
        self._accel.zero_()

        # 4-6. Compute forces based on constitutive model
        # Stress is read from state_in (previous) and written to state_out (new).
        # Force computation then uses the updated stress from state_out.
        if self._config.simulation_method == "dp":
            self._compute_velocity_gradient(state_in)
            self._compute_strain_rate(state_in)
            self._compute_stress_dp(state_in, state_out, dt)
            self._compute_stress_forces(state_in, state_out)
        elif self._config.simulation_method == "mui":
            self._compute_velocity_gradient(state_in)
            self._compute_strain_rate(state_in)
            self._compute_stress_mui(state_in, state_out, dt)
            self._compute_stress_forces(state_in, state_out)
        else:
            raise ValueError(f"Unknown simulation_method={self._config.simulation_method!r}")

        # 7. Artificial viscosity
        if self._config.artificial_viscosity_alpha > 0.0:
            self._compute_artificial_viscosity(state_in)

        # 8. Boundary forces (penalty method only; dummy particles
        #    handle boundaries through the SPH kernels themselves)
        if self._config.boundary_type == "penalty":
            self._apply_boundary_forces(state_in)

        # 9. Time integration (symplectic Euler, includes gravity)
        self._integrate(state_in, state_out, dt)

        # 10. XSPH velocity correction
        if self._config.xsph_epsilon > 0.0:
            self._xsph_correction(state_out)

        # 11. Copy density to state_out for continuity across steps
        wp.copy(state_out.sph.density, state_in.sph.density)

    def _step_position_verlet(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Position-based Verlet (2nd order) time integration step.

        Reference:
            Zhang et al. (2024) Computers and Geotechnics 167:106052.
        """
        model = self.model
        n = model.particle_count

        # 1. Half-step position: x_mid = x_n + (dt/2) * v_n
        wp.launch(
            half_step_position_kernel,
            dim=n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_flags,
                self._particle_type,
                dt * 0.5,
            ],
            outputs=[self._pos_mid],
            device=model.device,
        )

        # 2. Build neighbor list at midpoint positions
        self._build_neighbor_list(state_in, pos=self._pos_mid)

        # 3. Compute density at midpoint -> write to state_out
        #    Use state_in density as Shepard correction reference (not stale state_out)
        self._compute_density(state_out, pos=self._pos_mid, density_prev_source=state_in.sph.density)

        # 4. Zero acceleration
        self._accel.zero_()

        # 5-6. Compute forces at midpoint configuration
        if self._config.simulation_method == "dp":
            self._compute_velocity_gradient(state_out, pos=self._pos_mid, vel=state_in.particle_qd)
            self._compute_strain_rate(state_out)
            self._compute_stress_dp(
                state_in,
                state_out,
                dt,
                strain_rate=state_out.sph.strain_rate,
                velocity_gradient=state_out.sph.velocity_gradient,
            )
            self._compute_stress_forces(state_in, state_out, pos=self._pos_mid, density=state_out.sph.density)
        elif self._config.simulation_method == "mui":
            self._compute_velocity_gradient(state_out, pos=self._pos_mid, vel=state_in.particle_qd)
            self._compute_strain_rate(state_out)
            self._compute_stress_mui(
                state_in,
                state_out,
                dt,
                strain_rate=state_out.sph.strain_rate,
                density=state_out.sph.density,
            )
            self._compute_stress_forces(state_in, state_out, pos=self._pos_mid, density=state_out.sph.density)
        else:
            raise ValueError(f"Unknown simulation_method={self._config.simulation_method!r}")

        # 7. Artificial viscosity at midpoint
        if self._config.artificial_viscosity_alpha > 0.0:
            self._compute_artificial_viscosity(
                state_in, pos=self._pos_mid, vel=state_in.particle_qd, density=state_out.sph.density
            )

        # 8. Boundary forces at midpoint
        if self._config.boundary_type == "penalty":
            self._apply_boundary_forces(state_in, pos=self._pos_mid, vel=state_in.particle_qd)

        # 9. Full-step velocity + final position
        self._integrate_verlet_final(state_in, state_out, dt)

        # 10. XSPH velocity correction (uses midpoint hash grid)
        if self._config.xsph_epsilon > 0.0:
            self._xsph_correction(state_out)

        # Density was computed directly into state_out in step 3 — no copy needed.

    def _integrate_verlet_final(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Verlet final step: full-step velocity and position from midpoint."""
        model = self.model
        wp.launch(
            integrate_verlet_final_kernel,
            dim=model.particle_count,
            inputs=[
                self._pos_mid,
                state_in.particle_qd,
                self._accel,
                model.particle_flags,
                self._particle_type,
                model.particle_world,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_neighbor_list(self, state: newton.State, *, pos: wp.array[wp.vec3] | None = None) -> None:
        """Build hash grid from current particle positions."""
        positions = pos if pos is not None else state.particle_q
        self._hash_grid.build(positions, self._support_radius)

    def _compute_density(
        self,
        state: newton.State,
        *,
        pos: wp.array[wp.vec3] | None = None,
        density_prev_source: wp.array[float] | None = None,
    ) -> None:
        """Compute SPH density via Shepard-corrected direct summation.

        Args:
            state: State whose density array will be written.
            pos: Override positions (e.g. midpoint for Verlet).
            density_prev_source: Previous-step density for Shepard correction.
                Defaults to ``state.sph.density`` (correct for symplectic Euler
                where state_in has the previous density). For position Verlet,
                pass ``state_in.sph.density`` explicitly.
        """
        positions = pos if pos is not None else state.particle_q
        src = density_prev_source if density_prev_source is not None else state.sph.density
        wp.copy(self._density_prev, src)
        wp.launch(
            compute_density_kernel,
            dim=self._fluid_count,
            inputs=[
                self._hash_grid.id,
                positions,
                self.model.particle_mass,
                self._density_prev,
                self.model.particle_flags,
                self._particle_type,
                self._h,
                self._support_radius,
                self._config.reference_density,
            ],
            outputs=[state.sph.density],
            device=self.model.device,
        )

    def _compute_velocity_gradient(
        self,
        state: newton.State,
        *,
        pos: wp.array[wp.vec3] | None = None,
        vel: wp.array[wp.vec3] | None = None,
    ) -> None:
        """Compute SPH velocity gradient tensor."""
        positions = pos if pos is not None else state.particle_q
        velocities = vel if vel is not None else state.particle_qd
        wp.launch(
            compute_velocity_gradient_kernel,
            dim=self._fluid_count,
            inputs=[
                self._hash_grid.id,
                positions,
                velocities,
                self.model.particle_mass,
                state.sph.density,
                self.model.particle_flags,
                self._particle_type,
                self._wall_normal,
                self._h,
                self._support_radius,
                self._config.dummy_beta,
                self._config.reference_density,
            ],
            outputs=[state.sph.velocity_gradient],
            device=self.model.device,
        )

    def _compute_strain_rate(self, state: newton.State) -> None:
        """Compute strain rate D = 0.5 * (L + L^T)."""
        wp.launch(
            compute_strain_rate_kernel,
            dim=self._fluid_count,
            inputs=[
                state.sph.velocity_gradient,
                self.model.particle_flags,
            ],
            outputs=[state.sph.strain_rate],
            device=self.model.device,
        )

    def _compute_stress_dp(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        *,
        strain_rate: wp.array[wp.mat33] | None = None,
        velocity_gradient: wp.array[wp.mat33] | None = None,
    ) -> None:
        """Compute stress via Drucker-Prager elastic-plastic model.

        Reads previous stress from state_in, writes new stress to state_out.
        Uses Jaumann objective stress rate for frame-indifference.
        """
        sr = strain_rate if strain_rate is not None else state_in.sph.strain_rate
        vg = velocity_gradient if velocity_gradient is not None else state_in.sph.velocity_gradient
        # Copy plastic_strain from state_in to state_out for accumulation
        wp.copy(state_out.sph.plastic_strain, state_in.sph.plastic_strain)
        wp.launch(
            update_stress_dp_kernel,
            dim=self._fluid_count,
            inputs=[
                sr,
                vg,
                state_in.sph.stress,
                self.model.sph.young_modulus,
                self.model.sph.poisson_ratio,
                self.model.sph.friction,
                self.model.sph.cohesion,
                self.model.sph.dilatancy,
                self.model.particle_flags,
                self._particle_type,
                dt,
            ],
            outputs=[
                state_out.sph.stress,
                state_out.sph.pressure,
                state_out.sph.plastic_strain,
            ],
            device=self.model.device,
        )

    def _compute_stress_mui(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        *,
        strain_rate: wp.array[wp.mat33] | None = None,
        density: wp.array[float] | None = None,
    ) -> None:
        """Compute stress via mu(I) rheology model.

        Reads strain rate from state_in, writes new stress to state_out.
        """
        sr = strain_rate if strain_rate is not None else state_in.sph.strain_rate
        rho = density if density is not None else state_in.sph.density
        wp.launch(
            update_stress_mui_kernel,
            dim=self._fluid_count,
            inputs=[
                sr,
                rho,
                self.model.sph.friction,
                self.model.sph.cohesion,
                self.model.sph.viscosity,
                self.model.particle_flags,
                self._particle_type,
                self._config.reference_density,
                self._config.sound_speed,
                self._config.eos_exponent,
                self._h,
            ],
            outputs=[
                state_out.sph.stress,
                state_out.sph.pressure,
            ],
            device=self.model.device,
        )

    def _compute_stress_forces(
        self,
        state_in: newton.State,
        state_out: newton.State,
        *,
        pos: wp.array[wp.vec3] | None = None,
        density: wp.array[float] | None = None,
    ) -> None:
        """Compute acceleration from stress tensor divergence.

        Uses density from state_in and updated stress from state_out.
        """
        positions = pos if pos is not None else state_in.particle_q
        rho = density if density is not None else state_in.sph.density
        gravity_vec = self._gravity_vec
        wp.launch(
            compute_stress_force_kernel,
            dim=self._fluid_count,
            inputs=[
                self._hash_grid.id,
                positions,
                self.model.particle_mass,
                rho,
                state_out.sph.stress,
                self.model.particle_flags,
                self._particle_type,
                self._h,
                self._support_radius,
                self._config.reference_density,
                gravity_vec,
            ],
            outputs=[self._accel],
            device=self.model.device,
        )

    def _compute_artificial_viscosity(
        self,
        state: newton.State,
        *,
        pos: wp.array[wp.vec3] | None = None,
        vel: wp.array[wp.vec3] | None = None,
        density: wp.array[float] | None = None,
    ) -> None:
        """Add Monaghan-type artificial viscosity acceleration."""
        positions = pos if pos is not None else state.particle_q
        velocities = vel if vel is not None else state.particle_qd
        rho = density if density is not None else state.sph.density
        wp.launch(
            compute_artificial_viscosity_kernel,
            dim=self._fluid_count,
            inputs=[
                self._hash_grid.id,
                positions,
                velocities,
                self.model.particle_mass,
                rho,
                self.model.particle_flags,
                self._particle_type,
                self._wall_normal,
                self._h,
                self._support_radius,
                self._config.artificial_viscosity_alpha,
                self._config.sound_speed,
                self._config.dummy_beta,
                self._config.reference_density,
            ],
            outputs=[self._accel],
            device=self.model.device,
        )

    def _apply_boundary_forces(
        self,
        state: newton.State,
        *,
        pos: wp.array[wp.vec3] | None = None,
        vel: wp.array[wp.vec3] | None = None,
    ) -> None:
        """Apply cached ground plane penalty forces.

        Uses pre-computed plane normals and offsets from :meth:`__init__`
        to avoid GPU→CPU synchronization every substep.
        """
        positions = pos if pos is not None else state.particle_q
        velocities = vel if vel is not None else state.particle_qd
        n = self.model.particle_count
        for normal, offset in self._ground_planes:
            wp.launch(
                ground_plane_penalty_kernel,
                dim=n,
                inputs=[
                    positions,
                    velocities,
                    self.model.particle_flags,
                    normal,
                    offset,
                    self._config.penalty_stiffness,
                    self._config.penalty_damping,
                    self._config.boundary_friction,
                ],
                outputs=[self._accel],
                device=self.model.device,
            )

    def _integrate(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Symplectic Euler time integration."""
        model = self.model
        n = model.particle_count
        wp.launch(
            integrate_symplectic_euler_kernel,
            dim=n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self._accel,
                model.particle_flags,
                self._particle_type,
                model.particle_world,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )

    def _xsph_correction(self, state: newton.State) -> None:
        """Apply XSPH velocity smoothing correction in-place."""
        wp.launch(
            xsph_correction_kernel,
            dim=self._fluid_count,
            inputs=[
                self._hash_grid.id,
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                state.sph.density,
                self.model.particle_flags,
                self._particle_type,
                self._wall_normal,
                self._h,
                self._support_radius,
                self._config.xsph_epsilon,
                self._config.dummy_beta,
                self._config.reference_density,
            ],
            outputs=[state.particle_qd],
            device=self.model.device,
        )

    # ------------------------------------------------------------------
    # Geostatic stress initialization
    # ------------------------------------------------------------------

    def initialize_geostatic_stress(self, state: newton.State, y_max: float) -> None:
        """Initialize stress field using K0 earth pressure condition.

        Sets the initial stress tensor for each particle based on the
        geostatic K0 condition::

            sigma_zz = rho_0 * g_z * (z_max - z_i)
            sigma_xx = sigma_yy = K0 * sigma_zz
            K0 = 1 - sin(phi)

        where z is the vertical axis (up_axis of the model).

        Args:
            state: State whose ``sph:stress`` field will be initialized.
            y_max: Maximum vertical coordinate of the particle column [m].

        Reference:
            tiSPHi ``init_soil_stress()``.
        """
        n = self.model.particle_count
        if n == 0:
            return

        # Get gravity magnitude along up axis
        gravity_np = self.model.gravity.numpy()
        # Assume Z-up by default (gravity along axis 2)
        g_mag = abs(float(gravity_np[0][2]))

        wp.launch(
            initialize_geostatic_stress_kernel,
            dim=n,
            inputs=[
                state.particle_q,
                self.model.sph.friction,
                self.model.particle_flags,
                self._particle_type,
                self._config.reference_density,
                g_mag,
                y_max,
            ],
            outputs=[state.sph.stress],
            device=self.model.device,
        )

    # ------------------------------------------------------------------
    # CFL diagnostics
    # ------------------------------------------------------------------

    def compute_cfl_dt(self, state: newton.State, courant_number: float = 0.3) -> float:
        """Return the acoustic CFL-stable timestep for the current state.

        .. math::
            \\Delta t_{\\text{CFL}} = C \\frac{h}{c_s + \\|\\mathbf{v}\\|_{\\max}}

        Args:
            state: Current simulation state used to obtain the maximum
                particle velocity magnitude.
            courant_number: Courant safety factor *C* (default 0.3).

        Returns:
            Maximum stable timestep [s].
        """
        v_np = state.particle_qd.numpy()
        v_max = float(np.linalg.norm(v_np, axis=-1).max()) if len(v_np) else 0.0
        return courant_number * self._h / (self._config.sound_speed + v_max)
