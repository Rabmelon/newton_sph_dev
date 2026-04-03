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

from dataclasses import dataclass

import warp as wp

import newton

from ..solver import SolverBase
from .sph_boundary import apply_ground_plane_penalty
from .sph_constitutive import (
    initialize_geostatic_stress_kernel,
    update_stress_dp_kernel,
    update_stress_mui_kernel,
)
from .sph_kernels import (
    compute_artificial_viscosity_kernel,
    compute_density_kernel,
    compute_strain_rate_kernel,
    compute_stress_force_kernel,
    compute_velocity_gradient_kernel,
    integrate_symplectic_euler_kernel,
    xsph_correction_kernel,
    zero_accel_kernel,
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
        smoothing_length: float = 0.05
        """SPH smoothing length h [m]. Typically 1.2-1.5x particle spacing."""
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
        integration_scheme: str = "symplectic_euler"
        """Time integration: ``'symplectic_euler'`` or ``'leapfrog'``."""

        # --- Neighbor search ---
        hash_grid_dim: int = 128
        """Dimension of :class:`wp.HashGrid` per axis."""

        # --- Boundary ---
        penalty_stiffness: float = 1.0e6
        """SDF penalty boundary stiffness [N/m]."""
        penalty_damping: float = 1.0e3
        """SDF penalty boundary damping [N*s/m]."""
        restitution: float = 0.3
        """Domain boundary inelastic collision restitution coefficient."""
        domain_lo: tuple | None = None
        """Optional domain lower bound for boundary clamping [m]."""
        domain_hi: tuple | None = None
        """Optional domain upper bound for boundary clamping [m]."""

        # --- Corrections ---
        xsph_epsilon: float = 0.5
        """XSPH velocity smoothing factor (0 = off, 1 = full)."""

        # --- Granular damping ---
        viscous_damping: float = 0.0
        """Viscous damping factor: F_d = -eps * sqrt(E / (rho * h^2)) * v."""

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

        # Derived parameters
        self._h = config.smoothing_length
        self._support_radius = config.support_radius_factor * config.smoothing_length

        # Hash grid for neighbor search
        dim = config.hash_grid_dim
        with wp.ScopedDevice(model.device):
            self._hash_grid = wp.HashGrid(dim, dim, dim)

        # Scratch arrays
        n = model.particle_count
        self._accel = wp.zeros(n, dtype=wp.vec3, device=model.device)

    @property
    def config(self) -> Config:
        """Current solver configuration."""
        return self._config

    @property
    def smoothing_length(self) -> float:
        """SPH smoothing length h [m]."""
        return self._h

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

        with wp.ScopedDevice(model.device):
            # 1. Build neighbor list
            self._build_neighbor_list(state_in)

            # 2. Compute density (stored on state_in for use during force computation)
            self._compute_density(state_in)

            # 3. Zero acceleration
            wp.launch(zero_accel_kernel, dim=n, inputs=[], outputs=[self._accel], device=model.device)

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

            # 7. Artificial viscosity
            if self._config.artificial_viscosity_alpha > 0.0:
                self._compute_artificial_viscosity(state_in)

            # 8. Boundary forces
            self._apply_boundary_forces(state_in)

            # 9. Time integration (symplectic Euler, includes gravity)
            self._integrate(state_in, state_out, dt)

            # 10. XSPH velocity correction
            if self._config.xsph_epsilon > 0.0:
                self._xsph_correction(state_out)

            # 11. Copy density to state_out for continuity across steps
            wp.copy(state_out.sph.density, state_in.sph.density)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_neighbor_list(self, state: newton.State) -> None:
        """Build hash grid from current particle positions."""
        self._hash_grid.build(state.particle_q, self._support_radius)

    def _compute_density(self, state: newton.State) -> None:
        """Compute SPH density via direct summation."""
        n = self.model.particle_count
        wp.launch(
            compute_density_kernel,
            dim=n,
            inputs=[
                self._hash_grid.id,
                state.particle_q,
                self.model.particle_mass,
                self.model.particle_flags,
                self._h,
                self._support_radius,
            ],
            outputs=[state.sph.density],
            device=self.model.device,
        )

    def _compute_velocity_gradient(self, state: newton.State) -> None:
        """Compute SPH velocity gradient tensor."""
        n = self.model.particle_count
        wp.launch(
            compute_velocity_gradient_kernel,
            dim=n,
            inputs=[
                self._hash_grid.id,
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                state.sph.density,
                self.model.particle_flags,
                self._h,
                self._support_radius,
            ],
            outputs=[state.sph.velocity_gradient],
            device=self.model.device,
        )

    def _compute_strain_rate(self, state: newton.State) -> None:
        """Compute strain rate D = 0.5 * (L + L^T)."""
        n = self.model.particle_count
        wp.launch(
            compute_strain_rate_kernel,
            dim=n,
            inputs=[
                state.sph.velocity_gradient,
                self.model.particle_flags,
            ],
            outputs=[state.sph.strain_rate],
            device=self.model.device,
        )

    def _compute_stress_dp(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Compute stress via Drucker-Prager elastic-plastic model.

        Reads previous stress from state_in, writes new stress to state_out.
        """
        n = self.model.particle_count
        # Copy plastic_strain from state_in to state_out for accumulation
        wp.copy(state_out.sph.plastic_strain, state_in.sph.plastic_strain)
        wp.launch(
            update_stress_dp_kernel,
            dim=n,
            inputs=[
                state_in.sph.strain_rate,
                state_in.sph.stress,
                self.model.sph.young_modulus,
                self.model.sph.poisson_ratio,
                self.model.sph.friction,
                self.model.sph.cohesion,
                self.model.sph.dilatancy,
                self.model.particle_flags,
                dt,
            ],
            outputs=[
                state_out.sph.stress,
                state_out.sph.pressure,
                state_out.sph.plastic_strain,
            ],
            device=self.model.device,
        )

    def _compute_stress_mui(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Compute stress via mu(I) rheology model.

        Reads strain rate from state_in, writes new stress to state_out.
        """
        n = self.model.particle_count
        wp.launch(
            update_stress_mui_kernel,
            dim=n,
            inputs=[
                state_in.sph.strain_rate,
                state_in.sph.density,
                self.model.sph.friction,
                self.model.sph.cohesion,
                self.model.sph.viscosity,
                self.model.particle_flags,
                self._config.reference_density,
                self._config.sound_speed,
                self._config.eos_exponent,
            ],
            outputs=[
                state_out.sph.stress,
                state_out.sph.pressure,
            ],
            device=self.model.device,
        )

    def _compute_stress_forces(self, state_in: newton.State, state_out: newton.State) -> None:
        """Compute acceleration from stress tensor divergence.

        Uses density from state_in and updated stress from state_out.
        """
        n = self.model.particle_count
        wp.launch(
            compute_stress_force_kernel,
            dim=n,
            inputs=[
                self._hash_grid.id,
                state_in.particle_q,
                self.model.particle_mass,
                state_in.sph.density,
                state_out.sph.stress,
                self.model.particle_flags,
                self._h,
                self._support_radius,
            ],
            outputs=[self._accel],
            device=self.model.device,
        )

    def _compute_artificial_viscosity(self, state: newton.State) -> None:
        """Add Monaghan-type artificial viscosity acceleration."""
        n = self.model.particle_count
        wp.launch(
            compute_artificial_viscosity_kernel,
            dim=n,
            inputs=[
                self._hash_grid.id,
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                state.sph.density,
                self.model.particle_flags,
                self._h,
                self._support_radius,
                self._config.artificial_viscosity_alpha,
                self._config.sound_speed,
            ],
            outputs=[self._accel],
            device=self.model.device,
        )

    def _apply_boundary_forces(self, state: newton.State) -> None:
        """Apply boundary penalty forces from ground planes and collider shapes."""
        apply_ground_plane_penalty(
            self.model,
            state,
            self._accel,
            self._config.penalty_stiffness,
            self._config.penalty_damping,
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
        n = self.model.particle_count
        wp.launch(
            xsph_correction_kernel,
            dim=n,
            inputs=[
                self._hash_grid.id,
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                state.sph.density,
                self.model.particle_flags,
                self._h,
                self._support_radius,
                self._config.xsph_epsilon,
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
                self._config.reference_density,
                g_mag,
                y_max,
            ],
            outputs=[state.sph.stress],
            device=self.model.device,
        )
