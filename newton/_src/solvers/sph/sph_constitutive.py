# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH constitutive models for granular materials.

Provides two constitutive models:

1. **Drucker-Prager elastic-plastic** (``dp``): Hooke's law for the elastic
   trial stress increment, followed by return mapping onto the Drucker-Prager
   yield surface with three regimes (elastic, plastic flow, tension cracking).

2. **mu(I) rheology** (``mui``): rate-dependent granular rheology. Computes
   pressure via the Tait equation of state and an effective viscosity from
   the Drucker-Prager friction law, producing a viscous stress tensor.

References:
    - tiSPHi ``solver_sph_dp.py`` — ``adapt_stress()``
    - tiSPHi ``solver_sph_muI.py``
    - Newton MPM ``rheology_solver_kernels.py`` — ``project_stress()``
"""

import warp as wp

from ...geometry import ParticleFlags
from .sph_dummy_boundary import SPH_FLUID

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@wp.func
def drucker_prager_params(friction: float, cohesion: float) -> wp.vec2:
    """Compute Drucker-Prager parameters from friction angle and cohesion.

    Args:
        friction: Internal friction angle phi [rad].
        cohesion: Cohesion c [Pa].

    Returns:
        (alpha_phi, k_c) where:
            alpha_phi = tan(phi) / sqrt(9 + 12*tan(phi)^2)
            k_c = 3*c / sqrt(9 + 12*tan(phi)^2)
    """
    tan_phi = wp.tan(friction)
    denom = wp.sqrt(9.0 + 12.0 * tan_phi * tan_phi)
    alpha_phi = tan_phi / denom
    k_c = 3.0 * cohesion / denom
    return wp.vec2(alpha_phi, k_c)


@wp.func
def deviatoric_stress(sigma: wp.mat33) -> wp.mat33:
    """Compute deviatoric part: s = sigma - (tr(sigma)/3) * I."""
    I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return sigma - (I1 / 3.0) * eye


@wp.func
def mat33_double_contraction(A: wp.mat33, B: wp.mat33) -> float:
    """Compute A:B = sum(A_ij * B_ij)."""
    result = float(0.0)
    for row in range(3):
        for col in range(3):
            result += A[row, col] * B[row, col]
    return result


@wp.func
def hooke_stress_increment(strain_rate: wp.mat33, K: float, G: float, dt: float) -> wp.mat33:
    """Compute elastic stress increment using Hooke's law.

    d_sigma = (2*G*D_dev + K*tr(D)*I) * dt

    Args:
        strain_rate: Strain rate tensor D.
        K: Bulk modulus [Pa].
        G: Shear modulus [Pa].
        dt: Time step [s].

    Returns:
        Stress increment d_sigma.
    """
    tr_D = strain_rate[0, 0] + strain_rate[1, 1] + strain_rate[2, 2]
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D_dev = strain_rate - (tr_D / 3.0) * eye
    return (2.0 * G * D_dev + K * tr_D * eye) * dt


@wp.func
def dp_return_mapping(sigma: wp.mat33, alpha_phi: float, k_c: float) -> wp.mat33:
    """Project stress onto the Drucker-Prager yield surface.

    Three regimes:
        1. Elastic: f_DP <= 0 — no change.
        2. Tension cracking: f_DP >= J2 — project to apex.
        3. Plastic flow: 0 < f_DP < J2 — scale deviatoric part.

    Args:
        sigma: Trial Cauchy stress tensor [Pa].
        alpha_phi: DP friction parameter.
        k_c: DP cohesion parameter.

    Returns:
        Stress projected onto or inside the yield surface.

    Reference:
        tiSPHi ``adapt_stress()`` in ``solver_sph_dp.py``.
    """
    I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    s = sigma - (I1 / 3.0) * eye

    # J2 = sqrt(0.5 * s:s)
    s_norm_sq = mat33_double_contraction(s, s)
    J2 = wp.sqrt(0.5 * s_norm_sq)

    # Yield function: f = J2 + alpha_phi * I1 - k_c
    f_DP = J2 + alpha_phi * I1 - k_c

    if f_DP <= 0.0:
        # Elastic — inside yield surface
        return sigma

    if f_DP >= J2:
        # Tension cracking — project to apex
        # sigma = sigma - (I1 - k_c / alpha_phi) / 3 * I
        if alpha_phi > _EPSILON:
            sigma_new = sigma - ((I1 - k_c / alpha_phi) / 3.0) * eye
        else:
            sigma_new = sigma
        # Re-evaluate after apex projection
        I1_new = sigma_new[0, 0] + sigma_new[1, 1] + sigma_new[2, 2]
        s_new = sigma_new - (I1_new / 3.0) * eye
        s_new_norm_sq = mat33_double_contraction(s_new, s_new)
        J2_new = wp.sqrt(0.5 * s_new_norm_sq)
        f_new = J2_new + alpha_phi * I1_new - k_c
        if f_new <= 0.0:
            return sigma_new
        # Scale deviatoric part on the re-evaluated state
        if J2_new > _EPSILON:
            r = (-alpha_phi * I1_new + k_c) / J2_new
            r = wp.max(r, 0.0)
            return r * s_new + (I1_new / 3.0) * eye
        return (I1_new / 3.0) * eye

    # Plastic flow — scale deviatoric stress so that f_DP = 0
    # r = (-alpha_phi * I1 + k_c) / J2
    if J2 > _EPSILON:
        r = (-alpha_phi * I1 + k_c) / J2
        r = wp.max(r, 0.0)
        return r * s + (I1 / 3.0) * eye
    return (I1 / 3.0) * eye


# ---------------------------------------------------------------------------
# Drucker-Prager stress update kernel
# ---------------------------------------------------------------------------


@wp.kernel
def update_stress_dp_kernel(
    strain_rate: wp.array(dtype=wp.mat33),
    stress_prev: wp.array(dtype=wp.mat33),
    young_modulus: wp.array(dtype=float),
    poisson_ratio: wp.array(dtype=float),
    friction: wp.array(dtype=float),
    cohesion: wp.array(dtype=float),
    dilatancy: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_type: wp.array(dtype=wp.int32),
    dt: float,
    # output (in-place for stress)
    stress_out: wp.array(dtype=wp.mat33),
    pressure_out: wp.array(dtype=float),
    plastic_strain_out: wp.array(dtype=float),
):
    """Update Cauchy stress via Drucker-Prager elastic-plastic model.

    Steps:
        1. Compute elastic moduli K and G from E and nu.
        2. Compute trial stress: sigma_trial = sigma_prev + Hooke(D, K, G, dt).
        3. Apply Drucker-Prager return mapping.
        4. Compute pressure P = -tr(sigma) / 3.
        5. Accumulate equivalent plastic strain.

    Dummy particles are skipped.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    E = young_modulus[i]
    nu = poisson_ratio[i]
    phi = friction[i]
    c = cohesion[i]

    # Elastic moduli
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # Trial stress increment
    D = strain_rate[i]
    d_sigma = hooke_stress_increment(D, K, G, dt)

    # Trial stress
    sigma_trial = stress_prev[i] + d_sigma

    # Drucker-Prager parameters
    dp = drucker_prager_params(phi, c)
    alpha_phi = dp[0]
    k_c = dp[1]

    # Return mapping
    sigma_corrected = dp_return_mapping(sigma_trial, alpha_phi, k_c)

    # Pressure = -tr(sigma) / 3
    I1 = sigma_corrected[0, 0] + sigma_corrected[1, 1] + sigma_corrected[2, 2]
    pressure_out[i] = -I1 / 3.0

    # Equivalent plastic strain increment
    d_sigma_plastic = sigma_trial - sigma_corrected
    s_plastic = deviatoric_stress(d_sigma_plastic)
    d_eps_p_sq = (2.0 / 3.0) * mat33_double_contraction(s_plastic, s_plastic)
    if d_eps_p_sq > 0.0:
        if G > _EPSILON:
            d_eps_p = wp.sqrt(d_eps_p_sq) / (2.0 * G * dt + _EPSILON)
        else:
            d_eps_p = 0.0
        plastic_strain_out[i] = plastic_strain_out[i] + d_eps_p * dt
    stress_out[i] = sigma_corrected


# ---------------------------------------------------------------------------
# mu(I) rheology stress update kernel
# ---------------------------------------------------------------------------


@wp.kernel
def update_stress_mui_kernel(
    strain_rate: wp.array(dtype=wp.mat33),
    density: wp.array(dtype=float),
    friction: wp.array(dtype=float),
    cohesion: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_type: wp.array(dtype=wp.int32),
    reference_density: float,
    sound_speed: float,
    eos_gamma: float,
    smoothing_length: float,
    # output
    stress_out: wp.array(dtype=wp.mat33),
    pressure_out: wp.array(dtype=float),
):
    """Update stress via mu(I) rheology.

    Steps:
        1. Compute pressure via Tait equation of state.
        2. Compute deviatoric strain rate and equivalent strain rate D_equ.
        3. Compute effective viscosity: eta = (c + P * tan(phi)) / D_equ.
        4. Stress: sigma = 2 * eta * D_dev - P * I.
        5. Apply Drucker-Prager return mapping for safety.

    Dummy particles are skipped.

    Reference:
        tiSPHi ``solver_sph_muI.py``.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    rho = density[i]
    phi = friction[i]
    c = cohesion[i]

    D = strain_rate[i]
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Tait equation of state: P = (rho_0 * c_s^2 / gamma) * ((rho/rho_0)^gamma - 1)
    P = float(0.0)
    if rho > _EPSILON and reference_density > _EPSILON:
        ratio = rho / reference_density
        # Use wp.pow for exponentiation
        P = (reference_density * sound_speed * sound_speed / eos_gamma) * (wp.pow(ratio, eos_gamma) - 1.0)
    P = wp.max(P, 0.0)  # no tensile pressure for granular

    # Deviatoric strain rate
    tr_D = D[0, 0] + D[1, 1] + D[2, 2]
    D_dev = D - (tr_D / 3.0) * eye

    # Equivalent strain rate: D_equ = sqrt(2/3 * D_dev:D_dev)
    D_dev_sq = mat33_double_contraction(D_dev, D_dev)
    D_equ = wp.sqrt(2.0 / 3.0 * D_dev_sq)

    # Effective viscosity: eta = (c + P * tan(phi)) / max(D_equ, epsilon)
    tan_phi = wp.tan(phi)
    eta = (c + P * tan_phi) / wp.max(D_equ, _EPSILON)

    # Clamp viscosity to avoid numerical explosion when D_equ → 0 (particles at
    # rest).  User-specified cap takes priority; otherwise fall back to a
    # physically motivated upper bound: eta_max = rho_0 * c_s * h, which is the
    # same order as the artificial-viscosity dissipation scale.
    max_visc = viscosity[i]
    if max_visc <= 0.0:
        max_visc = reference_density * sound_speed * smoothing_length
    eta = wp.min(eta, max_visc)

    # Stress: sigma = 2 * eta * D_dev - P * I
    sigma = 2.0 * eta * D_dev - P * eye

    # Apply DP return mapping for safety
    dp = drucker_prager_params(phi, c)
    sigma = dp_return_mapping(sigma, dp[0], dp[1])

    stress_out[i] = sigma

    I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
    pressure_out[i] = -I1 / 3.0


# ---------------------------------------------------------------------------
# Geostatic stress initialization kernel
# ---------------------------------------------------------------------------


@wp.kernel
def initialize_geostatic_stress_kernel(
    pos: wp.array(dtype=wp.vec3),
    friction: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_type: wp.array(dtype=wp.int32),
    reference_density: float,
    g_mag: float,
    z_max: float,
    # output
    stress: wp.array(dtype=wp.mat33),
):
    """Initialize stress using K0 earth pressure condition.

    sigma_zz = -rho_0 * g * (z_max - z_i)   (compressive = negative)
    sigma_xx = sigma_yy = K0 * sigma_zz
    K0 = 1 - sin(phi)

    Assumes Z-up coordinate system. Dummy particles are skipped.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_type[i] != SPH_FLUID:
        return

    z_i = pos[i][2]
    depth = z_max - z_i
    if depth < 0.0:
        depth = 0.0

    phi = friction[i]
    K0 = 1.0 - wp.sin(phi)

    # Compressive stress is negative in continuum mechanics sign convention
    sigma_zz = -reference_density * g_mag * depth
    sigma_xx = K0 * sigma_zz
    sigma_yy = K0 * sigma_zz

    stress[i] = wp.mat33(
        sigma_xx,
        0.0,
        0.0,
        0.0,
        sigma_yy,
        0.0,
        0.0,
        0.0,
        sigma_zz,
    )
