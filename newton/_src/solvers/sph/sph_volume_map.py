# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Volume-map boundary representation for SPH.

Implements the half-space volume map V_B(d) of Bender, Kugelstadt, Weiler
& Koschier (2020, IEEE TVCG) for SPH boundary handling. Restricted to
shapes with analytical signed distance functions — plane, box, sphere —
so that the full 32-node Serendipity precomputation of the original
paper collapses to a single 1-D table parameterised on signed distance.

Provides:

- :func:`compute_volume_map_table` — CPU-side precomputation of the
  V_B(d) integral against the cubic-spline-shaped extension gamma*. Called
  once at solver init; the resulting :class:`warp.array` is broadcast
  to every boundary-pressure / friction-Laplacian kernel launch.
- :func:`sample_volume_map` (``@wp.func``) — runtime piecewise-linear
  interpolation of the V_B(d) table.
- :func:`frisvad_tangent_basis` (``@wp.func``) — branchless orthonormal
  frame from a surface normal, used to place the two tangent surface
  points required by the Bender 2020 friction Laplacian (Eq. 16).

References:
    Bender, J., Kugelstadt, T., Weiler, M., & Koschier, D. (2020).
    Volume Maps: An Implicit Boundary Representation for SPH.
    IEEE Transactions on Visualization and Computer Graphics.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)
_PI = wp.constant(3.14159265358979323846)


# ---------------------------------------------------------------------------
# Volume-map table — runtime sampler
# ---------------------------------------------------------------------------


@wp.func
def sample_volume_map(d: float, r: float, table: wp.array[float]) -> float:
    """Sample the precomputed V_B(d) volume-map table at signed distance d.

    The table covers ``d / r ∈ [-1, 1]`` with uniform spacing. Outside this
    range the value is extrapolated by clamping:

    - ``d ≤ -r``: full kernel-support volume ``(4/3) π r³`` (entire support
      lies inside the solid; the extension gamma* is unity throughout).
    - ``d ≥ r``: zero (entire support lies outside the solid).

    Args:
        d: Signed distance from the query point to the surface [m]. Positive
            outside the solid, negative inside (matches Bender 2020 Eq. 10).
        r: SPH support radius [m].
        table: Precomputed V_B values [m³] from
            :func:`compute_volume_map_table`.

    Returns:
        V_B(d) evaluated by piecewise-linear interpolation [m³].
    """
    n = table.shape[0]
    if r < _EPSILON:
        return 0.0
    if d >= r:
        return 0.0
    if d <= -r:
        return table[0]

    # d / r ∈ [-1, 1] → fractional index in [0, n-1]
    u = 0.5 * (d / r + 1.0)
    fidx = u * float(n - 1)
    i0 = int(wp.floor(fidx))
    if i0 < 0:
        i0 = 0
    i1 = i0 + 1
    if i1 >= n:
        i1 = n - 1
        i0 = n - 2
    t = fidx - float(i0)
    return (1.0 - t) * table[i0] + t * table[i1]


# ---------------------------------------------------------------------------
# Tangent basis — Frisvad orthonormal frame
# ---------------------------------------------------------------------------


@wp.func
def frisvad_tangent_basis(n: wp.vec3):
    """Build a branchless orthonormal tangent basis around a unit normal.

    Implements Frisvad's "Building an Orthonormal Basis from a 3D Unit
    Vector Without Normalization" (J. Graphics Tools 16(3), 2012). The
    branchless form avoids the singularity at ``n = (0, 0, -1)`` that
    naive Gram-Schmidt suffers from, which matters for boundary friction
    points placed on arbitrary box / sphere normals.

    Args:
        n: Unit-length surface normal.

    Returns:
        ``(t1, t2)``, two unit vectors orthogonal to ``n`` and to each
        other. Together with ``n`` they form a right-handed frame.
    """
    if n[2] < -0.9999999:
        # Reverse polarity: handle the south-pole singularity explicitly.
        return wp.vec3(0.0, -1.0, 0.0), wp.vec3(-1.0, 0.0, 0.0)
    a = 1.0 / (1.0 + n[2])
    b = -n[0] * n[1] * a
    t1 = wp.vec3(1.0 - n[0] * n[0] * a, b, -n[0])
    t2 = wp.vec3(b, 1.0 - n[1] * n[1] * a, -n[1])
    return t1, t2


# ---------------------------------------------------------------------------
# CPU-side V_B(d) table generation
# ---------------------------------------------------------------------------


def _gamma_star_cubic(s_over_r: np.ndarray) -> np.ndarray:
    """Cubic-spline extension gamma*(s) of Bender 2020 Eq. 13, parameterised on s/r.

    gamma* is a smooth ramp from 1 at ``s = 0`` to 0 at ``s = r`` whose shape
    matches the cubic-spline kernel profile (so that the extension blends
    into the SPH summation without introducing higher-order kink artefacts
    that the older density-map representation suffered from).

    Piecewise definition with ``q = 2 s / r``:

        s ≤ 0:                gamma* = 1
        s ∈ (0, r/2]  q ≤ 1:  gamma* = 1 - 1.5 q² + 0.75 q³
        s ∈ (r/2, r)  1<q<2:  gamma* = 0.25 (2 - q)³
        s ≥ r:                gamma* = 0

    Args:
        s_over_r: Array of ``s / r`` values (any shape).

    Returns:
        Array of ``gamma*`` values, same shape as input.
    """
    out = np.zeros_like(s_over_r, dtype=np.float64)
    out[s_over_r <= 0.0] = 1.0
    band = (s_over_r > 0.0) & (s_over_r < 1.0)
    q = 2.0 * s_over_r[band]
    inner = q < 1.0
    val = np.empty_like(q)
    val[inner] = 1.0 - 1.5 * q[inner] ** 2 + 0.75 * q[inner] ** 3
    outer = ~inner
    val[outer] = 0.25 * (2.0 - q[outer]) ** 3
    out[band] = val
    return out


def compute_volume_map_table(
    r: float,
    n_samples: int = 256,
    n_quadrature: int = 1024,
    device: wp.context.Device | str | None = None,
) -> wp.array:
    """Precompute the half-space volume map V_B(d) as a 1-D table.

    Specialises Bender 2020 Eq. 12 to a planar boundary, valid because for
    shapes restricted to plane / box / sphere the kernel-support intersection
    with the solid is, to second order in the curvature, a half-space.

    The integral

        V_B(d) = ∫_{-r}^{r} gamma*(d + u) · π (r² - u²) du

    is evaluated by trapezoidal quadrature for each sample of ``d / r ∈ [-1, 1]``.
    The integrand is the cross-sectional area of the kernel-support sphere at
    depth ``u`` along the surface normal, modulated by the cubic-spline
    extension gamma*.

    Args:
        r: SPH support radius [m].
        n_samples: Number of ``d / r`` table entries (default 256). The table
            is sampled inclusively from -1 to 1.
        n_quadrature: Trapezoidal quadrature points for the inner integral
            (default 1024). 1024 points give ≈1e-5 relative accuracy on the
            cubic-spline integrand at single precision.
        device: Warp device for the output array (defaults to the current
            device).

    Returns:
        Warp ``float`` array of length ``n_samples`` holding ``V_B`` values
        [m³]. Index 0 corresponds to ``d / r = -1`` (deepest inside),
        index ``n_samples - 1`` to ``d / r = +1`` (just outside support).

    Notes:
        Sanity bounds:

        - ``V_B(-r) ≈ (4/3) π r³`` (full sphere, gamma* ≡ 1 over support).
        - ``V_B(0)`` slightly above ``(2/3) π r³`` (the inside half-volume
          plus the smoothed gamma* tail through the outside half).
        - ``V_B(+r) = 0``.
    """
    d_over_r = np.linspace(-1.0, 1.0, n_samples, dtype=np.float64)
    u = np.linspace(-r, r, n_quadrature, dtype=np.float64)
    du = u[1] - u[0]
    area = np.pi * (r * r - u * u)

    table = np.empty(n_samples, dtype=np.float32)
    for i, d_r in enumerate(d_over_r):
        d = d_r * r
        gamma = _gamma_star_cubic((d + u) / r)
        integrand = gamma * area
        # Trapezoidal rule on uniform grid.
        table[i] = du * (np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1]))

    return wp.array(table, dtype=float, device=device)
