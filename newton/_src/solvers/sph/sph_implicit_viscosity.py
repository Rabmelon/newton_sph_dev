# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free conjugate-gradient solver for the implicit boundary friction step.

Solves the linear system

    (I - dt · diag(μ_B / rho_i) · L_B) v^{n+1} = v*

where ``L_B`` is the boundary-friction SPH Laplacian operator (Bender et
al. 2020 Eqs. 16, 18, 19) restricted to a fluid particle's four
boundary surface points (closest point + tangent samples). For interior
particles with no nearby boundary, ``L_B v = 0`` and the operator
reduces to the identity, so the CG iteration converges essentially
through near-boundary rows alone.

Why boundary-only implicit (rather than full Bender 2020 bulk +
boundary): Newton SPH applies *explicit* artificial viscosity for bulk
dissipation in :func:`~newton._src.solvers.sph.sph_kernels.compute_artificial_viscosity_kernel`.
Adding an implicit bulk Laplacian on top would double-count dissipation
and inflate scope; restricting the implicit step to boundary friction
matches the user-facing decree "implicit frictional boundary".

The apply-operator delegates the per-shape Laplacian contribution to
kernels in :mod:`sph_volume_map_kernels` (one kernel per shape kind,
launched once per cached boundary shape). This file contains only the
PCG glue: vector operations on ``wp.array[wp.vec3]`` arrays, the
identity-init step, and the iteration driver.

References:
    - Bender, J., et al. (2020). Volume Maps. IEEE TVCG.
    - Weiler, M., Koschier, D., Brand, M., & Bender, J. (2018).
      A Physically Consistent Implicit Viscosity Solver for SPH Fluids.
      Computer Graphics Forum.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from .sph_dummy_boundary import SPH_FLUID

wp.set_module_options({"enable_backward": False})

_EPSILON = wp.constant(1.0e-8)
_EPSILON_HOST = 1.0e-8


# ---------------------------------------------------------------------------
# Vector ops on wp.array[wp.vec3] arrays — fluid-only iteration support
# ---------------------------------------------------------------------------


@wp.kernel
def vec3_copy_kernel(
    src: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    # output
    dst: wp.array[wp.vec3],
):
    """``dst[i] = src[i]`` for fluid particles; non-fluid rows zeroed.

    Used to initialise CG state vectors from the velocity guess. Non-fluid
    rows are not part of the linear system; zeroing them keeps subsequent
    inner products clean.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        dst[i] = wp.vec3(0.0)
        return
    dst[i] = src[i]


@wp.kernel
def vec3_axpy_kernel(
    alpha: float,
    x: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    # in/out
    y: wp.array[wp.vec3],
):
    """``y[i] += alpha * x[i]`` for fluid particles."""
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        return
    y[i] = y[i] + alpha * x[i]


@wp.kernel
def vec3_xpay_kernel(
    alpha: float,
    x: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    # in/out
    y: wp.array[wp.vec3],
):
    """``y[i] = x[i] + alpha * y[i]`` for fluid particles (CG search-direction update)."""
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        return
    y[i] = x[i] + alpha * y[i]


@wp.kernel
def vec3_dot_kernel(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    # output (1-element scalar accumulator)
    out: wp.array[float],
):
    """Reduce ``Σ_i a[i] · b[i]`` over fluid particles into ``out[0]``."""
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        return
    s = wp.dot(a[i], b[i])
    wp.atomic_add(out, 0, s)


@wp.kernel
def apply_identity_kernel(
    x: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_type: wp.array[wp.int32],
    # output
    y: wp.array[wp.vec3],
):
    """``y[i] = x[i]`` (identity contribution to the apply-operator).

    Per-shape friction-Laplacian kernels then accumulate the negative
    boundary contribution ``-dt · (μ_B / rho_i) · L_B[x]_i`` into ``y[i]``.
    """
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_type[i] != SPH_FLUID:
        y[i] = wp.vec3(0.0)
        return
    y[i] = x[i]


# ---------------------------------------------------------------------------
# PCG driver
# ---------------------------------------------------------------------------


class ImplicitFrictionSolver:
    """Conjugate-gradient driver for ``A v = v*`` with friction-only A.

    The driver is matrix-free: it never assembles ``A`` explicitly,
    instead delegating the apply-operator to a caller-provided callback.
    This keeps the per-shape friction Laplacian dispatch decoupled from
    the iteration mechanics.

    Args:
        n: Number of particles (used to size scratch arrays).
        device: Warp device for scratch storage.
        max_iter: Maximum CG iterations before giving up.
        tol: Stopping criterion ``||r||² / max(||b||², ε) < tol²``.

    Usage:
        Construct once at solver init. Call :meth:`solve` from inside
        each integration substep, passing an ``apply_op`` callable that
        performs ``y = A x`` (typically launching identity + per-shape
        friction Laplacian kernels in sequence).
    """

    def __init__(
        self,
        n: int,
        device,
        max_iter: int = 100,
        tol: float = 1.0e-3,
    ) -> None:
        self.n = n
        self.device = device
        self.max_iter = max_iter
        self.tol = tol

        # CG scratch vectors — sized for the full particle array; non-fluid
        # rows are written to zero by the kernels above.
        self._r = wp.zeros(n, dtype=wp.vec3, device=device)
        self._p = wp.zeros(n, dtype=wp.vec3, device=device)
        self._q = wp.zeros(n, dtype=wp.vec3, device=device)
        self._tmp = wp.zeros(n, dtype=wp.vec3, device=device)
        # Single-element scalar accumulator for inner products.
        self._scalar = wp.zeros(1, dtype=float, device=device)

        # Iteration count from the most recent solve, exposed for diagnostics.
        self.last_iter_count: int = 0
        self.last_residual: float = 0.0

    def _dot(
        self,
        a: wp.array,
        b: wp.array,
        particle_flags: wp.array,
        particle_type: wp.array,
    ) -> float:
        """Compute ``Σ_i a[i] · b[i]`` over fluid particles (host scalar)."""
        self._scalar.zero_()
        wp.launch(
            vec3_dot_kernel,
            dim=self.n,
            inputs=[a, b, particle_flags, particle_type],
            outputs=[self._scalar],
            device=self.device,
        )
        return float(self._scalar.numpy()[0])

    def solve(
        self,
        x: wp.array,
        b: wp.array,
        particle_flags: wp.array,
        particle_type: wp.array,
        apply_op,
    ) -> int:
        """Solve ``A x = b`` in-place; initial guess is zeroed.

        The initial guess is forced to zero rather than warm-started from
        ``b`` (the tentative-velocity ``v*``). With a warm-start at ``x = b``,
        a correction smaller than ``tol * ||b||`` would leave ``x`` equal to
        ``v*`` — i.e. the friction correction would be silently discarded
        whenever the per-step relative residual falls below ``tol``. Forcing
        ``x_0 = 0`` makes ``r_0 = b`` and CG always iterates at least once;
        for the trivial interior case ``A = I`` the exact solution
        ``x = b`` is reached in a single iteration.

        Args:
            x: Output velocity array (``wp.array[wp.vec3]``, length ``n``).
                Caller passes ``state.particle_qd``; on exit it holds the
                CG solution. Whatever ``x`` contains on entry is overwritten.
            b: Right-hand side ``v*`` (``wp.array[wp.vec3]``).
            particle_flags: Particle activity flags.
            particle_type: Particle type array.
            apply_op: Callable ``apply_op(x, y)`` that computes
                ``y[i] = (A x)[i]`` for every fluid particle. Caller
                composes it from the identity kernel + per-shape
                friction-Laplacian kernels.

        Returns:
            Number of CG iterations performed (also stored on
            ``self.last_iter_count``).
        """
        n = self.n
        device = self.device

        # x_0 = 0 (zero initial guess, see docstring rationale).
        x.zero_()
        # With x_0 = 0, A x_0 = 0 and so r_0 = b - A x_0 = b directly.
        wp.launch(
            vec3_copy_kernel,
            dim=n,
            inputs=[b, particle_flags, particle_type],
            outputs=[self._r],
            device=device,
        )

        # p_0 = r_0
        wp.launch(
            vec3_copy_kernel,
            dim=n,
            inputs=[self._r, particle_flags, particle_type],
            outputs=[self._p],
            device=device,
        )

        bb = self._dot(b, b, particle_flags, particle_type)
        bb = max(bb, _EPSILON_HOST)
        rr = self._dot(self._r, self._r, particle_flags, particle_type)

        target = self.tol * self.tol * bb

        if rr <= target:
            self.last_iter_count = 0
            self.last_residual = (rr / bb) ** 0.5
            return 0

        for k in range(self.max_iter):
            apply_op(self._p, self._q)
            pq = self._dot(self._p, self._q, particle_flags, particle_type)
            if abs(pq) < _EPSILON_HOST:
                # System is degenerate or the search direction is zero;
                # bail rather than dividing by ε and producing NaNs.
                break
            alpha = rr / pq

            wp.launch(
                vec3_axpy_kernel,
                dim=n,
                inputs=[alpha, self._p, particle_flags, particle_type],
                outputs=[x],
                device=device,
            )
            wp.launch(
                vec3_axpy_kernel,
                dim=n,
                inputs=[-alpha, self._q, particle_flags, particle_type],
                outputs=[self._r],
                device=device,
            )

            rr_new = self._dot(self._r, self._r, particle_flags, particle_type)
            if rr_new <= target:
                self.last_iter_count = k + 1
                self.last_residual = (rr_new / bb) ** 0.5
                return k + 1

            beta = rr_new / max(rr, _EPSILON_HOST)
            wp.launch(
                vec3_xpay_kernel,
                dim=n,
                inputs=[beta, self._r, particle_flags, particle_type],
                outputs=[self._p],
                device=device,
            )
            rr = rr_new

        self.last_iter_count = self.max_iter
        self.last_residual = (rr / bb) ** 0.5
        return self.max_iter
