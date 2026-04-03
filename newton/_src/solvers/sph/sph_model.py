# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH model wrapper.

Augments a :class:`newton.Model` with SPH-specific derived data such as
initial particle volumes and boundary geometry references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import newton

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_volumes_kernel(
    mass: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    reference_density: float,
    # output
    volume: wp.array(dtype=float),
):
    """Compute particle volume from mass and reference density: V_i = m_i / rho_0."""
    i = wp.tid()
    m = mass[i]
    if inv_mass[i] > 0.0 and reference_density > 0.0:
        volume[i] = m / reference_density
    else:
        volume[i] = 0.0


class SPHModel:
    """Wraps :class:`newton.Model` with SPH-specific derived data.

    Computes initial particle volumes from mass and reference density.
    Caches references to ground-plane shapes for boundary force computation.

    Args:
        model: Newton model with particles and shapes.
        reference_density: Reference density rho_0 [kg/m^3].
    """

    def __init__(self, model: newton.Model, reference_density: float):
        self.model = model
        self.reference_density = reference_density

        n = model.particle_count
        self.particle_volume = wp.zeros(n, dtype=float, device=model.device)

        if n > 0:
            wp.launch(
                _compute_volumes_kernel,
                dim=n,
                inputs=[
                    model.particle_mass,
                    model.particle_inv_mass,
                    reference_density,
                ],
                outputs=[self.particle_volume],
                device=model.device,
            )
