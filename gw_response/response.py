from __future__ import annotations

import chex
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import field
from typing import TYPE_CHECKING

from .constants import PhysicalConstants
from .single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    get_single_link_response,
)
from .space_based.tdi import tdi_matrix

if TYPE_CHECKING:
    from .detector import Detector


@jax.jit
def linear_response_angular(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # single_link has shape configuration, x_vector, arms, pixels

    # linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)


@jax.jit
def quadratic_response_angular(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    # linear response is configuration, x_vector, TDI, pixels
    linear_response = linear_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )

    # quadratic response is configuration, x_vector, TDI, TDI, pixels
    quadratic_response = jnp.einsum(
        "...ijl,...ikl->...ijkl",
        linear_response,
        jnp.conjugate(linear_response),
    )

    # The first 2 is sum over polarization the second is for the h.c. sum
    return 2 * 2 * quadratic_response / jnp.pi / 4


@jax.jit
def quadratic_integrand(
    TDI_idx,
    single_link,
    arms_matrix_rescaled,
    x_vector,
):
    # Defines the integrand using the TDI factors
    return quadratic_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )


@jax.jit
def quadratic_response_integrated(angular_response):
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)


@chex.dataclass
class Response(object):
    """
    Generic class to handle GW response computations for any Detector (e.g. LISA, LIGO).

    Identity-based `__hash__`/`__eq__` (overriding chex's default field-based ones,
    which reject `Response` as unhashable) are what let `self` be used as a static
    argument to `jax.jit` below: every `get_*` method here is a pure function of its
    arguments (`det` is also passed as a static argument and resolved via ordinary
    Python attribute access at trace time), so wrapping them individually allows to
    trace and compile the whole computation for one call as a single XLA program.
    instead of stitching together instead of stitching together separately-jitted
    kernels with Python overhead in between. `compute_detector` is intentionally left
    unjitted since it mutates `self`'s dict attributes, and a jitted function's
    Python-level side effects only run once (at trace time) rather than on every call,
    which would silently stop updating them on a cache hit.
    """

    ps: PhysicalConstants = PhysicalConstants()
    single_link_response: dict = field(default_factory=dict)
    linear_integrand: dict = field(default_factory=dict)
    quadratic_integrand: dict = field(default_factory=dict)
    quadratic_integrated: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("polarization",))
    def get_single_link_response(
        self,
        det: "Detector",
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
    ) -> dict:
        """
        Computes the per-link, per-pixel response to a GW arriving from
        (theta_array, phi_array), for each polarization mode. The underlying
        Michelson-link physics is the same regardless of detector geometry
        (LISA's 6 arms or LIGO's Michelson arms alike), so this isn't
        delegated to `det` at all.

        Returns a dict keyed by polarization, e.g. {"L": ..., "R": ...}.
        """
        pol = polarization.upper()
        wavevector = unit_vec(theta_array, phi_array)
        u, v = uv_analytical(theta_array, phi_array)

        positions_rescaled = det.vertex_positions(times_in_years) / det.armlength
        arms_matrix_rescaled = det.detector_arms(times_in_years) / det.armlength
        x_vector = det.x(frequency_array)

        if pol == "PC":
            p1, p2 = polarization_tensors_PC(u, v)
        elif pol == "LR":
            p1, p2 = polarization_tensors_LR(u, v)
        else:
            raise ValueError("Incorrect polarization type")

        ppol = {pol[0]: p1, pol[1]: p2}
        return {
            p: get_single_link_response(
                ppol[p], arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
            )
            for p in ppol.keys()
        }

    @partial(
        jax.jit,
        static_argnums=(0, 1),
        static_argnames=("combination", "polarization"),
    )
    def get_linear_integrand(
        self,
        det: "Detector",
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
        combination=None,
    ) -> dict:
        """
        Computes the linear integrand for a GW arriving from
        (theta_array, phi_array), building the single-link response and the
        combination matrix internally (`combination` is a readout
        combination name, e.g. a TDI variable such as "XYZ"/"AET" for LISA,
        or "Michelson" for LIGO -- it defaults to `det.default_combination`).
        """
        combination = combination or det.default_combination

        single_link = self.get_single_link_response(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )
        arms_matrix_rescaled = det.detector_arms(times_in_years) / det.armlength
        x_vector = det.x(frequency_array)
        combination_matrix = det.combination_matrix(
            combination, arms_matrix_rescaled, x_vector
        )

        return det.linear_response_from_single_link(single_link, combination_matrix)

    @partial(
        jax.jit,
        static_argnums=(0, 1),
        static_argnames=("combination", "polarization"),
    )
    def get_quadratic_integrand(
        self,
        det: "Detector",
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
        combination=None,
    ) -> dict:
        """
        Computes the quadratic integrand for a GW arriving from
        (theta_array, phi_array), building the linear integrand internally
        (see `get_linear_integrand`).
        """
        linear = self.get_linear_integrand(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )
        return det.quadratic_response_from_single_link(linear)

    @partial(
        jax.jit,
        static_argnums=(0, 1),
        static_argnames=("combination", "polarization"),
    )
    def get_quadratic_integrated(
        self,
        det: "Detector",
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
        combination=None,
    ) -> dict:
        """
        Computes the sky-integrated quadratic response for a GW arriving
        from (theta_array, phi_array), building the quadratic integrand
        internally (see `get_quadratic_integrand`).
        """
        quadratic = self.get_quadratic_integrand(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )
        return det.integrate_quadratic_response(quadratic)

    def compute_detector(
        self,
        det: "Detector",
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        combination=None,
        polarization="LR",
    ):
        combination = combination or det.default_combination

        self.single_link_response = self.get_single_link_response(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )

        self.linear_integrand[combination] = self.get_linear_integrand(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )

        self.quadratic_integrand[combination] = self.get_quadratic_integrand(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )

        self.quadratic_integrated[combination] = self.get_quadratic_integrated(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )
