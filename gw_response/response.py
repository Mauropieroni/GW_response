from __future__ import annotations

import chex
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import field
from typing import TYPE_CHECKING

from jax.typing import ArrayLike

from gw_response.constants import PhysicalConstants
from gw_response.single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    get_single_link_response,
)
from gw_response.space_based.tdi import tdi_matrix

if TYPE_CHECKING:
    from gw_response.detector import Detector


@jax.jit
def linear_response_angular(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Projects the single-link strain response onto a TDI combination, giving
    the (sky-resolved) linear response of that TDI variable.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`gw_response.single_link.get_single_link_response`, with
            shape (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The linear TDI response, with shape (configurations,
            x_vector, TDI, pixels).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # single_link has shape configuration, x_vector, arms, pixels

    # linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)


@jax.jit
def quadratic_response_angular(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the (sky-resolved) quadratic response of a TDI combination,
    i.e. the cross-spectrum of the linear response with its own conjugate,
    summed over polarizations and Hermitian conjugation.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`gw_response.single_link.get_single_link_response`, with
            shape (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The quadratic TDI response, with shape (configurations,
            x_vector, TDI, TDI, pixels).
    """
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
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the sky-resolved integrand later averaged, over the sky, by
    :func:`quadratic_response_integrated` to give the quadratic TDI
    response.

    This is currently a thin wrapper around
    :func:`quadratic_response_angular`.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`gw_response.single_link.get_single_link_response`, with
            shape (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The quadratic response integrand, with shape
            (configurations, x_vector, TDI, TDI, pixels).
    """
    # Defines the integrand using the TDI factors
    return quadratic_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )


@jax.jit
def quadratic_response_integrated(angular_response: ArrayLike) -> jax.Array:
    """
    Averages the sky-resolved quadratic response over the sky (pixels) to
    give the quadratic TDI response as a function of frequency.

    Args:
        angular_response (ArrayLike): Sky-resolved quadratic response, as
            returned by :func:`quadratic_integrand`, with shape
            (configurations, x_vector, TDI, TDI, pixels).

    Returns:
        jax.Array: The sky-averaged quadratic response, with shape
            (configurations, x_vector, TDI, TDI), normalized by ``4 * pi`` to
            account for the solid angle of the sphere.
    """
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)


@chex.dataclass
class Response(object):
    """
    Generic class to handle GW response computations for any Detector (e.g.
    LISA, LIGO).

    Identity-based `__hash__`/`__eq__` (overriding chex's default field-based
    ones, which reject `Response` as unhashable) are what let `self` be used
    as a static argument to `jax.jit` below: every `get_*` method here is a
    pure function of its arguments (`det` is also passed as a static argument
    and resolved via ordinary Python attribute access at trace time), so
    wrapping them individually allows tracing and compiling the whole
    computation for one call as a single XLA program, instead of stitching
    together separately-jitted kernels with Python overhead in between.
    `compute_detector` is intentionally left unjitted since it mutates
    `self`'s dict attributes, and a jitted function's Python-level side
    effects only run once (at trace time) rather than on every call, which
    would silently stop updating them on a cache hit.

    Attributes:
        ps (chex.dataclass): Physical constants used in the response
            computations.
        single_link_response (dict): Cache of the single-link response
            computed by :meth:`compute_detector`, keyed by polarization
            letter.
        linear_integrand (dict): Cache of the linear response integrand
            computed by :meth:`compute_detector`, keyed by combination name.
        quadratic_integrand (dict): Cache of the sky-resolved quadratic
            response computed by :meth:`compute_detector`, keyed by
            combination name.
        quadratic_integrated (dict): Cache of the sky-averaged quadratic
            response computed by :meth:`compute_detector`, keyed by
            combination name.
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
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        polarization: str = "LR",
    ) -> dict[str, jax.Array]:
        """
        Computes the per-link, per-pixel strain response to a GW arriving
        from (theta_array, phi_array), for each polarization mode. The
        underlying Michelson-link physics is the same regardless of detector
        geometry (LISA's 6 arms or LIGO's Michelson arms alike), so this
        isn't delegated to `det` at all.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".

        Returns:
            dict: A dictionary mapping each polarization letter (e.g. "L"
                and "R") to its single-link strain response, with shape
                (configurations, x_vector, arms, pixels).

        Raises:
            ValueError: If ``polarization`` is not "LR" or "PC".
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
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Computes the linear integrand for a GW arriving from
        (theta_array, phi_array), for each polarization, projected onto a
        readout combination.

        Builds the single-link response and the combination matrix
        internally, then applies `det.linear_response_from_single_link` to
        project the former onto the latter (`combination` is a readout
        combination name, e.g. a TDI variable such as "XYZ"/"AET" for LISA,
        or "Michelson" for LIGO -- it defaults to `det.default_combination`).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            dict: A dictionary mapping each polarization letter to the
                linear response integrand, with shape (configurations,
                x_vector, channels, pixels).
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
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Computes the sky-resolved quadratic response integrand for a GW
        arriving from (theta_array, phi_array), for each polarization,
        projected onto a readout combination.

        Builds the linear integrand internally (see
        :meth:`get_linear_integrand`), then applies
        `det.quadratic_response_from_single_link` to it.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            dict: A dictionary mapping each doubled polarization letter
                (e.g. "LL", "RR") to the sky-resolved quadratic response,
                with shape (configurations, x_vector, channels, channels,
                pixels).
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
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Computes the sky-averaged quadratic response for a GW arriving from
        (theta_array, phi_array), for each polarization, projected onto a
        readout combination.

        Builds the quadratic integrand internally (see
        :meth:`get_quadratic_integrand`), then applies
        `det.integrate_quadratic_response` to it.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            dict: A dictionary mapping each doubled polarization letter
                (e.g. "LL", "RR") to the sky-averaged quadratic response,
                with shape (configurations, x_vector, channels, channels).
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
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        combination: str | None = None,
        polarization: str = "LR",
    ) -> None:
        """
        Computes and caches the full response chain (single-link, linear
        integrand, quadratic integrand, and sky-averaged quadratic response)
        for a given readout combination.

        Results are stored in :attr:`single_link_response`,
        :attr:`linear_integrand`, :attr:`quadratic_integrand`, and
        :attr:`quadratic_integrated` (the latter three keyed by
        ``combination``).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".
        """
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
