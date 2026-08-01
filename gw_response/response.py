from __future__ import annotations

import chex
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import field
from typing import Callable, TYPE_CHECKING

from jax.typing import ArrayLike

from gw_response.constants import PhysicalConstants
from gw_response.single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    get_single_link_response,
    get_single_link_response_retarded,
)
from gw_response.space_based.tdi import tdi_matrix
from gw_response.FFT_utils import (
    strain_to_frequency_domain,
    frequency_domain_to_time_domain,
    fft_positive_time_and_freqs,
    spectral_derivative,
)

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


# Single-link arm labels and their (receiver, emitter) satellite indices, for
# a LISA-like 3-satellite, one-way-laser-link constellation. Matches
# gw_response.utils.arms_matrix_from_vertex_positions's fixed arm ordering
# and vertex-index convention (satellites 1, 2, 3): for arm label "ij" (a
# two-digit int with digits i, j), the arm vector points from satellite i to
# satellite j, and -- as validated against the independent `lisagwresponse`
# package -- satellite j is the emitter and satellite i is the receiver.
_SINGLE_LINK_ARM_LABELS = (12, 23, 31, 21, 32, 13)


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
    def get_single_link_response_fd(
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
    def get_linear_integrand_fd(
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

        single_link = self.get_single_link_response_fd(
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
    def get_quadratic_integrand_fd(
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
        :meth:`get_linear_integrand_fd`), then applies
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
        linear = self.get_linear_integrand_fd(
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
    def get_quadratic_integrated_fd(
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
        :meth:`get_quadratic_integrand_fd`), then applies
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
        quadratic = self.get_quadratic_integrand_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )
        return det.integrate_quadratic_response(quadratic)

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("combination",))
    def get_response_frozen_td(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        h_plus: jax.Array,
        h_cross: jax.Array,
        dt: ArrayLike,
        combination: str | None = None,
    ) -> jax.Array:
        """
        Projects a deterministic plus/cross waveform onto a detector's
        readout channel(s), returning the real time-domain response.

        The detector is treated as frozen at `time_in_years` (exact for
        LIGO; a good approximation for LISA if the signal is short compared
        to the constellation's orbital motion). `time_in_years`/`theta`/`phi`
        may hold several configurations/sky positions at once, as in
        :meth:`get_linear_integrand_fd`. `h_plus`/`h_cross` are FFT'd,
        multiplied by the transfer function from :meth:`get_linear_integrand_fd`,
        summed over polarizations, and IFFT'd back. The corresponding time
        axis, if needed, is ``jnp.arange(n) * dt`` where ``n =
        h_plus.shape[-1]``.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            time_in_years (ArrayLike): Time(s), in years, at which the
                detector configuration is frozen.
            theta (ArrayLike): Colatitude(s) of the sky position(s) the
                signal arrives from, in radians.
            phi (ArrayLike): Longitude(s) of the sky position(s) the signal
                arrives from, in radians.
            h_plus (ArrayLike): Plus-polarization time-domain waveform,
                uniformly sampled with spacing `dt`, with shape (time,).
            h_cross (ArrayLike): Cross-polarization time-domain waveform,
                uniformly sampled with spacing `dt`, with shape (time,).
            dt (ArrayLike): Sample spacing of `h_plus`/`h_cross`, in seconds.
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            jax.Array: The real time-domain readout, with shape
                (configurations, channels, pixels, time) for multi-channel
                combinations (e.g. LISA's `XYZ`/`AET`), or (configurations,
                pixels, time) for single-channel ones (e.g. LIGO's
                `Michelson`).
        """
        n = h_plus.shape[-1]
        frequency_array = jnp.fft.rfftfreq(n, d=dt)

        h_f_plus = strain_to_frequency_domain(h_plus, dt)
        h_f_cross = strain_to_frequency_domain(h_cross, dt)

        linear = self.get_linear_integrand_fd(
            det,
            time_in_years,
            theta,
            phi,
            frequency_array,
            polarization="PC",
            combination=combination,
        )

        # linear["P"/"C"] is (configurations, frequency, [channels,] pixels); einsum
        # contracts with h_f_plus/h_f_cross and moves frequency to the last axis
        signal_plus = jnp.einsum("cf...,f->c...f", linear["P"], h_f_plus)
        signal_cross = jnp.einsum("cf...,f->c...f", linear["C"], h_f_cross)
        return frequency_domain_to_time_domain(signal_plus + signal_cross, n, dt)

    def _combine_modes(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        theta: ArrayLike,
        phi: ArrayLike,
        frequency_array: jax.Array,
        h_t_plus: jax.Array,
        h_t_cross: jax.Array,
        combination: str | None,
    ) -> jax.Array:
        """
        Shared tail end of the time-domain methods. `frequency_array`,
        `h_t_plus`, `h_t_cross` all have shape (modes, time): the
        (already-combined) transfer function from :meth:`get_linear_integrand_fd`
        is evaluated at each sample's own time and its (possibly several
        simultaneous) frequencies (`jax.vmap` over time
        samples, passing the whole per-sample mode vector through in one
        call), and the real response is read off directly in the time
        domain, ``Re[sum_modes R_plus(t) * h_t_plus(t) + R_cross(t) *
        h_t_cross(t)]``.
        """

        def linear_at_sample(
            time_in_years: ArrayLike, frequency_modes: jax.Array
        ) -> dict[str, jax.Array]:
            return self.get_linear_integrand_fd(
                det,
                time_in_years,
                theta,
                phi,
                frequency_modes,
                polarization="PC",
                combination=combination,
            )

        linear = jax.vmap(linear_at_sample, in_axes=(0, 1))(
            times_in_years, frequency_array
        )
        # linear["P"/"C"] is (time, configs=1, freq=modes, [channels,], pixels=1);
        # squeeze drops configs/pixels then einsum contracts with h_t (modes, time).
        R_plus = jnp.squeeze(linear["P"], axis=(1, -1))
        R_cross = jnp.squeeze(linear["C"], axis=(1, -1))
        signal_plus = jnp.einsum("tm...,mt->...t", R_plus, h_t_plus)
        signal_cross = jnp.einsum("tm...,mt->...t", R_cross, h_t_cross)
        return jnp.real(signal_plus + signal_cross)

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("combination",))
    def get_response_spectral_td(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        theta: ArrayLike,
        phi: ArrayLike,
        h_plus: jax.Array,
        h_cross: jax.Array,
        combination: str | None = None,
    ) -> jax.Array:
        """
        Like :meth:`get_response_frozen_td`, but evaluates the detector's
        actual, evolving configuration at every sample instead of freezing
        it at one configuration -- so it captures orbital modulation over
        the signal's duration, and matches :meth:`get_response_frozen_td`'s
        output whenever the detector's configuration doesn't change
        appreciably over `times_in_years`' span (e.g. LIGO's static
        geometry, exactly; or a short-duration/narrowband case for LISA).

        `h_plus`/`h_cross` are re-expressed as their complex analytic signal
        (:func:`gw_response.FFT_utils.fft_positive_time_and_freqs`, an exact, lossless
        Hilbert-transform construction: ``h_plus(t) = Re[h_t_plus(t)]``),
        giving a local  amplitude/phase with no separate FFT of
        the detector response -- unlike :meth:`get_response_frozen_td`,
        which FFTs `h_plus`/`h_cross` into `h_f_plus`/`h_f_cross` and
        multiplies by one frozen transfer function, here the (already-
        combined, e.g. TDI/Michelson) transfer function from
        :meth:`get_linear_integrand_fd` is evaluated at each sample's own time
        and  frequency (`jax.vmap` over samples, so cost is
        linear in the number of samples), and the response is read off
        directly in the time domain (``Re[R(t) * h_t_plus(t)]``).

        `h_plus`/`h_cross` may hold several simultaneous, independently-
        tracked modes (e.g. well-separated tones or harmonics) instead of a
        single (time,) waveform, with shape (modes, time) -- each mode's
        Hilbert-transform envelope and  frequency are computed independently (the FFT in
        :func:`gw_response.FFT_utils.fft_positive_time_and_freqs`
        acts along the last axis only), then each mode's contribution is
        projected through the transfer function at its own frequency and
        summed, relying on the detector response being linear in the strain.
        This is why multiple known frequencies should be passed as separate
        modes rather than summed into one `h_plus` array first: the analytic
        signal of a sum of well-separated tones doesn't decompose back into
        their individual  frequencies.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, uniformly spaced,
                matching `h_plus`/`h_cross`'s sample grid.
            theta (ArrayLike): Colatitude of the single sky position the
                signal arrives from, in radians.
            phi (ArrayLike): Longitude of the single sky position the signal
                arrives from, in radians.
            h_plus (ArrayLike): Real plus-polarization time-domain waveform,
                with shape (time,), or (modes, time) for several
                simultaneous modes.
            h_cross (ArrayLike): Real cross-polarization time-domain
                waveform, with shape (time,), or (modes, time) for several
                simultaneous modes.
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            jax.Array: The real time-domain readout, with shape (channels,
                time) for multi-channel combinations (e.g. LISA's `XYZ`/
                `AET`), or (time,) for single-channel ones (e.g. LIGO's
                `Michelson`).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky
                position.
        """
        if jnp.atleast_1d(theta).shape[0] != 1 or jnp.atleast_1d(phi).shape[0] != 1:
            raise ValueError("get_response_spectral_td requires a single sky position.")

        dt = (times_in_years[1] - times_in_years[0]) * self.ps.yr

        # _combine_modes expects a (modes, time) convention; a bare
        # (time,) input is just the single-mode case. Each mode's Hilbert
        # transform is computed independently (fft_positive_time_and_freqs
        # and spectral_derivative both act along axis=-1 and broadcast over
        # any leading axis), so modes never mix.
        h_t_plus = fft_positive_time_and_freqs(jnp.atleast_2d(h_plus))
        h_t_cross = fft_positive_time_and_freqs(jnp.atleast_2d(h_cross))
        frequency_array = jnp.imag(spectral_derivative(h_t_plus, dt) / h_t_plus) / (
            2 * jnp.pi
        )

        return self._combine_modes(
            det,
            times_in_years,
            theta,
            phi,
            frequency_array,
            h_t_plus,
            h_t_cross,
            combination,
        )

    @partial(jax.jit, static_argnums=(0, 1, 5, 6, 7), static_argnames=("combination",))
    def get_response_autodiff_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        amplitude_plus: Callable[[ArrayLike], jax.Array],
        amplitude_cross: Callable[[ArrayLike], jax.Array],
        phase: Callable[[ArrayLike], jax.Array],
        combination: str | None = None,
    ) -> jax.Array:
        """
        Like :meth:`get_response_spectral_td`, but for a waveform
        given as a genuine function of time rather than a fixed array of
        samples: ``h_plus(t) = amplitude_plus(t) * cos(phase(t))``,
        ``h_cross(t) = amplitude_cross(t) * sin(phase(t))``, the standard
        quadrature decomposition (e.g. a compact-binary waveform's
        amplitude/phase). `amplitude_plus`/`amplitude_cross`/`phase` may
        each return a vector instead of a scalar, to hold several
        simultaneous modes (e.g. GW harmonics) per time sample, each with
        its own  frequency -- matching
        :meth:`get_response_spectral_td`'s (modes, time)
        convention for array input. The  frequency is ``(1 / 2
        pi) * d(phase)/dt``, computed with one `jax.jvp` per time sample
        (exact autodiff for the whole mode vector at once, in a single
        forward pass, since the input is a scalar time) instead of
        :func:`gw_response.FFT_utils._frequency`'s spectral
        differentiation -- exact for any sample spacing, with no
        periodicity/leakage caveats, since there is no FFT anywhere in this
        method.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the response.
            theta (ArrayLike): Colatitude of the single sky position the
                signal arrives from, in radians.
            phi (ArrayLike): Longitude of the single sky position the signal
                arrives from, in radians.
            amplitude_plus (Callable): Maps a time, in seconds, to the
                plus-polarization amplitude(s) (scalar, or one per mode).
            amplitude_cross (Callable): Maps a time, in seconds, to the
                cross-polarization amplitude(s) (scalar, or one per mode).
            phase (Callable): Maps a time, in seconds, to the (shared)
                waveform phase(s), in radians (scalar, or one per mode).
            combination (str, optional): Name of the readout combination.
                Defaults to ``det.default_combination``.

        Returns:
            jax.Array: The real time-domain readout, with shape (channels,
                time) for multi-channel combinations (e.g. LISA's `XYZ`/
                `AET`), or (time,) for single-channel ones (e.g. LIGO's
                `Michelson`).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky
                position.
        """
        if jnp.atleast_1d(theta).shape[0] != 1 or jnp.atleast_1d(phi).shape[0] != 1:
            raise ValueError(
                "get_response_autodiff_td requires a " "single sky position."
            )

        times_in_years = jnp.atleast_1d(times_in_years)
        times_seconds = times_in_years * self.ps.yr

        def phase_and_frequency(
            time_seconds: ArrayLike,
        ) -> tuple[jax.Array, jax.Array]:
            phase_value, phase_dot = jax.jvp(phase, (time_seconds,), (1.0,))
            return jnp.atleast_1d(phase_value), jnp.atleast_1d(phase_dot) / (2 * jnp.pi)

        # out_axes=-1 puts the (modes,) leading axis last directly, matching
        # _combine_modes's (modes, time) convention with no separate
        # moveaxis step.
        phase_values, frequency_array = jax.vmap(phase_and_frequency, out_axes=-1)(
            times_seconds
        )
        amplitude_plus_array = jax.vmap(
            lambda t: jnp.atleast_1d(amplitude_plus(t)), out_axes=-1
        )(times_seconds)
        amplitude_cross_array = jax.vmap(
            lambda t: jnp.atleast_1d(amplitude_cross(t)), out_axes=-1
        )(times_seconds)

        h_t_plus = amplitude_plus_array * jnp.exp(1j * phase_values)
        h_t_cross = amplitude_cross_array * jnp.exp(1j * (phase_values - jnp.pi / 2))

        return self._combine_modes(
            det,
            times_in_years,
            theta,
            phi,
            frequency_array,
            h_t_plus,
            h_t_cross,
            combination,
        )

    @partial(
        jax.jit,
        static_argnums=(0, 1, 5, 6, 7),
        static_argnames=("freeze_geometry",),
    )
    def get_single_link_response_delay_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        amplitude_plus: Callable[[ArrayLike], jax.Array],
        amplitude_cross: Callable[[ArrayLike], jax.Array],
        phase: Callable[[ArrayLike], jax.Array],
        times_geometry_years: ArrayLike | None = None,
        freeze_geometry: bool = False,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1j,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) time-domain response for a LISA-like,
        one-way-laser-link constellation, computed by directly evaluating
        the waveform at each link's retarded emission/reception times and
        differencing -- ``h(t_emission) - h(t_reception)`` -- rather than
        building a frequency-domain transfer function.

        Unlike :meth:`get_response_spectral_td`/
        :meth:`get_response_autodiff_td`, this makes no narrowband
        (single instantaneous-frequency-per-sample) approximation: emitter
        and receiver positions are evaluated at their own (different) true
        retarded times via `det.vertex_positions`, so the result is exact
        for arbitrarily evolving geometry, not just a good approximation.
        Cross-checked against the independent `lisagwresponse` package
        (https://gitlab.in2p3.fr/lisa-simulation/gw-response) for both
        frozen and genuinely time-evolving LISA geometry, agreeing to
        floating-point precision there (once translated into that
        package's conventions, via `wavevector_sign`/`final_factor` below);
        see ``examples/compare_with_lisagwresponse.ipynb``.

        This is deliberately narrower in scope than
        :meth:`get_response_spectral_td`/`_autodiff`: it returns
        the raw per-link response (no TDI/Michelson `combination`), doesn't
        support multiple simultaneous modes, and its retarded-time-delay
        derivation assumes a one-way inter-satellite link topology (LISA,
        Taiji) -- it is not applicable to ground-based, round-trip Michelson
        detectors (LIGO, CE, ET).

        `amplitude_plus`/`amplitude_cross`/`phase` follow
        :meth:`get_response_autodiff_td`'s convention:
        ``h_plus(t) = amplitude_plus(t) * cos(phase(t))``, ``h_cross(t) =
        amplitude_cross(t) * sin(phase(t))``. The default `final_factor`
        of ``1j`` (matching this class's fractional-frequency observable
        convention -- effectively a time derivative of phase under this
        codebase's plane-wave sign convention) is what makes the result
        match :meth:`get_single_link_response_fd`'s own convention exactly
        for frozen geometry.

        `wavevector_sign`/`final_factor` exist to reproduce other codes'
        conventions without duplicating this method: `lisagwresponse`
        (https://gitlab.in2p3.fr/lisa-simulation/gw-response) is matched by
        passing `wavevector_sign=-1.0` (its wave propagation vector points
        *from* the sky position *to* the detector, antiparallel to this
        method's own `unit_vec(theta, phi)`) and `final_factor=-jnp.sqrt(2.0)`
        (a real factor -- no `i` this time, a different residual of the
        same tensor-normalization/observable-convention differences that
        give :meth:`get_single_link_response_frozen_td` a complex factor of
        `√2 i` instead, for this differently-derived formula); see
        ``examples/compare_with_lisagwresponse.ipynb`` for the derivation.

        Args:
            det (Detector): The detector (e.g. LISA, Taiji) the response is
                computed for.
            times_in_years (ArrayLike): Reception time(s), in years, at
                which to evaluate the response.
            theta (ArrayLike): Colatitude of the single sky position the
                signal arrives from, in radians.
            phi (ArrayLike): Longitude of the single sky position the
                signal arrives from, in radians.
            amplitude_plus (Callable): Maps a time, in seconds, to the
                plus-polarization amplitude.
            amplitude_cross (Callable): Maps a time, in seconds, to the
                cross-polarization amplitude.
            phase (Callable): Maps a time, in seconds, to the waveform
                phase, in radians.
            times_geometry_years (ArrayLike, optional): Time(s), in years,
                at which to evaluate the *detector geometry* (broadcast
                against `times_in_years`, which always supplies the
                *phase*/reception time). Defaults to `times_in_years`
                itself -- i.e. genuinely evolving geometry. Passing a
                separate, e.g. constant, array is only useful to reproduce
                a frozen-geometry reference built at a fixed instant while
                the waveform phase still evolves.
            freeze_geometry (bool): If True, also evaluates each emitter's
                *position* at the same reception time as the receiver,
                instead of backdating it by the light-travel time -- for
                matching a frozen reference that shares one simultaneous
                geometry for both. The light-travel-time delay in the
                *phase* argument (a genuine retardation effect even for a
                non-moving detector) is still applied either way -- only
                the emitter's position lookup is affected. Leave False for
                genuinely moving geometry, where both the position and the
                phase need the real, backdated emission time.
            wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)`
                before it's used as the wavevector; -1.0 reproduces
                `lisagwresponse`'s antiparallel convention (see above).
            final_factor (ArrayLike): Complex factor applied to the result
                just before taking its real part; see above.

        Returns:
            jax.Array: The real single-link time-domain response, with
                shape (arms=6, time), in arm order
                :data:`_SINGLE_LINK_ARM_LABELS` (12, 23, 31, 21, 32, 13).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky
                position.
        """
        if jnp.atleast_1d(theta).shape[0] != 1 or jnp.atleast_1d(phi).shape[0] != 1:
            raise ValueError(
                "get_single_link_response_delay_td requires a single sky position."
            )

        times_in_years = jnp.atleast_1d(times_in_years)
        times_phase_seconds = times_in_years * self.ps.yr
        times_geometry_years = (
            times_in_years
            if times_geometry_years is None
            else jnp.atleast_1d(times_geometry_years)
        )
        times_geometry_years, times_phase_seconds = jnp.broadcast_arrays(
            times_geometry_years, times_phase_seconds
        )

        u, v = uv_analytical(theta, phi)
        p_plus, p_cross = polarization_tensors_PC(u, v)
        p_plus_mat = p_plus[0]  # (3, 3), single sky position
        p_cross_mat = p_cross[0]
        k = wavevector_sign * unit_vec(theta, phi)[:, 0]  # (3,), single sky position

        def h_t_plus(tau: ArrayLike) -> jax.Array:
            return amplitude_plus(tau) * jnp.exp(1j * phase(tau))

        def h_t_cross(tau: ArrayLike) -> jax.Array:
            return amplitude_cross(tau) * jnp.exp(1j * (phase(tau) - jnp.pi / 2))

        def single_arm_at_sample(
            positions_rec: jax.Array,
            t_geometry_years: ArrayLike,
            t_rec_seconds: ArrayLike,
            label: int,
        ) -> jax.Array:
            receiver_sat, emitter_sat = label // 10, label % 10
            x_rec = positions_rec[:, receiver_sat - 1]
            # Approximate light-travel time from the reception-time arm
            # length, to get a first estimate of the emission time -- LISA's
            # ~8.3 s light travel time is far shorter than the constellation
            # dynamics, so a single (non-iterative) pass is already exact to
            # floating-point precision (validated in the notebook above).
            ltt_approx = (
                jnp.linalg.norm(x_rec - positions_rec[:, emitter_sat - 1])
                / self.ps.light_speed
            )
            if freeze_geometry:
                x_emi = positions_rec[:, emitter_sat - 1]
            else:
                t_emi_years = t_geometry_years - ltt_approx / self.ps.yr
                x_emi = det.vertex_positions(t_emi_years)[0][:, emitter_sat - 1]

            arm_length = jnp.linalg.norm(x_rec - x_emi)
            n_vec = (x_rec - x_emi) / arm_length
            t_emi_seconds = t_rec_seconds - ltt_approx
            t_emi_shifted = t_emi_seconds - jnp.dot(x_emi, k) / self.ps.light_speed
            t_rec_shifted = t_rec_seconds - jnp.dot(x_rec, k) / self.ps.light_speed

            xiplus = n_vec @ p_plus_mat @ n_vec
            xicross = n_vec @ p_cross_mat @ n_vec
            termplus = h_t_plus(t_emi_shifted) - h_t_plus(t_rec_shifted)
            termcross = h_t_cross(t_emi_shifted) - h_t_cross(t_rec_shifted)
            return (termplus * xiplus + termcross * xicross) / (
                2 * (1 - jnp.dot(n_vec, k))
            )

        def all_arms_at_sample(
            t_geometry_years: ArrayLike, t_rec_seconds: ArrayLike
        ) -> jax.Array:
            # The receiver's own position at reception time is the same for
            # every arm (only the emitter differs), so it's looked up once
            # here and shared, rather than re-deriving it 6 times over.
            positions_rec = det.vertex_positions(t_geometry_years)[0]
            return jnp.stack(
                [
                    single_arm_at_sample(
                        positions_rec, t_geometry_years, t_rec_seconds, label
                    )
                    for label in _SINGLE_LINK_ARM_LABELS
                ]
            )

        y_complex = jax.vmap(all_arms_at_sample)(
            times_geometry_years, times_phase_seconds
        )  # (time, arms)
        return jnp.real(y_complex.T * final_factor)  # (arms, time)

    @partial(jax.jit, static_argnums=(0, 1))
    def get_single_link_response_frozen_td(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        h_plus: jax.Array,
        h_cross: jax.Array,
        dt: ArrayLike,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1.0,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) frozen-geometry response: FFTs
        `h_plus`/`h_cross`, multiplies by the finite-arm-length single-link
        transfer function (:func:`gw_response.single_link.get_single_link_response`),
        and IFFTs back. The single-link sibling of
        :meth:`get_response_frozen_td`, stopping before its TDI/
        Michelson combination step.

        `wavevector_sign`/`final_factor` exist to reproduce other codes'
        conventions without duplicating this method: the independent
        `lisagwresponse` package
        (https://gitlab.in2p3.fr/lisa-simulation/gw-response) is matched by
        passing `wavevector_sign=-1.0` (its wave propagation vector points
        *from* the sky position *to* the detector, antiparallel to this
        method's own `unit_vec(theta, phi)`) and
        `final_factor=1j * jnp.sqrt(2.0)` (a factor of `√2` from
        `lisagwresponse`'s unnormalized antenna-pattern convention versus
        this method's unit-normalized polarization tensors, times a
        residual `i` from how each code's convention defines the
        observable -- see ``examples/compare_with_lisagwresponse.ipynb``
        for the derivation).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is
                computed for.
            time_in_years (ArrayLike): Time, in years, at which the
                detector configuration is frozen.
            theta (ArrayLike): Colatitude of the single sky position the
                signal arrives from, in radians.
            phi (ArrayLike): Longitude of the single sky position the
                signal arrives from, in radians.
            h_plus (ArrayLike): Real plus-polarization time-domain
                waveform, with shape (time,).
            h_cross (ArrayLike): Real cross-polarization time-domain
                waveform, with shape (time,).
            dt (ArrayLike): Sample spacing of `h_plus`/`h_cross`, in
                seconds.
            wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)`
                before it's used as the wavevector; -1.0 reproduces
                `lisagwresponse`'s antiparallel convention (see above).
            final_factor (ArrayLike): Complex factor applied to the
                frequency-domain response before the final IFFT; see above.

        Returns:
            jax.Array: The real single-link time-domain response, with
                shape (arms=6, time), in arm order
                :data:`_SINGLE_LINK_ARM_LABELS` (12, 23, 31, 21, 32, 13).
        """
        n = h_plus.shape[-1]
        freqs = jnp.fft.rfftfreq(n, d=dt)
        positions_rescaled = det.vertex_positions(time_in_years) / det.armlength
        arms_matrix_rescaled = det.detector_arms(time_in_years) / det.armlength
        x_vector = det.x(freqs)

        u, v = uv_analytical(theta, phi)
        p_plus, p_cross = polarization_tensors_PC(u, v)
        wavevector = wavevector_sign * unit_vec(theta, phi)

        h_f_plus = strain_to_frequency_domain(h_plus, dt)
        h_f_cross = strain_to_frequency_domain(h_cross, dt)
        P = get_single_link_response(
            p_plus, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )[0, :, :, 0]
        C = get_single_link_response(
            p_cross, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )[0, :, :, 0]
        freq_domain_per_arm = jnp.einsum("fa,f->af", P, h_f_plus) + jnp.einsum(
            "fa,f->af", C, h_f_cross
        )
        freq_domain_per_arm = freq_domain_per_arm * final_factor
        return frequency_domain_to_time_domain(freq_domain_per_arm, n, dt)

    @partial(jax.jit, static_argnums=(0, 1, 5, 6, 7))
    def get_single_link_response_autodiff_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        amplitude_plus: Callable[[ArrayLike], jax.Array],
        amplitude_cross: Callable[[ArrayLike], jax.Array],
        phase: Callable[[ArrayLike], jax.Array],
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1.0,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) narrowband response via autodiff'd
        instantaneous frequency -- the same per-sample, single-instantaneous-
        frequency approximation :meth:`get_response_autodiff_td` makes, but
        single-link, stopping before its TDI/Michelson combination step.
        Exists to quantify that approximation's accuracy against the exact
        :meth:`get_single_link_response_delay_td` for genuinely evolving
        geometry -- see ``examples/compare_with_lisagwresponse.ipynb``.

        `wavevector_sign`/`final_factor`: see
        :meth:`get_single_link_response_frozen_td` -- the same conversion
        factors reproduce `lisagwresponse`'s convention here too, since
        this shares :func:`gw_response.single_link.get_single_link_response`
        with that method.

        Args: see :meth:`get_response_autodiff_td`.

        Returns:
            jax.Array: shape (arms=6, time), arm order
                :data:`_SINGLE_LINK_ARM_LABELS`.
        """
        times_in_years = jnp.atleast_1d(times_in_years)
        times_seconds = times_in_years * self.ps.yr

        u, v = uv_analytical(theta, phi)
        p_plus, p_cross = polarization_tensors_PC(u, v)
        wavevector = wavevector_sign * unit_vec(theta, phi)

        def phase_and_frequency(t_seconds: ArrayLike) -> tuple[jax.Array, jax.Array]:
            phase_value, phase_dot = jax.jvp(phase, (t_seconds,), (1.0,))
            return phase_value, phase_dot / (2 * jnp.pi)

        phase_values, frequency_array = jax.vmap(phase_and_frequency)(times_seconds)

        def single_link_at_sample(
            t_year: ArrayLike, freq: ArrayLike
        ) -> tuple[jax.Array, jax.Array]:
            positions_rescaled = det.vertex_positions(t_year) / det.armlength
            arms_matrix_rescaled = det.detector_arms(t_year) / det.armlength
            x_vector = det.x(jnp.atleast_1d(freq))
            P = get_single_link_response(
                p_plus, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
            )[0, 0, :, 0]
            C = get_single_link_response(
                p_cross, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
            )[0, 0, :, 0]
            return P, C

        P_t, C_t = jax.vmap(single_link_at_sample)(times_in_years, frequency_array)

        h_t_plus = amplitude_plus(times_seconds) * jnp.exp(1j * phase_values)
        h_t_cross = amplitude_cross(times_seconds) * jnp.exp(
            1j * (phase_values - jnp.pi / 2)
        )

        y_complex = P_t * h_t_plus[:, None] + C_t * h_t_cross[:, None]  # (time, arms)
        return jnp.real(y_complex.T * final_factor)  # (arms, time)

    def _per_arm_retarded_geometry(
        self, det: "Detector", time_in_years: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Shared geometry lookup behind
        :meth:`get_single_link_response_frozen_retarded_td`: for each of the 6
        arms, backdates the emitter's position by that arm's own
        (simultaneous-distance-based) light-travel-time estimate, giving
        the inputs :func:`gw_response.single_link.get_single_link_response_retarded`
        needs to treat the arm as genuinely asymmetric rather than static.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: `(arm_vector_retarded,
            light_travel_time, receiver_position)`, each stacked over the 6
            arms in :data:`_SINGLE_LINK_ARM_LABELS` order, with shape
            (vectorial_index (3), arms) for the vectors and (arms,) for the
            light-travel-time.
        """
        positions_rec = det.vertex_positions(time_in_years)[0]  # (xyz, sat)

        arm_vectors, ltts, receivers = [], [], []
        for label in _SINGLE_LINK_ARM_LABELS:
            receiver_sat, emitter_sat = label // 10, label % 10
            x_rec = positions_rec[:, receiver_sat - 1]
            x_emi_simul = positions_rec[:, emitter_sat - 1]
            ltt_approx = jnp.linalg.norm(x_rec - x_emi_simul) / self.ps.light_speed
            t_emi_years = time_in_years - ltt_approx / self.ps.yr
            x_emi_retarded = det.vertex_positions(t_emi_years)[0][:, emitter_sat - 1]
            arm_vectors.append(x_emi_retarded - x_rec)
            ltts.append(ltt_approx)
            receivers.append(x_rec)

        return (
            jnp.stack(arm_vectors, axis=-1),
            jnp.stack(ltts),
            jnp.stack(receivers, axis=-1),
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def get_single_link_response_frozen_retarded_td(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        h_plus: jax.Array,
        h_cross: jax.Array,
        dt: ArrayLike,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1.0,
    ) -> jax.Array:
        """
        Like :meth:`get_single_link_response_frozen_td`, but exact for a
        genuinely moving detector evaluated at one instant, rather than
        assuming a truly static arm: each arm's emitter position is taken
        at its own light-travel-time-retarded instant (`det.vertex_positions`
        queried at a second, arm-specific, earlier time) instead of
        simultaneously with the receiver, via
        :func:`gw_response.single_link.get_single_link_response_retarded`.
        Reduces to :meth:`get_single_link_response_frozen_td`'s output exactly
        whenever the arm is genuinely static (retarded and simultaneous
        emitter positions coincide). Standalone FFT-based building block
        for anyone specifically wanting a frequency-domain, bandwidth-aware
        transfer function with the emission/reception asymmetry handled
        correctly (unlike :meth:`get_single_link_response_frozen_td`, which
        shares one simultaneous geometry for both) -- not used internally
        by :meth:`get_single_link_response_segmented_td`, which instead
        linearizes :meth:`get_single_link_response_delay_td`'s own exact
        formula per segment (see its docstring).

        `wavevector_sign`/`final_factor`: see :meth:`get_single_link_response_frozen_td`
        -- the same conversion factors apply here, unaffected by the
        asymmetric (retarded-emitter) geometry used in this method.

        Args: see :meth:`get_single_link_response_frozen_td`.

        Returns:
            jax.Array: shape (arms=6, time), arm order
                :data:`_SINGLE_LINK_ARM_LABELS`.
        """
        n = h_plus.shape[-1]
        freqs = jnp.fft.rfftfreq(n, d=dt)
        x_vector = det.x(freqs)

        u, v = uv_analytical(theta, phi)
        p_plus, p_cross = polarization_tensors_PC(u, v)
        wavevector = wavevector_sign * unit_vec(theta, phi)

        arm_vector_retarded, ltt, receiver_position = self._per_arm_retarded_geometry(
            det, time_in_years
        )
        arm_vector_retarded_rescaled = arm_vector_retarded[None] / det.armlength
        ltt_rescaled = ltt[None] * self.ps.light_speed / det.armlength
        receiver_positions_rescaled = receiver_position[None] / det.armlength

        h_f_plus = strain_to_frequency_domain(h_plus, dt)
        h_f_cross = strain_to_frequency_domain(h_cross, dt)
        P = get_single_link_response_retarded(
            p_plus,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )[0, :, :, 0]
        C = get_single_link_response_retarded(
            p_cross,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )[0, :, :, 0]
        freq_domain_per_arm = jnp.einsum("fa,f->af", P, h_f_plus) + jnp.einsum(
            "fa,f->af", C, h_f_cross
        )
        freq_domain_per_arm = freq_domain_per_arm * final_factor
        return frequency_domain_to_time_domain(freq_domain_per_arm, n, dt)

    def _per_arm_linearized_geometry(
        self,
        det: "Detector",
        t_ref_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        wavevector_sign: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Shared geometry lookup behind :meth:`get_single_link_response_segmented_td`:
        for each of the 6 arms, computes the retarded-emitter delay
        formula's full set of geometric quantities -- the wavevector-
        projected emission/reception delays (`shift_emi`, `shift_rec`,
        matching :meth:`get_single_link_response_delay_td`'s own) and the
        antenna-pattern factors they're combined with (`xiplus`, `xicross`,
        `denom`) -- at `t_ref_years`, and *all five* of their time
        derivatives at once, via a single `jax.jvp` through the whole
        pipeline (positions, light-travel-time, antenna pattern) per arm.
        That single automatic-differentiation call is what makes this
        exact to first order: it captures every coupling between how the
        light-travel-time, the arm direction, and the wavevector
        projection change together, rather than only correcting the delay
        term (which -- as `examples/compare_with_lisagwresponse.ipynb`
        shows numerically -- leaves a residual that scales *linearly*
        with segment duration, not the quadratically-shrinking residual a
        genuine first-order expansion gives).

        Returns:
            tuple[jax.Array, jax.Array]: `(values, derivatives)`, each
            shape (5, arms) stacked in :data:`_SINGLE_LINK_ARM_LABELS`
            order, rows `(shift_rec, shift_emi, xiplus, xicross, denom)`;
            `derivatives` are with respect to time in seconds.
        """
        u, v = uv_analytical(theta, phi)
        p_plus, p_cross = polarization_tensors_PC(u, v)
        p_plus_mat = p_plus[0]
        p_cross_mat = p_cross[0]
        k = wavevector_sign * unit_vec(theta, phi)[:, 0]

        def geometry_at(t_rec_years: ArrayLike, label: int) -> jax.Array:
            receiver_sat, emitter_sat = label // 10, label % 10
            x_rec = det.vertex_positions(t_rec_years)[0][:, receiver_sat - 1]
            x_emi_simul = det.vertex_positions(t_rec_years)[0][:, emitter_sat - 1]
            ltt_approx = jnp.linalg.norm(x_rec - x_emi_simul) / self.ps.light_speed
            t_emi_years = t_rec_years - ltt_approx / self.ps.yr
            x_emi = det.vertex_positions(t_emi_years)[0][:, emitter_sat - 1]

            shift_rec = jnp.dot(x_rec, k) / self.ps.light_speed
            shift_emi = ltt_approx + jnp.dot(x_emi, k) / self.ps.light_speed
            n_vec = (x_rec - x_emi) / jnp.linalg.norm(x_rec - x_emi)
            xiplus = n_vec @ p_plus_mat @ n_vec
            xicross = n_vec @ p_cross_mat @ n_vec
            denom = 2 * (1 - jnp.dot(n_vec, k))
            return jnp.stack([shift_rec, shift_emi, xiplus, xicross, denom])

        values, derivatives = [], []
        for label in _SINGLE_LINK_ARM_LABELS:
            v_mid, dv_dyear = jax.jvp(
                lambda ty, label=label: geometry_at(ty, label), (t_ref_years,), (1.0,)
            )
            values.append(v_mid)
            derivatives.append(dv_dyear / self.ps.yr)  # d/d(seconds)

        return jnp.stack(values, axis=-1), jnp.stack(derivatives, axis=-1)

    def _single_link_response_linearized(
        self,
        det: "Detector",
        t_ref_years: ArrayLike,
        t_ref_seconds: ArrayLike,
        t_phase_seconds: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        amplitude_plus: Callable[[ArrayLike], jax.Array],
        amplitude_cross: Callable[[ArrayLike], jax.Array],
        phase: Callable[[ArrayLike], jax.Array],
        wavevector_sign: ArrayLike,
        final_factor: ArrayLike,
    ) -> jax.Array:
        """
        Shared computation behind :meth:`get_single_link_response_segmented_td`,
        for one segment: evaluates :meth:`_per_arm_linearized_geometry` once at the
        segment's own `t_ref_years`, then reconstructs each geometric
        quantity as a first-order Taylor expansion around it
        (`value(t) = value(t_ref) + derivative(t_ref) * (t - t_ref)`) at
        every sample in `t_phase_seconds`, and combines them exactly as
        :meth:`get_single_link_response_delay_td` does -- differing only in
        using this linear model instead of a fresh
        `det.vertex_positions` lookup per sample.

        Returns:
            jax.Array: shape (arms=6, time), arm order
                :data:`_SINGLE_LINK_ARM_LABELS`.
        """
        values, derivatives = self._per_arm_linearized_geometry(
            det, t_ref_years, theta, phi, wavevector_sign
        )
        shift_rec_ref, shift_emi_ref, xiplus_ref, xicross_ref, denom_ref = values
        dshift_rec, dshift_emi, dxiplus, dxicross, ddenom = derivatives

        t_phase_seconds = jnp.asarray(t_phase_seconds)
        dt_arr = (t_phase_seconds - t_ref_seconds)[None, :]  # (1, time)
        shift_rec = shift_rec_ref[:, None] + dshift_rec[:, None] * dt_arr
        shift_emi = shift_emi_ref[:, None] + dshift_emi[:, None] * dt_arr
        xiplus = xiplus_ref[:, None] + dxiplus[:, None] * dt_arr
        xicross = xicross_ref[:, None] + dxicross[:, None] * dt_arr
        denom = denom_ref[:, None] + ddenom[:, None] * dt_arr

        t_rec_shifted = t_phase_seconds[None, :] - shift_rec  # (arms, time)
        t_emi_shifted = t_phase_seconds[None, :] - shift_emi

        def h_t_plus(tau: ArrayLike) -> jax.Array:
            return amplitude_plus(tau) * jnp.exp(1j * phase(tau))

        def h_t_cross(tau: ArrayLike) -> jax.Array:
            return -1j * amplitude_cross(tau) * jnp.exp(1j * phase(tau))

        termplus = h_t_plus(t_emi_shifted) - h_t_plus(t_rec_shifted)
        termcross = h_t_cross(t_emi_shifted) - h_t_cross(t_rec_shifted)
        y_complex = (termplus * xiplus + termcross * xicross) / denom
        return jnp.real(y_complex * final_factor)

    @partial(jax.jit, static_argnums=(0, 1, 5, 6, 7, 8))
    def get_single_link_response_segmented_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        amplitude_plus: Callable[[ArrayLike], jax.Array],
        amplitude_cross: Callable[[ArrayLike], jax.Array],
        phase: Callable[[ArrayLike], jax.Array],
        segment_length: int,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1j,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) response for evolving detector
        geometry, via segment-stacking: the full duration is split into
        short chunks, each evaluated as a first-order (linear-in-time)
        expansion of the exact delay formula
        (:meth:`get_single_link_response_delay_td`) around that chunk's own
        midpoint, using one automatic-differentiation call per chunk
        (:meth:`_per_arm_linearized_geometry`) rather than one
        `det.vertex_positions` evaluation per sample. The error from this
        truncation shrinks *quadratically* with `segment_length` (a
        genuine first-order Taylor expansion, not merely "geometry updated
        less often") -- see ``examples/compare_with_lisagwresponse.ipynb``
        for the numerical scaling. Reduces to
        :meth:`get_single_link_response_delay_td`'s own per-sample-exact
        result exactly once `segment_length` is short enough that the
        linear approximation and the true geometry coincide to the
        precision being asked for.

        This error is *local* to each segment's own span and does not
        accumulate across segments: each segment's linearization is built
        fresh from `det.vertex_positions` at that segment's own midpoint,
        never from a neighboring segment's (approximate) output, so the
        error for a given `segment_length` stays the same regardless of
        how many segments the full duration is split into -- confirmed
        numerically in ``examples/compare_with_lisagwresponse.ipynb``
        (same worst-case error from 1 to 30 days at fixed `segment_length`).
        `segment_length` is consequently the only knob trading accuracy for
        speed here, with no separate cost (in either direction) tied to the
        total duration being processed.

        Unlike an FFT-based approach, there's no minimum segment length
        for this to remain valid -- `segment_length = 1` is allowed and
        simply reproduces :meth:`get_single_link_response_delay_td` sample by
        sample (at the cost of the same per-sample `det.vertex_positions`
        evaluations that method already pays for; the computational
        benefit of segmenting only shows up for `segment_length > 1`).

        `wavevector_sign`/`final_factor`: see
        :meth:`get_single_link_response_delay_td` -- the same conversion
        factors reproduce `lisagwresponse`'s convention here too (a real
        `final_factor=-jnp.sqrt(2.0)`, unaffected by the linearization used
        here).

        Args:
            det, theta, phi, amplitude_plus, amplitude_cross, phase: see
                :meth:`get_single_link_response_delay_td`.
            times_in_years (ArrayLike): Time(s), in years, uniformly
                spaced.
            segment_length (int): Number of samples per segment; must
                evenly divide `times_in_years`'s length.
            wavevector_sign (ArrayLike): see
                :meth:`get_single_link_response_delay_td`.
            final_factor (ArrayLike): see
                :meth:`get_single_link_response_delay_td`.

        Returns:
            jax.Array: shape (arms=6, time), arm order
                :data:`_SINGLE_LINK_ARM_LABELS`.

        Raises:
            ValueError: If `segment_length` doesn't evenly divide the
                number of samples.
        """
        times_in_years = jnp.atleast_1d(times_in_years)
        n = times_in_years.shape[-1]
        if n % segment_length != 0:
            raise ValueError(
                f"times_in_years length ({n}) must be a multiple of "
                f"segment_length ({segment_length})."
            )
        n_segments = n // segment_length

        times_seconds = times_in_years * self.ps.yr
        times_years_segments = times_in_years.reshape(n_segments, segment_length)
        times_seconds_segments = times_seconds.reshape(n_segments, segment_length)
        t_ref_years = times_years_segments[:, segment_length // 2]
        t_ref_seconds = times_seconds_segments[:, segment_length // 2]

        def one_segment(
            t_ref_y: ArrayLike, t_ref_s: ArrayLike, t_phase_s: jax.Array
        ) -> jax.Array:
            return self._single_link_response_linearized(
                det,
                t_ref_y,
                t_ref_s,
                t_phase_s,
                theta,
                phi,
                amplitude_plus,
                amplitude_cross,
                phase,
                wavevector_sign,
                final_factor,
            )

        y_segments = jax.vmap(one_segment)(
            t_ref_years, t_ref_seconds, times_seconds_segments
        )  # (segments, arms, segment_length)
        n_arms = y_segments.shape[1]
        return jnp.moveaxis(y_segments, 0, 1).reshape(n_arms, n)

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

        self.single_link_response = self.get_single_link_response_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )

        self.linear_integrand[combination] = self.get_linear_integrand_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )

        self.quadratic_integrand[combination] = self.get_quadratic_integrand_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )

        self.quadratic_integrated[combination] = self.get_quadratic_integrated_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
            combination=combination,
        )
