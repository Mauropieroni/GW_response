from __future__ import annotations

import chex
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import field
from typing import Any, Callable, TYPE_CHECKING

from jax.typing import ArrayLike

from gw_response.constants import PhysicalConstants
from gw_response.polarization import polarization_tensors_and_signed_wavevector
from gw_response.response_utils import (
    contract_with_h,
    Waveform,
)
from gw_response.single_link_retarded import get_single_link_response_retarded
from gw_response.single_link_utils import get_single_link_response_long_wavelength
from gw_response.space_based.single_link_geometry import (
    per_arm_retarded_geometry_rescaled,
    single_link_response_linearized,
    single_link_response_delay_td,
    tdi_response_delay_td,
)
from gw_response.FFT_utils import (
    strain_to_frequency_domain,
    frequency_domain_to_time_domain,
)

if TYPE_CHECKING:
    from gw_response.detector import Detector


@chex.dataclass
class Response(object):
    """
    Generic class to handle GW response computations for any Detector (e.g. LISA, LIGO).

    Identity-based `__hash__`/`__eq__` let `self` be a static `jax.jit` argument below,
    so each `get_*` method traces to one XLA program instead of stitching together
    separately-jitted kernels. `compute_detector` and the `waveform`-reading single-link
    methods are left unjitted since they read/write mutable attributes (`self`'s dict
    caches, `self.waveform`) that a jitted trace would only touch once, at trace time,
    not on every call.

    Attributes:
        ps (PhysicalConstants): Physical constants used in the response computations.
        waveform (Waveform, optional): Current waveform, used by the single-link methods
            that need values at run-time-determined (retarded) times. Set this before
            calling those methods.
        single_link_response (dict): Cache from `compute_detector`, keyed by
            polarization letter.
        linear_integrand (dict): Cache from `compute_detector`, keyed by combination
            name.
        quadratic_integrand (dict): Cache from `compute_detector`, keyed by combination
            name.
        quadratic_integrated (dict): Cache from `compute_detector`, keyed by combination
            name.
    """

    ps: PhysicalConstants = PhysicalConstants()
    waveform: Waveform | None = None
    single_link_response: dict = field(default_factory=dict)
    linear_integrand: dict = field(default_factory=dict)
    quadratic_integrand: dict = field(default_factory=dict)
    quadratic_integrated: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @partial(jax.jit, static_argnums=(0, 1, 6))
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
        Per-link, per-pixel strain response to a GW arriving from (theta_array,
        phi_array), for each polarization mode. Uses the exact retarded geometry
        (`detector_arms_retarded` + `get_single_link_response_retarded`), which
        reduces to the symmetric-arm pipeline in :mod:`gw_response.single_link_static`
        for a non-moving detector (LIGO) -- unless `det.long_wavelength_approximation`
        is set, in which case `get_single_link_response_long_wavelength` is used
        instead.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s), in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the response.
            polarization (str, optional): "LR" or "PC". Default is "LR".

        Returns:
            dict: Maps each polarization letter to its single-link strain response, with
                shape (configurations, x_vector, arms, pixels).

        Raises:
            ValueError: If `polarization` is not "LR" or "PC".
        """
        x_vector = det.x(frequency_array)

        wavevector, p1, p2 = polarization_tensors_and_signed_wavevector(
            polarization, theta_array, phi_array, 1.0
        )

        pol = polarization.upper()
        ppol = {pol[0]: p1, pol[1]: p2}

        if getattr(det, "long_wavelength_approximation", False):
            arms_matrix_rescaled = det.detector_arms(times_in_years) / det.armlength
            return {
                p: get_single_link_response_long_wavelength(
                    ppol[p], arms_matrix_rescaled, x_vector
                )
                for p in ppol.keys()
            }

        (
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            receiver_positions_rescaled,
        ) = per_arm_retarded_geometry_rescaled(det, times_in_years, self.ps)
        return {
            p: get_single_link_response_retarded(
                ppol[p],
                arm_vector_retarded_rescaled,
                ltt_rescaled,
                wavevector,
                x_vector,
                receiver_positions_rescaled,
            )
            for p in ppol.keys()
        }

    @partial(jax.jit, static_argnums=(0, 1, 6, 7))
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
        Computes the linear integrand for a GW arriving from (theta_array, phi_array),
        for each polarization, projected onto a readout combination (e.g. a TDI variable
        like "XYZ"/"AET" for LISA, or "Michelson" for LIGO) via
        `det.linear_response_from_single_link`.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s), in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the response.
            polarization (str, optional): "LR" or "PC". Default is "LR".
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.

        Returns:
            dict: Maps each polarization letter to the linear response integrand, with
                shape (configurations, x_vector, channels, pixels).
        """
        combination = combination or det.default_combination
        arms_matrix_rescaled = det.detector_arms(times_in_years) / det.armlength
        x_vector = det.x(frequency_array)
        combination_matrix = det.combination_matrix(
            combination, arms_matrix_rescaled, x_vector
        )

        single_link = self.get_single_link_response_fd(
            det,
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )

        return det.linear_response_from_single_link(single_link, combination_matrix)

    @partial(jax.jit, static_argnums=(0, 1, 6, 7))
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
        Sky-resolved quadratic response integrand for a GW arriving from (theta_array,
        phi_array), for each polarization, projected onto a readout combination: builds
        the linear integrand (see :meth:`get_linear_integrand_fd`), then applies
        `det.quadratic_response_from_single_link` to it.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s), in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the response.
            polarization (str, optional): "LR" or "PC". Default is "LR".
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.

        Returns:
            dict: Maps each doubled polarization letter (e.g. "LL", "RR") to the
                sky-resolved quadratic response, with shape (configurations, x_vector,
                channels, channels, pixels).
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

    @partial(jax.jit, static_argnums=(0, 1, 6, 7))
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
        Sky-averaged quadratic response for a GW arriving from (theta_array, phi_array),
        for each polarization, projected onto a readout combination: builds the
        quadratic integrand (see :meth:`get_quadratic_integrand_fd`), then applies
        `det.integrate_quadratic_response` to it.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s), in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the response.
            polarization (str, optional): "LR" or "PC". Default is "LR".
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.

        Returns:
            dict: Maps each doubled polarization letter (e.g. "LL", "RR") to the
                sky-averaged quadratic response, with shape (configurations, x_vector,
                channels, channels).
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

    @partial(jax.jit, static_argnums=(0, 1, 5, 9, 10))
    def _response_frozen_td_from_freq_domain(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        n: int,
        dt: ArrayLike,
        h_f_plus: jax.Array,
        h_f_cross: jax.Array,
        combination: str | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        Shared tail behind :meth:`_response_frozen_td_from_strain_td`/
        :meth:`_response_frozen_td_from_strain_fd`: multiplies `h_f_plus`/`h_f_cross`
        (already on the ``jnp.fft.rfftfreq(n, d=dt)`` grid) by the transfer function
        from :meth:`get_linear_integrand_fd` (detector frozen at `time_in_years`), sums
        over polarizations, and IFFTs back.
        """
        frequency_array = jnp.fft.rfftfreq(n, d=dt)
        pol = polarization.upper()
        linear = self.get_linear_integrand_fd(
            det,
            time_in_years,
            theta,
            phi,
            frequency_array,
            polarization=pol,
            combination=combination,
        )

        # linear[pol[0]/pol[1]] is (configurations, frequency, [channels,] pixels);
        # move frequency to the last axis so it lines up with h_f_plus/h_f_cross for a
        # plain elementwise contract_with_h, then IFFT (still frequency/time-last) and
        # finally move time to front.
        R_plus = jnp.moveaxis(linear[pol[0]], 1, -1)
        R_cross = jnp.moveaxis(linear[pol[1]], 1, -1)
        signal = contract_with_h(R_plus, R_cross, h_f_plus, h_f_cross)
        result = frequency_domain_to_time_domain(signal, n, dt)
        return jnp.moveaxis(result, -1, 0)

    @partial(jax.jit, static_argnums=(0, 1, 5, 7, 9, 10))
    def _response_frozen_td_from_strain_td(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        n: int,
        dt: ArrayLike,
        strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
        waveform_params: Any,
        combination: str | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        Backs :meth:`get_response_frozen_td` for a `Waveform` with only `strain_td` set:
        evaluates it on the real time grid ``jnp.arange(n) * dt`` (taking each
        polarization's real part) and FFTs it, then delegates to
        :meth:`_response_frozen_td_from_freq_domain`.
        """
        t_seconds = jnp.arange(n) * dt
        h_plus, h_cross = strain_td(t_seconds, waveform_params)
        h_f_plus = strain_to_frequency_domain(jnp.real(h_plus), dt)
        h_f_cross = strain_to_frequency_domain(jnp.real(h_cross), dt)
        return self._response_frozen_td_from_freq_domain(
            det,
            time_in_years,
            theta,
            phi,
            n,
            dt,
            h_f_plus,
            h_f_cross,
            combination=combination,
            polarization=polarization,
        )

    @partial(jax.jit, static_argnums=(0, 1, 5, 7, 9, 10))
    def _response_frozen_td_from_strain_fd(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        n: int,
        dt: ArrayLike,
        strain_fd: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
        waveform_params: Any,
        combination: str | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        Backs :meth:`get_response_frozen_td` for a `Waveform` with `strain_fd` set:
        evaluates it directly on the ``jnp.fft.rfftfreq(n, d=dt)`` grid -- no FFT --
        then delegates to :meth:`_response_frozen_td_from_freq_domain`.
        """
        frequency_array = jnp.fft.rfftfreq(n, d=dt)
        h_f_plus, h_f_cross = strain_fd(frequency_array, waveform_params)
        return self._response_frozen_td_from_freq_domain(
            det,
            time_in_years,
            theta,
            phi,
            n,
            dt,
            h_f_plus,
            h_f_cross,
            combination=combination,
            polarization=polarization,
        )

    def get_response_frozen_td(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        waveform_params: Any,
        n: int,
        dt: ArrayLike,
        combination: str | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        Projects :attr:`waveform` (set beforehand) onto a detector's readout channel(s),
        sampled on the ``jnp.fft.rfftfreq(n, d=dt)`` grid -- via `waveform.strain_fd`
        directly if set (see :meth:`_response_frozen_td_from_strain_fd`), otherwise by
        FFT-ing `waveform.strain_td` (see :meth:`_response_frozen_td_from_strain_td`).

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            time_in_years (ArrayLike): Time(s), in years, at which the detector
                configuration is frozen.
            theta (ArrayLike): Colatitude(s) of the sky position(s) the signal arrives
                from, in radians.
            phi (ArrayLike): Longitude(s) of the sky position(s) the signal arrives
                from, in radians.
            waveform_params (Any): Source parameters passed through to
                `waveform.strain_td`/`waveform.strain_fd`.
            n (int): Number of time samples.
            dt (ArrayLike): Sample spacing, in seconds.
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.
            polarization (str, optional): "PC" or "LR". Default is "PC".

        Returns:
            jax.Array: The real time-domain readout, with shape (time, configurations,
                channels, pixels) for multi-channel combinations (e.g. LISA's
                `XYZ`/`AET`), or (time, configurations, pixels) for single-channel ones
                (e.g. LIGO's `Michelson`).

        Raises:
            ValueError: If :attr:`waveform` hasn't been set, or has neither
                `strain_td` nor `strain_fd` set.
        """
        if self.waveform is None:
            raise ValueError(
                "Response.waveform must be set before calling "
                "get_response_frozen_td."
            )

        if self.waveform.strain_fd is not None:
            return self._response_frozen_td_from_strain_fd(
                det,
                time_in_years,
                theta,
                phi,
                n,
                dt,
                self.waveform.strain_fd,
                waveform_params,
                combination=combination,
                polarization=polarization,
            )
        if self.waveform.strain_td is not None:
            return self._response_frozen_td_from_strain_td(
                det,
                time_in_years,
                theta,
                phi,
                n,
                dt,
                self.waveform.strain_td,
                waveform_params,
                combination=combination,
                polarization=polarization,
            )
        raise ValueError(
            "Response.waveform must have strain_td or strain_fd set before "
            "calling get_response_frozen_td."
        )

    def get_single_link_response_delay_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        waveform_params: Any,
        times_geometry_years: ArrayLike | None = None,
        freeze_geometry: bool = False,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1j,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) time-domain response for a LISA-like,
        one-way-laser-link constellation (LISA, Taiji), computed exactly by directly
        evaluating the waveform at each link's retarded emission/reception times and
        differencing, rather than building a frequency-domain transfer function -- see
        `single_link_response_delay_td` (in
        :mod:`gw_response.space_based.single_link_geometry`), which this delegates to
        (with `strain_td` taken from :attr:`waveform`, set beforehand), for the full
        derivation and convention notes. Not applicable to ground-based, round-trip
        Michelson detectors (LIGO, CE, ET).

        A frozen-geometry response (the detector's configuration held fixed while the
        waveform still evolves) is just this with a constant `times_geometry_years`,
        decoupled from `times_in_years`, which keeps supplying the (still-varying)
        phase/reception times.

        Left unjitted (unlike most methods here) so that reassigning :attr:`waveform`
        between calls is always picked up -- see the class docstring. The actual
        numerical work is jitted separately, with `strain_td` as its own static
        argument (`waveform_params` stays a regular, traced argument, so that work is
        reused -- not retraced -- across different parameter values, e.g. across an
        inference run's likelihood evaluations).

        Args: see `single_link_response_delay_td` (`strain_td` excepted -- taken from
        :attr:`waveform` instead).

        Returns:
            jax.Array: The real single-link time-domain response, with shape (time,
                arms=6), in arm order :data:`_SINGLE_LINK_ARM_LABELS` (12, 23, 31, 21,
                32, 13).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky position, or if
                :attr:`waveform`/`waveform.strain_td` hasn't been set.
        """
        strain_td = self.waveform.strain_td if self.waveform is not None else None
        if strain_td is None:
            raise ValueError(
                "Response.waveform.strain_td must be set before calling "
                "get_single_link_response_delay_td."
            )
        return single_link_response_delay_td(
            det,
            self.ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            times_geometry_years,
            freeze_geometry,
            wavevector_sign,
            final_factor,
        )

    def get_response_delay_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        waveform_params: Any,
        combination: str = "XYZ",
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1j,
    ) -> jax.Array:
        """
        TDI-combined time-domain response for a LISA-like constellation, exact for
        genuinely evolving geometry -- see `tdi_response_delay_td` (in
        :mod:`gw_response.space_based.single_link_geometry`), which this delegates to
        (with `strain_td` taken from :attr:`waveform`, set beforehand), for the TDI 1.5
        (unequal but locally-constant arms) delay-operator formulas this implements
        (Muratore, Vetrugno & Vitale, arXiv:2303.15929, eq. 2.24) and their reuse of
        :meth:`get_single_link_response_delay_td`. TDI 2.0 (accounting for arm-length
        evolution *during* the nested delays themselves) is not implemented.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring.

        Args: see `tdi_response_delay_td` (`strain_td` excepted -- taken from
        :attr:`waveform` instead).

        Returns:
            jax.Array: The real TDI-combined time-domain response, with shape (time,
                channels=3).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky position, if
                `combination` isn't a supported name, or if
                :attr:`waveform`/`waveform.strain_td` hasn't been set.
        """
        strain_td = self.waveform.strain_td if self.waveform is not None else None
        if strain_td is None:
            raise ValueError(
                "Response.waveform.strain_td must be set before calling "
                "get_response_delay_td."
            )
        return tdi_response_delay_td(
            det,
            self.ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            combination,
            wavevector_sign,
            final_factor,
        )

    def get_single_link_response_segmented_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
        waveform_params: Any,
        segment_length: int,
        wavevector_sign: ArrayLike = 1.0,
        final_factor: ArrayLike = 1j,
    ) -> jax.Array:
        """
        Single-link response for evolving detector geometry, via segment-stacking: the
        full duration is split into short chunks, each evaluated as a first-order Taylor
        expansion of the exact delay formula (:meth:`get_single_link_response_delay_td`)
        around that chunk's own midpoint (one autodiff call per chunk, via
        `per_arm_linearized_geometry`, instead of one `det.vertex_positions` evaluation
        per sample). The error shrinks *quadratically* with `segment_length` and is
        *local* to each segment (doesn't accumulate across segments) -- see
        ``examples/compare_with_lisagwresponse.ipynb`` for the numerical scaling.
        `segment_length = 1` is allowed and reproduces
        :meth:`get_single_link_response_delay_td` sample by sample.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring.

        Args:
            det, theta, phi, waveform_params: see
                :meth:`get_single_link_response_delay_td`.
            times_in_years (ArrayLike): Time(s), in years, uniformly spaced.
            segment_length (int): Number of samples per segment; must evenly divide
                `times_in_years`'s length.
            wavevector_sign (ArrayLike): see :meth:`get_single_link_response_delay_td`.
            final_factor (ArrayLike): see :meth:`get_single_link_response_delay_td`.

        Returns:
            jax.Array: shape (time, arms=6), arm order :data:`_SINGLE_LINK_ARM_LABELS`.

        Raises:
            ValueError: If `segment_length` doesn't evenly divide the number of samples,
                or if waveform hasn't been set.
        """
        strain_td = self.waveform.strain_td if self.waveform is not None else None
        if strain_td is None:
            raise ValueError(
                "Response.waveform.strain_td must be set before calling "
                "get_single_link_response_segmented_td."
            )
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

        one_segment = partial(
            single_link_response_linearized,
            det,
            self.ps,
            theta=theta,
            phi=phi,
            strain_td=strain_td,
            params=waveform_params,
            wavevector_sign=wavevector_sign,
            final_factor=final_factor,
        )

        # (segments, segment_length, arms), already in time-major order
        y_segments = jax.vmap(one_segment)(
            t_ref_years, t_ref_seconds, times_seconds_segments
        )
        n_arms = y_segments.shape[-1]
        return y_segments.reshape(n, n_arms)

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
        Computes and caches the full response chain (single-link, linear integrand,
        quadratic integrand, and sky-averaged quadratic response) for a given readout
        combination.

        Results are stored in :attr:`single_link_response`, :attr:`linear_integrand`,
        :attr:`quadratic_integrand`, and :attr:`quadratic_integrated` (the latter three
        keyed by ``combination``).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s), in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the response.
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.
            polarization (str, optional): "LR" or "PC". Default is "LR".
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
