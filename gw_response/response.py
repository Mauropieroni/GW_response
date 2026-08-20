from __future__ import annotations

import chex
from functools import partial
import jax
import jax.numpy as jnp
from dataclasses import field
from typing import Any, TYPE_CHECKING

from jax.typing import ArrayLike

from gw_response.constants import PhysicalConstants
from gw_response.polarization import polarization_tensors_and_wavevector
from gw_response.response_utils import (
    contract_with_h,
    Waveform,
)
from gw_response.single_link_retarded import get_single_link_response_retarded
from gw_response.single_link_utils import get_single_link_response_long_wavelength
from gw_response.space_based.single_link_geometry import (
    per_arm_retarded_geometry_rescaled,
    single_link_response_delay_td,
    single_link_response_segmented_td,
)
from gw_response.space_based.tdi import (
    build_tdi,
    tdi_response_delay_td,
    tdi_response_segmented_td,
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
    Projects the single-link strain response onto a TDI combination, giving the
    (sky-resolved) linear response of that TDI variable: Hartwig, Lilley, Muratore &
    Pieroni (arXiv:2303.15929) eq. 2.27's ``V(f) = Σ c^V_ij η_ij(f)``.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`gw_response.single_link_static.get_single_link_response_static`, with
            shape (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The linear TDI response, with shape (configurations, x_vector, TDI,
            pixels).
    """
    # Delegates to build_tdi, which also handles single_link with no
    # trailing pixels axis (already sky-integrated).
    return build_tdi(TDI_idx, single_link, arms_matrix_rescaled, x_vector)


@jax.jit
def quadratic_from_linear(linear_response: ArrayLike) -> jax.Array:
    """
    Squares a linear (TDI- or Michelson-projected) response into its quadratic
    cross-spectrum, summed over polarizations and Hermitian conjugation -- the
    per-sky-pixel ``C^U_ij C^V*_mn`` product underlying Hartwig, Lilley, Muratore &
    Pieroni (arXiv:2303.15929) eq. 2.29's ``S^UV(f)``, before the ``dΩ_k̂`` sky
    integral (eq. 2.15-2.16) that :func:`quadratic_response_integrated` performs.

    Args:
        linear_response (ArrayLike): Linear response, with shape (configurations,
            x_vector, TDI, pixels).

    Returns:
        jax.Array: The quadratic response, with shape (configurations, x_vector, TDI,
            TDI, pixels).
    """
    quadratic_response = jnp.einsum(
        "...ijl,...ikl->...ijkl",
        linear_response,
        jnp.conjugate(linear_response),
    )
    # The first 2 is sum over polarization the second is for the h.c. sum
    return 2 * 2 * quadratic_response / jnp.pi / 4


@jax.jit
def quadratic_response_angular(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the (sky-resolved) quadratic response of a TDI combination:
    :func:`linear_response_angular` squared via :func:`quadratic_from_linear`.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The quadratic TDI response, with shape (configurations, x_vector,
            TDI, TDI, pixels).
    """
    linear_response = linear_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )
    return quadratic_from_linear(linear_response)


@jax.jit
def quadratic_response_integrated(angular_response: ArrayLike) -> jax.Array:
    """
    Averages the sky-resolved quadratic response over the sky (pixels) to give the
    quadratic TDI response as a function of frequency -- the sky-averaged response
    ``R(f)`` of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.4-2.5
    (``S^GW(f) = R(f) P_h(f)``), via a pixelized ``dΩ_k̂`` average (eq. 2.16).

    Args:
        angular_response (ArrayLike): Sky-resolved quadratic response, as returned by
            :func:`quadratic_integrand`, with shape (configurations, x_vector, TDI, TDI,
            pixels).

    Returns:
        jax.Array: The sky-averaged quadratic response, with shape (configurations,
            x_vector, TDI, TDI), normalized by ``4 * pi`` to account for the solid angle
            of the sphere.
    """
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)


@chex.dataclass
class Response(object):
    """
    Generic class to handle GW response computations for any Detector (e.g. LISA, LIGO).
    `get_response` is the high-level entry point for projecting :attr:`waveform` onto a
    detector's readout channel(s); the many `get_*_td`/`get_*_fd` methods it delegates
    to (and the more granular FD-only integrand methods used by `compute_detector`) can
    also be called directly for finer control.

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
        theta_array: jax.Array,
        phi_array: jax.Array,
        frequency_array: jax.Array,
        polarization: str = "LR",
    ) -> dict[str, jax.Array]:
        """
        Per-link, per-pixel strain response to a GW arriving from (theta_array,
        phi_array), for each polarization mode -- the ``ξ_ij^A``-based single-link
        kernel of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eqs. 2.12-2.15
        (see `get_single_link_response_retarded`/`single_link_static` for the exact
        per-equation mapping). Uses the exact retarded geometry
        (`detector_arms_retarded` + `get_single_link_response_retarded`), which
        reduces to the symmetric-arm pipeline in :mod:`gw_response.single_link_static`
        for a non-moving detector (LIGO) -- unless `det.long_wavelength_approximation`
        is set, in which case `get_single_link_response_long_wavelength` is used
        instead (not itself tied to a specific equation in that paper -- see its own
        docstring).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (jax.Array): Colatitude(s) of the sky position(s), in radians.
            phi_array (jax.Array): Longitude(s) of the sky position(s), in radians.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the response.
            polarization (str, optional): "LR" or "PC". Default is "LR".

        Returns:
            dict: Maps each polarization letter to its single-link strain response, with
                shape (configurations, x_vector, arms, pixels).

        Raises:
            ValueError: If `polarization` is not "LR" or "PC".
        """
        x_vector = det.x(frequency_array)
        wavevector, p1, p2 = polarization_tensors_and_wavevector(
            polarization, theta_array, phi_array
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
        theta_array: jax.Array,
        phi_array: jax.Array,
        frequency_array: jax.Array,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Computes the linear integrand for a GW arriving from (theta_array, phi_array),
        for each polarization, projected onto a readout combination (e.g. a TDI variable
        like "XYZ"/"AET" for LISA, or "Michelson" for LIGO) via
        `det.linear_response_from_single_link` -- Hartwig, Lilley, Muratore & Pieroni
        (arXiv:2303.15929) eq. 2.27's ``V(f) = Σ c^V_ij η_ij(f)`` TDI-variable
        combination, generalized to any `Detector`'s own readout combination via
        `det.combination_matrix` (see :class:`gw_response.detector.Detector`).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (jax.Array): Colatitude(s) of the sky position(s), in radians.
            phi_array (jax.Array): Longitude(s) of the sky position(s), in radians.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
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

    def get_response_fd(
        self,
        det: "Detector",
        time_in_years: ArrayLike,
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        frequency_array: jax.Array,
        combination: str | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        Projects :attr:`waveform.strain_fd` (set beforehand) onto a detector's readout
        channel(s) at `frequency_array`, natively in the frequency domain -- no FFT/IFFT
        needed, since `strain_fd` is already frequency-domain -- reusing
        :meth:`get_linear_integrand_fd` for the transfer function.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            time_in_years (ArrayLike): Time(s), in years, at which the detector
                configuration is frozen.
            theta (jax.Array): Colatitude(s) of the sky position(s) the signal arrives
                from, in radians.
            phi (jax.Array): Longitude(s) of the sky position(s) the signal arrives
                from, in radians.
            waveform_params (Any): Source parameters passed through to
                `waveform.strain_fd`.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the response.
            combination (str, optional): Name of the readout combination. Defaults to
                ``det.default_combination``.
            polarization (str, optional): "PC" or "LR". Default is "PC".

        Returns:
            jax.Array: The complex frequency-domain readout, with shape
                (configurations, frequency, channels, pixels) for multi-channel
                combinations (e.g. LISA's `XYZ`/`AET`), or (configurations, frequency,
                pixels) for single-channel ones (e.g. LIGO's `Michelson`) -- matching
                :meth:`get_linear_integrand_fd`'s own axis convention.

        Raises:
            ValueError: If :attr:`waveform`/`waveform.strain_fd` hasn't been set.
        """
        strain_fd = self.waveform.strain_fd if self.waveform is not None else None
        if strain_fd is None:
            raise ValueError(
                "Response.waveform.strain_fd must be set before calling "
                "get_response_fd."
            )
        h_f_plus, h_f_cross = strain_fd(jnp.asarray(frequency_array), waveform_params)

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
        # linear[pol[0]/pol[1]] is (configurations, frequency, [channels,]
        # pixels); move frequency to the last axis so it lines up with
        # h_f_plus/h_f_cross for a plain elementwise contract_with_h, then
        # move it back to match get_linear_integrand_fd's own convention.
        R_plus = jnp.moveaxis(linear[pol[0]], 1, -1)
        R_cross = jnp.moveaxis(linear[pol[1]], 1, -1)
        result = contract_with_h(R_plus, R_cross, h_f_plus, h_f_cross)
        return jnp.moveaxis(result, -1, 1)

    def get_single_link_response_delay_td(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        freeze_geometry: bool = False,
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
                32, 13). Real by construction (``jnp.real(y_complex)``, no residual
                phase factor needed) -- see `single_link_response_delay_td`'s
                docstring/comments for the caveat on reconstructing other conventions'
                differently-normalized real output.

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
            freeze_geometry,
        )

    def get_response_delay_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        combination: str = "XYZ",
        tdi_order: float = 1.5,
    ) -> jax.Array:
        """
        TDI-combined time-domain response for a LISA-like constellation, exact for
        genuinely evolving geometry -- see `tdi_response_delay_td` (in
        :mod:`gw_response.space_based.tdi`), which this delegates to
        (with `strain_td` taken from :attr:`waveform`, set beforehand), for the TDI 1.5
        and 2.0 delay-operator formulas this implements (Hartwig, Lilley, Muratore &
        Pieroni, arXiv:2303.15929, eqs. 2.24 and 2.23) and their reuse of
        :meth:`get_single_link_response_delay_td`.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring. The actual numerical work
        (`tdi_response_delay_td`) is jitted separately, as one fused program, same
        pattern as `get_single_link_response_delay_td`.

        Args: see `tdi_response_delay_td` (`strain_td` excepted -- taken from
        :attr:`waveform` instead).

        Returns:
            jax.Array: The real TDI-combined time-domain response, with shape (time,
                channels=3).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky position, if
                `combination`/`tdi_order` isn't a supported value, or if
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
            tdi_order,
        )

    def get_single_link_response_segmented_td(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        segment_length: int,
    ) -> jax.Array:
        """
        Single-link (not TDI-combined) response for evolving detector geometry, via
        segment-stacking -- see `single_link_response_segmented_td` (in
        :mod:`gw_response.space_based.single_link_geometry`), which this delegates to
        (with `strain_td` taken from :attr:`waveform`, set beforehand), for the full
        derivation and convention notes.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring.

        Args: see `single_link_response_segmented_td` (`strain_td` excepted -- taken
        from :attr:`waveform` instead).

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
        return single_link_response_segmented_td(
            det,
            self.ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            segment_length,
        )

    def get_response_segmented_td(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        segment_length: int,
        combination: str = "XYZ",
    ) -> jax.Array:
        """
        TDI 1.5 time-domain response for a LISA-like constellation, via
        segment-stacking -- see `tdi_response_segmented_td` (in
        :mod:`gw_response.space_based.tdi`), which this delegates to (with `strain_td`
        taken from :attr:`waveform`, set beforehand), for the full derivation and its
        reuse of :meth:`get_single_link_response_segmented_td`. Faster than, and not
        much less accurate than, :meth:`get_response_delay_td`; TDI 2.0 isn't supported
        here -- use :meth:`get_response_delay_td` for that.

        Left unjitted so that reassigning :attr:`waveform` between calls is always
        picked up -- see the class docstring. The actual numerical work
        (`tdi_response_segmented_td`) is jitted separately, as one fused program, same
        pattern (and same reasoning) as `get_response_delay_td`.

        Args: see `tdi_response_segmented_td` (`strain_td` excepted -- taken from
        :attr:`waveform` instead).

        Returns:
            jax.Array: The real TDI-combined time-domain response, with shape (time,
                channels=3).

        Raises:
            ValueError: If `theta`/`phi` resolve to more than one sky position, if
                `segment_length` doesn't evenly divide the number of samples, if
                `combination` isn't a supported value, or if
                :attr:`waveform`/`waveform.strain_td` hasn't been set.
        """
        strain_td = self.waveform.strain_td if self.waveform is not None else None
        if strain_td is None:
            raise ValueError(
                "Response.waveform.strain_td must be set before calling "
                "get_response_segmented_td."
            )
        return tdi_response_segmented_td(
            det,
            self.ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            segment_length,
            combination,
        )

    def get_response(
        self,
        det: "Detector",
        theta: jax.Array,
        phi: jax.Array,
        waveform_params: Any,
        which_domain: str = "TD",
        which_method: str = "delay",
        which_TDI: str | None = None,
        TDI_order: float = 1.5,
        times_in_years: ArrayLike | None = None,
        frequency_array: jax.Array | None = None,
        segment_length: int | None = None,
        polarization: str = "PC",
    ) -> jax.Array:
        """
        High-level entry point: projects :attr:`waveform` (set beforehand) onto a
        detector's readout channel(s), in whichever domain/method/TDI combination is
        requested, by delegating to the specific worker method below -- see that
        method's own docstring for the full derivation/conventions. `which_domain="TD"`
        is exact for evolving geometry (see :meth:`get_response_delay_td`) or, via
        `which_method="segmented"`, faster and not much less accurate (see
        :meth:`get_response_segmented_td`); `which_domain="FD"` is natively
        frequency-domain (see :meth:`get_response_fd`).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            theta (jax.Array): Colatitude(s) of the sky position(s), in radians.
            phi (jax.Array): Longitude(s) of the sky position(s), in radians.
            waveform_params (Any): Source parameters passed through to
                `waveform.strain_td`/`waveform.strain_fd`.
            which_domain (str, optional): "TD" or "FD". Default is "TD".
            which_method (str, optional): "delay" or "segmented"; only used for
                `which_domain="TD"`. Default is "delay".
            which_TDI (str, optional): Name of the readout combination (e.g. a TDI
                variable like "XYZ"/"AET" for LISA, or "Michelson" for LIGO). Defaults
                to ``det.default_combination``.
            TDI_order (float, optional): 1.5 or 2.0; only used for
                `which_method="delay"` -- see :meth:`get_response_delay_td`.
                `which_method="segmented"` only supports 1.5. Default is 1.5.
            times_in_years (ArrayLike, optional): Time(s), in years -- required for both
                `which_domain="TD"` and `which_domain="FD"` (for the latter, the single
                time the detector configuration is frozen at).
            frequency_array (jax.Array, optional): Frequency values, in Hz -- required
                for `which_domain="FD"`.
            segment_length (int, optional): Number of samples per segment; required for
                `which_method="segmented"` -- see :meth:`get_response_segmented_td`.
            polarization (str, optional): "PC" or "LR". Only used for `which_domain=
                "FD"`. Default is "PC".

        Returns:
            jax.Array: The response, with shape and domain depending on
                `which_domain` -- see the delegated method's own docstring.

        Raises:
            ValueError: If `which_domain`/`which_method` isn't recognized, or a
                required argument for the selected domain/method is missing, or
                `which_method="segmented"` is combined with `TDI_order=2.0`.
        """
        combination = which_TDI or det.default_combination

        if which_domain == "TD":
            if times_in_years is None:
                raise ValueError(
                    "get_response requires times_in_years for which_domain='TD'."
                )
            if which_method == "delay":
                return self.get_response_delay_td(
                    det,
                    times_in_years,
                    theta,
                    phi,
                    waveform_params,
                    combination=combination,
                    tdi_order=TDI_order,
                )
            if which_method == "segmented":
                if segment_length is None:
                    raise ValueError(
                        "get_response requires segment_length for "
                        "which_method='segmented'."
                    )
                if TDI_order != 1.5:
                    raise ValueError(
                        "get_response only supports TDI_order=1.5 for "
                        "which_method='segmented'."
                    )
                return self.get_response_segmented_td(
                    det,
                    times_in_years,
                    theta,
                    phi,
                    waveform_params,
                    segment_length,
                    combination=combination,
                )
            raise ValueError(
                f"Unknown which_method '{which_method}'; expected 'delay' or "
                "'segmented'."
            )
        if which_domain == "FD":
            if times_in_years is None or frequency_array is None:
                raise ValueError(
                    "get_response requires times_in_years and frequency_array for "
                    "which_domain='FD'."
                )
            return self.get_response_fd(
                det,
                times_in_years,
                theta,
                phi,
                waveform_params,
                frequency_array,
                combination=combination,
                polarization=polarization,
            )
        raise ValueError(
            f"Unknown which_domain '{which_domain}'; expected 'TD' or 'FD'."
        )

    @partial(jax.jit, static_argnums=(0, 1, 6, 7))
    def get_quadratic_integrand_fd(
        self,
        det: "Detector",
        times_in_years: ArrayLike,
        theta_array: jax.Array,
        phi_array: jax.Array,
        frequency_array: jax.Array,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Sky-resolved quadratic response integrand for a GW arriving from (theta_array,
        phi_array), for each polarization, projected onto a readout combination: builds
        the linear integrand (see :meth:`get_linear_integrand_fd`), then applies
        `det.quadratic_response_from_single_link` to it -- the per-sky-pixel analog of
        Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.29's ``S^UV(f) =
        Σ C^U_ij C^V*_mn S^η_ij,lm(f)`` CSD, before the ``dΩ_k̂`` sky integral that
        :meth:`get_quadratic_integrated_fd` performs (see eq. 2.15-2.16).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (jax.Array): Colatitude(s) of the sky position(s), in radians.
            phi_array (jax.Array): Longitude(s) of the sky position(s), in radians.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
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
        theta_array: jax.Array,
        phi_array: jax.Array,
        frequency_array: jax.Array,
        polarization: str = "LR",
        combination: str | None = None,
    ) -> dict[str, jax.Array]:
        """
        Sky-averaged quadratic response for a GW arriving from (theta_array, phi_array),
        for each polarization, projected onto a readout combination: builds the
        quadratic integrand (see :meth:`get_quadratic_integrand_fd`), then applies
        `det.integrate_quadratic_response` to it -- the sky-averaged response function
        ``R(f)`` of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.4-2.5
        (``S^GW(f) = R(f) P_h(f)``), performing the same ``dΩ_k̂`` integral as eq.
        2.16's ``Υ^A_ij,mn(f)`` via a pixelized sky average.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the response is computed for.
            times_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                vertex positions and detector arms.
            theta_array (jax.Array): Colatitude(s) of the sky position(s), in radians.
            phi_array (jax.Array): Longitude(s) of the sky position(s), in radians.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
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

    def compute_detector(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        theta_array: jax.Array,
        phi_array: jax.Array,
        frequency_array: jax.Array,
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
