from __future__ import annotations

# Global imports
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, TYPE_CHECKING

from jax.typing import ArrayLike

# Local imports
from gw_response.constants import PhysicalConstants
from gw_response.utils import as_time_array

if TYPE_CHECKING:
    from gw_response.noise import Noise
    from gw_response.response import Response


class Detector(ABC):
    """
    Abstract base class for a gravitational wave detector.

    Concrete subclasses (e.g. ``LISA``, ``LIGO``) must provide the detector's basic
    characteristics as class attributes, and implement the methods that describe its
    geometry, response and noise.

    Attributes:
        name: Human-readable name of the detector.
        fmin: Minimum frequency of the detector's sensitive band, in Hz.
        fmax: Maximum frequency of the detector's sensitive band, in Hz.
        armlength: Nominal detector arm length, in meters.
        res: Expected relative resolution/precision of the detector.
        ps: Detector-specific physical/instrumental parameters.
        default_combination: Name of the default readout combination (e.g. a TDI
            variable for LISA) used when none is specified.
        response: Response object used to compute the detector's response to
            gravitational waves.
        noise: Noise object used to compute the detector's noise budget.
    """

    name: str
    fmin: float
    fmax: float
    armlength: float
    res: float
    ps: Any
    default_combination: str
    # Concrete detectors (LISA, LIGO) will instantiate real Response/Noise objects
    response: Response
    noise: Noise

    @abstractmethod
    def vertex_positions(self, time_in_years: jax.Array) -> jax.Array:
        """
        Computes satellite/vertex positions at the given time(s).

        Concrete implementations are responsible for normalizing `time_in_years` to an
        array (of at least 1 dimension) via :func:`gw_response.utils.as_time_array` as
        their own first step, so that a bare scalar as well as an array is accepted --
        this is no longer done generically by a shared wrapper, to avoid an extra
        method per concrete detector that did nothing but that one line.

        Args:
            time_in_years (jax.Array): Time(s), in years, at which to evaluate the
                vertex positions.

        Returns:
            jax.Array: Array of vertex positions.
        """

    @abstractmethod
    def detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes the detector's arm matrix at the given time(s).

        Concrete implementations are responsible for normalizing `time_in_years` via
        :func:`gw_response.utils.as_time_array` as their own first step, same as
        :meth:`vertex_positions`.

        Args:
            time_in_years (ArrayLike): Time(s), in years, at which to evaluate the
                detector arms.

        Returns:
            jax.Array: Array representing the vector between each pair of vertices (i.e.
                each detector arm).
        """

    @property
    @abstractmethod
    def arm_vertex_pairs(self) -> tuple[tuple[int, int], ...]:
        """
        For each of `detector_arms`'s arm columns, in the same order, the
        (receiver_vertex_index, emitter_vertex_index) pair into `vertex_positions`'s own
        vertex axis -- i.e. that arm's vector equals ``vertex_positions[...,
        emitter_index] -vertex_positions[..., receiver_index]``. Lets generic
        single-link code (e.g. :meth:`detector_arms_retarded`) compute each arm's
        retarded (backdated-emitter) geometry without knowing the detector's specific
        topology.

        Returns:
            tuple[tuple[int, int], ...]: One (receiver_index, emitter_index) pair per
                arm.
        """

    @staticmethod
    @partial(jax.jit, static_argnames=("axis",))
    def arm_length_and_unit_vector(
        arm_vector: jax.Array, axis: int = -2
    ) -> tuple[jax.Array, jax.Array]:
        """
        Length and unit direction of an arm vector (or a batch of them), along its
        vectorial (length-3) axis. Shared helper for the handful of places that need an
        arm's raw length and/or unit direction directly from a vector -- as opposed to
        the already-combined per-arm length computed from a full arm matrix by
        :func:`gw_response.utils.arm_lengths_from_matrix` -- namely
        :meth:`detector_arms_retarded` (below),
        :func:`gw_response.single_link_retarded.xi_k_A_retarded`, and
        :func:`gw_response.space_based.single_link_geometry.all_arms_geometry`. A
        `staticmethod` (no detector-specific state involved) so it can be called both
        as `self.arm_length_and_unit_vector(...)` here and as
        `Detector.arm_length_and_unit_vector(...)` from those detector-agnostic, pure
        array functions elsewhere.

        Args:
            arm_vector (jax.Array): Arm vector(s), with the vectorial (length-3) axis at
                `axis`.
            axis (int): Which axis of `arm_vector` is the vectorial index. Default -2,
                matching an arm-matrix-shaped input (..., vectorial_index, arms); pass
                -1 for a single arm vector with no separate trailing arms axis.

        Returns:
            tuple[jax.Array, jax.Array]: `(length, unit_vector)` -- `length` is
                `arm_vector`'s shape with `axis` removed; `unit_vector` is
                `arm_vector`'s own shape, divided by `length` along `axis`.
        """
        length = jnp.linalg.norm(arm_vector, axis=axis)
        unit_vector = arm_vector / jnp.expand_dims(length, axis)
        return length, unit_vector

    def detector_arms_retarded(
        self,
        time_in_years: ArrayLike,
        ps: PhysicalConstants,
        freeze_geometry: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Retarded counterpart of :meth:`detector_arms`: for each arm (per
        `arm_vertex_pairs`), backdates the emitter's position by that arm's own
        light-travel-time estimate, giving a genuinely asymmetric arm vector for a
        moving detector -- unless `freeze_geometry`, which evaluates the emitter
        simultaneously with the receiver instead, making this exactly
        ``detector_arms(time_in_years)`` (per arm, alongside the same light-travel-time
        estimate). Concrete (not abstract) and detector-agnostic, since it's driven
        entirely by `vertex_positions`/`arm_vertex_pairs` -- shared by the frequency-
        domain retarded pipeline and the exact time-domain single-link methods (see
        :mod:`gw_response.space_based.single_link_geometry`).

        Args:
            time_in_years (ArrayLike): Time(s), in years, at which the receiver end of
                each arm (and, if `freeze_geometry`, the emitter end too) is evaluated.
            ps (PhysicalConstants): Physical constants (`light_speed`, `yr`).
            freeze_geometry (bool): If True, evaluate the emitter at the same time as
                the receiver instead of backdating it.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: `(arm_vector, light_travel_time,
                receiver_position)`, each with shape (configurations, vectorial_index
                (3), arms) -- except `light_travel_time`, shape (configurations, arms)
                -- stacked over the arms in `arm_vertex_pairs` order.
        """
        time_in_years = as_time_array(time_in_years)
        positions_rec = self.vertex_positions(time_in_years)

        arm_vectors, ltts, receivers = [], [], []
        for receiver_idx, emitter_idx in self.arm_vertex_pairs:
            x_rec = positions_rec[:, :, receiver_idx]
            x_emi_simul = positions_rec[:, :, emitter_idx]
            ltt_approx = (
                self.arm_length_and_unit_vector(x_rec - x_emi_simul, axis=-1)[0]
                / ps.light_speed
            )
            if freeze_geometry:
                x_emi = x_emi_simul
            else:
                t_emi_years = time_in_years - ltt_approx / ps.yr
                x_emi = self.vertex_positions(t_emi_years)[:, :, emitter_idx]
            arm_vectors.append(x_emi - x_rec)
            ltts.append(ltt_approx)
            receivers.append(x_rec)

        return (
            jnp.stack(arm_vectors, axis=-1),
            jnp.stack(ltts, axis=-1),
            jnp.stack(receivers, axis=-1),
        )

    def frequency_vec(self, freq_pts: int) -> jax.Array:
        """
        Generates a frequency vector within the detector's frequency range.

        Args:
            freq_pts (int): The number of frequency points to generate.

        Returns:
            jax.Array: A linearly spaced array of frequency points within the detector's
                operational frequency range, starting from ``self.fmin`` to
                ``self.fmax``.
        """
        return jnp.linspace(self.fmin, self.fmax, freq_pts)

    def klvector(self, frequency_vec: jax.Array) -> jax.Array:
        """
        Computes the kl-vector for a given frequency vector, i.e. the detector's arm
        length in units of the reduced wavelength.

        Args:
            frequency_vec (jax.Array): An array of frequency values, in Hz, for which
                the kl-vector is to be computed.

        Returns:
            jax.Array: An array representing the kl-vector, which is a product of the
                frequency vector, the detector arm length, and the inverse of the speed
                of light.
        """
        return frequency_vec * self.armlength / self.ps.light_speed

    def x(self, frequency_vec: jax.Array) -> jax.Array:
        """
        Computes the x-parameter (``2 pi f L / c``) for a given frequency vector.

        Args:
            frequency_vec (jax.Array): An array of frequency values, in Hz, for which
                the x-parameter is to be computed.

        Returns:
            jax.Array: An array representing the x-parameter, calculated as 2π times the
                kl-vector for the given frequency vector.
        """
        return 2 * jnp.pi * self.klvector(frequency_vec)

    @abstractmethod
    def combination_matrix(
        self, combination: str, arms_matrix_rescaled: jax.Array, x_vector: jax.Array
    ) -> jax.Array:
        """
        Builds the mixing matrix that turns per-link responses into the detector's
        readout channel(s) for the requested combination (e.g. a TDI variable for LISA,
        the Michelson combination for LIGO). For LISA, the ``c^V_ij`` coefficients of
        Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.27, generalized
        here to any `Detector`'s own readout combination (this paper covers LISA/TDI
        only, so the abstract interface itself isn't tied to it -- see
        :mod:`gw_response.space_based.tdi` for the LISA-specific matrices that do
        implement it).

        Args:
            combination (str): Name of the readout combination to build the mixing
                matrix for.
            arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
                length.
            x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

        Returns:
            jax.Array: The mixing matrix, of shape (..., x_vector, channels, arms).
        """

    @abstractmethod
    def linear_response_from_single_link(
        self, single_link: dict[str, jax.Array], combination_matrix: jax.Array
    ) -> dict[str, jax.Array]:
        """
        Applies `combination_matrix` (as built by `combination_matrix`) to each
        polarization's single-link response, returning the dict of linear integrands
        (keyed by polarization) for this detector -- for LISA, eq. 2.27's ``V(f) =
        Σ c^V_ij η_ij(f)`` (see :meth:`combination_matrix`).

        Args:
            single_link (dict): Single-link response per polarization, e.g. as returned
                by :meth:`gw_response.response.Response.get_single_link_response_fd`.
            combination_matrix (jax.Array): Mixing matrix as built by
                :meth:`combination_matrix`.

        Returns:
            dict: The linear response integrand per polarization.
        """

    @abstractmethod
    def quadratic_response_from_single_link(
        self, linear_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Given the dict of linear integrands (keyed by polarization), returns the dict of
        quadratic integrands for this detector.

        Args:
            linear_integrand (dict): Linear response integrand per polarization, as
                returned by :meth:`linear_response_from_single_link`.

        Returns:
            dict: The quadratic response integrand per doubled polarization letter (e.g.
                "LL", "RR").
        """

    @abstractmethod
    def integrate_quadratic_response(
        self, quadratic_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Given the dict of quadratic integrands (keyed by polarization) for a single
        combination, returns the corresponding integrated response.

        Args:
            quadratic_integrand (dict): Quadratic response integrand per doubled
                polarization letter, as returned by
                :meth:`quadratic_response_from_single_link`.

        Returns:
            dict: The integrated (e.g. sky-averaged) quadratic response per doubled
                polarization letter.
        """

    @abstractmethod
    def single_link_noise(
        self,
        frequency_array: jax.Array,
        arms_matrix_rescaled: jax.Array,
        x_vector: jax.Array,
        **noise_parameters,
    ) -> jax.Array:
        """
        Builds the per-link noise covariance (or, for detectors with no per-link
        decomposition, the already-combined noise) before projection into a readout
        combination. `noise_parameters` are whatever detector-specific noise parameters
        this needs (e.g. LISA's TM_acceleration_parameters/OMS_parameters); LIGO takes
        none.

        Args:
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the noise.
            arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
                length.
            x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
                **noise_parameters: Detector-specific noise parameters.

        Returns:
            jax.Array: The per-link (or already-combined) noise covariance.
        """

    @abstractmethod
    def project_noise(
        self, combination_matrix: jax.Array, single_link_noise: jax.Array
    ) -> jax.Array:
        """
        Projects `single_link_noise` (as built by `single_link_noise`) into the readout
        basis defined by `combination_matrix` (as built by `combination_matrix`), e.g.
        via the congruence transform combination_matrix @ single_link_noise @
        combination_matrix^H -- for LISA, Hartwig, Lilley, Muratore & Pieroni
        (arXiv:2303.15929) eq. 2.29b/2.30's ``S^UV,N = C^UV S^η,N``. Detectors without a
        per-link decomposition (e.g. LIGO) can simply return `single_link_noise`
        unchanged.

        Args:
            combination_matrix (jax.Array): Mixing matrix as built by
                :meth:`combination_matrix`.
            single_link_noise (jax.Array): Per-link noise covariance, as built by
                :meth:`single_link_noise`.

        Returns:
            jax.Array: The noise covariance, projected into the readout basis.
        """
