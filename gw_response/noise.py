# Global imports
import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from dataclasses import field

# Local imports
from .constants import PhysicalConstants
from .lisa import LISA
from .tdi import tdi_matrix, TDI_map
from .utils import arm_length_exponential

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)
# jax.checking_leaks = True


def LISA_acceleration_noise(frequency: ArrayLike, acc_param: float = 3.0) -> jax.Array:
    """
    Computes the single test-mass acceleration noise power spectral density
    for LISA, as a function of frequency.

    Args:
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        acc_param (float, optional): Acceleration noise amplitude parameter,
            in units of :math:`10^{-15}\\, \\mathrm{m\\,s^{-2}/\\sqrt{Hz}}`.
            Default is 3.0 (the LISA requirement).

    Returns:
        jax.Array: The acceleration noise power spectral density, with the
            same shape as ``frequency``.
    """

    first = 1.0 + (4e-4 / frequency) ** 2
    second = 1.0 + (frequency / 8e-3) ** 4
    third = (2.0 * jnp.pi * frequency) ** (-4) * (2.0 * jnp.pi * frequency / 3e8) ** 2
    # TODO: Change 3e8 to ps.light_speed
    return acc_param**2 * 1e-30 * first * second * third


def LISA_interferometric_noise(
    frequency: ArrayLike, inter_param: float = 15.0
) -> jax.Array:
    """
    Computes the single-link interferometric (optical metrology system,
    OMS) noise power spectral density for LISA, as a function of frequency.

    Args:
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        inter_param (float, optional): Interferometric noise amplitude
            parameter, in units of :math:`10^{-12}\\, \\mathrm{m/\\sqrt{Hz}}`.
            Default is 15.0 (the LISA requirement).

    Returns:
        jax.Array: The interferometric noise power spectral density, with
            the same shape as ``frequency``.
    """

    first = 1.0 + (2e-3 / frequency) ** 4
    second = (2.0 * jnp.pi * frequency / 3e8) ** 2

    return inter_param**2 * 1e-24 * first * second


def single_link_TM_acceleration_noise_variance(
    frequency: ArrayLike,
    TM_acceleration_parameters: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the single-link test-mass (acceleration) noise covariance
    matrix, including the cross-correlation between an arm and its
    reverse-direction counterpart introduced by the light-travel-time delay.

    Args:
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        TM_acceleration_parameters (ArrayLike): Per-arm acceleration noise
            amplitude parameters, a vector of length 6 (or an array with a
            trailing dimension of length 6 for multiple configurations).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The single-link test-mass noise covariance matrix, with
            shape (configurations, frequency, arms (6), arms (6)).
    """

    # the shape of t_retarded_factor is configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    # This would be a diag matrix on the last 2 indexes,
    # the shape is configurations, x_vector, arms, arms
    t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(6), t_retarded_factor
    )

    # This would be a diag matrix on the last 2 indexes,
    # the shape is configurations, x_vector, arms, arms
    flipped_t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij",
        jnp.identity(6),
        jnp.roll(t_retarded_factor, 3, axis=-1),
    )

    # The shape will be configurations, arms, arms
    parameters_matrix = jnp.einsum(
        "ij,...j->...ij", jnp.identity(6), TM_acceleration_parameters**2
    )

    # The shape will be configurations, arms, arms
    flipped_parameters_matrix = jnp.einsum(
        "ij,...j->...ij",
        jnp.identity(6),
        jnp.roll(TM_acceleration_parameters**2, 3),
    )

    # The shape will be frequency
    N_acc = LISA_acceleration_noise(frequency, acc_param=1.0)

    # The shape will be configurations, frequency, arms, arms
    noise_matrix = jnp.einsum(
        "...ij,k->...kij", parameters_matrix + flipped_parameters_matrix, N_acc
    )

    # t_retarded_coeffs is configurations, x_vector, arms, arms
    # flipped_parameters_matrix is configurations, arms, arms
    # The shape will be configurations, frequency, arms, arms
    delayed_1 = jnp.einsum(
        "...kij,...ij->...kij", t_retarded_coeffs, flipped_parameters_matrix
    )

    delayed_2 = jnp.einsum(
        "...kij,...ij->...kij",
        jnp.conjugate(flipped_t_retarded_coeffs),
        parameters_matrix,
    )

    cross_matrix = jnp.einsum(
        "...kij,k->...kij",
        (delayed_1 + delayed_2),
        N_acc,
    )

    # The shape will be configurations, frequency, arms, arms
    return noise_matrix + jnp.roll(cross_matrix, 3, axis=-1)


def single_link_OMS_noise_variance(
    frequency: ArrayLike,
    OMS_parameters: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the single-link interferometric (OMS) noise covariance matrix.

    Unlike the test-mass noise, the OMS noise is uncorrelated between arms,
    so the resulting covariance matrix is diagonal.

    Args:
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        OMS_parameters (ArrayLike): Per-arm interferometric noise amplitude
            parameters, a vector of length 6 (or an array with a trailing
            dimension of length 6 for multiple configurations).
        arms_matrix_rescaled (ArrayLike): Unused. Present so this function
            has the same signature as
            :func:`single_link_TM_acceleration_noise_variance`.
        x_vector (ArrayLike): Unused. Present for the same reason as
            ``arms_matrix_rescaled``.

    Returns:
        jax.Array: The single-link OMS noise covariance matrix, with shape
            (configurations, frequency, arms (6), arms (6)).
    """

    # The shape will be configurations, arms, arms
    parameters_matrix = jnp.einsum("ij,...j->...ij", jnp.identity(6), OMS_parameters**2)

    # The shape will be frequency
    N_int = LISA_interferometric_noise(frequency, inter_param=1.0)

    # The shape will be configurations, frequency, arms, arms
    return jnp.einsum("...ij,k->...kij", parameters_matrix, N_int)


def tdi_projection(
    TDI_idx: ArrayLike,
    single_link_mat: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Projects a single-link noise covariance matrix onto a TDI combination.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link_mat (ArrayLike): Single-link noise covariance matrix, as
            returned by e.g. :func:`single_link_TM_acceleration_noise_variance`
            or :func:`single_link_OMS_noise_variance`, with shape
            (configurations, frequency, arms (6), arms (6)).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The TDI noise covariance matrix, with shape
            (configurations, frequency, TDI, TDI).
    """

    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # The shape will be configurations, frequency, tdi, arms
    first_contraction = jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link_mat)

    # The shape will be configurations, frequency, tdi, tdi
    res = jnp.einsum("...ijk,...ilk->...ijl", jnp.conjugate(tdi_mat), first_contraction)

    return res


def noise_TM_matrix(
    TDI_idx: ArrayLike,
    frequency: ArrayLike,
    TM_acceleration_parameters: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the test-mass (acceleration) noise covariance matrix for a TDI
    combination.

    Combines :func:`single_link_TM_acceleration_noise_variance` with
    :func:`tdi_projection`.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            selecting the TDI combination to project onto.
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        TM_acceleration_parameters (ArrayLike): Per-arm acceleration noise
            amplitude parameters, a vector of length 6.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The TDI test-mass noise covariance matrix, with shape
            (configurations, frequency, TDI, TDI).
    """

    # if (
    #     len(TM_acceleration_parameters.shape) != len(arms_matrix_rescaled.shape) - 1
    # ) or (TM_acceleration_parameters.shape[-1] != arms_matrix_rescaled.shape[-1]):
    #     raise ValueError(
    #         "TM_acceleration_parameters and arms_matrix_rescaled"
    #         + " do not have compatible shapes",
    #         TM_acceleration_parameters.shape,
    #         arms_matrix_rescaled.shape,
    #     )

    single_link_mat = single_link_TM_acceleration_noise_variance(
        frequency, TM_acceleration_parameters, arms_matrix_rescaled, x_vector
    )

    return tdi_projection(TDI_idx, single_link_mat, arms_matrix_rescaled, x_vector)


def noise_OMS_matrix(
    TDI_idx: ArrayLike,
    frequency: ArrayLike,
    OMS_parameters: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the interferometric (OMS) noise covariance matrix for a TDI
    combination.

    Combines :func:`single_link_OMS_noise_variance` with
    :func:`tdi_projection`.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            selecting the TDI combination to project onto.
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        OMS_parameters (ArrayLike): Per-arm interferometric noise amplitude
            parameters, a vector of length 6.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The TDI OMS noise covariance matrix, with shape
            (configurations, frequency, TDI, TDI).
    """

    # if (len(OMS_parameters.shape) != len(arms_matrix_rescaled.shape) - 1) or (
    #     OMS_parameters.shape[-1] != arms_matrix_rescaled.shape[-1]
    # ):
    #     raise ValueError(
    #         "OMS_parameters and arms_matrix_rescaled"
    #         + " do not have compatible shapes",
    #         OMS_parameters.shape,
    #         arms_matrix_rescaled.shape,
    #     )

    single_link_mat = single_link_OMS_noise_variance(
        frequency, OMS_parameters, arms_matrix_rescaled, x_vector
    )

    return tdi_projection(TDI_idx, single_link_mat, arms_matrix_rescaled, x_vector)


def noise_matrix(
    TDI_idx: ArrayLike,
    frequency: ArrayLike,
    TM_acceleration_parameters: ArrayLike,
    OMS_parameters: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the total (test-mass + OMS) noise covariance matrix for a TDI
    combination.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            selecting the TDI combination to project onto.
        frequency (ArrayLike): Frequency values, in Hz, at which to evaluate
            the noise.
        TM_acceleration_parameters (ArrayLike): Per-arm acceleration noise
            amplitude parameters, a vector of length 6.
        OMS_parameters (ArrayLike): Per-arm interferometric noise amplitude
            parameters, a vector of length 6.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The total TDI noise covariance matrix, with shape
            (configurations, frequency, TDI, TDI).
    """
    return noise_TM_matrix(
        TDI_idx,
        frequency,
        TM_acceleration_parameters,
        arms_matrix_rescaled,
        x_vector,
    ) + noise_OMS_matrix(
        TDI_idx, frequency, OMS_parameters, arms_matrix_rescaled, x_vector
    )


@chex.dataclass
class Noise(object):
    """
    A wrapper around the noise functions in this module that ties them to a
    detector, mirroring the way Response wraps the single_link/tdi functions.

    Attributes:
        ps (chex.dataclass): Physical constants used in the noise
            computations.
        det (chex.dataclass): The detector (e.g. LISA) the noise is computed
            for.
        frequency_array (ArrayLike): Frequency values, in Hz, at which the
            noise is evaluated.
        TM_noise_matrix (dict): Cache of test-mass noise covariance
            matrices computed by :meth:`compute_detector`, keyed by TDI
            combination name.
        OMS_noise_matrix (dict): Cache of OMS noise covariance matrices
            computed by :meth:`compute_detector`, keyed by TDI combination
            name.
        noise_matrix (dict): Cache of total (test-mass + OMS) noise
            covariance matrices computed by :meth:`compute_detector`, keyed by
            TDI combination name.
    """

    ps: chex.dataclass = PhysicalConstants()
    det: chex.dataclass = field(default_factory=lambda: LISA())
    frequency_array: jax.Array = None
    TM_noise_matrix = {}
    OMS_noise_matrix = {}
    noise_matrix = {}

    def __post_init__(self) -> None:
        """Precomputes the ``x_vector`` (``2 pi f L / c``) for the detector."""
        self.x_vector = self.det.x(self.frequency_array)

    def get_arms_matrix_rescaled(self, times_in_years: ArrayLike) -> jax.Array:
        """
        Computes the detector's arm matrix, rescaled by the arm length, at
        the given time(s).

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.

        Returns:
            jax.Array: The rescaled arm matrix, with shape (configurations,
                vectorial_index (3), arms (6)).
        """
        return self.det.detector_arms(times_in_years) / self.det.armlength

    def get_single_link_TM_noise(
        self, times_in_years: ArrayLike, TM_acceleration_parameters: ArrayLike
    ) -> jax.Array:
        """
        Computes the single-link test-mass noise covariance matrix at the
        given time(s). See :func:`single_link_TM_acceleration_noise_variance`.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            TM_acceleration_parameters (ArrayLike): Per-arm acceleration
                noise amplitude parameters, a vector of length 6.

        Returns:
            jax.Array: The single-link test-mass noise covariance matrix,
                with shape (configurations, frequency, arms (6), arms (6)).
        """
        return single_link_TM_acceleration_noise_variance(
            self.frequency_array,
            TM_acceleration_parameters,
            self.get_arms_matrix_rescaled(times_in_years),
            self.x_vector,
        )

    def get_single_link_OMS_noise(
        self, times_in_years: ArrayLike, OMS_parameters: ArrayLike
    ) -> jax.Array:
        """
        Computes the single-link OMS noise covariance matrix at the given
        time(s). See :func:`single_link_OMS_noise_variance`.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            OMS_parameters (ArrayLike): Per-arm interferometric noise
                amplitude parameters, a vector of length 6.

        Returns:
            jax.Array: The single-link OMS noise covariance matrix, with
                shape (configurations, frequency, arms (6), arms (6)).
        """
        return single_link_OMS_noise_variance(
            self.frequency_array,
            OMS_parameters,
            self.get_arms_matrix_rescaled(times_in_years),
            self.x_vector,
        )

    def get_TM_noise_matrix(
        self,
        times_in_years: ArrayLike,
        TM_acceleration_parameters: ArrayLike,
        TDI: str = "XYZ",
    ) -> jax.Array:
        """
        Computes the test-mass noise covariance matrix for a TDI combination
        at the given time(s). See :func:`noise_TM_matrix`.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            TM_acceleration_parameters (ArrayLike): Per-arm acceleration
                noise amplitude parameters, a vector of length 6.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".

        Returns:
            jax.Array: The TDI test-mass noise covariance matrix, with shape
                (configurations, frequency, TDI, TDI).
        """
        return noise_TM_matrix(
            TDI_map[TDI],
            self.frequency_array,
            TM_acceleration_parameters,
            self.get_arms_matrix_rescaled(times_in_years),
            self.x_vector,
        )

    def get_OMS_noise_matrix(
        self,
        times_in_years: ArrayLike,
        OMS_parameters: ArrayLike,
        TDI: str = "XYZ",
    ) -> jax.Array:
        """
        Computes the OMS noise covariance matrix for a TDI combination at
        the given time(s). See :func:`noise_OMS_matrix`.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            OMS_parameters (ArrayLike): Per-arm interferometric noise
                amplitude parameters, a vector of length 6.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".

        Returns:
            jax.Array: The TDI OMS noise covariance matrix, with shape
                (configurations, frequency, TDI, TDI).
        """
        return noise_OMS_matrix(
            TDI_map[TDI],
            self.frequency_array,
            OMS_parameters,
            self.get_arms_matrix_rescaled(times_in_years),
            self.x_vector,
        )

    def compute_detector(
        self,
        times_in_years: ArrayLike,
        TM_acceleration_parameters: ArrayLike,
        OMS_parameters: ArrayLike,
        TDI: str = "XYZ",
    ) -> None:
        """
        Computes and caches the test-mass, OMS, and total noise covariance
        matrices for a TDI combination.

        Results are stored in :attr:`TM_noise_matrix`, :attr:`OMS_noise_matrix`,
        and :attr:`noise_matrix`, keyed by ``TDI``.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            TM_acceleration_parameters (ArrayLike): Per-arm acceleration
                noise amplitude parameters, a vector of length 6.
            OMS_parameters (ArrayLike): Per-arm interferometric noise
                amplitude parameters, a vector of length 6.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".
        """
        self.TM_noise_matrix[TDI] = self.get_TM_noise_matrix(
            times_in_years, TM_acceleration_parameters, TDI=TDI
        )

        self.OMS_noise_matrix[TDI] = self.get_OMS_noise_matrix(
            times_in_years, OMS_parameters, TDI=TDI
        )

        self.noise_matrix[TDI] = self.TM_noise_matrix[TDI] + self.OMS_noise_matrix[TDI]
