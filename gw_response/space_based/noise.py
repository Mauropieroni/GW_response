# Global imports
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

# Local imports
from gw_response.utils import arm_length_exponential, project_noise_matrix
from gw_response.space_based.tdi import tdi_matrix

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@jax.jit
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
    return jnp.asarray(acc_param**2 * 1e-30 * first * second * third)


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

    return jnp.asarray(inter_param**2 * 1e-24 * first * second)


@jax.jit
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


@jax.jit
def tdi_projection(
    TDI_idx: ArrayLike,
    single_link_mat: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Projects a single-link noise covariance matrix onto a TDI combination.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
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

    # The shape will be configurations, frequency, tdi, tdi
    return project_noise_matrix(tdi_mat, single_link_mat)


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
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
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
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
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
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
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
