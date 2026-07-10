# Global imports
import chex
import jax
import jax.numpy as jnp
from dataclasses import field

# Local imports
from .constants import PhysicalConstants
from .lisa import LISA
from .tdi import tdi_matrix, TDI_map
from .utils import arm_length_exponential

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)
# jax.checking_leaks = True


def LISA_acceleration_noise(frequency, acc_param=3.0):
    """
    This compute the acceleration noise spectrum in frequency for a
    certain value of the associated parameter

    Parameters
    ----------
    frequency : np.array (of floats)
    acc_param : float

    Returns
    ----------
    acceleration_noise : np.array (of floats)
    """

    first = 1.0 + (4e-4 / frequency) ** 2
    second = 1.0 + (frequency / 8e-3) ** 4
    third = (2.0 * jnp.pi * frequency) ** (-4) * (2.0 * jnp.pi * frequency / 3e8) ** 2
    # TODO: Change 3e8 to ps.light_speed
    return acc_param**2 * 1e-30 * first * second * third


def LISA_interferometric_noise(frequency, inter_param=15.0):
    """
    This compute the interferometric noise spectrum in frequency for a
    certain value of the associated parameter

    Parameters
    ----------
    frequency   : np.array (of floats)
    inter_param : float

    Returns
    ----------
    interferometric_noise : np.array (of floats)
    """

    first = 1.0 + (2e-3 / frequency) ** 4
    second = (2.0 * jnp.pi * frequency / 3e8) ** 2

    return inter_param**2 * 1e-24 * first * second


def single_link_TM_acceleration_noise_variance(
    frequency,
    TM_acceleration_parameters,
    arms_matrix_rescaled,
    x_vector,
):
    """
    TM_acceleration_parameters is a vector of len 6
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
    frequency, OMS_parameters, arms_matrix_rescaled, x_vector
):
    """TO ADD."""

    # The shape will be configurations, arms, arms
    parameters_matrix = jnp.einsum("ij,...j->...ij", jnp.identity(6), OMS_parameters**2)

    # The shape will be frequency
    N_int = LISA_interferometric_noise(frequency, inter_param=1.0)

    # The shape will be configurations, frequency, arms, arms
    return jnp.einsum("...ij,k->...kij", parameters_matrix, N_int)


def tdi_projection(
    TDI_idx,
    single_link_mat,
    arms_matrix_rescaled,
    x_vector,
):
    """
    TM_acceleration_parameters is a configuration  of len 6
    """

    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # The shape will be configurations, frequency, tdi, arms
    first_contraction = jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link_mat)

    # The shape will be configurations, frequency, tdi, tdi
    res = jnp.einsum("...ijk,...ilk->...ijl", jnp.conjugate(tdi_mat), first_contraction)

    return res


def noise_TM_matrix(
    TDI_idx,
    frequency,
    TM_acceleration_parameters,
    arms_matrix_rescaled,
    x_vector,
):
    """TO ADD."""

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
    TDI_idx,
    frequency,
    OMS_parameters,
    arms_matrix_rescaled,
    x_vector,
):
    """TO ADD."""

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
    TDI_idx,
    frequency,
    TM_acceleration_parameters,
    OMS_parameters,
    arms_matrix_rescaled,
    x_vector,
):
    """TO ADD."""
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
    """

    ps: chex.dataclass = PhysicalConstants()
    det: chex.dataclass = field(default_factory=lambda: LISA())
    frequency_array: jnp.array = None
    TM_noise_matrix = {}
    OMS_noise_matrix = {}
    noise_matrix = {}

    def __post_init__(self):
        self.x_vector = self.det.x(self.frequency_array)

    def get_arms_matrix(self, times_in_years):
        return self.det.detector_arms(times_in_years) / self.det.armlength

    def get_single_link_TM_noise(self, times_in_years, TM_acceleration_parameters):
        return single_link_TM_acceleration_noise_variance(
            self.frequency_array,
            TM_acceleration_parameters,
            self.get_arms_matrix(times_in_years),
            self.x_vector,
        )

    def get_single_link_OMS_noise(self, times_in_years, OMS_parameters):
        return single_link_OMS_noise_variance(
            self.frequency_array,
            OMS_parameters,
            self.get_arms_matrix(times_in_years),
            self.x_vector,
        )

    def get_TM_noise_matrix(
        self, times_in_years, TM_acceleration_parameters, TDI="XYZ"
    ):
        return noise_TM_matrix(
            TDI_map[TDI],
            self.frequency_array,
            TM_acceleration_parameters,
            self.get_arms_matrix(times_in_years),
            self.x_vector,
        )

    def get_OMS_noise_matrix(self, times_in_years, OMS_parameters, TDI="XYZ"):
        return noise_OMS_matrix(
            TDI_map[TDI],
            self.frequency_array,
            OMS_parameters,
            self.get_arms_matrix(times_in_years),
            self.x_vector,
        )

    def compute_detector(
        self, times_in_years, TM_acceleration_parameters, OMS_parameters, TDI="XYZ"
    ):
        self.TM_noise_matrix[TDI] = self.get_TM_noise_matrix(
            times_in_years, TM_acceleration_parameters, TDI=TDI
        )

        self.OMS_noise_matrix[TDI] = self.get_OMS_noise_matrix(
            times_in_years, OMS_parameters, TDI=TDI
        )

        self.noise_matrix[TDI] = self.TM_noise_matrix[TDI] + self.OMS_noise_matrix[TDI]
