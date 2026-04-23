# Global imports
import jax
import jax.numpy as jnp


# Local imports
from .utils import arm_length_exponential


# Update JAX configuration to enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def sin_factors(arms_matrix_rescaled, x_vector):
    # arms_matrix_rescaled is configurations, vectorial_index, arms
    # arm_lengths has shape configurations, arms
    # arms ordered as 2-1, 3-2, 1-3, 2-1, 2-3, 3-1
    arm_lengths = jnp.sqrt(
        jnp.sqrt(
            jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
        )
    )

    # xij is configurations, x_vector, arms
    xij = jnp.einsum("i,...j->...ij", x_vector, arm_lengths)

    # This is averaging ij, ji
    single_arm_mean = (xij + jnp.roll(xij, 3, axis=-1)) / 2

    # the output is configurations, x_vector, arms / 2
    return 2j * jnp.sin(single_arm_mean) * jnp.exp(-1j * single_arm_mean)


@jax.jit
def detector_output(arms_matrix_rescaled, x_vector):
    """
    Constructs a Michelson-like response for LIGO:
    h(f) = h12 + D12 h21 - h23 - D23 h32

    Parameters
    ----------
    arms_matrix_rescaled : (..., 3, 4)
        Rescaled arm vectors (unit * length)
    x_vector : (F,)
        Frequency array scaled as x = 2π f L / c

    Returns
    -------
    mix_matrix : (..., F, 1, 4)
        Mixing matrix to be used with single-link h̃(f)
    """

    # Compute frequency-domain delay operator: exp(-i x)
    delays = arm_length_exponential(arms_matrix_rescaled, x_vector)  # (..., F, 4)

    # Shape: (..., F, 1, 4)
    mix_matrix = jnp.zeros((*delays.shape[:-1], 1, 4), dtype=jnp.complex128)

    # Assign coefficients:
    # link 0 ≡ h12       → coefficient +1
    # link 1 ≡ h23       → coefficient -1
    # link 2 ≡ h21       → coefficient +D12
    # link 3 ≡ h32       → coefficient -D23

    mix_matrix = mix_matrix.at[..., 0, 0].set(1.0 + 0j)  # h12
    mix_matrix = mix_matrix.at[..., 0, 1].set(-1.0 + 0j)  # h23
    mix_matrix = mix_matrix.at[..., 0, 2].set(delays[..., 2])  # D12 h21
    mix_matrix = mix_matrix.at[..., 0, 3].set(-delays[..., 3])  # -D23 h32

    return mix_matrix


@jax.jit
def build_datastream(detector_mat, single_link):
    """
    Apply the Michelson detector matrix to per-link data.

    detector_mat: (..., F, 1, 4)   from detector_output()
    single_link:  (..., F, 4, P)   per-link response (e.g. time samples)
    returns:      (..., F, 1, P)
    """
    return jnp.einsum("...ijk,...ikl->...ijl", detector_mat, single_link)
