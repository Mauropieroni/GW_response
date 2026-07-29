# Global imports
import jax
import jax.numpy as jnp

# Local imports
from gw_response.utils import arm_length_exponential

# Update JAX configuration to enable 64-bit precision
jax.config.update("jax_enable_x64", True)


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
