# Global imports
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

# Local imports
from gw_response.utils import arm_length_exponential

# Update JAX configuration to enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def detector_output(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Constructs the Michelson-combination mixing matrix for an L-shaped
    ground-based detector (e.g. LIGO): ``h(f) = h12 + D12 h21 - h23 - D23
    h32``, where ``hij`` is the single-link response on arm ``ij`` and
    ``Dij`` is the corresponding light-travel-time delay factor.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (..., vectorial_index (3), arms (4)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency, with shape (frequency,).

    Returns:
        jax.Array: The Michelson-combination mixing matrix, to be applied to
            the single-link response, with shape (..., frequency, channels
            (1), arms (4)).
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
