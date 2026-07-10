# Global imports
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Local imports
from .tdi import tdi_matrix
from .utils import arm_length_exponential

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def unit_vec(theta: ArrayLike, phi: ArrayLike) -> jax.Array:
    """
    Computes the unit wavevector pointing from the sky towards the detector
    for each requested sky position.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in
            radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in
            radians.

    Returns:
        jax.Array: The unit wavevector(s) in Cartesian coordinates, with
            shape (vectorial_index (3), pixels).
    """
    theta = jax.lax.cond(
        isinstance(theta, float),
        lambda theta: jnp.reshape(jnp.array([theta]), (-1,)),
        lambda theta: jnp.reshape(jnp.array(theta), (-1,)),
        theta,
    )
    phi = jax.lax.cond(
        isinstance(phi, float),
        lambda phi: jnp.reshape(jnp.array([phi]), (-1,)),
        lambda phi: jnp.reshape(jnp.array(phi), (-1,)),
        phi,
    )
    # The output will be vectorial index, pixels
    return jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )


@jax.jit
def uv_analytical(theta: ArrayLike, phi: ArrayLike) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the two unit vectors spanning the plane transverse to the
    propagation direction, for each requested sky position.

    These vectors (u, v) form, together with the wavevector from
    :func:`unit_vec`, an orthonormal triad used to build the gravitational
    wave polarization basis.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in
            radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in
            radians.

    Returns:
        tuple: A tuple ``(u, v)`` of jax.Array, each with shape (pixels,
            vectorial_index (3)), giving the two transverse unit vectors for
            every sky position.
    """
    theta = jax.lax.cond(
        isinstance(theta, float),
        lambda theta: jnp.reshape(jnp.array([theta]), (-1,)),
        lambda theta: jnp.reshape(jnp.array(theta), (-1,)),
        theta,
    )
    phi = jax.lax.cond(
        isinstance(phi, float),
        lambda phi: jnp.reshape(jnp.array([phi]), (-1,)),
        lambda phi: jnp.reshape(jnp.array(phi), (-1,)),
        phi,
    )
    # The output will be pixels, vectorial index
    return (
        jnp.array(
            [
                jnp.cos(theta) * jnp.cos(phi),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        ).T,
        jnp.array([jnp.sin(phi), -jnp.cos(phi), 0.0 * phi]).T,
    )


@jax.jit
def polarization_vectors(u: ArrayLike, v: ArrayLike) -> Tuple[jax.Array, jax.Array]:
    """
    Builds the complex left/right circular polarization vectors from the two
    transverse unit vectors.

    Args:
        u (ArrayLike): First transverse unit vector, shape (pixels,
            vectorial_index (3)).
        v (ArrayLike): Second transverse unit vector, shape (pixels,
            vectorial_index (3)).

    Returns:
        tuple: A tuple of two complex jax.Array, each with shape (pixels,
            vectorial_index (3)), corresponding to the ``(u - i v) / sqrt(2)``
            and ``(u + i v) / sqrt(2)`` combinations.
    """
    # The output will be pixels, vectorial index
    return (u - 1j * v) / jnp.sqrt(2), (u + 1j * v) / jnp.sqrt(2)


@jax.jit
def polarization_tensors_PC(u: ArrayLike, v: ArrayLike) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the plus/cross gravitational wave polarization tensors.

    Args:
        u (ArrayLike): First transverse unit vector, shape (pixels,
            vectorial_index (3)).
        v (ArrayLike): Second transverse unit vector, shape (pixels,
            vectorial_index (3)).

    Returns:
        tuple: A tuple ``(e_plus, e_cross)`` of jax.Array, each with shape
            (pixels, vectorial_index (3), vectorial_index (3)), giving the plus
            and cross polarization tensors for every sky position.
    """
    e1p = jnp.einsum("...i,...j->...ij", u, u) - jnp.einsum("...i,...j->...ij", v, v)
    e1c = jnp.einsum("...i,...j->...ij", u, v) + jnp.einsum("...i,...j->...ij", v, u)

    # The output will be pixels, vectorial index, vectorial index
    return e1p / jnp.sqrt(2), e1c / jnp.sqrt(2)


@jax.jit
def polarization_tensors_LR(u: ArrayLike, v: ArrayLike) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the left/right circular gravitational wave polarization
    tensors.

    Args:
        u (ArrayLike): First transverse unit vector, shape (pixels,
            vectorial_index (3)).
        v (ArrayLike): Second transverse unit vector, shape (pixels,
            vectorial_index (3)).

    Returns:
        tuple: A tuple ``(e_L, e_R)`` of complex jax.Array, each with shape
            (pixels, vectorial_index (3), vectorial_index (3)), giving the left
            and right circular polarization tensors for every sky position.
    """
    first, second = polarization_vectors(u, v)
    e1L = jnp.einsum("...i,...j->...ij", first, first)
    e1R = jnp.einsum("...i,...j->...ij", second, second)

    # The output will be pixels, vectorial index, vectorial index
    return e1L, e1R


@jax.jit
def xi_k_no_G(
    unit_wavevector: ArrayLike, x_vector: ArrayLike, arms_mat_rescaled: ArrayLike
) -> jax.Array:
    """x_vector is a vector over frequency (2 pi f L / c) unit_wavevector is
    vectorial index (3), pixels arms_mat_norm is (configurations (num time
    slices),) vectorial_index (3), arms (6)

    This returns an object with shape:
    configurations, x_vector, arms, pixels

    2.5 of the present draft without G
    """

    k_dot_arms = jnp.einsum("...ij,ik->...jk", arms_mat_rescaled, unit_wavevector)
    # These guys will be configurations, arms, pixel
    comb_plus = 1 + k_dot_arms
    comb_minus = 1 - k_dot_arms

    # These guys are  configurations, x_vector, arms, pixels
    prod_plus = jnp.einsum("i,...kl->...ikl", x_vector, comb_plus)
    prod_minus = jnp.einsum("i,...kl->...ikl", x_vector, comb_minus)

    # These guys are configurations, x_vector, arms, pixels
    return jnp.exp(0.5j * prod_minus) * jnp.sinc(prod_plus / 2 / jnp.pi)


@jax.jit
def position_exponential(
    positions_detector_frame: ArrayLike, unit_wavevector: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """x_vector is a vector unit_wavevector is vectorial index, pixels
    positions_detector_frame is configurations, vectorial_index, satellite (3)
    # Need shifted positions to get the numerical precision in the dot
    product."""

    # This is configurations, satellite, pixels
    scalar = jnp.einsum("...ij,ik->...jk", positions_detector_frame, unit_wavevector)
    exponent = jnp.einsum("i,...jk->...ijk", -1j * x_vector, scalar)

    # Output is configurations, x_vector, satellite, pixels
    return jnp.exp(exponent)


@jax.jit
def geometrical_factor(
    arms_matrix: ArrayLike, polarization_tensor: ArrayLike
) -> jax.Array:
    """
    Projects the gravitational wave polarization tensor onto each detector
    arm, giving the geometrical antenna-pattern factor of the single-link
    response.

    Args:
        arms_matrix (ArrayLike): Detector arm vectors, with shape
            (configurations, vectorial_index (3), arms (6)).
        polarization_tensor (ArrayLike): Polarization tensor(s) as returned
            by e.g. :func:`polarization_tensors_LR`, with shape (pixels,
            vectorial_index (3), vectorial_index (3)).

    Returns:
        jax.Array: The geometrical factor, with shape (configurations, arms,
            pixels).
    """
    # arms_matrix is configurations, vectorial_index, arms
    # polarization_tensor is pixels, vectorial_index, vectorial_index

    aux = jnp.einsum("...ik,...jk->...ijk", arms_matrix, arms_matrix / 2)

    # aux is configurations, arms, vectorial_index, pixels
    # aux = jnp.einsum("...ij,ilk->...jlk", arms_matrix, polarization_tensor.T)

    # the output is configurations, arms, pixels
    return jnp.einsum("...ijk,...ijl->...kl", aux, polarization_tensor.T)


@jax.jit
def xi_k_Avec_func(
    arms_matrix_rescaled: ArrayLike,
    unit_wavevector: ArrayLike,
    x_vector: ArrayLike,
    geometrical: ArrayLike,
) -> jax.Array:
    """
    Combines the finite-arm-length transfer function (:func:`xi_k_no_G`)
    with the geometrical antenna-pattern factor to give the single-link
    response kernel, prior to the light-travel-time and position phase
    factors.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        unit_wavevector (ArrayLike): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.
        geometrical (ArrayLike): Geometrical antenna-pattern factor as
            returned by :func:`geometrical_factor`, with shape (configurations,
            arms, pixels).

    Returns:
        jax.Array: The single-link response kernel, with shape
            (configurations, x_vector, arms, pixels).
    """

    # xi_vec is configurations, x_vector, arms, pixels
    xi_vec = xi_k_no_G(unit_wavevector, x_vector, arms_matrix_rescaled)

    # The output is configurations, x_vector, arms, pixels
    return jnp.einsum("...ijk,...jk->...ijk", xi_vec, geometrical)


@jax.jit
def single_link_response(
    positions: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    wavevector: ArrayLike,
    x_vector: ArrayLike,
    xi_k_Avec: ArrayLike,
) -> jax.Array:
    """
    Computes the full single-link (arm) strain response, combining the
    response kernel with the light-travel-time delay and the satellite
    position phase factors.

    Args:
        positions (ArrayLike): Satellite positions, with shape
            (configurations, vectorial_index (3), satellite (3)).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        wavevector (ArrayLike): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.
        xi_k_Avec (ArrayLike): Single-link response kernel as returned by
            :func:`xi_k_Avec_func`, with shape (configurations, x_vector, arms,
            pixels).

    Returns:
        jax.Array: The single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
    """
    # positions is configuration, vector, masses (masses are 1,2,3)
    all_position = jnp.concatenate(
        (positions, jnp.roll(positions, -1, axis=-1)), axis=-1
    )

    # exp has shape configurations, x_vector, arms, pixels
    position_exp_factor = position_exponential(all_position, wavevector, x_vector)

    # this guy will be configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    # this guy will be configurations, arms
    arm_lengths = jnp.sqrt(
        jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
    )

    # This will be configurations, x_vector, arms
    prefactor = jnp.einsum("...j,...ij->...ij", arm_lengths, t_retarded_factor)
    # Need to pre-multiply by x to convert to fractional frequency in single
    # link response
    prefactor = jnp.einsum("i,...ij->...ij", x_vector, prefactor)

    # This will be configurations, x_vector, arms, pixels
    return jnp.einsum(
        "...ij,...ijk->...ijk", prefactor, position_exp_factor * xi_k_Avec
    )


@jax.jit
def get_single_link_response(
    polarization: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    wavevector: ArrayLike,
    x_vector: ArrayLike,
    positions: ArrayLike,
) -> jax.Array:
    """
    Computes the single-link strain response for a given polarization
    tensor, tying together the geometrical factor, the response kernel, and
    the phase factors.

    Args:
        polarization (ArrayLike): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)), e.g. one of the tensors
            returned by :func:`polarization_tensors_LR` or
            :func:`polarization_tensors_PC`.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        wavevector (ArrayLike): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.
        positions (ArrayLike): Satellite positions, with shape
            (configurations, vectorial_index (3), satellite (3)).

    Returns:
        jax.Array: The single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
    """
    # This computes the geometrical factor
    geometrical = geometrical_factor(arms_matrix_rescaled, polarization)

    # This computes the xi vectors
    xi_k_vec = xi_k_Avec_func(arms_matrix_rescaled, wavevector, x_vector, geometrical)

    # This will be configurations, x_vector, arms, pixels
    return single_link_response(
        positions, arms_matrix_rescaled, wavevector, x_vector, xi_k_vec
    )


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
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            (or :data:`gw_response.tdi.tdi_fun_list`) selecting the TDI
            combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`get_single_link_response`, with shape (configurations,
            x_vector, arms, pixels).
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
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            (or :data:`gw_response.tdi.tdi_fun_list`) selecting the TDI
            combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`get_single_link_response`, with shape (configurations,
            x_vector, arms, pixels).
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
        TDI_idx (ArrayLike): Index into :data:`gw_response.tdi.TDI_map`
            (or :data:`gw_response.tdi.tdi_fun_list`) selecting the TDI
            combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`get_single_link_response`, with shape (configurations,
            x_vector, arms, pixels).
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
            returned by :func:`quadratic_integrand`, with shape (configurations,
            x_vector, TDI, TDI, pixels).

    Returns:
        jax.Array: The sky-averaged quadratic response, with shape
            (configurations, x_vector, TDI, TDI), normalized by ``4 * pi`` to
            account for the solid angle of the sphere.
    """
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)
