# Global imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def unit_vec(theta: ArrayLike, phi: ArrayLike) -> jax.Array:
    """
    Computes the unit wavevector pointing from the sky towards the detector for each
    requested sky position -- the sky-position unit vector k̂ used throughout Hartwig,
    Lilley, Muratore & Pieroni (arXiv:2303.15929) Sec. II A (e.g. eq. 2.8's plane-wave
    decomposition); that paper leaves k̂'s sign convention implicit, so this isn't cited
    to a specific equation for the sign itself.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        jax.Array: The unit wavevector(s) in Cartesian coordinates, with shape
            (vectorial_index (3), pixels).
    """
    theta = jnp.atleast_1d(theta)
    phi = jnp.atleast_1d(phi)

    # The output will be vectorial index, pixels
    return jnp.array(
        [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)]
    )


@jax.jit
def uv_analytical(theta: ArrayLike, phi: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """
    Computes the two unit vectors spanning the plane transverse to the propagation
    direction, for each requested sky position.

    These vectors (u, v) form, together with the wavevector from :func:`unit_vec`, a
    right-handed orthonormal triad (``u x v = unit_vec(theta, phi)``) used to build the
    gravitational wave polarization basis -- matching the LDC Manual's (LISA-LCST-SGS-
    MAN-001) Sec. 6.1.2 convention that its own ``(u, v, k)`` be a direct triad, once
    its ``k = -unit_vec(theta, phi)`` antiparallel convention (reproducible by
    evaluating this whole package at the antipodal sky position ``(pi - theta, phi +
    pi)`` instead, rather than any parameter here) is accounted for: flipping the sign
    of one wavevector (k -> -k) requires flipping exactly one of its two transverse
    partners to keep the triad's handedness consistent, which is why only `dk_dphi`
    (not `dk_dtheta`) carries a relative sign here. Equivalently (verified numerically,
    not just by matching variable names), this is the same right-handed construction as
    Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.9's ``û(k̂) = (k̂ x
    ê_z)/|k̂ x ê_z|``, ``v̂(k̂) = k̂ x û``: their ``(û, v̂)`` relate to this function's
    ``(dk_dtheta, dk_dphi)`` as ``dk_dtheta = v̂`` and ``dk_dphi = -û`` (a swap, with
    `dk_dphi` carrying an extra sign) -- not a literal name-for-name match, but the same
    right-handed triad construction relative to ``k̂``.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(u, v)`` of jax.Array, each with shape (pixels, vectorial_index
            (3)), giving the two transverse unit vectors for every sky position.
    """
    theta = jnp.atleast_1d(theta)
    phi = jnp.atleast_1d(phi)

    dk_dtheta = jnp.array(
        [
            jnp.cos(theta) * jnp.cos(phi),
            jnp.cos(theta) * jnp.sin(phi),
            -jnp.sin(theta),
        ]
    ).T
    dk_dphi = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.0 * phi]).T

    # The output will be pixels, vectorial index
    return dk_dtheta, dk_dphi


@jax.jit
def polarization_vectors(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Builds the complex left/right circular polarization vectors from the two transverse
    unit vectors.

    Args:
        u (jax.Array): First transverse unit vector, shape (pixels, vectorial_index
            (3)).
        v (jax.Array): Second transverse unit vector, shape (pixels, vectorial_index
            (3)).

    Returns:
        tuple: A tuple of two complex jax.Array, each with shape (pixels,
            vectorial_index (3)), corresponding to the ``(u - i v) / sqrt(2)`` and ``(u
            + i v) / sqrt(2)`` combinations.
    """
    # The output will be pixels, vectorial index
    return (u - 1j * v) / jnp.sqrt(2), (u + 1j * v) / jnp.sqrt(2)


@jax.jit
def polarization_vectors_angles(
    theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the two transverse unit vectors spanning the plane perpendicular to the
    propagation direction, given the sky position.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(u, v)`` of jax.Array, each with shape (pixels, vectorial_index
            (3)), giving the two transverse unit vectors for every sky position.
    """
    u, v = uv_analytical(theta, phi)
    return polarization_vectors(u, v)


@jax.jit
def polarization_tensors_PC(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Computes the plus/cross gravitational wave polarization tensors: Hartwig, Lilley,
    Muratore & Pieroni (arXiv:2303.15929) eq. 2.10's ``e^+_ab = û_a û_b - v̂_a v̂_b``,
    ``e^×_ab = û_a v̂_b + v̂_a û_b`` exactly, with no residual normalization
    difference.

    Args:
        u (jax.Array): First transverse unit vector, shape (pixels, vectorial_index
            (3)).
        v (jax.Array): Second transverse unit vector, shape (pixels, vectorial_index
            (3)).

    Returns:
        tuple: A tuple ``(e_plus, e_cross)`` of jax.Array, each with shape (pixels,
            vectorial_index (3), vectorial_index (3)), giving the plus and cross
            polarization tensors for every sky position.
    """
    e1p = jnp.einsum("...i,...j->...ij", u, u) - jnp.einsum("...i,...j->...ij", v, v)
    e1c = jnp.einsum("...i,...j->...ij", u, v) + jnp.einsum("...i,...j->...ij", v, u)

    # The output will be pixels, vectorial index, vectorial index
    return e1p, e1c


@jax.jit
def polarization_tensors_PC_angles(
    theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the plus/cross gravitational wave polarization tensors, given the sky
    position.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(e_plus, e_cross)`` of jax.Array, each with shape (pixels,
            vectorial_index (3), vectorial_index (3)), giving the plus and cross
            polarization tensors for every sky position.
    """
    u, v = uv_analytical(theta, phi)
    return polarization_tensors_PC(u, v)


@jax.jit
def polarization_tensors_LR(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Computes the left/right circular gravitational wave polarization tensors: Hartwig,
    Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.10's ``e^{L/R}_ab = e^+_ab ∓
    i*e^×_ab`` exactly, with no residual normalization difference. Built directly from
    :func:`polarization_tensors_PC`'s (unnormalized) ``e_plus``/``e_cross`` rather than
    by squaring :func:`polarization_vectors`' separately-normalized ``(u ∓ i v) /
    sqrt(2)`` vectors, which would reintroduce an extra factor of 1/2.

    Args:
        u (jax.Array): First transverse unit vector, shape (pixels, vectorial_index
            (3)).
        v (jax.Array): Second transverse unit vector, shape (pixels, vectorial_index
            (3)).

    Returns:
        tuple: A tuple ``(e_L, e_R)`` of complex jax.Array, each with shape (pixels,
            vectorial_index (3), vectorial_index (3)), giving the left and right
            circular polarization tensors for every sky position.
    """
    e_plus, e_cross = polarization_tensors_PC(u, v)
    e1L = e_plus - 1j * e_cross
    e1R = e_plus + 1j * e_cross

    # The output will be pixels, vectorial index, vectorial index
    return e1L, e1R


@jax.jit
def polarization_tensors_LR_angles(
    theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the left/right circular gravitational wave polarization tensors, given the
    sky position.

    Args:
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(e_L, e_R)`` of complex jax.Array, each with shape (pixels,
            vectorial_index (3), vectorial_index (3)), giving the left and right
            circular polarization tensors for every sky position.
    """
    u, v = uv_analytical(theta, phi)
    return polarization_tensors_LR(u, v)


def pc_tensors_and_wavevector(
    theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Plus/cross polarization tensors and the unit wavevector for a sky position.

    Args:
        theta (ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: ``(wavevector, p_plus, p_cross)``.
    """
    u, v = uv_analytical(theta, phi)
    p_plus, p_cross = polarization_tensors_PC(u, v)
    wavevector = unit_vec(theta, phi)
    return wavevector, p_plus, p_cross


def lr_tensors_and_wavevector(
    theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Left/right polarization tensors and the unit wavevector for a sky position. LR
    sibling of :func:`pc_tensors_and_wavevector`.

    Args:
        theta (ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: ``(wavevector, p_L, p_R)``.
    """
    u, v = uv_analytical(theta, phi)
    p_L, p_R = polarization_tensors_LR(u, v)
    wavevector = unit_vec(theta, phi)
    return wavevector, p_L, p_R


def polarization_tensors_and_wavevector(
    polarization: str, theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Dispatches to :func:`pc_tensors_and_wavevector` or
    :func:`lr_tensors_and_wavevector` for the requested polarization basis.

    Args:
        polarization (str): "PC" or "LR" (case-insensitive).
        theta (ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: ``(wavevector, p1, p2)``, `p1`/`p2` its two polarization tensors (e.g.
            `p_plus`/`p_cross` for "PC").

    Raises:
        ValueError: If `polarization` is not "PC" or "LR".
    """
    pol = polarization.upper()
    if pol == "PC":
        return pc_tensors_and_wavevector(theta, phi)
    elif pol == "LR":
        return lr_tensors_and_wavevector(theta, phi)
    raise ValueError("Incorrect polarization type")
