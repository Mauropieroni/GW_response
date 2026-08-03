# Global imports
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
import interpax
from dataclasses import field

from jax.typing import ArrayLike

# Local imports
from gw_response.constants import PhysicalConstants
from gw_response.detector import Detector
from gw_response.noise import Noise
from gw_response.response import Response
from gw_response.utils import combine_single_link
from gw_response.ground_based.datastream import (
    detector_output,
    detector_output_long_wavelength,
)

# -----------------------------------------------------------------------------
# -- Earth & Arm Constants ---------------------------------------------------
# -----------------------------------------------------------------------------
REARTH = 6.371e6  # Earth radius in meters
LIGOARM = 4e3  # LIGO arm length in meters

# -----------------------------------------------------------------------------
# -- LIGO Design Sensitivity Curve --------------------------------------------
# -----------------------------------------------------------------------------
_path_to_LIGO_design = os.path.join(os.path.dirname(__file__), "noise_data", "LIGO.npz")
with np.load(_path_to_LIGO_design) as _ligo_design_curves:
    _ligo_freqs = _ligo_design_curves["Frequency"]
    _ligo_psd = _ligo_design_curves["Mid high/Late low"]
_ligo_interp = interpax.Interpolator1D(
    _ligo_freqs, _ligo_psd, method="linear", extrap=True
)


def LIGO_noise(frequencies: ArrayLike) -> jax.Array:
    """
    Returns the LIGO design PSD at `frequencies`, interpolated from the tabulated design
    curve. `_ligo_interp` is a jax-traceable `interpax.Interpolator1D`, so it can be
    called directly with either a concrete array (outside jit) or a traced one (e.g.
    from a jitted caller like `Noise.get_single_link_noise`).

    Args:
        frequencies (ArrayLike): Frequency values, in Hz, at which to evaluate the
            noise.

    Returns:
        jax.Array: The LIGO design noise PSD, with the same shape as ``frequencies``.
    """
    return _ligo_interp(jnp.asarray(frequencies))


def single_link_LIGO_noise_variance(frequencies: ArrayLike) -> jax.Array:
    """
    Returns the 1D Michelson-output PSD S_n(f) for LIGO.

    Args:
        frequencies (ArrayLike): Frequency values, in Hz, at which to evaluate the
            noise.

    Returns:
        jax.Array: The Michelson-output noise PSD, with shape (frequency,).
    """
    return LIGO_noise(frequencies)


# -----------------------------------------------------------------------------
# -- LIGO Static Site Geometries (ECEF) --------------------------------------
# -----------------------------------------------------------------------------
_SITE_GEOMETRIES = {
    "Hanford": {
        "center": jnp.array([-0.33827472, -0.60015338, 0.72483525]) * REARTH,
        "arm1": jnp.array([-0.22389266154, 0.79983062746, 0.55690487831]),
        "arm2": jnp.array([-0.91397818574, 0.02609403989, -0.40492342125]),
    },
    "Livingston": {
        "center": jnp.array([-0.01163537, -0.8609929, 0.50848387]) * REARTH,
        "arm1": jnp.array([-0.95457412153, -0.14158077340, -0.26218911324]),
        "arm2": jnp.array([0.29774156894, -0.48791033647, -0.82054461286]),
    },
}


# -----------------------------------------------------------------------------
# -- JIT-Compiled Helpers for Static Geometries ------------------------------
# -----------------------------------------------------------------------------
@jax.jit
def _rotate_about_z(vec: jax.Array, angle: ArrayLike) -> jax.Array:
    """
    Rotates a fixed 3-vector about the (ECEF) z-axis -- Earth's rotation axis, to
    standard approximation ignoring precession/nutation/polar motion -- by one angle per
    configuration.

    Args:
        vec (jax.Array): A fixed vector, with shape (vectorial_index (3),).
        angle (ArrayLike): Rotation angle(s), in radians, with shape (configurations,).

    Returns:
        jax.Array: The rotated vector(s), with shape (configurations, vectorial_index
            (3)).
    """
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    x, y, z = vec[0], vec[1], vec[2]
    new_x = cos_a * x - sin_a * y
    new_y = sin_a * x + cos_a * y
    new_z = jnp.broadcast_to(z, cos_a.shape)
    return jnp.stack([new_x, new_y, new_z], axis=-1)


@jax.jit
def LIGO_positions(
    center: jax.Array,
    arm1: jax.Array,
    arm2: jax.Array,
    armlength: ArrayLike,
    rotation_angle: ArrayLike,
) -> jax.Array:
    """
    Computes the Cartesian positions of a LIGO-like L-shaped detector's 3 vertices --
    the corner (beamsplitter) station and the two end mirrors.

    LIGO's site geometry is static in the (rotating) ECEF frame; whether the returned
    positions are also static in the (non-rotating) frame that sky positions are defined
    in depends on `rotation_angle`: pass an all-zeros array (e.g. from
    `LIGO.include_earth_rotation=False`) to keep `center`/`arm1`/`arm2` fixed, matching
    the non-rotating-Earth approximation.

    Args:
        center (ArrayLike): ECEF unit-vector position of the detector's corner
            (beamsplitter) station, with shape (vectorial_index (3),).
        arm1 (ArrayLike): Unit vector along the first arm, with shape (vectorial_index
            (3),).
        arm2 (ArrayLike): Unit vector along the second arm, with shape (vectorial_index
            (3),).
        armlength (ArrayLike): Arm length, in meters.
        rotation_angle (ArrayLike): Earth's rotation angle(s) about the ECEF z-axis, in
            radians, with shape (configurations,).

    Returns:
        jax.Array: The 3 vertex positions -- ordered [corner, end mirror 1, end mirror
            2], matching :meth:`LIGO.arm_vertex_pairs`'s vertex indexing -- with shape
            (configurations, vectorial_index (3), vertices (3)).
    """
    center_t = _rotate_about_z(center, rotation_angle)
    xm_t = _rotate_about_z(center + arm1 * armlength, rotation_angle)
    ym_t = _rotate_about_z(center + arm2 * armlength, rotation_angle)
    return jnp.stack([center_t, xm_t, ym_t], axis=-1)


@jax.jit
def LIGO_arms_matrix(
    arm1: jax.Array,
    arm2: jax.Array,
    armlength: ArrayLike,
    rotation_angle: ArrayLike,
) -> jax.Array:
    """
    Computes the arm matrix of a LIGO-like L-shaped detector: the two physical arm
    vectors and their reverse-direction counterparts (mirroring the "forward + reverse"
    arm-doubling convention used for LISA's 6-arm matrix).

    See :func:`LIGO_positions` regarding `rotation_angle`.

    Args:
        arm1 (ArrayLike): Unit vector along the first arm, with shape (vectorial_index
            (3),).
        arm2 (ArrayLike): Unit vector along the second arm, with shape (vectorial_index
            (3),).
        armlength (ArrayLike): Arm length, in meters.
        rotation_angle (ArrayLike): Earth's rotation angle(s) about the ECEF z-axis, in
            radians, with shape (configurations,).

    Returns:
        jax.Array: The arm matrix, with shape (configurations, vectorial_index (3), arms
            (4)), ordered [arm1, arm2, -arm1, -arm2].
    """
    vec1 = _rotate_about_z(arm1, rotation_angle) * armlength
    vec2 = _rotate_about_z(arm2, rotation_angle) * armlength
    return jnp.stack([vec1, vec2, -vec1, -vec2], axis=-1)


# -----------------------------------------------------------------------------
# -- Unified LIGO Detector Class ---------------------------------------------
# -----------------------------------------------------------------------------
@chex.dataclass(frozen=True)
class LIGO(Detector):
    """
    A data class representing a LIGO-like L-shaped ground-based gravitational-wave
    detector. Unlike LISA, its site geometry is static (fixed to the Earth's surface)
    with a single Michelson readout channel rather than several TDI combinations.

    Attributes:
        which_detector (str): Name of the LIGO site to use, a key of `_SITE_GEOMETRIES`
            ("Hanford" or "Livingston").
        name (str): Human-readable name, set in `__post_init__`.
        ps (PhysicalConstants): Physical constants used in the noise/ response
            computations.
        fmin (float): Minimum frequency of LIGO's sensitive band, in Hz.
        fmax (float): Maximum frequency of LIGO's sensitive band, in Hz.
        armlength (float): LIGO's arm length, in meters.
        res (float): Frequency resolution used by `frequency_vec`, in Hz.
        default_combination (str): The only combination LIGO supports, "Michelson".
        center (jax.Array): ECEF unit-vector position of the site's corner
            (beamsplitter) station. Set in `__post_init__`.
        arm1 (jax.Array): Unit vector along the first arm. Set in `__post_init__`.
        arm2 (jax.Array): Unit vector along the second arm. Set in `__post_init__`.
        include_earth_rotation (bool): If True, `vertex_positions`/ `detector_arms`
            rotate about the ECEF z-axis with Earth's sidereal period, so orientation
            evolves relative to the (non-rotating) `theta`/`phi` sky frame. Default
            False (fixed).
        long_wavelength_approximation (bool): If True, drops the finite-arm-length
            correction and uses only the (frequency-independent) antenna-pattern factor,
            as in `gw_fast`/ `gw_fish`. Default False (exact, frequency-dependent).
        response (Response): Computes LIGO's response to gravitational waves.
        noise (Noise): Computes LIGO's noise budget.
    """

    which_detector: str = "Hanford"
    name: str = "LIGO Hanford"
    ps: PhysicalConstants = PhysicalConstants()
    fmin: float = 1.0
    fmax: float = 2e3
    armlength: float = 4e3
    res: float = 1e-1
    default_combination: str = "Michelson"
    include_earth_rotation: bool = False
    long_wavelength_approximation: bool = False
    center: jax.Array = field(init=False, default_factory=lambda: jnp.zeros(3))
    arm1: jax.Array = field(init=False, default_factory=lambda: jnp.zeros(3))
    arm2: jax.Array = field(init=False, default_factory=lambda: jnp.zeros(3))
    response: Response = field(init=False, default_factory=Response)
    noise: Noise = field(init=False, default_factory=Noise)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __post_init__(self) -> None:
        """
        Looks up `which_detector` in `_SITE_GEOMETRIES` and sets `name`, `center`,
        `arm1`, and `arm2` accordingly.

        Raises:
            ValueError: If `which_detector` is not a key of `_SITE_GEOMETRIES`.
        """
        if self.which_detector not in _SITE_GEOMETRIES:
            raise ValueError(f"Unknown LIGO site '{self.which_detector}'")
        object.__setattr__(self, "name", f"LIGO {self.which_detector}")
        geom = _SITE_GEOMETRIES[self.which_detector]
        object.__setattr__(self, "center", geom["center"])
        object.__setattr__(self, "arm1", geom["arm1"])
        object.__setattr__(self, "arm2", geom["arm2"])

    def _earth_rotation_angle(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Earth's rotation angle(s) about the ECEF z-axis, relative to `time_in_years=0`.
        Zero everywhere when `include_earth_rotation` is False, via a plain boolean-gate
        multiply rather than branching, so `LIGO_positions`/`LIGO_arms_matrix` always
        take the same code path.

        Args:
            time_in_years (ArrayLike): Time(s), in years.

        Returns:
            jax.Array: The rotation angle(s), in radians, with the same shape as
                `time_in_years`.
        """
        omega_earth = 2 * jnp.pi / self.ps.sidereal_day
        return jnp.asarray(
            self.include_earth_rotation * omega_earth * time_in_years * self.ps.yr
        )

    def _vertex_positions(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes the positions of LIGO's 3 vertices -- the corner station and the two
        end mirrors -- at the given time(s). See :func:`LIGO_positions`.

        Args:
            time_in_years (ArrayLike): Time(s), in years.

        Returns:
            jax.Array: The vertex positions, as returned by :func:`LIGO_positions`.
        """
        return LIGO_positions(
            self.center,
            self.arm1,
            self.arm2,
            self.armlength,
            self._earth_rotation_angle(time_in_years),
        )

    def _detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes LIGO's arm matrix at the given time(s). See :func:`LIGO_arms_matrix`.

        Args:
            time_in_years (ArrayLike): Time(s), in years.

        Returns:
            jax.Array: The arm matrix, as returned by :func:`LIGO_arms_matrix`.
        """
        return LIGO_arms_matrix(
            self.arm1,
            self.arm2,
            self.armlength,
            self._earth_rotation_angle(time_in_years),
        )

    @property
    def arm_vertex_pairs(self) -> tuple[tuple[int, int], ...]:
        """
        The 4 (receiver, emitter) vertex-index pairs for LIGO's round-trip Michelson,
        decomposed into one-way legs exactly like LISA's links: vertex 0 is the corner
        station, 1 and 2 are the two end mirrors. Each arm's forward leg (corner to
        mirror) and return leg (mirror to corner) is its own one-way link, matching
        :func:`LIGO_arms_matrix`'s existing column order ``[corner->mirror1,
        corner->mirror2, mirror1->corner, mirror2->corner]``.

        Returns:
            tuple[tuple[int, int], ...]: ``((0, 1), (0, 2), (1, 0), (2, 0))``.
        """
        return ((0, 1), (0, 2), (1, 0), (2, 0))

    def combination_matrix(
        self, combination: str, arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
    ) -> jax.Array:
        """
        Builds the Michelson-combination mixing matrix. See
        :func:`gw_response.ground_based.datastream.detector_output` (or, if
        `long_wavelength_approximation` is set,
        :func:`gw_response.ground_based.datastream.detector_output_long_wavelength`).

        Args:
            combination (str): Must be `self.default_combination` ("Michelson"); LIGO
                supports no other readout combination.
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
                length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

        Returns:
            jax.Array: The Michelson-combination mixing matrix.

        Raises:
            ValueError: If ``combination`` is not "Michelson".
        """
        if combination != self.default_combination:
            raise ValueError(
                f"LIGO only supports the '{self.default_combination}' combination"
            )
        if self.long_wavelength_approximation:
            return detector_output_long_wavelength(arms_matrix_rescaled, x_vector)
        return detector_output(arms_matrix_rescaled, x_vector)

    def linear_response_from_single_link(
        self, single_link: dict[str, jax.Array], combination_matrix: ArrayLike
    ) -> dict[str, jax.Array]:
        """
        Applies the Michelson-combination mixing matrix to each polarization's
        single-link response.

        LIGO's Michelson combination has a single readout channel, so that trivial axis
        is dropped from the result rather than carried around as a dangling dimension.

        Args:
            single_link (dict): Single-link response per polarization.
            combination_matrix (ArrayLike): Mixing matrix as built by
                :meth:`combination_matrix`.

        Returns:
            dict: The Michelson-combination linear response per polarization.
        """
        return {
            p: jnp.squeeze(combine_single_link(combination_matrix, sl), axis=-2)
            for p, sl in single_link.items()
        }

    def quadratic_response_from_single_link(
        self, linear_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Computes the quadratic response as the squared modulus of the linear response.

        Args:
            linear_integrand (dict): Linear response per polarization, as returned by
                :meth:`linear_response_from_single_link`.

        Returns:
            dict: The quadratic response per polarization.
        """
        return {p: jnp.abs(linear) ** 2 for p, linear in linear_integrand.items()}

    def integrate_quadratic_response(
        self, quadratic_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Passes the quadratic response through unchanged.

        Unlike LISA's sky-averaging, LIGO's PSD weighting and integration over pixels is
        left to the caller, so there is nothing to do here.

        Args:
            quadratic_integrand (dict): Quadratic response per polarization, as returned
                by :meth:`quadratic_response_from_single_link`.

        Returns:
            dict: ``quadratic_integrand``, unchanged.
        """
        return quadratic_integrand

    def single_link_noise(
        self,
        frequency_array: ArrayLike,
        arms_matrix_rescaled: ArrayLike,
        x_vector: ArrayLike,
        **_,
    ) -> jax.Array:
        """
        Returns the LIGO design noise PSD. See :func:`single_link_LIGO_noise_variance`.

        LIGO's noise model here is a single already-combined PSD curve; there is no
        per-link decomposition to build from ``arms_matrix_rescaled`` or ``x_vector``,
        so they're accepted (for interface parity with `Detector.single_link_noise`) but
        unused. LIGO also takes no detector-specific noise parameters.

        Args:
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the noise.
            arms_matrix_rescaled (ArrayLike): Unused.
            x_vector (ArrayLike): Unused.

        Returns:
            jax.Array: The LIGO design noise PSD.
        """
        return single_link_LIGO_noise_variance(frequency_array)

    def project_noise(
        self, combination_matrix: ArrayLike, single_link_noise: jax.Array
    ) -> jax.Array:
        """
        Returns `single_link_noise` unchanged.

        `single_link_noise` (as built by :meth:`single_link_noise`) is already the
        readout-domain PSD, so there is nothing left to project.

        Args:
            combination_matrix (ArrayLike): Unused.
            single_link_noise (jax.Array): The noise PSD as built by
                :meth:`single_link_noise`.

        Returns:
            jax.Array: `single_link_noise`, unchanged.
        """
        return single_link_noise
