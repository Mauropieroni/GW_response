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
from gw_response.ground_based.datastream import detector_output

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
    Returns the LIGO design PSD at `frequencies`, interpolated from the
    tabulated design curve. `_ligo_interp` is a jax-traceable
    `interpax.Interpolator1D`, so it can be called directly with either a
    concrete array (outside jit) or a traced one (e.g. from a jitted caller
    like `Noise.get_single_link_noise`).

    Args:
        frequencies (ArrayLike): Frequency values, in Hz, at which to
            evaluate the noise.

    Returns:
        jax.Array: The LIGO design noise PSD, with the same shape as
            ``frequencies``.
    """
    return _ligo_interp(jnp.asarray(frequencies))


def single_link_LIGO_noise_variance(frequencies: ArrayLike) -> jax.Array:
    """
    Returns the 1D Michelson-output PSD S_n(f) for LIGO.

    Args:
        frequencies (ArrayLike): Frequency values, in Hz, at which to
            evaluate the noise.

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
def LIGO_positions(
    time_in_years: ArrayLike,
    center: jax.Array,
    arm1: jax.Array,
    arm2: jax.Array,
    armlength: ArrayLike,
) -> jax.Array:
    """
    Computes the Cartesian (ECEF) positions of a LIGO-like L-shaped
    detector's two end-mirror vertices, at the given time(s).

    LIGO's site geometry is static (no orbital motion), so the returned
    positions are the same at every time; `time_in_years` only sets how many
    times the (constant) positions are tiled, to match the time-dependent
    signature used by orbiting detectors like LISA.

    Args:
        time_in_years (ArrayLike): Time(s), in years, at which to evaluate
            the vertex positions. Only its length is used.
        center (ArrayLike): ECEF unit-vector position of the detector's
            corner (beamsplitter) station, with shape (vectorial_index (3),).
        arm1 (ArrayLike): Unit vector along the first arm, with shape
            (vectorial_index (3),).
        arm2 (ArrayLike): Unit vector along the second arm, with shape
            (vectorial_index (3),).
        armlength (ArrayLike): Arm length, in meters.

    Returns:
        jax.Array: The two end-mirror vertex positions, tiled over time,
            with shape (configurations, vectorial_index (3), vertices (2)).
    """
    n = jnp.atleast_1d(time_in_years).shape[0]
    xm = center + arm1 * armlength
    ym = center + arm2 * armlength
    P = jnp.stack([xm, ym], axis=1)
    tiled = jnp.tile(P[:, :, None], (1, 1, n))
    return tiled.transpose(2, 0, 1)


@jax.jit
def LIGO_arms_matrix(
    time_in_years: ArrayLike,
    arm1: jax.Array,
    arm2: jax.Array,
    armlength: ArrayLike,
) -> jax.Array:
    """
    Computes the arm matrix of a LIGO-like L-shaped detector, at the given
    time(s): the two physical arm vectors and their reverse-direction
    counterparts (mirroring the "forward + reverse" arm-doubling convention
    used for LISA's 6-arm matrix).

    LIGO's site geometry is static (no orbital motion), so the returned arm
    matrix is the same at every time; `time_in_years` only sets how many
    times it is tiled, to match the time-dependent signature used by
    orbiting detectors like LISA.

    Args:
        time_in_years (ArrayLike): Time(s), in years, at which to evaluate
            the detector arms. Only its length is used.
        arm1 (ArrayLike): Unit vector along the first arm, with shape
            (vectorial_index (3),).
        arm2 (ArrayLike): Unit vector along the second arm, with shape
            (vectorial_index (3),).
        armlength (ArrayLike): Arm length, in meters.

    Returns:
        jax.Array: The arm matrix, tiled over time, with shape
            (configurations, vectorial_index (3), arms (4)), ordered
            [arm1, arm2, -arm1, -arm2].
    """
    n = jnp.atleast_1d(time_in_years).shape[0]
    vec1 = arm1 * armlength
    vec2 = arm2 * armlength
    links = jnp.stack([vec1, vec2, -vec1, -vec2], axis=1)
    tiled = jnp.tile(links[:, :, None], (1, 1, n)).transpose(2, 0, 1)
    return tiled


# -----------------------------------------------------------------------------
# -- Unified LIGO Detector Class ---------------------------------------------
# -----------------------------------------------------------------------------
@chex.dataclass(frozen=True)
class LIGO(Detector):
    """
    A data class representing a LIGO-like L-shaped ground-based
    gravitational-wave detector.

    Unlike LISA, LIGO's site geometry is static (fixed to the Earth's
    surface, no orbital motion) and it has a single Michelson readout
    channel rather than several TDI combinations, so several of the
    `Detector` methods below are considerably simpler than their LISA
    counterparts.

    Attributes:
        which_detector (str): Name of the LIGO site to use, a key of
            `_SITE_GEOMETRIES` (currently "Hanford" or "Livingston").
            Default is "Hanford".
        name (str): Human-readable detector name, set to "LIGO
            {which_detector}" in `__post_init__`.
        ps (PhysicalConstants): Physical constants used in the noise/response
            computations.
        fmin (float): Minimum frequency of LIGO's sensitive band, in Hz.
        fmax (float): Maximum frequency of LIGO's sensitive band, in Hz.
        armlength (float): LIGO's arm length, in meters.
        res (float): Frequency resolution used by `frequency_vector`, in Hz.
        default_combination (str): The only combination LIGO supports,
            "Michelson".
        center (jax.Array): ECEF unit-vector position of the site's corner
            (beamsplitter) station. Set in `__post_init__` from
            `_SITE_GEOMETRIES[which_detector]`.
        arm1 (jax.Array): Unit vector along the first arm. Set in
            `__post_init__`.
        arm2 (jax.Array): Unit vector along the second arm. Set in
            `__post_init__`.
        response (Response): Response object used to compute LIGO's response
            to gravitational waves.
        noise (Noise): Noise object used to compute LIGO's noise budget.
    """

    which_detector: str = "Hanford"
    name: str = "LIGO Hanford"
    ps: PhysicalConstants = PhysicalConstants()
    fmin: float = 1.0
    fmax: float = 2e3
    armlength: float = 4e3
    res: float = 1e-1
    default_combination: str = "Michelson"
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
        Looks up `which_detector` in `_SITE_GEOMETRIES` and sets `name`,
        `center`, `arm1`, and `arm2` accordingly.

        Raises:
            ValueError: If `which_detector` is not a key of
                `_SITE_GEOMETRIES`.
        """
        if self.which_detector not in _SITE_GEOMETRIES:
            raise ValueError(f"Unknown LIGO site '{self.which_detector}'")
        object.__setattr__(self, "name", f"LIGO {self.which_detector}")
        geom = _SITE_GEOMETRIES[self.which_detector]
        object.__setattr__(self, "center", geom["center"])
        object.__setattr__(self, "arm1", geom["arm1"])
        object.__setattr__(self, "arm2", geom["arm2"])

    def _vertex_positions(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes the positions of LIGO's two end-mirror vertices at the
        given time(s). See :func:`LIGO_positions`.

        Args:
            time_in_years (ArrayLike): Time(s), in years. LIGO's geometry is
                static, so this only sets how many times the (constant)
                positions are tiled.

        Returns:
            jax.Array: The end-mirror vertex positions, as returned by
                :func:`LIGO_positions`.
        """
        return LIGO_positions(
            time_in_years, self.center, self.arm1, self.arm2, self.armlength
        )

    def _detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes LIGO's arm matrix at the given time(s). See
        :func:`LIGO_arms_matrix`.

        Args:
            time_in_years (ArrayLike): Time(s), in years. LIGO's geometry is
                static, so this only sets how many times the (constant) arm
                matrix is tiled.

        Returns:
            jax.Array: The arm matrix, as returned by
                :func:`LIGO_arms_matrix`.
        """
        return LIGO_arms_matrix(time_in_years, self.arm1, self.arm2, self.armlength)

    def detector_position(self) -> jax.Array:
        """
        Returns the ECEF unit-vector position of LIGO's corner
        (beamsplitter) station.

        Returns:
            jax.Array: `self.center`.
        """
        return self.center

    def frequency_vector(self) -> jax.Array:
        """
        Generates a frequency vector spanning LIGO's sensitive band at its
        fixed resolution `self.res`.

        Returns:
            jax.Array: An array of frequency points from `self.fmin` to
                `self.fmax` (inclusive), spaced by `self.res`.
        """
        return jnp.arange(self.fmin, self.fmax + self.res, self.res)

    def combination_matrix(
        self, combination: str, arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
    ) -> jax.Array:
        """
        Builds the Michelson-combination mixing matrix. See
        :func:`gw_response.ground_based.datastream.detector_output`.

        Args:
            combination (str): Must be `self.default_combination`
                ("Michelson"); LIGO supports no other readout combination.
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled
                by the arm length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
                frequency.

        Returns:
            jax.Array: The Michelson-combination mixing matrix.

        Raises:
            ValueError: If ``combination`` is not "Michelson".
        """
        if combination != self.default_combination:
            raise ValueError(
                f"LIGO only supports the '{self.default_combination}' combination"
            )
        return detector_output(arms_matrix_rescaled, x_vector)

    def linear_response_from_single_link(
        self, single_link: dict[str, jax.Array], combination_matrix: ArrayLike
    ) -> dict[str, jax.Array]:
        """
        Applies the Michelson-combination mixing matrix to each
        polarization's single-link response.

        LIGO's Michelson combination has a single readout channel, so that
        trivial axis is dropped from the result rather than carried around
        as a dangling dimension.

        Args:
            single_link (dict): Single-link response per polarization.
            combination_matrix (ArrayLike): Mixing matrix as built by
                :meth:`combination_matrix`.

        Returns:
            dict: The Michelson-combination linear response per
                polarization.
        """
        return {
            p: jnp.squeeze(combine_single_link(combination_matrix, sl), axis=-2)
            for p, sl in single_link.items()
        }

    def quadratic_response_from_single_link(
        self, linear_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Computes the quadratic response as the squared modulus of the
        linear response.

        Args:
            linear_integrand (dict): Linear response per polarization, as
                returned by :meth:`linear_response_from_single_link`.

        Returns:
            dict: The quadratic response per polarization.
        """
        return {p: jnp.abs(linear) ** 2 for p, linear in linear_integrand.items()}

    def integrate_quadratic_response(
        self, quadratic_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Passes the quadratic response through unchanged.

        Unlike LISA's sky-averaging, LIGO's PSD weighting and integration
        over pixels is left to the caller, so there is nothing to do here.

        Args:
            quadratic_integrand (dict): Quadratic response per polarization,
                as returned by :meth:`quadratic_response_from_single_link`.

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
        Returns the LIGO design noise PSD. See
        :func:`single_link_LIGO_noise_variance`.

        LIGO's noise model here is a single already-combined PSD curve;
        there is no per-link decomposition to build from
        ``arms_matrix_rescaled`` or ``x_vector``, so they're accepted (for
        interface parity with `Detector.single_link_noise`) but unused.
        LIGO also takes no detector-specific noise parameters.

        Args:
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the noise.
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

        `single_link_noise` (as built by :meth:`single_link_noise`) is
        already the readout-domain PSD, so there is nothing left to
        project.

        Args:
            combination_matrix (ArrayLike): Unused.
            single_link_noise (jax.Array): The noise PSD as built by
                :meth:`single_link_noise`.

        Returns:
            jax.Array: `single_link_noise`, unchanged.
        """
        return single_link_noise
