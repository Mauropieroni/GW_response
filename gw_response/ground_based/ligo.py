# Global imports
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pandas as pd
from dataclasses import field
from scipy.interpolate import interp1d

# Local imports
from ..constants import PhysicalConstants
from ..detector import Detector
from ..noise import Noise
from ..response import Response
from ..utils import combine_single_link
from .datastream import detector_output

# -----------------------------------------------------------------------------
# -- Earth & Arm Constants ---------------------------------------------------
# -----------------------------------------------------------------------------
REARTH = 6.371e6  # Earth radius in meters
LIGOARM = 4e3  # LIGO arm length in meters

# -----------------------------------------------------------------------------
# -- LIGO Design Sensitivity Curve --------------------------------------------
# -----------------------------------------------------------------------------
_path_to_LIGO_design = os.path.join(
    os.path.dirname(__file__), "noise_data", "LIGO.pkl"
)
_ligo_design_curves = pd.read_pickle(_path_to_LIGO_design)

_ligo_freqs = np.asarray(_ligo_design_curves["Frequency"])
_ligo_psd = np.asarray(_ligo_design_curves["Mid high/Late low"])
_ligo_interp = interp1d(
    _ligo_freqs,
    _ligo_psd,
    kind="linear",
    bounds_error=False,
    # scipy stubs don't type this literal
    fill_value="extrapolate",  # type: ignore[arg-type]
)


def LIGO_noise(frequencies):
    """
    Returns the LIGO design PSD at `frequencies`, interpolated from the
    tabulated design curve. `_ligo_interp` is a plain scipy/numpy
    interpolator, not a jax-traceable operation, so this goes through
    `jax.pure_callback` -- letting it be called with a concrete array
    (outside jit) or with a traced one (e.g. from a jitted caller like
    `Noise.get_single_link_noise`) alike.
    """
    frequencies = jnp.asarray(frequencies)
    result_shape = jax.ShapeDtypeStruct(frequencies.shape, jnp.float64)
    return jax.pure_callback(
        lambda f: np.asarray(_ligo_interp(f)), result_shape, frequencies
    )


def single_link_LIGO_noise_variance(frequencies):
    """
    Returns the 1D Michelson-output PSD S_n(f) for LIGO.
    Shape: (F,)
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
def LIGO_positions(time_in_years, center, arm1, arm2, armlength):
    n = jnp.atleast_1d(time_in_years).shape[0]
    xm = center + arm1 * armlength
    ym = center + arm2 * armlength
    P = jnp.stack([xm, ym], axis=1)
    tiled = jnp.tile(P[:, :, None], (1, 1, n))
    return tiled.transpose(2, 0, 1)


@jax.jit
def LIGO_arms_matrix(time_in_years, arm1, arm2, armlength):
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

    def __post_init__(self):
        if self.which_detector not in _SITE_GEOMETRIES:
            raise ValueError(f"Unknown LIGO site '{self.which_detector}'")
        object.__setattr__(self, "name", f"LIGO {self.which_detector}")
        geom = _SITE_GEOMETRIES[self.which_detector]
        object.__setattr__(self, "center", geom["center"])
        object.__setattr__(self, "arm1", geom["arm1"])
        object.__setattr__(self, "arm2", geom["arm2"])

    def _vertex_positions(self, time_in_years):
        return LIGO_positions(
            time_in_years, self.center, self.arm1, self.arm2, self.armlength
        )

    def _detector_arms(self, time_in_years):
        return LIGO_arms_matrix(time_in_years, self.arm1, self.arm2, self.armlength)

    def detector_position(self):
        return self.center

    def frequency_vector(self):
        return jnp.arange(self.fmin, self.fmax + self.res, self.res)

    def combination_matrix(self, combination, arms_matrix_rescaled, x_vector):
        if combination != self.default_combination:
            raise ValueError(
                f"LIGO only supports the '{self.default_combination}' combination"
            )
        return detector_output(arms_matrix_rescaled, x_vector)

    def linear_response_from_single_link(self, single_link, combination_matrix):
        # LIGO's Michelson combination has a single readout channel, so drop
        # that trivial axis rather than carry it around as a dangling dim.
        return {
            p: jnp.squeeze(combine_single_link(combination_matrix, sl), axis=-2)
            for p, sl in single_link.items()
        }

    def quadratic_response_from_single_link(self, linear_integrand):
        return {p: jnp.abs(linear) ** 2 for p, linear in linear_integrand.items()}

    def integrate_quadratic_response(self, quadratic_integrand):
        # PSD weighting and integration over pixels is left to the caller
        return quadratic_integrand

    def single_link_noise(self, frequency_array, arms_matrix_rescaled, x_vector, **_):
        # LIGO's noise model here is a single already-combined PSD curve;
        # there is no per-link decomposition to build from arms_matrix_rescaled
        # or x_vector, so they're accepted (for interface parity) but unused.
        return single_link_LIGO_noise_variance(frequency_array)

    def project_noise(self, combination_matrix, single_link_noise):
        # single_link_noise is already the readout-domain PSD, so there is
        # nothing left to project.
        return single_link_noise
