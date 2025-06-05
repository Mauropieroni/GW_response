import jax
import jax.numpy as jnp
import chex
from functools import partial
from .constants import PhysicalConstants
from .detector import Detector

# -----------------------------------------------------------------------------
# -- Earth & Arm Constants ---------------------------------------------------
# -----------------------------------------------------------------------------
REARTH   = 6.371e6  # Earth radius in meters
LIGOARM  = 4e3      # LIGO arm length in meters

# -----------------------------------------------------------------------------
# -- LIGO Static Site Geometries (ECEF) --------------------------------------
# -----------------------------------------------------------------------------
_SITE_GEOMETRIES = {
    "Hanford": {
        "center": jnp.array([-0.33827472, -0.60015338,  0.72483525]) * REARTH,
        "arm1":   jnp.array([-0.22389266154,  0.79983062746,  0.55690487831]),
        "arm2":   jnp.array([-0.91397818574,  0.02609403989, -0.40492342125]),
    },
    "Livingston": {
        "center": jnp.array([-0.01163537, -0.8609929 ,  0.50848387]) * REARTH,
        "arm1":   jnp.array([-0.95457412153, -0.14158077340, -0.26218911324]),
        "arm2":   jnp.array([ 0.29774156894, -0.48791033647, -0.82054461286]),
    }
}

# -----------------------------------------------------------------------------
# -- JIT-Compiled Helpers for Static Geometries ------------------------------
# -----------------------------------------------------------------------------
@jax.jit
def LIGO_satellite_positions(time_in_years, center, arm1, arm2, armlength):
    t = jnp.atleast_1d(time_in_years)
    n = t.shape[0]
    xm = center + arm1 * armlength
    ym = center + arm2 * armlength
    P  = jnp.stack([xm, ym], axis=1)
    tiled = jnp.tile(P[:, :, None], (1, 1, n))
    return tiled.transpose(2, 0, 1)

@jax.jit
def LIGO_arms_matrix(time_in_years, arm1, arm2, armlength):
    t = jnp.atleast_1d(time_in_years)
    n = t.shape[0]
    vec1 = arm1 * armlength
    vec2 = arm2 * armlength
    links = jnp.stack([vec1, vec2, -vec1, -vec2], axis=1)
    tiled = jnp.tile(links[:, :, None], (1, 1, n)).transpose(2, 0, 1)
    return tiled

# -----------------------------------------------------------------------------
# -- Unified LIGO Detector Class --------------------------------------------- ---------------------------------------------
# -----------------------------------------------------------------------------
@chex.dataclass(frozen=True)
class LIGO(Detector):
    which_detector: str = "Hanford"
    name: str           = "LIGO Hanford"
    ps: chex.dataclass  = PhysicalConstants()  # type: ignore
    fmin: float         = 1.0
    fmax: float         = 2e3
    armlength: float    = 4e3
    res: float          = 1e-1

    def __post_init__(self):
        if self.which_detector not in _SITE_GEOMETRIES:
            raise ValueError(f"Unknown LIGO site '{self.which_detector}'")
        object.__setattr__(self, 'name', f"LIGO {self.which_detector}")
        geom = _SITE_GEOMETRIES[self.which_detector]
        object.__setattr__(self, 'center', geom["center"])
        object.__setattr__(self, 'arm1',   geom["arm1"])
        object.__setattr__(self, 'arm2',   geom["arm2"])

    def satellite_positions(self, time_in_years):
        return LIGO_satellite_positions(
            time_in_years,
            self.center, self.arm1,
            self.arm2, self.armlength
        )

    def detector_arms(self, time_in_years):
        return LIGO_arms_matrix(
            time_in_years,
            self.arm1, self.arm2,
            self.armlength
        )

    def detector_position(self):
        return self.center

    def frequency_vector(self):
        return jnp.arange(self.fmin, self.fmax + self.res, self.res)

