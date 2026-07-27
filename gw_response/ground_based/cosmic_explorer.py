"""
Placeholder for the Cosmic Explorer (CE) detector.

CE is architecturally a LIGO-like L-shaped Michelson interferometer (same
2-arm, single-readout-channel topology, just a longer armlength and a
different site), so most of the physics this needs should already exist as
generic, parameter-driven code:
  - `ground_based/datastream.py`'s `detector_output` for the Michelson
    single-link mixing matrix.
  - `LIGO_positions`/`LIGO_arms_matrix` in `ground_based/ligo.py` for the
    site-geometry -> vertex-positions/arm-matrix computation (currently
    living there since LIGO is still the only consumer; once CE needs them
    too, consider factoring them out into a shared `ground_based/geometry.py`
    the same way `space_based/orbits.py` was split out of `lisa.py`).

Filling this in should mostly mean choosing CE's own parameters (site
location/orientation, armlength, noise design curve, etc.) and wiring them
into a concrete `Detector` the same way `ground_based/ligo.py` does for
LIGO -- `LIGO` is the reference implementation to mirror here.
"""

# Local imports
from gw_response.detector import Detector


class CosmicExplorer(Detector):
    """TODO: implement Cosmic Explorer (see module docstring)."""
