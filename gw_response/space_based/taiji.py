"""
Placeholder for the Taiji detector.

Taiji is architecturally a LISA-like triangular constellation (same 3-arm,
6-link topology), so most of the physics this needs should already exist as
generic, parameter-driven code in this package:
  - `space_based/orbits.py` for the constellation orbit models (rigid /
    Keplerian / numeric).
  - `space_based/tdi.py` for the TDI combinations built from the arm matrix.
  - `space_based/noise.py` for the TM/OMS single-link noise budget shape.

Filling this in should mostly mean choosing Taiji's own parameters
(armlength, orbit radius/eccentricity, noise budget parameters, etc.) and
wiring them into a concrete `Detector` the same way `space_based/lisa.py`
does for LISA -- `LISA` is the reference implementation to mirror here.
"""

# Local imports
from gw_response.detector import Detector


class Taiji(Detector):
    """TODO: implement Taiji (see module docstring)."""
