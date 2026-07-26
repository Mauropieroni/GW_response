"""
Placeholder for the Einstein Telescope (ET) detector.

Unlike LIGO/CE, ET is a triangular, 3-arm/6-link underground detector -- the
same topology as LISA's constellation, just fixed to the Earth instead of
orbiting. That means:
  - It does *not* fit `ground_based/datastream.py`'s Michelson
    `detector_output` (that's specific to a single L-shaped readout channel).
  - Its readout combination is likely closer to the TDI-style combinatorics
    in `space_based/tdi.py` (built from a 6-link arm matrix) than to LIGO's
    `combination_matrix`, even though ET needs no orbital mechanics at all
    (its vertex positions only need to track Earth's rotation, not an orbit).

Filling this in will likely mean picking whichever of `space_based/tdi.py`'s
combination logic applies (or a triangular-specific readout of its own) and
a `_vertex_positions`/`_detector_arms` implementation based on ET's fixed,
underground site geometry and orientation, wired into a concrete `Detector`
the same way `ground_based/ligo.py` does for LIGO.
"""

# Local imports
from ..detector import Detector


class EinsteinTelescope(Detector):
    """TODO: implement Einstein Telescope (see module docstring)."""
