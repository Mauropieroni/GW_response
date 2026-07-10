from abc import ABC, abstractmethod

import jax
from jax.typing import ArrayLike


class Detector(ABC):
    """Abstract base class for a gravitational wave detector.

    Concrete subclasses (e.g. ``LISA``) must provide the detector's basic
    characteristics as class attributes, and implement the methods that
    describe its geometry over time.

    Attributes:
        name: Human-readable name of the detector.
        fmin: Minimum frequency of the detector's sensitive band, in Hz.
        fmax: Maximum frequency of the detector's sensitive band, in Hz.
        armlength: Nominal detector arm length, in meters.
        res: Expected relative resolution/precision of the detector.
    """

    name: str
    fmin: float
    fmax: float
    armlength: float
    res: float

    @abstractmethod
    def __init__(self) -> None:
        """Initializes the detector configuration."""
        pass

    @abstractmethod
    def satellite_positions(self, time_in_years: ArrayLike) -> jax.Array:
        """Computes satellite positions at the given time(s).

        Args:
            time_in_years: Time(s), in years, at which to evaluate the
                satellite positions.

        Returns:
                Array of satellite positions.
        """
        pass

    @abstractmethod
    def detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """Computes the detector's arm matrix at the given time(s).

        Args:
            time_in_years: Time(s), in years, at which to evaluate the
                detector arms.

        Returns:
                Array representing the vector between each pair of satellites
                (i.e. each detector arm).
        """
        pass
