# Global imports
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Any


class Detector(ABC):
    name: str
    fmin: float
    fmax: float
    armlength: float
    res: float
    ps: Any

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def satellite_positions(self, *args, **kwargs):
        pass

    @abstractmethod
    def detector_arms(self, *args, **kwargs):
        pass

    def frequency_vec(self, freq_pts):
        """
        Generates a frequency vector within the LISA frequency range.

        Args:
            freq_pts (int): The number of frequency points to generate.

        Returns:
            jnp.ndarray: A linearly spaced array of frequency points within
            LISA's operational frequency range, starting from LISA_fmin to
            LISA_fmax.
        """
        return jnp.linspace(self.fmin, self.fmax, freq_pts)

    def klvector(self, frequency_vec):
        """
        Computes the kl-vector for a given frequency vector in the context of
        the LISA configuration.

        Args:
            frequency_vec (jnp.ndarray): An array of frequency values for which
            the kl-vector is to be computed.

        Returns:
            jnp.ndarray: An array representing the kl-vector, which is a product
            of the frequency vector, the LISA arm length, and the inverse of the
            speed of light.
        """
        return frequency_vec * self.armlength / self.ps.light_speed

    def x(self, frequency_vec):
        """
        Computes the x-parameter for a given frequency vector based on the LISA
        configuration.

        Args:
            frequency_vec (jnp.ndarray): An array of frequency values for which
            the x-parameter is to be computed.

        Returns:
            jnp.ndarray: An array representing the x-parameter, calculated as
            2π times the kl-vector for the given frequency vector.
        """
        return 2 * jnp.pi * self.klvector(frequency_vec)
