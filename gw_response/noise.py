from __future__ import annotations

# Global imports
import chex
import functools
import jax
from dataclasses import field
from typing import TYPE_CHECKING

# Local imports
from gw_response.constants import PhysicalConstants

if TYPE_CHECKING:
    from gw_response.detector import Detector

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@chex.dataclass
class Noise(object):
    """
    Generic per-link/projected noise computation for any Detector (e.g. LISA, LIGO),
    mirroring the way Response wraps the single-link/combination functions. `Noise`
    holds no reference to any particular detector. The `det` object passed to each
    method implements the detector-specific behavior (per-link noise and projection).

    Identity-based `__hash__`/`__eq__` (overriding chex's default field-based ones,
    which reject `Noise` as unhashable) let `self` be used as a static in `jax.jit`.

    Attributes:
        ps (chex.dataclass): Physical constants used in the noise computations.
        single_link_noise (jax.Array or None): Cache of the most recently computed
            per-link noise (or already-combined noise, for detectors without a per-link
            decomposition), as set by :meth:`compute_detector`.
        noise_matrix (dict): Cache of projected noise covariance matrices computed by
            :meth:`compute_detector`, keyed by combination name.
    """

    ps: PhysicalConstants = PhysicalConstants()
    # The per-link noise doesn't depend on the readout combination
    single_link_noise: jax.Array | None = None
    # The projected noise depends on the readout combination, so it is keyed.
    noise_matrix: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def get_arms_matrix_rescaled(
        self, det: "Detector", times_in_years: jax.Array
    ) -> jax.Array:
        """
        Computes the (rescaled) detector's arm matrix at the given time(s).

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the noise is computed for.
            times_in_years (jax.Array): Time(s), in years, at which to evaluate the
                detector arms.

        Returns:
            jax.Array: The rescaled arm matrix.
        """
        return det.detector_arms(times_in_years) / det.armlength

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def get_single_link_noise(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        frequency_array: jax.Array,
        **noise_parameters,
    ) -> jax.Array:
        """
        Computes the per-link noise covariance (or, for detectors with no per-link
        decomposition, the already-combined noise) at the given time(s) and frequencies.
        See :meth:`Detector.single_link_noise`.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the noise is computed for.
            times_in_years (jax.Array): Time(s), in years, at which to evaluate the
                detector arms.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the noise.
            **noise_parameters: Detector-specific noise parameters (e.g. LISA's
                ``TM_acceleration_parameters``/``OMS_parameters``); LIGO takes none.

        Returns:
            jax.Array: The per-link (or already-combined) noise covariance.
        """
        return det.single_link_noise(
            frequency_array,
            self.get_arms_matrix_rescaled(det, times_in_years),
            det.x(frequency_array),
            **noise_parameters,
        )

    @functools.partial(jax.jit, static_argnums=(0, 1, 4))
    def get_noise_matrix(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        frequency_array: jax.Array,
        combination: str | None = None,
        **noise_parameters,
    ) -> jax.Array:
        """
        Computes the noise covariance matrix projected into a readout combination, at
        the given time(s) and frequencies as in eq. 2.29b/2.30 of Hartwig, Lilley,
        Muratore & Pieroni (arXiv:2303.15929):

             ``S^UV,N = C^UV S^η,N``

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the noise is computed for.
            times_in_years (jax.Array): Time(s), in years, at which to evaluate the
                detector arms.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the noise.
            combination (str, optional): Name of the readout combination (e.g. a TDI
                variable for LISA). Defaults to ``det.default_combination``.
            **noise_parameters: Detector-specific noise parameters (e.g. LISA's
                ``TM_acceleration_parameters``/``OMS_parameters``); LIGO takes none.

        Returns:
            jax.Array: The noise covariance matrix, projected into therequested
                combination.
        """
        combination = combination or det.default_combination
        arms_matrix_rescaled = self.get_arms_matrix_rescaled(det, times_in_years)
        x_vector = det.x(frequency_array)
        combination_matrix = det.combination_matrix(
            combination, arms_matrix_rescaled, x_vector
        )
        single_link_noise = self.get_single_link_noise(
            det, times_in_years, frequency_array, **noise_parameters
        )
        return det.project_noise(combination_matrix, single_link_noise)

    def compute_detector(
        self,
        det: "Detector",
        times_in_years: jax.Array,
        frequency_array: jax.Array,
        combination: str | None = None,
        **noise_parameters,
    ) -> None:
        """
        Computes and caches the per-link and projected noise covariance matrices for a
        readout combination.

        Results are stored in :attr:`single_link_noise` and :attr:`noise_matrix`, the
        latter keyed by ``combination``.

        Args:
            det (Detector): The detector (e.g. LISA, LIGO) the noise is computed for.
            times_in_years (jax.Array): Time(s), in years, at which to evaluate the
                detector arms.
            frequency_array (jax.Array): Frequency values, in Hz, at which to evaluate
                the noise.
            combination (str, optional): Name of the readout combination (e.g. a TDI
                variable for LISA). Defaults to ``det.default_combination``.
            **noise_parameters: Detector-specific noise parameters (e.g. LISA's
                ``TM_acceleration_parameters``/``OMS_parameters``); LIGO takes none.
        """
        combination = combination or det.default_combination

        single_link_noise = self.get_single_link_noise(
            det, times_in_years, frequency_array, **noise_parameters
        )
        self.single_link_noise = single_link_noise

        arms_matrix_rescaled = self.get_arms_matrix_rescaled(det, times_in_years)
        x_vector = det.x(frequency_array)
        combination_matrix = det.combination_matrix(
            combination, arms_matrix_rescaled, x_vector
        )
        self.noise_matrix[combination] = det.project_noise(
            combination_matrix, single_link_noise
        )
