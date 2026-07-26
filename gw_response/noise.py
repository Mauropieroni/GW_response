from __future__ import annotations

# Global imports
import chex
import functools
import jax
from dataclasses import field
from typing import TYPE_CHECKING

# Local imports
from .constants import PhysicalConstants

if TYPE_CHECKING:
    from .detector import Detector

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@chex.dataclass
class Noise(object):
    """
    Generic per-link/projected noise computation for any Detector (e.g.
    LISA, LIGO), mirroring the way Response wraps the single-link/combination
    functions. Detector-specific behavior (what the per-link noise looks
    like, and how it projects into a readout combination) is delegated to
    the `det` object passed into each method.

    `Noise` holds no reference to any particular detector -- a detector
    owns its `Noise` (e.g. `lisa.noise`), not the other way around, so
    `det` is passed explicitly to every method here instead of being
    stored on `self`.

    Identity-based `__hash__`/`__eq__` (overriding chex's default
    field-based ones, which reject `Noise` as unhashable) let `self` be
    used as a static `jax.jit` argument below -- see `Response` for the
    full rationale. `compute_detector` stays unjitted since it mutates
    `self`'s dict attributes.
    """

    ps: PhysicalConstants = PhysicalConstants()
    # The per-link noise doesn't depend on the readout combination (only its
    # projection does), so unlike `noise_matrix` below it isn't keyed by one.
    single_link_noise: jax.Array | None = None
    noise_matrix: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def get_arms_matrix_rescaled(self, det: "Detector", times_in_years):
        return det.detector_arms(times_in_years) / det.armlength

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def get_single_link_noise(
        self, det: "Detector", times_in_years, frequency_array, **noise_parameters
    ):
        return det.single_link_noise(
            frequency_array,
            self.get_arms_matrix_rescaled(det, times_in_years),
            det.x(frequency_array),
            **noise_parameters,
        )

    @functools.partial(jax.jit, static_argnums=(0, 1), static_argnames=("combination",))
    def get_noise_matrix(
        self,
        det: "Detector",
        times_in_years,
        frequency_array,
        combination=None,
        **noise_parameters,
    ):
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
        times_in_years,
        frequency_array,
        combination=None,
        **noise_parameters,
    ):
        combination = combination or det.default_combination

        self.single_link_noise = self.get_single_link_noise(
            det, times_in_years, frequency_array, **noise_parameters
        )

        arms_matrix_rescaled = self.get_arms_matrix_rescaled(det, times_in_years)
        x_vector = det.x(frequency_array)
        combination_matrix = det.combination_matrix(
            combination, arms_matrix_rescaled, x_vector
        )
        self.noise_matrix[combination] = det.project_noise(
            combination_matrix, self.single_link_noise
        )
