import jax
import jax.numpy as jnp
import chex
from dataclasses import field

from .constants import PhysicalConstants
from .ligo import LIGO
from .single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    get_single_link_response,
)
from .earth_based_datastream import detector_output, build_datastream


@chex.dataclass
class ResponseLIGO:
    ps: chex.dataclass = PhysicalConstants()
    det: chex.dataclass = field(default_factory=lambda: LIGO())
    single_link_response = {}
    linear_integrand = {}
    quadratic_integrand = {}
    quadratic_integrated = {}

    def __post_init__(self, **kwargs):
        self.get_positions = lambda times: self.det.satellite_positions(times, **kwargs)
        self.get_arms = lambda times: self.det.detector_arms(times, **kwargs)

    def get_single_link_response(
        self,
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
    ):
        pol = polarization.upper()
        k_vec = unit_vec(theta_array, phi_array)
        u, v = uv_analytical(theta_array, phi_array)

        positions = self.get_positions(times_in_years) / self.det.armlength
        arms_matrix = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        if pol == "PC":
            p1, p2 = polarization_tensors_PC(u, v)
        elif pol == "LR":
            p1, p2 = polarization_tensors_LR(u, v)
        else:
            raise ValueError("Unknown polarization type")

        out = {}
        for label, tensor in zip(("L", "R"), (p1, p2)):
            out[label] = get_single_link_response(
                tensor, arms_matrix, k_vec, x_array, positions
            )
        return out  # {'L': (..., F, 4, P), 'R': (...)}

    def get_linear_integrand(
        self,
        times_in_years,
        single_link,
        frequency_array,
        polarization="LR",
    ):
        arms = self.get_arms(times_in_years) / self.det.armlength
        xvec = self.det.x(frequency_array)
        lin = {}
        for p_label, sl in single_link.items():
            det_mat = detector_output(arms, xvec)      # (..., F, 1, 4)
            mic = build_datastream(det_mat, sl)        # (..., F, 1, P)
            lin[p_label] = mic[..., 0, :]              # squeeze detector axis
        return lin  # {'L': (..., F, P), ...}

    def get_quadratic_integrand(
        self,
        times_in_years,
        single_link,
        frequency_array,
        TDI="XYZ",  # ignored
        polarization="LR",
    ):
        pol = polarization.upper()
        quad = {}
        lin = self.get_linear_integrand(
            times_in_years, single_link, frequency_array, polarization=polarization
        )
        for p_label, htilde in lin.items():
            quad[p_label] = jnp.abs(htilde) ** 2  # (..., F, P)
        return quad

    def get_quadratic_integrated(
        self,
        quad_integrand,
        TDI="XYZ",  # ignored
        polarization="LR",
        verbose=True,
    ):
        return quad_integrand  # user performs PSD weighting and integration externally

    def compute_detector(
        self,
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        TDI="XYZ",  # ignored
        polarization="LR",
    ):
        self.single_link_response = self.get_single_link_response(
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )

        self.linear_integrand = self.get_linear_integrand(
            times_in_years,
            self.single_link_response,
            frequency_array,
            polarization=polarization,
        )

        self.quadratic_integrand = self.get_quadratic_integrand(
            times_in_years,
            self.single_link_response,
            frequency_array,
            polarization=polarization,
        )

        self.quadratic_integrated = self.get_quadratic_integrated(
            self.quadratic_integrand,
            polarization=polarization,
        )
