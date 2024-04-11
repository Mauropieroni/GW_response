import chex
from dataclasses import field

from .constants import PhysicalConstants
from .lisa import LISA
from .single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    get_single_link_response,
)
from .tdi import TDI_map
from .single_link import (
    linear_response_angular,
    integrand,
    response_integrated,
)


@chex.dataclass
class Response(object):
    ps: chex.dataclass = PhysicalConstants()
    det: chex.dataclass = field(default_factory=lambda: LISA())
    single_link_response = {}
    linear_integrand = {}
    ### TO DO all these should become quadratic integrand/integrated
    integrand = {}
    integrated = {}

    def __post_init__(self, **kwargs):
        self.get_positions = lambda times: self.det.satellite_positions(times, **kwargs)
        self.get_arms = lambda times: self.det.detector_arms(times, **kwargs)

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def get_single_link_response(
        self,
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        polarization="LR",
    ):
        pol = polarization.upper()

        k_vector = unit_vec(theta_array, phi_array)
        u, v = uv_analytical(theta_array, phi_array)

        positions = self.get_positions(times_in_years) / self.det.armlength
        arms_matrix = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        if pol == "PC":
            p1, p2 = polarization_tensors_PC(u, v)
        elif pol == "LR":
            p1, p2 = polarization_tensors_LR(u, v)
        else:
            raise ValueError("Incrorrect polarization type")

        ppol = {pol[0]: p1, pol[1]: p2}
        val = {}

        for p in ppol.keys():
            val[2 * p] = get_single_link_response(
                ppol[p],
                arms_matrix,
                k_vector,
                x_array,
                positions,
            )

        return val

    def get_linear_integrand(
        self,
        times_in_years,
        single_link,
        frequency_array,
        TDI="XYZ",
        polarization="LR",
    ):
        pol = polarization.upper()
        val = {}

        arms_matrix = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        for p in pol:
            ### TO DO this should use only 1 p
            val[2 * p] = linear_response_angular(
                TDI_map[TDI],
                single_link[2 * p],
                arms_matrix,
                x_array,
            )

        return val

    def get_integrand(
        self,
        times_in_years,
        single_link,
        frequency_array,
        TDI="XYZ",
        polarization="LR",
    ):
        pol = polarization.upper()
        val = {}

        arms_matrix = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        for p in pol:
            val[2 * p] = integrand(
                TDI_map[TDI],
                single_link[2 * p],
                arms_matrix,
                x_array,
            )

        return val

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_integrated(
        self,
        TDI="XYZ",
        polarization="LR",
        verbose=True,
    ):
        integrated = {}
        for p in polarization:
            ### Computes the integral for the TDI variable
            integrated[2 * p] = response_integrated(self.integrand[TDI][2 * p])

        return integrated

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def compute_detector(
        self,
        times_in_years,
        theta_array,
        phi_array,
        frequency_array,
        TDI="XYZ",
        polarization="LR",
    ):
        self.single_link_response = self.get_single_link_response(
            times_in_years,
            theta_array,
            phi_array,
            frequency_array,
            polarization=polarization,
        )

        self.linear_integrand[TDI] = self.get_linear_integrand(
            times_in_years,
            self.single_link_response,
            frequency_array,
            TDI=TDI,
            polarization=polarization,
        )

        ### Computes the integral for the TDI variable
        self.integrand[TDI] = self.get_integrand(
            times_in_years,
            self.single_link_response,
            frequency_array,
            TDI=TDI,
            polarization=polarization,
        )

        ### Computes the integral for the TDI variable
        self.integrated[TDI] = self.get_integrated(TDI=TDI, polarization=polarization)
