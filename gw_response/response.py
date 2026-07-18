import chex
import jax
from jax.typing import ArrayLike
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
    quadratic_integrand,
    quadratic_response_integrated,
)


@chex.dataclass
class Response(object):
    """
    A wrapper tying the single-link and TDI response functions in this
    package to a specific detector, and caching the results of a full
    response computation.

    Attributes:
        ps (chex.dataclass): Physical constants used in the response
            computations.
        det (chex.dataclass): The detector (e.g. LISA) the response is
            computed for.
        single_link_response (dict): Cache of the single-link response
            computed by :meth:`compute_detector`, keyed by polarization letter.
        linear_integrand (dict): Cache of the linear TDI response integrand
            computed by :meth:`compute_detector`, keyed by TDI combination name.
        quadratic_integrand (dict): Cache of the sky-resolved quadratic TDI
            response computed by :meth:`compute_detector`, keyed by TDI
            combination name.
        quadratic_integrated (dict): Cache of the sky-averaged quadratic TDI
            response computed by :meth:`compute_detector`, keyed by TDI
            combination name.
    """

    ps: chex.dataclass = PhysicalConstants()
    det: chex.dataclass = field(default_factory=lambda: LISA())
    single_link_response = {}
    linear_integrand = {}
    quadratic_integrand = {}
    quadratic_integrated = {}

    def __post_init__(self, **kwargs) -> None:
        """
        Binds the detector's ``satellite_positions`` and ``detector_arms``
        methods into the ``get_positions`` and ``get_arms`` helpers used
        throughout this class.

        Args:
                **kwargs: Extra keyword arguments forwarded to
                ``det.satellite_positions`` and ``det.detector_arms`` on every
                call. Reserved for future per-call detector options; ``ps``
                and ``det`` are regular dataclass fields (not ``InitVar``),
                so no keyword arguments reach here through normal
                ``Response(...)`` construction today.
        """
        self.get_positions = lambda times: self.det.satellite_positions(times, **kwargs)
        self.get_arms = lambda times: self.det.detector_arms(times, **kwargs)

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def get_single_link_response(
        self,
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        polarization: str = "LR",
    ) -> dict[str, jax.Array]:
        """
        Computes the single-link strain response for both polarizations of
        the requested basis.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the satellite positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".

        Returns:
            dict: A dictionary mapping each polarization letter (e.g. "L"
                and "R") to its single-link strain response, with shape
                (configurations, x_vector, arms, pixels).

        Raises:
            ValueError: If ``polarization`` is not "LR" or "PC".
        """
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
            val[p] = get_single_link_response(
                ppol[p],
                arms_matrix,
                k_vector,
                x_array,
                positions,
            )

        return val

    def get_linear_integrand(
        self,
        times_in_years: ArrayLike,
        single_link: dict[str, ArrayLike],
        frequency_array: ArrayLike,
        TDI: str = "XYZ",
        polarization: str = "LR",
    ) -> dict[str, jax.Array]:
        """
        Projects a single-link response onto a TDI combination, for each
        polarization.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            single_link (dict): Single-link strain response per
                polarization, as returned by :meth:`get_single_link_response`.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".
            polarization (str, optional): Polarization letters to iterate
                over (e.g. "LR"). Default is "LR".

        Returns:
            dict: A dictionary mapping each polarization letter to the
                linear TDI response, with shape (configurations, x_vector, TDI,
                pixels).
        """
        pol = polarization.upper()
        val = {}

        arms_matrix_rescaled = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        for p in pol:
            val[p] = linear_response_angular(
                TDI_map[TDI],
                single_link[p],
                arms_matrix_rescaled,
                x_array,
            )

        return val

    def get_quadratic_integrand(
        self,
        times_in_years: ArrayLike,
        single_link: dict[str, ArrayLike],
        frequency_array: ArrayLike,
        TDI: str = "XYZ",
        polarization: str = "LR",
    ) -> dict[str, jax.Array]:
        """
        Computes the sky-resolved quadratic TDI response, for each
        polarization.

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.
            single_link (dict): Single-link strain response per
                polarization, as returned by :meth:`get_single_link_response`.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".
            polarization (str, optional): Polarization letters to iterate
                over (e.g. "LR"). Default is "LR".

        Returns:
            dict: A dictionary mapping each doubled polarization letter
                (e.g. "LL", "RR") to the sky-resolved quadratic TDI response,
                with shape (configurations, x_vector, TDI, TDI, pixels).
        """
        pol = polarization.upper()
        val = {}

        arms_matrix = self.get_arms(times_in_years) / self.det.armlength
        x_array = self.det.x(frequency_array)

        for p in pol:
            val[2 * p] = quadratic_integrand(
                TDI_map[TDI],
                single_link[p],
                arms_matrix,
                x_array,
            )

        return val

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def get_quadratic_integrated(
        self,
        quadratic_integrand: dict[str, dict[str, ArrayLike]],
        TDI: str = "XYZ",
        polarization: str = "LR",
        verbose: bool = True,
    ) -> dict[str, jax.Array]:
        """
        Averages the sky-resolved quadratic TDI response over the sky, for
        each polarization.

        Args:
            quadratic_integrand (dict): Sky-resolved quadratic TDI response
                per TDI combination and doubled polarization letter, as returned
                by :meth:`get_quadratic_integrand` (nested under the TDI
                combination name, matching :attr:`quadratic_integrand`).
            TDI (str, optional): Name of the TDI combination to look up in
                ``quadratic_integrand``. Default is "XYZ".
            polarization (str, optional): Polarization letters to iterate
                over (e.g. "LR"). Default is "LR".
            verbose (bool, optional): Unused. Present for interface
                compatibility. Default is True.

        Returns:
            dict: A dictionary mapping each doubled polarization letter
                (e.g. "LL", "RR") to the sky-averaged quadratic TDI response,
                with shape (configurations, x_vector, TDI, TDI).
        """
        quadratic_integrated = {}
        for p in polarization:
            # Computes the integral for the TDI variable
            quadratic_integrated[2 * p] = quadratic_response_integrated(
                quadratic_integrand[TDI][2 * p]
            )

        return quadratic_integrated

    # @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def compute_detector(
        self,
        times_in_years: ArrayLike,
        theta_array: ArrayLike,
        phi_array: ArrayLike,
        frequency_array: ArrayLike,
        TDI: str = "XYZ",
        polarization: str = "LR",
    ) -> None:
        """
        Computes and caches the full response chain (single-link, linear TDI
        integrand, quadratic TDI integrand, and sky-averaged quadratic TDI
        response) for a given TDI combination.

        Results are stored in :attr:`single_link_response`,
        :attr:`linear_integrand`, :attr:`quadratic_integrand`, and
        :attr:`quadratic_integrated` (the latter three keyed by ``TDI``).

        Args:
            times_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the satellite positions and detector arms.
            theta_array (ArrayLike): Colatitude(s) of the sky position(s),
                in radians.
            phi_array (ArrayLike): Longitude(s) of the sky position(s), in
                radians.
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the response.
            TDI (str, optional): Name of the TDI combination (a key of
                :data:`gw_response.tdi.TDI_map`). Default is "XYZ".
            polarization (str, optional): Polarization basis to use, either
                "LR" (left/right circular) or "PC" (plus/cross). Default is
                "LR".
        """
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

        # Computes the integral for the TDI variable
        self.quadratic_integrand[TDI] = self.get_quadratic_integrand(
            times_in_years,
            self.single_link_response,
            frequency_array,
            TDI=TDI,
            polarization=polarization,
        )

        # Computes the integral for the TDI variable
        self.quadratic_integrated[TDI] = self.get_quadratic_integrated(
            self.quadratic_integrand, TDI=TDI, polarization=polarization
        )
