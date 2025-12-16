"""
TDI presets and guidance for GW response computations.

This module provides documentation and guidance on TDI (Time Delay Interferometry)
combinations and common configuration presets.
"""

from typing import Literal
from dataclasses import dataclass


# Type aliases for valid options
TDIBasis = Literal["XYZ", "AET", "Sagnac", "AET_Sagnac", "AE_zeta", "AE_Sagnac_zeta"]
PolarizationBasis = Literal["LR", "PC"]


@dataclass(frozen=True)
class TDIInfo:
    """Information about a TDI combination."""

    name: str
    channels: tuple
    description: str
    use_case: str
    noise_properties: str


TDI_GUIDE = {
    "XYZ": TDIInfo(
        name="XYZ (Michelson)",
        channels=("X", "Y", "Z"),
        description="First-generation Michelson TDI variables.",
        use_case="Standard choice for most analyses. Straightforward physical interpretation.",
        noise_properties="Correlated noise between channels. X, Y, Z related by 120-degree rotations.",
    ),
    "AET": TDIInfo(
        name="AET (Orthogonal)",
        channels=("A", "E", "T"),
        description="Orthogonalized combination of XYZ with decorrelated noise.",
        use_case="Preferred for parameter estimation and signal detection. A and E are signal-sensitive, T is null channel.",
        noise_properties="A and E have identical noise. T channel is insensitive to GW signals at low frequencies.",
    ),
    "Sagnac": TDIInfo(
        name="Sagnac",
        channels=("alpha", "beta", "gamma"),
        description="Sagnac interferometer combinations.",
        use_case="Useful for removing laser frequency noise. Good for studying instrumental systematics.",
        noise_properties="Different sensitivity pattern than Michelson. Insensitive to GW at DC.",
    ),
    "AET_Sagnac": TDIInfo(
        name="AET Sagnac",
        channels=("A_Sagnac", "E_Sagnac", "T_Sagnac"),
        description="Orthogonalized Sagnac combinations.",
        use_case="Combines benefits of AET decorrelation with Sagnac properties.",
        noise_properties="Decorrelated noise with Sagnac sensitivity pattern.",
    ),
    "AE_zeta": TDIInfo(
        name="AE + zeta",
        channels=("A", "E", "zeta"),
        description="A and E channels plus the symmetric Sagnac combination (zeta).",
        use_case="Three independent channels with different noise properties. Good for null tests.",
        noise_properties="zeta has different frequency dependence than A/E.",
    ),
    "AE_Sagnac_zeta": TDIInfo(
        name="AE Sagnac + zeta",
        channels=("A_Sagnac", "E_Sagnac", "zeta"),
        description="Sagnac-based A and E plus symmetric zeta.",
        use_case="Maximum independence between channels.",
        noise_properties="All three channels have different noise characteristics.",
    ),
}


def list_tdi_options() -> None:
    """Print a guide to available TDI combinations."""
    print("Available TDI Combinations for gw_response")
    print("=" * 60)
    for key, info in TDI_GUIDE.items():
        print(f"\n{key}: {info.name}")
        print(f"  Channels: {', '.join(info.channels)}")
        print(f"  Description: {info.description}")
        print(f"  Use case: {info.use_case}")
        print(f"  Noise: {info.noise_properties}")
    print("\n" + "=" * 60)
    print("Recommendation: Use 'AET' for most GW analyses.")


def get_tdi_info(tdi: str) -> TDIInfo:
    """
    Get detailed information about a TDI combination.

    Args:
        tdi: TDI combination name (e.g., "XYZ", "AET").

    Returns:
        TDIInfo dataclass with description, channels, use case, and noise properties.

    Raises:
        ValueError: If tdi is not a valid TDI combination.
    """
    if tdi not in TDI_GUIDE:
        valid = ", ".join(TDI_GUIDE.keys())
        raise ValueError(f"Unknown TDI '{tdi}'. Valid options: {valid}")
    return TDI_GUIDE[tdi]


# Common frequency ranges (Hz)
LISA_BAND = (3e-5, 0.5)
LISA_SWEET_SPOT = (1e-3, 1e-2)


# Default configurations
DEFAULT_CONFIG = {
    "detector": "LISA",
    "tdi": "AET",
    "polarization": "LR",
    "nside": 8,
    "n_frequencies": 300,
    "freq_range": LISA_BAND,
    "time": 0.0,
}
