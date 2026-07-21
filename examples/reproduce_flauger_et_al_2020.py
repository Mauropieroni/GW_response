#!/usr/bin/env python
r"""
Reproduce Figures 2 and 3 of

    Flauger, Karnesis, Nardini, Pieroni, Ricciardone & Torrado,
    "Improved reconstruction of a stochastic gravitational wave background
     with LISA", arXiv:2009.11845 (JCAP 01 (2021) 059)

using the ``gw_response`` package.

- Figure 2: LISA noise spectra N_ij(f) (left) and strain sensitivities
  sqrt(S_n^ij) (right) for the XX, XY, AA and TT channels.
- Figure 3: geometrical response factors R~_ij(f) (left) and the full
  quadratic response functions R_ij(f) (right) for the same channels.

The relevant paper relations, all recovered numerically by the package:

    R_ij(f)   = 16 sin^2(2 pi f L / c) (2 pi f L / c)^2  R~_ij(f)   (eq 2.22)
    S_n^ij(f) = N_ij(f) / R_ij(f)                                   (eq 2.26)
    N_AA = N_XX - N_XY ,   N_TT = N_XX + 2 N_XY                      (eq 2.17)
    R_AA = R_XX - R_XY ,   R_TT = R_XX + 2 R_XY                      (eq 2.18)

with the ESA noise amplitudes P = 15 (IMS) and A = 3 (acceleration), and the
equilateral LISA arm length L = 2.5e9 m.

Run:
    python reproduce_2009_11845_fig2_fig3.py
Outputs ``fig2_noise.png`` and ``fig3_response.png`` in the working directory.
"""

import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import gw_response as gwr
from gw_response.noise import LISA_acceleration_noise, LISA_interferometric_noise

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Write figures next to this script (in ``examples/figures/``) so the example
# reproduces identically regardless of the current working directory.
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
P_IMS_AMP = 15.0  # IMS (optical metrology) noise amplitude  [paper: P = 15]
A_ACC_AMP = 3.0  # acceleration (test-mass) noise amplitude  [paper: A = 3]
NSIDE = 16  # HEALPix sky resolution for the response sky-integral
N_FREQ = 2000  # number of (log-spaced) frequency points

# Channels to plot: label -> (TDI basis, matrix row, matrix col, colour, dash)
CHANNELS = {
    "XX": ("XYZ", 0, 0, "tab:blue", "-"),
    "XY": ("XYZ", 0, 1, "tab:purple", "--"),
    "AA": ("AET", 0, 0, "tab:orange", "-"),
    "TT": ("AET", 2, 2, "tab:green", "-"),
}


def build_detector_quantities():
    """Compute noise spectra and response functions for all channels."""
    det = gwr.LISA()
    c, L = det.ps.light_speed, det.armlength

    freqs = jnp.logspace(np.log10(det.fmin), np.log10(det.fmax), N_FREQ)
    f = np.asarray(freqs)
    times = jnp.array([0.0])  # response is evaluated at a single time slice

    # -- Response functions R_ij (quadratic, sky-integrated) ---------------- #
    # The package stores the polarization-summed response per TDI channel; the
    # "LL" entry already carries the sum over L/R polarizations, so it equals
    # the paper's R_ij directly (verified against R~_XX -> 3/10, R~_AA -> 9/20).
    response = gwr.Response(ps=gwr.PhysicalConstants(), det=det)
    pixel = gwr.Pixel(NSIDE=NSIDE)
    for tdi in ("XYZ", "AET"):
        response.compute_detector(
            times_in_years=times,
            theta_array=pixel.theta_pixel,
            phi_array=pixel.phi_pixel,
            frequency_array=freqs,
            TDI=tdi,
            polarization="LR",
        )

    # -- Noise matrices N_ij ------------------------------------------------- #
    # TM (acceleration) parameters carry A, OMS parameters carry P; one entry
    # per LISA link (6 links), all identical in the equilateral, equal-noise
    # configuration assumed in the paper.
    noise = gwr.Noise(det=det, frequency_array=freqs)
    for tdi in ("XYZ", "AET"):
        noise.compute_detector(
            times,
            jnp.full(6, A_ACC_AMP),  # TM_acceleration_parameters  (A)
            jnp.full(6, P_IMS_AMP),  # OMS_parameters              (P)
            TDI=tdi,
        )

    # Geometry prefactor of eq. (2.22): R_ij = 16 sin^2(x) x^2 R~_ij, x=2pifL/c
    x_L = 2.0 * np.pi * f * L / c
    geom_prefactor = 16.0 * np.sin(x_L) ** 2 * x_L**2

    data = {"f": f, "geom_prefactor": geom_prefactor}
    for label, (tdi, i, j, _c, _d) in CHANNELS.items():
        R = np.asarray(response.quadratic_integrated[tdi]["LL"])[0, :, i, j].real
        N = np.asarray(noise.noise_matrix[tdi])[0, :, i, j].real
        data[label] = {
            "R": R,  # full response function      (Fig 3 R)
            "Rtilde": R / geom_prefactor,  # geometrical factor          (Fig 3 R~)
            "N": N,  # noise spectrum              (Fig 2 N)
            "Sn": N / R,  # strain sensitivity S_n=N/R  (Fig 2 sqrt)
        }
    return det, data


def verify(det, data):
    """Cross-check the package output against the paper's analytic results."""
    f, c, L = data["f"], det.ps.light_speed, det.armlength
    x_L = 2.0 * np.pi * f * L / c

    # Analytic noise spectra, eqs (2.7)-(2.9) & (2.17)
    Pacc = A_ACC_AMP**2 * np.asarray(LISA_acceleration_noise(jnp.asarray(f), 1.0))
    Pims = P_IMS_AMP**2 * np.asarray(LISA_interferometric_noise(jnp.asarray(f), 1.0))
    Naa = 16 * np.sin(x_L) ** 2 * ((3 + np.cos(2 * x_L)) * Pacc + Pims)
    Nab = -8 * np.sin(x_L) ** 2 * np.cos(x_L) * (4 * Pacc + Pims)
    analytic_N = {"XX": Naa, "XY": Nab, "AA": Naa - Nab, "TT": Naa + 2 * Nab}

    # Analytic low-frequency geometrical-factor limits, eqs (2.23)-(2.25)
    analytic_Rtilde0 = {"XX": 3 / 10, "XY": -3 / 20, "AA": 9 / 20, "TT": 0.0}

    print("Verification against arXiv:2009.11845 (max relative error):")
    for label in CHANNELS:
        n_err = np.max(
            np.abs(data[label]["N"] - analytic_N[label])
            / (np.abs(analytic_N[label]) + 1e-300)
        )
        r0 = data[label]["Rtilde"][0]
        print(
            f"  {label}:  N_ij vs analytic = {n_err:.2e}"
            f"   R~_ij(f->0) = {r0:+.4f}  (paper {analytic_Rtilde0[label]:+.3f})"
        )
    print()


def plot_figure2(data):
    """Figure 2: noise spectra (left) and strain sensitivities (right)."""
    f = data["f"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    for label, (_t, _i, _j, colour, dash) in CHANNELS.items():
        axL.loglog(
            f,
            np.abs(data[label]["N"]),
            dash,
            color=colour,
            label=rf"$N_{{\rm {label}}}$",
        )
        axR.loglog(
            f,
            np.sqrt(np.abs(data[label]["Sn"])),
            dash,
            color=colour,
            label=rf"$\sqrt{{|S_n^{{\rm {label}}}|}}$",
        )

    axL.set_title("LISA noise spectra")
    axL.set_xlabel("Frequency [Hz]")
    axL.set_ylabel(r"$N(f)$")
    axL.set_ylim(1e-52, 1e-36)

    axR.set_title("LISA strain sensitivities")
    axR.set_xlabel("Frequency [Hz]")
    axR.set_ylabel(r"$\sqrt{S_n(f)}\ [{\rm Hz}^{-1/2}]$")

    for ax in (axL, axR):
        ax.set_xlim(f[0], f[-1])
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Reproduction of Fig. 2 of arXiv:2009.11845", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "fig2_noise.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_figure3(data):
    """Figure 3: geometrical factors (left) and response functions (right)."""
    f = data["f"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    for label, (_t, _i, _j, colour, dash) in CHANNELS.items():
        axL.loglog(
            f,
            np.abs(data[label]["Rtilde"]),
            dash,
            color=colour,
            label=rf"$\tilde R_{{\rm {label}}}$",
        )
        axR.loglog(
            f,
            np.abs(data[label]["R"]),
            dash,
            color=colour,
            label=rf"$\mathcal{{R}}_{{\rm {label}}}$",
        )

    axL.set_title("LISA geometrical factors")
    axL.set_xlabel("Frequency [Hz]")
    axL.set_ylabel(r"$\tilde R(f)$")
    axL.set_ylim(1e-5, 1e0)

    axR.set_title("LISA response functions")
    axR.set_xlabel("Frequency [Hz]")
    axR.set_ylabel(r"$\mathcal{R}(f)$")
    axR.set_ylim(1e-13, 2e1)

    for ax in (axL, axR):
        ax.set_xlim(f[0], f[-1])
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Reproduction of Fig. 3 of arXiv:2009.11845", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "fig3_response.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    det, data = build_detector_quantities()
    verify(det, data)
    plot_figure2(data)
    plot_figure3(data)


if __name__ == "__main__":
    main()
