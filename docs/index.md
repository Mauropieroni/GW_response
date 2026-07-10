# gw_response

A [JAX](https://github.com/google/jax)-accelerated Python package for computing the
**response function of a space-based gravitational-wave interferometer** such as
[LISA](https://www.elisascience.org/). Everything is written in terms of JAX
primitives, so the full pipeline — from satellite orbits to time-delay
interferometry (TDI) and detector noise — is JIT-compilable and runs on CPU or GPU.

## Features

- **Single-link response** — sky- and frequency-resolved response of each of the
  six LISA laser links to a gravitational-wave background.
- **Time-delay interferometry** — build `XYZ`, `AET`, Sagnac, and `zeta`
  combinations from the single-link response.
- **Linear & quadratic response** — angular integrands and their sky-integrated
  quadratic response for any TDI channel and polarization basis (`LR` or `PC`).
- **Instrument noise** — test-mass (acceleration) and OMS (interferometric) noise
  projected into the TDI basis via the `Noise` class.
- **Analytic LISA orbits** — closed-form satellite positions and arm vectors, with
  a `Detector` base class for extending to other missions.
- **HEALPix sky pixelisation** through a JAX-friendly `Pixel` helper.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api/index
```

## Links

- [GitHub repository](https://github.com/Mauropieroni/GW_response)
- [Tutorial notebook](https://github.com/Mauropieroni/GW_response/blob/main/tutorial.ipynb)
- [License](https://github.com/Mauropieroni/GW_response/blob/main/LICENSE)
