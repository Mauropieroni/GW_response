# `gw_response`

[![Python package](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml/badge.svg)](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

A [JAX](https://github.com/google/jax)-accelerated Python package for computing the
**response function of a space-based gravitational-wave interferometer** such as
[LISA](https://www.elisascience.org/). Everything is written in terms of JAX
primitives, so the full pipeline — from satellite orbits to time-delay
interferometry (TDI) and detector noise — is JIT-compilable and runs on CPU or GPU.

> **Release:** `v1.0.0` — tested on Python 3.10, 3.11, 3.12, and 3.13.
> **Status:** under active development.

---

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

## Installation

The package requires Python ≥ 3.10. JAX and all other dependencies are resolved
automatically.

### Using `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) gives fast, reproducible installs. From a clone
of the repository:

```bash
# create an isolated environment and install the package (editable)
uv venv --python 3.10
uv pip install -e .
```

Optional extras:

```bash
# development / testing tools (pytest, flake8, black)
uv pip install -e ".[test]"

# GPU acceleration via NVIDIA CUDA 12 wheels
uv pip install -e ".[cuda]"

# everything at once
uv pip install -e ".[test,cuda]"
```

### Using `pip`

```bash
pip install .                 # base install
pip install ".[test]"         # with testing tools
pip install ".[cuda]"         # with GPU (CUDA 12) support
```

### GPU support

The `cuda` extra installs the CUDA-enabled build of JAX. To confirm the GPU is
visible:

```bash
python -c "import jax; print(jax.devices())"
# e.g. [CudaDevice(id=0)]
```

## Quickstart

Compute the single-link response and the sky-integrated quadratic response of LISA
in the `XYZ` TDI basis:

```python
import jax.numpy as jnp
import gw_response as gwr

# Detector and sky pixelisation
response = gwr.Response(ps=gwr.PhysicalConstants(), det=gwr.LISA())
pixel = gwr.Pixel(NSIDE=8)

frequencies = jnp.logspace(-4, 0, 300)      # Hz
times = jnp.array([0.0])                     # observation times [years]

# Full pipeline: single-link -> linear & quadratic integrands -> integrated response
response.compute_detector(
    times_in_years=times,
    theta_array=pixel.theta_pixel,
    phi_array=pixel.phi_pixel,
    frequency_array=frequencies,
    TDI="XYZ",
    polarization="LR",
)

# Sky-integrated quadratic response of the LL and RR channels
print(response.quadratic_integrated["XYZ"]["LL"].shape)
```

Project the instrument noise into the same TDI basis:

```python
noise = gwr.Noise(det=gwr.LISA(), frequency_array=frequencies)

TM = jnp.full(6, 3.0)    # test-mass acceleration noise parameters (one per link)
OMS = jnp.full(6, 15.0)  # optical metrology system noise parameters

noise.compute_detector(times, TM, OMS, TDI="XYZ")
print(noise.noise_matrix["XYZ"].shape)
```

See [`tutorial.ipynb`](tutorial.ipynb) for a full, worked walk-through.

## Package layout

| Module | Contents |
| --- | --- |
| `constants.py` | `PhysicalConstants`, basis transformations |
| `detector.py` | Abstract `Detector` base class |
| `lisa.py` | `LISA` detector: analytic orbits and arm vectors |
| `single_link.py` | Single-link response, polarization tensors |
| `tdi.py` | Time-delay interferometry combinations (`XYZ`, `AET`, Sagnac, …) |
| `response.py` | `Response`: linear & quadratic response driver |
| `noise.py` | `Noise`: test-mass and OMS noise in the TDI basis |
| `utils.py` | HEALPix `Pixel` helper and array utilities |

## Testing

```bash
uv pip install -e ".[test]"
pytest
```

## Contact

- [Mauro Pieroni](mailto:mauro.pieroni@cern.ch)
- [James Alvey](mailto:jbga2@cam.ac.uk)

## License

See [`LICENSE`](LICENSE).
