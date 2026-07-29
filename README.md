# `gw_response`

[![Python package](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml/badge.svg)](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

A [JAX](https://github.com/google/jax)-accelerated Python package for computing the
**response function of gravitational-wave interferometers**, both space-based
(such as [LISA](https://www.elisascience.org/)) and ground-based (such as LIGO).
Everything is written in terms of JAX primitives, so the full pipeline — from
detector geometry to the readout combination (e.g. time-delay interferometry for
LISA, or the Michelson combination for LIGO) and detector noise — is
JIT-compilable and runs on CPU or GPU. A common `Detector` interface lets the
single-link response, noise projection, and integration logic be shared across
detector types.

> **Release:** `v1.0.0` — tested on Python 3.10, 3.11, 3.12, and 3.13.
> **Status:** under active development.

---

## Features

- **`Detector` base class** — a common interface (vertex positions, arms, readout
  combination, noise) implemented by each concrete detector, so `Response` and
  `Noise` are written once and work for any of them.
- **LISA** (space-based, implemented) — six-link single-link response;
  time-delay interferometry combinations (`XYZ`, `AET`, Sagnac, `zeta`, and
  Sagnac-based variants) built from the single-link response; test-mass
  (acceleration) and OMS (interferometric) noise projected into the TDI basis.
  **Multiple orbit models** are supported — a perfectly rigid analytic
  constellation, an exact Keplerian cartwheel model (Martens & Joffre 2021,
  arXiv:2101.03040), or numerical orbits interpolated from a file (plain-text or
  [`lisaorbits`](https://pypi.org/project/lisaorbits/) HDF5), selected via
  `orbit_approximant`.
- **LIGO** (ground-based, implemented) — site geometry for the Hanford/Livingston
  interferometers, Michelson-combination response, and the LIGO design
  sensitivity curve as the noise model.
- **Cosmic Explorer, Einstein Telescope, Taiji** — stubbed out as `Detector`
  subclasses, not yet implemented.
- **Linear & quadratic response** — angular integrands and their sky-integrated
  quadratic response for any detector, readout combination, and polarization
  basis (`LR` or `PC`), via the generic `Response` class.
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
in the `XYZ` TDI basis. `Response` and `Noise` are generic: the detector is passed
in explicitly to each call, so the same `response`/`noise` objects work for any
`Detector` (e.g. swap `gwr.LISA()` for `gwr.LIGO()` and `combination="XYZ"` for
`combination="Michelson"`):

```python
import jax.numpy as jnp
import gw_response as gwr

# Detector and sky pixelisation
det = gwr.LISA()
response = gwr.Response()
pixel = gwr.Pixel(NSIDE=8)

frequencies = jnp.logspace(-4, 0, 300)      # Hz
times = jnp.array([0.0])                     # observation times [years]

# Full pipeline: single-link -> linear & quadratic integrands -> integrated response
response.compute_detector(
    det,
    times_in_years=times,
    theta_array=pixel.theta_pixel,
    phi_array=pixel.phi_pixel,
    frequency_array=frequencies,
    combination="XYZ",
    polarization="LR",
)

# Sky-integrated quadratic response of the LL and RR channels
print(response.quadratic_integrated["XYZ"]["LL"].shape)
```

Project the instrument noise into the same TDI basis:

```python
noise = gwr.Noise()

TM = jnp.full(6, 3.0)    # test-mass acceleration noise parameters (one per link)
OMS = jnp.full(6, 15.0)  # optical metrology system noise parameters

noise.compute_detector(
    det,
    times,
    frequencies,
    combination="XYZ",
    TM_acceleration_parameters=TM,
    OMS_parameters=OMS,
)
print(noise.noise_matrix["XYZ"].shape)
```

Switch between LISA orbit models via `orbit_approximant`:

```python
lisa_rigid = gwr.LISA()  # orbit_approximant="rigid" (default)
lisa_keplerian = gwr.LISA(orbit_approximant="keplerian")
lisa_numeric = gwr.LISA(
    orbit_approximant="numeric",
    orbit_file="orbits.h5",  # plain-text or lisaorbits HDF5
)
```

See [`tutorial_LISA.ipynb`](tutorial_LISA.ipynb) and
[`tutorial_LIGO.ipynb`](tutorial_LIGO.ipynb) for full, worked walk-throughs.

## Package layout

| Module | Contents |
| --- | --- |
| `constants.py` | `PhysicalConstants`, basis transformations |
| `detector.py` | Abstract `Detector` base class |
| `single_link.py` | Single-link response, polarization tensors |
| `response.py` | `Response`: linear & quadratic response driver, generic over detectors |
| `noise.py` | `Noise`: per-link/projected noise driver, generic over detectors |
| `utils.py` | HEALPix `Pixel` helper, orbit-file loading, and shared array/geometry utilities |
| `space_based/lisa.py` | `LISA` detector: rigid, Keplerian, and numerical orbits and arm vectors |
| `space_based/orbits.py` | LISA constellation orbit models |
| `space_based/tdi.py` | Time-delay interferometry combinations (`XYZ`, `AET`, Sagnac, …) |
| `space_based/noise.py` | LISA test-mass and OMS single-link noise budget |
| `space_based/taiji.py` | `Taiji` detector (not yet implemented) |
| `ground_based/ligo.py` | `LIGO` detector: site geometry, Michelson combination, design noise curve |
| `ground_based/datastream.py` | Michelson-combination response shared by ground-based detectors |
| `ground_based/cosmic_explorer.py` | `CosmicExplorer` detector (not yet implemented) |
| `ground_based/einstein_telescope.py` | `EinsteinTelescope` detector (not yet implemented) |

## Testing

```bash
uv pip install -e ".[test]"
pytest
```

## Contact

- [Mauro Pieroni](mailto:mauro.pieroni@csic.es)
- [James Alvey](mailto:jbga2@cam.ac.uk)
- [Androniki Dimitriou](mailto:androniki.dimitriou@ific.uv.es)

## License

See [`LICENSE`](LICENSE).
