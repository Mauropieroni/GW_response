# `gw_response`

[![Python package](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml/badge.svg)](https://github.com/Mauropieroni/GW_response/actions/workflows/python-package.yml)

A jit-enhanced python code to compute the response function of a GW interferometer.

**Release:** v2.0.0 available now. Tested on python 3.8, 3.9, 3.10, and 3.11.

**Current status:** Under active development

## Quick Start

```python
import gw_response as gwr
import jax.numpy as jnp

# Compute LISA response in one line
frequencies = jnp.logspace(-4, -1, 200)  # 0.1 mHz to 0.1 Hz
result = gwr.compute_response(frequencies, tdi="AET")

# Access results easily
print(result)              # Summary
print(result.AA)           # A-A channel response
print(result.diagonal())   # All diagonal TDI elements (AA, EE, TT)

# For unpolarized stochastic background
total = result.sum_polarizations()
```

### TDI Options

```python
# See available TDI combinations with guidance
gwr.list_tdi_options()
```

### Advanced Usage

For full control, use the lower-level `Response` class:

```python
response = gwr.Response(ps=gwr.PhysicalConstants(), det=gwr.LISA())
pixel = gwr.Pixel(NSIDE=16)
response.compute_detector(
    times_in_years=jnp.array([0.0]),
    theta_array=pixel.theta_pixel,
    phi_array=pixel.phi_pixel,
    frequency_array=frequencies,
    TDI="XYZ",
    polarization="LR",
)
result = response.quadratic_integrated["XYZ"]["LL"]
```

## Installation

```bash
pip install -r requirements.txt
pip install .
```

**Contact:**
- [Mauro Pieroni](mailto:mauro.pieroni@cern.ch)
- [James Alvey](mailto:j.b.g.alvey@uva.nl)
- [Uddipta Bhardwaj](u.bhardwaj@uva.nl)
