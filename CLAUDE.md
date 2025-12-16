# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GW_response is a JAX-based Python package for computing the response function of gravitational wave (GW) interferometers, specifically LISA (Laser Interferometer Space Antenna). It uses JIT compilation for performance-critical physics simulations.

## Common Commands

```bash
# Install dependencies and package
pip install -r requirements.txt
pip install .

# Run tests
pytest

# Lint
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Format
black . --line-length=88
```

## High-Level API (Recommended)

```python
import gw_response as gwr
import jax.numpy as jnp

# One-liner for LISA response
result = gwr.compute_response(jnp.logspace(-4, -1, 200), tdi="AET")

# Access results
result.LL          # Left-Left polarization response
result.AA          # A-A TDI diagonal element
result.diagonal()  # All diagonal TDI elements
result.sum_polarizations()  # Sum over polarizations

# TDI guidance
gwr.list_tdi_options()  # Print available TDI combinations
```

## Architecture

The package follows a layered architecture for gravitational wave detector response calculations:

**High-Level API** (`api.py`, `results.py`, `presets.py`):
- `compute_response()` - one-liner for common use cases
- `ResponseResult` - typed result object with convenience properties
- `list_tdi_options()` - TDI guidance for users

**Core Layer** (`constants.py`, `utils.py`):
- `PhysicalConstants` dataclass with astronomical/physics constants
- `Pixel` class for sky pixelization (HEALPix)
- Coordinate transformation utilities

**Detector Layer** (`detector.py`, `lisa.py`):
- `Detector` abstract base class
- `LISA` dataclass implementing satellite orbital mechanics with analytical parametrization
- Computes satellite positions and arm vectors over time

**Physics Layer** (`single_link.py`):
- Single interferometer link response calculations
- GW polarization tensors (L/R left/right circular, P/C plus/cross)
- Phase accumulation through optical paths

**Signal Processing** (`tdi.py`, `noise.py`, `response.py`):
- `TDI` for Time Delay Interferometry (XYZ, AET coordinate bases)
- Noise characterization (acceleration, interferometric)
- `Response` class orchestrates complete detector response computation

## Key Patterns

- All computational functions use `@jax.jit` for JIT compilation
- 64-bit floats enabled via `jax.config.update("jax_enable_x64", True)`
- Use `jax.numpy` (jnp) instead of standard numpy for JAX compatibility
- Data classes use `@chex.dataclass` (frozen for immutable constants)
- Heavy use of `jnp.einsum` for tensor contractions
- Tests validate against pre-computed reference arrays in `testing/test_data/*.npy`
