# Quickstart

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

See the
[`tutorial.ipynb`](https://github.com/Mauropieroni/GW_response/blob/main/tutorial.ipynb)
notebook for a full, worked walk-through.

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
