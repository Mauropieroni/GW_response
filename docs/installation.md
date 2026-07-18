# Installation

The package requires Python ≥ 3.10. JAX and all other dependencies are resolved
automatically.

## Using `uv` (recommended)

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

## Using `pip`

```bash
pip install .                 # base install
pip install ".[test]"         # with testing tools
pip install ".[cuda]"         # with GPU (CUDA 12) support
```

## GPU support

The `cuda` extra installs the CUDA-enabled build of JAX. To confirm the GPU is
visible:

```bash
python -c "import jax; print(jax.devices())"
# e.g. [CudaDevice(id=0)]
```

## Building the docs locally

```bash
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. These are the same pages
published on Read the Docs, built from the `docs` extra declared in
`pyproject.toml`.
