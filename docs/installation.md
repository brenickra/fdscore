# Installation

## Requirements

- Python 3.10 or newer
- `numpy>=1.24`
- `scipy>=1.10`
- `numba>=0.58`

The project also defines an optional `spectral` extra, which installs
`FLife` for Dirlik-based spectral fatigue workflows.

## Install From PyPI

```bash
pip install fdscore
```

To enable the spectral workflow as well:

```bash
pip install "fdscore[spectral]"
```

## Development Installation

For editable development installs:

```bash
pip install -e .
```

To install development tooling as well:

```bash
pip install -e ".[dev]"
```

## Install From Source

```bash
git clone https://github.com/brenickra/fdscore.git
cd fdscore
pip install -e .
```

To install the optional spectral dependency in editable mode:

```bash
pip install -e ".[spectral]"
```
