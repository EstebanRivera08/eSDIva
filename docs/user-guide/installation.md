---
icon: lucide/package
---

# Installation

## Requirements

- **Python** >= 3.11
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)

## Install from GitHub

```bash
uv add git+https://github.com/EstebanRivera08/eSDIva.git
```

Or with pip:

```bash
pip install git+https://github.com/EstebanRivera08/eSDIva.git
```

## Development installation

Clone the repository and sync all dependencies including dev tools:

```bash
git clone https://github.com/EstebanRivera08/eSDIva.git
cd eSDIva
uv sync
```

## Optional dependencies

| Package | Purpose |
|---------|---------|
| `pyvista` | 3-D interactive visualization |
| `brainglobe-atlasapi` | Brain atlas integration (Examples 6–7) |

These are installed automatically when running `uv sync` inside the cloned repository.

## Verify installation

```python
import esdiva
print(esdiva.__version__)
```

!!! note "PyPI release coming soon"
    eSDIva will be available on PyPI in a future release. For now, install from GitHub.
