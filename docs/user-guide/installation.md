---
icon: lucide/package
---

# Installation

## Requirements

- **Python** >= 3.11
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)

## Install from PyPI

```bash
uv add "esdiva[all]"
```

Or with pip:

```bash
pip install "esdiva[all]"
```

`[all]` is the recommended install: it switches on every optional feature.

To try the built-in demo without adding eSDIva to a project, use
[pipx](https://pipx.pypa.io/), which puts the `esdiva` command on your PATH in
its own isolated environment:

```bash
pipx install "esdiva[all]"
esdiva
```

For `import esdiva` in your own scripts, use `uv add` or `pip install` instead —
a pipx install is not visible to other environments.

## Install from GitHub

For the latest development version:

```bash
uv add "esdiva[all] @ git+https://github.com/EstebanRivera08/eSDIva.git"
```

## Development installation

Clone the repository and sync all dependencies, including every optional extra
and the dev tools:

```bash
git clone https://github.com/EstebanRivera08/eSDIva.git
cd eSDIva
uv sync
```

## Optional features (extras)

Simulating and plotting a field needs nothing beyond the core install. Three
features carry heavier dependencies, so they ship as opt-in extras. Install
plain `esdiva` and add back only what you use:

| Install | What you get |
|---------|--------------|
| `esdiva` | The simulator core: fields, transducers, reception, beamforming, 2-D and 3-D plotting. Enough for the whole Quick Start. |
| `esdiva[atlas]` | [`BG_Atlas`][esdiva.utilities.BG_Atlas], which maps computed fields onto brain anatomy through [BrainGlobe](https://brainglobe.info/). Adds pandas, pyarrow and tifffile. |
| `esdiva[video]` | Saving animations as `.mp4`, `.avi` or `.gif` from the plotting utilities. Adds imageio and a bundled ffmpeg. Without it, static plots and screenshots still work. |
| `esdiva[jupyter]` | Interactive 3-D PyVista scenes rendered *inside* a notebook. Adds the trame and ipywidgets stack. Without it, 3-D scenes still open in a desktop window. |
| `esdiva[all]` | All of the above. |

Extras combine:

```bash
pip install "esdiva[atlas,video]"
```

Using a feature whose extra is missing raises an error naming the extra to
install, so nothing fails silently.

## Verify installation

```python
import esdiva
print(esdiva.__version__)
```
