# autoACPC

Automatically set brain image origin to AC-PC (anterior commissure–posterior commissure) alignment.

## How it works

The algorithm (based on [QSIPrep](https://qsiprep.readthedocs.io/)):

1. **Template registration** — Registers the input brain image to a standard template (default: MNI152NLin2009cAsym) using ANTs with a two-stage strategy (Similarity + Affine).
2. **Rigid extraction** — Decomposes the full affine transform to extract only the 6-DOF rigid component (rotation + translation), discarding scaling and shearing.
3. **Resampling** — Applies the rigid transform to resample the input image into AC-PC aligned space.

This produces an output image whose origin is at AC-PC, with the same anatomical content as the input (no nonlinear warping).

## Prerequisites

- **Python** >= 3.10
- **ANTs** — `antsRegistration` and `antsApplyTransforms` must be on your `PATH`
- **TemplateFlow** templates are auto-downloaded on first use

## Installation

```bash
pip install .

# For development:
pip install -e ".[dev]"
```

## Usage

```bash
# Basic usage
autoacpc input.nii.gz output_acpc.nii.gz

# With options
autoacpc input.nii.gz output_acpc.nii.gz \
    --template MNI152NLin2009cAsym \
    --modality T1w \
    --interpolation LanczosWindowedSinc \
    --save-transform rigid_acpc.mat \
    --verbose

# Fast mode (less accurate, useful for testing)
autoacpc input.nii.gz output_acpc.nii.gz --fast
```

### Options

| Option | Default | Description |
|---|---|---|
| `--template` | `MNI152NLin2009cAsym` | TemplateFlow template name |
| `--modality` | `T1w` | Template modality (`T1w` or `T2w`) |
| `--interpolation` | `LanczosWindowedSinc` | Resampling interpolation method |
| `--fast` | off | Use fast registration (fewer iterations) |
| `--work-dir` | temp dir | Directory for intermediate files |
| `--save-transform` | — | Save the rigid transform file |
| `-v, --verbose` | off | Enable debug logging |

### Python API

```python
from autoacpc.pipeline import acpc_align

acpc_align(
    input_image="sub-01_T1w.nii.gz",
    output_image="sub-01_T1w_acpc.nii.gz",
)
```

## Running tests

```bash
pytest
```
