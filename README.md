# autoACPC

Automatically set brain image origin to AC-PC (anterior commissure–posterior commissure) alignment.

## How it works

The algorithm (based on [QSIPrep](https://qsiprep.readthedocs.io/)):

1. **Template registration** — Registers the input brain image to a standard template (default: MNI152NLin2009cAsym) using ANTs with a two-stage strategy (Similarity + Affine).
2. **Rigid extraction** — Decomposes the full affine transform to extract only the 6-DOF rigid component (rotation + translation), discarding scaling and shearing.
3. **Apply transform** — Either resamples the image into AC-PC aligned space (default) or modifies the NIfTI header affine to update world coordinates while preserving the original voxel data (`--header-only`).

This produces an output image whose origin is at AC-PC, with the same anatomical content as the input (no nonlinear warping).

## Prerequisites

- **ANTs** — `antsRegistration` and `antsApplyTransforms` must be on your `PATH`
- **TemplateFlow** templates are auto-downloaded on first use (or supply your own with `--template-path`)

## Tested environment

- Tested on macOS
- Tested with ANTs 2.6.5

## Installation

### Standalone binary

Download the latest archive for your platform from [GitHub Releases](../../releases/latest). No Python installation required.

```bash
# macOS / Linux
tar -xzf autoacpc-Darwin-arm64.tar.gz   # or autoacpc-Linux-x86_64.tar.gz
./autoacpc/autoacpc input.nii.gz output_acpc.nii.gz

# Windows — extract autoacpc-Windows-AMD64.zip, then:
autoacpc\autoacpc.exe input.nii.gz output_acpc.nii.gz
```

### From source

```bash
# Using uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Or with pip
pip install .
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

# Header-only mode (update affine, no resampling — preserves voxel data)
autoacpc input.nii.gz output_acpc.nii.gz --header-only

# Use a local template instead of downloading from TemplateFlow
autoacpc input.nii.gz output_acpc.nii.gz \
    --template-path /path/to/template.nii.gz \
    --template-mask /path/to/brain_mask.nii.gz
```

### Options

| Option | Default | Description |
|---|---|---|
| `--template` | `MNI152NLin2009cAsym` | TemplateFlow template name |
| `--modality` | `T1w` | Template modality (`T1w` or `T2w`) |
| `--interpolation` | `LanczosWindowedSinc` | Resampling interpolation method |
| `--fast` | off | Use fast registration (fewer iterations) |
| `--work-dir` | temp dir | Directory for intermediate files |
| `--template-path` | — | Path to a local template image (bypasses TemplateFlow) |
| `--template-mask` | — | Path to a local brain mask (use with `--template-path`) |
| `--header-only` | off | Modify NIfTI header affine instead of resampling (preserves voxel data) |
| `--save-transform` | — | Save the rigid transform file |
| `-v, --verbose` | off | Enable debug logging |

### Python API

```python
from autoacpc.pipeline import acpc_align

acpc_align(
    input_image="sub-01_T1w.nii.gz",
    output_image="sub-01_T1w_acpc.nii.gz",
)

# Header-only mode
acpc_align(
    input_image="sub-01_T1w.nii.gz",
    output_image="sub-01_T1w_acpc.nii.gz",
    header_only=True,
)

# With a local template
acpc_align(
    input_image="sub-01_T1w.nii.gz",
    output_image="sub-01_T1w_acpc.nii.gz",
    template_path="/path/to/template.nii.gz",
    template_mask="/path/to/brain_mask.nii.gz",
)
```

## Running tests

```bash
uv run pytest
```

## Building standalone binary locally

```bash
uv run python build.py        # outputs dist/autoacpc-<OS>-<arch>.tar.gz (.zip on Windows)
uv run python build.py clean  # remove build artifacts
```
