# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

autoACPC is a standalone CLI tool that automatically sets brain image origin to AC-PC alignment. It implements the AC-PC alignment algorithm from QSIPrep without nipype dependencies, using ANTs directly via subprocess.

## Build and development commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_registration.py::test_build_ants_command_contains_expected_flags

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Architecture

The pipeline has three stages, each in its own module:

1. **`template.py`** — Fetches standard template + brain mask from TemplateFlow
2. **`registration.py`** — Runs ANTs registration (Similarity + Affine), then decomposes the affine into a 6-DOF rigid transform using SimpleITK + dipy geometry
3. **`pipeline.py`** — Orchestrates the full flow: fetch template → register → extract rigid → apply transform

**`cli.py`** is a thin Click wrapper around `pipeline.acpc_align()`.

### Key algorithm detail

The rigid extraction (`affine_to_rigid`) decomposes the ANTs affine matrix using `dipy.core.geometry.decompose_matrix` to get Euler angles, rebuilds a pure rotation matrix, and creates a `sitk.Euler3DTransform` with only rotation + translation. This is the same approach as QSIPrep's `itk_affine_to_rigid` in `qsiprep/interfaces/itk.py`.

## External dependencies

- **ANTs**: `antsRegistration` and `antsApplyTransforms` must be on `PATH`
- **TemplateFlow**: Templates auto-download on first use; default is `MNI152NLin2009cAsym`

## Project layout

Uses `src/` layout with hatchling build backend. Source is in `src/autoacpc/`, tests in `tests/`.
