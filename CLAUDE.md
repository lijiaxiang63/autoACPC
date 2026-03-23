# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

autoACPC is a standalone CLI tool that automatically sets brain image origin to AC-PC alignment. It implements the AC-PC alignment algorithm from QSIPrep without nipype dependencies, using ANTs directly via subprocess.

## Build and development commands

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_registration.py::test_build_ants_command_contains_expected_flags

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Architecture

The pipeline has three stages, each in its own module:

1. **`template.py`** — Fetches standard template + brain mask from TemplateFlow (skipped when a local template path is provided)
2. **`registration.py`** — Runs ANTs registration (Similarity + Affine), then decomposes the affine into a 6-DOF rigid transform using SimpleITK + dipy geometry. Also provides `apply_transform_to_header` for header-only mode (modifies NIfTI affine instead of resampling).
3. **`pipeline.py`** — Orchestrates the full flow: fetch template → register → extract rigid → apply transform (resampling or header-only)

**`cli.py`** is a thin Click wrapper around `pipeline.acpc_align()`.

### Key algorithm detail

The rigid extraction (`affine_to_rigid`) decomposes the ANTs affine matrix using `dipy.core.geometry.decompose_matrix` to get Euler angles, rebuilds a pure rotation matrix, and creates a `sitk.Euler3DTransform` with only rotation + translation. This is the same approach as QSIPrep's `itk_affine_to_rigid` in `qsiprep/interfaces/itk.py`.

## External dependencies

- **ANTs**: `antsRegistration` and `antsApplyTransforms` must be on `PATH`
- **TemplateFlow**: Templates auto-download on first use; default is `MNI152NLin2009cAsym`. Can be bypassed with `--template-path` to use a local template

## Project layout

Uses `src/` layout with hatchling build backend. Source is in `src/autoacpc/`, tests in `tests/`.
