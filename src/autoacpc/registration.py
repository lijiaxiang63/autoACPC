"""Backwards-compatible re-exports from ants and transforms modules."""

from .ants import (
    ACPC_REGISTRATION_SETTINGS,
    FAST_REGISTRATION_SETTINGS,
    apply_transform,
    build_ants_command,
    run_registration,
)
from .transforms import (
    affine_to_rigid,
    apply_transform_to_header,
    read_itk_transform,
    write_itk_affine_tfm,
)

__all__ = [
    "ACPC_REGISTRATION_SETTINGS",
    "FAST_REGISTRATION_SETTINGS",
    "apply_transform",
    "build_ants_command",
    "run_registration",
    "affine_to_rigid",
    "apply_transform_to_header",
    "read_itk_transform",
    "write_itk_affine_tfm",
]
