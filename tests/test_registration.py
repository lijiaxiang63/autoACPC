"""Tests for registration module."""

from autoacpc.registration import (
    ACPC_REGISTRATION_SETTINGS,
    FAST_REGISTRATION_SETTINGS,
    build_ants_command,
)


def test_build_ants_command_contains_expected_flags():
    cmd = build_ants_command(
        moving_image="moving.nii.gz",
        reference_image="ref.nii.gz",
        output_prefix="/tmp/test_",
    )
    cmd_str = " ".join(cmd)
    assert "antsRegistration" in cmd_str
    assert "--dimensionality" in cmd_str
    assert "Similarity" in cmd_str
    assert "Affine" in cmd_str
    assert "moving.nii.gz" in cmd_str
    assert "ref.nii.gz" in cmd_str


def test_build_ants_command_with_masks():
    cmd = build_ants_command(
        moving_image="moving.nii.gz",
        reference_image="ref.nii.gz",
        output_prefix="/tmp/test_",
        moving_mask="moving_mask.nii.gz",
        reference_mask="ref_mask.nii.gz",
    )
    cmd_str = " ".join(cmd)
    assert "--masks" in cmd_str
    assert "ref_mask.nii.gz" in cmd_str


def test_fast_settings_fewer_iterations():
    normal_iters = ACPC_REGISTRATION_SETTINGS["number_of_iterations"]
    fast_iters = FAST_REGISTRATION_SETTINGS["number_of_iterations"]
    for normal, fast in zip(normal_iters, fast_iters):
        for n, f in zip(normal, fast):
            assert f <= n


def test_build_ants_command_fast_uses_fast_settings():
    cmd_normal = build_ants_command(
        moving_image="m.nii.gz",
        reference_image="r.nii.gz",
        output_prefix="/tmp/n_",
    )
    cmd_fast = build_ants_command(
        moving_image="m.nii.gz",
        reference_image="r.nii.gz",
        output_prefix="/tmp/f_",
        fast=True,
    )
    # Fast should use larger shrink factors (coarser)
    assert "8x4" in " ".join(cmd_fast)
    assert "4x2" in " ".join(cmd_normal)
