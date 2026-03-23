"""Tests for registration module."""

import nibabel as nb
import numpy as np
import SimpleITK as sitk

from autoacpc.registration import (
    ACPC_REGISTRATION_SETTINGS,
    FAST_REGISTRATION_SETTINGS,
    apply_transform_to_header,
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


def test_apply_transform_to_header_identity(tmp_path):
    """Identity transform should leave the affine unchanged."""
    # Create a simple NIfTI image
    affine = np.eye(4)
    affine[:3, 3] = [10, 20, 30]
    data = np.zeros((4, 4, 4), dtype=np.float32)
    img = nb.Nifti1Image(data, affine)
    input_path = str(tmp_path / "input.nii.gz")
    nb.save(img, input_path)

    # Write an identity Euler3DTransform
    rigid = sitk.Euler3DTransform()
    transform_path = str(tmp_path / "identity.mat")
    sitk.WriteTransform(rigid, transform_path)

    output_path = str(tmp_path / "output.nii.gz")
    apply_transform_to_header(input_path, transform_path, output_path)

    out = nb.load(output_path)
    np.testing.assert_array_almost_equal(out.affine, affine)
    np.testing.assert_array_equal(np.asarray(out.dataobj), data)


def test_apply_transform_to_header_translation(tmp_path):
    """Pure translation should shift the affine origin."""
    affine = np.eye(4)
    data = np.zeros((4, 4, 4), dtype=np.float32)
    img = nb.Nifti1Image(data, affine)
    input_path = str(tmp_path / "input.nii.gz")
    nb.save(img, input_path)

    # ITK Euler3DTransform with translation in LPS
    rigid = sitk.Euler3DTransform()
    # ITK translation is in LPS: (L, P, S)
    # After LPS→RAS conversion: (-L, -P, S) = (R, A, S)
    rigid.SetTranslation((5.0, 10.0, 15.0))
    transform_path = str(tmp_path / "translate.mat")
    sitk.WriteTransform(rigid, transform_path)

    output_path = str(tmp_path / "output.nii.gz")
    apply_transform_to_header(input_path, transform_path, output_path)

    out = nb.load(output_path)
    # LPS (5,10,15) → RAS (-5,-10,15)
    expected = np.eye(4)
    expected[:3, 3] = [-5.0, -10.0, 15.0]
    np.testing.assert_array_almost_equal(out.affine, expected)
    np.testing.assert_array_equal(np.asarray(out.dataobj), data)


def test_apply_transform_to_header_matches_point_transform(tmp_path):
    """Header affine should match the rigid point transform in world space."""
    affine = np.array([
        [0.0, -2.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, 20.0],
        [0.0, 0.0, 3.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    data = np.zeros((3, 3, 3), dtype=np.float32)
    img = nb.Nifti1Image(data, affine)
    input_path = str(tmp_path / "input_oblique.nii.gz")
    nb.save(img, input_path)

    rigid = sitk.Euler3DTransform()
    rigid.SetCenter((11.0, -7.0, 5.0))
    rigid.SetRotation(0.2, -0.1, 0.3)
    rigid.SetTranslation((4.0, -2.0, 1.0))
    transform_path = str(tmp_path / "rigid.mat")
    sitk.WriteTransform(rigid, transform_path)

    output_path = str(tmp_path / "output_oblique.nii.gz")
    apply_transform_to_header(input_path, transform_path, output_path)

    out = nb.load(output_path)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    for ijk in ((0, 0, 0), (1, 2, 1), (2, 1, 2)):
        voxel = np.array([*ijk, 1.0])
        original_world_ras = (affine @ voxel)[:3]
        original_world_lps = lps_to_ras @ original_world_ras
        expected_world_lps = np.array(rigid.TransformPoint(tuple(original_world_lps)))
        expected_world_ras = lps_to_ras @ expected_world_lps
        observed_world_ras = (out.affine @ voxel)[:3]
        np.testing.assert_allclose(observed_world_ras, expected_world_ras, atol=1e-6)


def test_apply_transform_to_header_preserves_scaled_intensities(tmp_path):
    """Header-only output should preserve unscaled storage and NIfTI scaling."""
    data = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    img = nb.Nifti1Image(data, np.eye(4))
    img.header.set_slope_inter(2.0, 10.0)
    input_path = str(tmp_path / "input_scaled.nii.gz")
    nb.save(img, input_path)

    rigid = sitk.Euler3DTransform()
    transform_path = str(tmp_path / "identity.mat")
    sitk.WriteTransform(rigid, transform_path)

    output_path = str(tmp_path / "output_scaled.nii.gz")
    apply_transform_to_header(input_path, transform_path, output_path)

    original = nb.load(input_path)
    out = nb.load(output_path)

    np.testing.assert_array_equal(np.asarray(out.dataobj), np.asarray(original.dataobj))
    np.testing.assert_array_equal(out.dataobj.get_unscaled(), original.dataobj.get_unscaled())
    assert out.dataobj.slope == original.dataobj.slope
    assert out.dataobj.inter == original.dataobj.inter
