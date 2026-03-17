"""Tests for pipeline module — local template path support."""

from unittest.mock import patch

import pytest

from autoacpc.pipeline import acpc_align


def test_local_template_skips_templateflow(tmp_path):
    """When template_path is given, get_template should not be called."""
    fake_template = tmp_path / "template.nii.gz"
    fake_template.write_bytes(b"fake")
    fake_input = tmp_path / "input.nii.gz"
    fake_input.write_bytes(b"fake")

    with (
        patch("autoacpc.pipeline.get_template") as mock_get,
        patch("autoacpc.pipeline.run_registration", return_value="affine.mat"),
        patch("autoacpc.pipeline.affine_to_rigid", return_value=("rigid.mat", "inv.mat")),
        patch("autoacpc.pipeline.apply_transform"),
        patch("autoacpc.pipeline.nb.load") as mock_load,
    ):
        mock_load.return_value.shape = (256, 256, 256)
        mock_load.return_value.header.get_zooms.return_value = (1.0, 1.0, 1.0)

        acpc_align(
            input_image=str(fake_input),
            output_image=str(tmp_path / "out.nii.gz"),
            template_path=str(fake_template),
            work_dir=str(tmp_path),
        )

        mock_get.assert_not_called()


def test_local_template_mask_is_optional(tmp_path):
    """mask_path should be None when template_mask is not given."""
    fake_template = tmp_path / "template.nii.gz"
    fake_template.write_bytes(b"fake")
    fake_input = tmp_path / "input.nii.gz"
    fake_input.write_bytes(b"fake")

    with (
        patch("autoacpc.pipeline.run_registration", return_value="affine.mat") as mock_reg,
        patch("autoacpc.pipeline.affine_to_rigid", return_value=("rigid.mat", "inv.mat")),
        patch("autoacpc.pipeline.apply_transform"),
        patch("autoacpc.pipeline.nb.load") as mock_load,
    ):
        mock_load.return_value.shape = (256, 256, 256)
        mock_load.return_value.header.get_zooms.return_value = (1.0, 1.0, 1.0)

        acpc_align(
            input_image=str(fake_input),
            output_image=str(tmp_path / "out.nii.gz"),
            template_path=str(fake_template),
            work_dir=str(tmp_path),
        )

        _, kwargs = mock_reg.call_args
        assert kwargs["reference_mask"] is None


def test_local_template_with_mask(tmp_path):
    """Both template and mask paths are forwarded to registration."""
    fake_template = tmp_path / "template.nii.gz"
    fake_template.write_bytes(b"fake")
    fake_mask = tmp_path / "mask.nii.gz"
    fake_mask.write_bytes(b"fake")
    fake_input = tmp_path / "input.nii.gz"
    fake_input.write_bytes(b"fake")

    with (
        patch("autoacpc.pipeline.run_registration", return_value="affine.mat") as mock_reg,
        patch("autoacpc.pipeline.affine_to_rigid", return_value=("rigid.mat", "inv.mat")),
        patch("autoacpc.pipeline.apply_transform"),
        patch("autoacpc.pipeline.nb.load") as mock_load,
    ):
        mock_load.return_value.shape = (256, 256, 256)
        mock_load.return_value.header.get_zooms.return_value = (1.0, 1.0, 1.0)

        acpc_align(
            input_image=str(fake_input),
            output_image=str(tmp_path / "out.nii.gz"),
            template_path=str(fake_template),
            template_mask=str(fake_mask),
            work_dir=str(tmp_path),
        )

        _, kwargs = mock_reg.call_args
        assert kwargs["reference_mask"] == str(fake_mask.resolve())


def test_local_template_not_found(tmp_path):
    """Raise FileNotFoundError when template_path doesn't exist."""
    fake_input = tmp_path / "input.nii.gz"
    fake_input.write_bytes(b"fake")

    with (
        patch("autoacpc.pipeline.nb.load") as mock_load,
    ):
        mock_load.return_value.shape = (256, 256, 256)
        mock_load.return_value.header.get_zooms.return_value = (1.0, 1.0, 1.0)

        with pytest.raises(FileNotFoundError, match="Template image not found"):
            acpc_align(
                input_image=str(fake_input),
                output_image=str(tmp_path / "out.nii.gz"),
                template_path="/nonexistent/template.nii.gz",
                work_dir=str(tmp_path),
            )
