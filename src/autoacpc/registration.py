"""ANTs-based registration to template for AC-PC alignment.

Implements the core algorithm from qsiprep:
1. Register input image to a standard template (Similarity + Affine)
2. Extract 6-DOF rigid component from the affine transform
3. Apply rigid transform to resample into AC-PC space
"""

import logging
import subprocess
from pathlib import Path

import nibabel as nb
import numpy as np
import SimpleITK as sitk
from dipy.core import geometry as geom

logger = logging.getLogger(__name__)

# ANTs registration settings for AC-PC alignment (from qsiprep intramodal_ACPC.json)
ACPC_REGISTRATION_SETTINGS = {
    "dimension": 3,
    "float": True,
    "winsorize_lower_quantile": 0.002,
    "winsorize_upper_quantile": 0.998,
    "collapse_output_transforms": True,
    "write_composite_transform": False,
    "use_histogram_matching": [True, True],
    "transforms": ["Similarity", "Affine"],
    "number_of_iterations": [[100000, 100000], [10000, 10000]],
    "output_warped_image": True,
    "transform_parameters": [[0.3], [0.1]],
    "convergence_threshold": [1e-6, 1e-6],
    "convergence_window_size": [10, 10],
    "metric": ["MI", "MI"],
    "sampling_percentage": [0.7, 0.2],
    "sampling_strategy": ["Regular", "Regular"],
    "shrink_factors": [[4, 2], [2, 1]],
    "sigma_units": ["vox", "vox"],
    "metric_weight": [1, 1],
    "smoothing_sigmas": [[0, 0], [2, 2]],
    "radius_or_number_of_bins": [32, 32],
    "interpolation": "Linear",
}

FAST_REGISTRATION_SETTINGS = {
    **ACPC_REGISTRATION_SETTINGS,
    "number_of_iterations": [[10, 10], [10, 10]],
    "transform_parameters": [[0.4], [0.2]],
    "convergence_window_size": [4, 4],
    "sampling_percentage": [0.2, 0.05],
    "shrink_factors": [[8, 4], [4, 2]],
}


def build_ants_command(
    moving_image: str,
    reference_image: str,
    output_prefix: str,
    moving_mask: str | None = None,
    reference_mask: str | None = None,
    fast: bool = False,
) -> list[str]:
    """Build the antsRegistration command for AC-PC alignment."""
    settings = FAST_REGISTRATION_SETTINGS if fast else ACPC_REGISTRATION_SETTINGS

    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1" if settings["float"] else "0",
        "--output", f"[{output_prefix},{output_prefix}Warped.nii.gz]",
        "--interpolation", settings["interpolation"],
        "--winsorize-image-intensities",
        f"[{settings['winsorize_lower_quantile']},{settings['winsorize_upper_quantile']}]",
        "--write-composite-transform", "0",
        "--collapse-output-transforms", "1",
        "--initial-moving-transform",
        f"[{reference_image},{moving_image},1]",
    ]

    if reference_mask:
        cmd.extend(["--masks", f"[{reference_mask},{moving_mask or 'NULL'}]"])

    for i, transform_type in enumerate(settings["transforms"]):
        iters = "x".join(str(n) for n in settings["number_of_iterations"][i])
        shrinks = "x".join(str(n) for n in settings["shrink_factors"][i])
        sigmas = "x".join(str(n) for n in settings["smoothing_sigmas"][i])

        cmd.extend([
            "--transform", f"{transform_type}[{settings['transform_parameters'][i][0]}]",
            "--metric",
            f"{settings['metric'][i]}[{reference_image},{moving_image},"
            f"{settings['metric_weight'][i]},{settings['radius_or_number_of_bins'][i]},"
            f"{settings['sampling_strategy'][i]},{settings['sampling_percentage'][i]}]",
            "--convergence",
            f"[{iters},{settings['convergence_threshold'][i]},"
            f"{settings['convergence_window_size'][i]}]",
            "--shrink-factors", shrinks,
            "--smoothing-sigmas", f"{sigmas}{settings['sigma_units'][i]}",
        ])

        if settings["use_histogram_matching"][i]:
            cmd.append("--use-histogram-matching")

    return cmd


def run_registration(
    moving_image: str,
    reference_image: str,
    output_dir: str,
    moving_mask: str | None = None,
    reference_mask: str | None = None,
    fast: bool = False,
) -> str:
    """Run ANTs registration and return the path to the affine transform.

    Returns:
        Path to the output affine transform file (.mat).
    """
    output_prefix = str(Path(output_dir) / "acpc_reg_")

    cmd = build_ants_command(
        moving_image=moving_image,
        reference_image=reference_image,
        output_prefix=output_prefix,
        moving_mask=moving_mask,
        reference_mask=reference_mask,
        fast=fast,
    )

    logger.info("Running ANTs registration...")
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsRegistration failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # ANTs outputs transform files as <prefix>0GenericAffine.mat for linear transforms
    affine_path = f"{output_prefix}0GenericAffine.mat"
    if not Path(affine_path).exists():
        raise FileNotFoundError(
            f"Expected affine transform not found: {affine_path}\n"
            "Check that antsRegistration completed successfully."
        )

    logger.info("Registration complete. Affine transform: %s", affine_path)
    return affine_path


def affine_to_rigid(transform_file: str, output_dir: str) -> tuple[str, str]:
    """Extract 6-DOF rigid component from an ITK affine transform.

    Based on qsiprep's itk_affine_to_rigid function. Decomposes the affine
    matrix to extract only rotation and translation, discarding scaling/shear.

    Returns:
        Tuple of (rigid_transform_path, rigid_inverse_transform_path).
    """
    output_dir = Path(output_dir)
    rigid_path = str(output_dir / "rigid_acpc.mat")
    inverse_path = str(output_dir / "rigid_acpc_inverse.mat")

    raw_transform = sitk.ReadTransform(transform_file)
    aff_transform = sitk.AffineTransform(3)
    aff_transform.SetFixedParameters(raw_transform.GetFixedParameters())
    aff_transform.SetParameters(raw_transform.GetParameters())

    # Decompose affine to extract rotation angles
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = np.array(aff_transform.GetMatrix()).reshape((3, 3), order="C")
    _, _, angles, _, _ = geom.decompose_matrix(full_matrix)
    rot_mat = geom.euler_matrix(angles[0], angles[1], angles[2])

    # Build rigid transform (rotation + translation only)
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(aff_transform.GetCenter())
    rigid.SetTranslation(aff_transform.GetTranslation())
    rigid.SetMatrix(tuple(rot_mat[:3, :3].flatten(order="C")))

    sitk.WriteTransform(rigid, rigid_path)
    sitk.WriteTransform(rigid.GetInverse(), inverse_path)

    logger.info("Rigid transform extracted: %s", rigid_path)
    return rigid_path, inverse_path


def apply_transform_to_header(
    input_image: str,
    inverse_transform: str,
    output_image: str,
) -> str:
    """Apply a rigid transform by modifying the NIfTI header affine.

    Instead of resampling, this composes the inverse rigid transform with
    the existing sform/qform so that downstream tools see AC-PC aligned
    coordinates while the voxel data stays untouched.

    Args:
        input_image: Path to input NIfTI image.
        inverse_transform: Path to the *inverse* rigid ITK transform (.mat).
        output_image: Path for the output image with modified header.

    Returns:
        Path to the output image.
    """
    # Load the inverse rigid transform (maps input space → template space)
    raw = sitk.ReadTransform(inverse_transform)
    rigid = sitk.Euler3DTransform()
    rigid.SetFixedParameters(raw.GetFixedParameters())
    rigid.SetParameters(raw.GetParameters())

    # Build 4x4 matrix in ITK (LPS) space
    rot_3x3 = np.array(rigid.GetMatrix()).reshape((3, 3), order="C")
    center = np.array(rigid.GetFixedParameters()[:3])
    translation = np.array(rigid.GetTranslation())
    # ITK applies: y = R*(x - center) + center + t  →  y = R*x + (center - R*center + t)
    offset = center - rot_3x3 @ center + translation

    lps_matrix = np.eye(4)
    lps_matrix[:3, :3] = rot_3x3
    lps_matrix[:3, 3] = offset

    # Convert LPS → RAS: flip x and y axes
    lps2ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_matrix = lps2ras @ lps_matrix @ lps2ras

    # Compose with existing NIfTI affine
    img = nb.load(str(input_image))
    new_affine = ras_matrix @ img.affine

    # Preserve on-disk voxel storage and NIfTI scaling exactly. Using
    # np.asarray(img.dataobj) would materialize scaled values and trigger
    # re-quantization on save for images with scl_slope/scl_inter.
    header = img.header.copy()
    dataobj = img.dataobj
    if hasattr(dataobj, "get_unscaled"):
        data = dataobj.get_unscaled()
        slope = dataobj.slope
        intercept = dataobj.inter
    else:
        data = np.asanyarray(dataobj)
        slope = None
        intercept = None

    out = nb.Nifti1Image(data, new_affine, header, extra=img.extra.copy())
    if slope is not None and intercept is not None:
        out.header.set_slope_inter(slope, intercept)
    out.set_sform(new_affine)
    out.set_qform(new_affine)
    nb.save(out, str(output_image))

    logger.info("Header-only transform applied. Output: %s", output_image)
    return output_image


def apply_transform(
    input_image: str,
    reference_image: str,
    transform: str,
    output_image: str,
    interpolation: str = "LanczosWindowedSinc",
) -> str:
    """Apply a transform to an image using antsApplyTransforms.

    Returns:
        Path to the output image.
    """
    cmd = [
        "antsApplyTransforms",
        "--dimensionality", "3",
        "--input", input_image,
        "--reference-image", reference_image,
        "--output", output_image,
        "--interpolation", interpolation,
        "--transform", transform,
        "--input-image-type", "0",
    ]

    logger.info("Applying transform...")
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed (exit code {result.returncode}):\n{result.stderr}"
        )

    logger.info("Transform applied. Output: %s", output_image)
    return output_image
