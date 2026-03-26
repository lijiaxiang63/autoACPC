"""ANTs command building & execution for AC-PC alignment."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ANTs registration settings for AC-PC alignment (from qsiprep intramodal_ACPC.json)
ACPC_REGISTRATION_SETTINGS = {
    "float": True,
    "winsorize_lower_quantile": 0.002,
    "winsorize_upper_quantile": 0.998,
    "collapse_output_transforms": True,
    "use_histogram_matching": [True, True],
    "transforms": ["Similarity", "Affine"],
    "number_of_iterations": [[100000, 100000], [10000, 10000]],
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


def _run_command(cmd: list[str], tool_name: str) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{tool_name} failed (exit code {result.returncode}):\n{result.stderr}")
    return result


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
        "--dimensionality",
        "3",
        "--float",
        "1" if settings["float"] else "0",
        "--output",
        f"[{output_prefix},{output_prefix}Warped.nii.gz]",
        "--interpolation",
        settings["interpolation"],
        "--winsorize-image-intensities",
        f"[{settings['winsorize_lower_quantile']},{settings['winsorize_upper_quantile']}]",
        "--write-composite-transform",
        "0",
        "--collapse-output-transforms",
        "1",
        "--initial-moving-transform",
        f"[{reference_image},{moving_image},1]",
    ]

    if reference_mask:
        cmd.extend(["--masks", f"[{reference_mask},{moving_mask or 'NULL'}]"])

    for i, transform_type in enumerate(settings["transforms"]):
        iters = "x".join(str(n) for n in settings["number_of_iterations"][i])
        shrinks = "x".join(str(n) for n in settings["shrink_factors"][i])
        sigmas = "x".join(str(n) for n in settings["smoothing_sigmas"][i])

        cmd.extend(
            [
                "--transform",
                f"{transform_type}[{settings['transform_parameters'][i][0]}]",
                "--metric",
                f"{settings['metric'][i]}[{reference_image},{moving_image},"
                f"{settings['metric_weight'][i]},{settings['radius_or_number_of_bins'][i]},"
                f"{settings['sampling_strategy'][i]},{settings['sampling_percentage'][i]}]",
                "--convergence",
                f"[{iters},{settings['convergence_threshold'][i]},"
                f"{settings['convergence_window_size'][i]}]",
                "--shrink-factors",
                shrinks,
                "--smoothing-sigmas",
                f"{sigmas}{settings['sigma_units'][i]}",
            ]
        )

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

    _run_command(cmd, "antsRegistration")

    # ANTs outputs transform files as <prefix>0GenericAffine.mat for linear transforms
    affine_path = f"{output_prefix}0GenericAffine.mat"
    if not Path(affine_path).exists():
        raise FileNotFoundError(
            f"Expected affine transform not found: {affine_path}\n"
            "Check that antsRegistration completed successfully."
        )

    logger.info("Registration complete. Affine transform: %s", affine_path)
    return affine_path


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
        "--dimensionality",
        "3",
        "--input",
        input_image,
        "--reference-image",
        reference_image,
        "--output",
        output_image,
        "--interpolation",
        interpolation,
        "--transform",
        transform,
        "--input-image-type",
        "0",
    ]

    logger.info("Applying transform...")
    logger.debug("Command: %s", " ".join(cmd))

    _run_command(cmd, "antsApplyTransforms")

    logger.info("Transform applied. Output: %s", output_image)
    return output_image
