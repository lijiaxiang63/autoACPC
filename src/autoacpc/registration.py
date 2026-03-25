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


# ---------------------------------------------------------------------------
# Pure-numpy ITK transform I/O
# ---------------------------------------------------------------------------


def _read_itk_text_transform(path: str) -> dict:
    """Read an ITK text transform file (.tfm / .mat with text header)."""
    transform_type = None
    parameters = None
    fixed_parameters = None

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if line.startswith("Transform:"):
                    transform_type = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Transform:"):
                transform_type = line.split(":", 1)[1].strip()
            elif line.startswith("Parameters:"):
                parameters = np.fromstring(line.split(":", 1)[1], sep=" ")
            elif line.startswith("FixedParameters:"):
                fixed_parameters = np.fromstring(line.split(":", 1)[1], sep=" ")

    if transform_type is None or parameters is None:
        raise ValueError(f"Invalid ITK transform file: {path}")

    if transform_type.startswith(("AffineTransform_", "MatrixOffsetTransformBase_")):
        A = parameters[:9].reshape((3, 3), order="C")
        t = parameters[9:12]
        c = fixed_parameters[:3] if fixed_parameters is not None else np.zeros(3)
        return {"matrix": A, "translation": t, "center": c}

    if transform_type.startswith("Euler3DTransform_"):
        angles = parameters[:3]
        t = parameters[3:6]
        c = fixed_parameters[:3] if fixed_parameters is not None else np.zeros(3)
        compute_zyx = (
            bool(round(fixed_parameters[3]))
            if fixed_parameters is not None and len(fixed_parameters) > 3
            else False
        )
        # Build rotation matrix from Euler angles
        ax, ay, az = angles
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        if compute_zyx:
            A = Rz @ Ry @ Rx
        else:
            A = Rz @ Rx @ Ry
        return {"matrix": A, "translation": t, "center": c}

    raise NotImplementedError(f"Unsupported transform type: {transform_type}")


def _read_ants_mat4(path: str) -> dict:
    """Read an ANTs MATLAB-v4 binary .mat transform file."""

    def _parse(endian: str) -> dict[str, np.ndarray]:
        variables: dict[str, np.ndarray] = {}
        with open(path, "rb") as f:
            while True:
                hdr = f.read(20)
                if not hdr:
                    return variables
                if len(hdr) != 20:
                    raise ValueError("truncated MAT v4 header")
                mopt, mrows, ncols, imagf, namelen = np.frombuffer(
                    hdr, dtype=endian + "i4", count=5
                )
                mopt, mrows, ncols, imagf, namelen = (
                    int(mopt),
                    int(mrows),
                    int(ncols),
                    int(imagf),
                    int(namelen),
                )
                if imagf != 0 or mrows <= 0 or ncols <= 0 or namelen <= 0 or namelen > 256:
                    raise ValueError("bad MAT v4 header")
                name = f.read(namelen)
                if len(name) != namelen:
                    raise ValueError("truncated MAT v4 name")
                name = name.rstrip(b"\x00").decode("ascii")
                p = (mopt % 100) // 10
                dtype_map = {
                    0: endian + "f8",
                    1: endian + "f4",
                    2: endian + "i4",
                    3: endian + "i2",
                    4: endian + "u2",
                    5: endian + "u1",
                }
                if p not in dtype_map:
                    raise ValueError(f"unsupported MAT v4 dtype code {p}")
                count = mrows * ncols
                data = np.fromfile(f, dtype=np.dtype(dtype_map[p]), count=count)
                if data.size != count:
                    raise ValueError("truncated MAT v4 data")
                variables[name] = data.reshape((mrows, ncols), order="F")
        return variables

    try:
        variables = _parse("<")
    except Exception:
        variables = _parse(">")

    affine_key = next(
        k
        for k in variables
        if k.startswith(("AffineTransform_", "MatrixOffsetTransformBase_"))
    )
    fixed_key = next(k for k in variables if k.lower().startswith("fixed"))
    params = np.asarray(variables[affine_key]).ravel(order="F").astype(float)
    center = np.asarray(variables[fixed_key]).ravel(order="F").astype(float)
    A = params[:9].reshape((3, 3), order="C")
    t = params[9:12]
    return {"matrix": A, "translation": t, "center": center[:3]}


def read_itk_transform(path: str) -> dict:
    """Read an ITK linear transform from either text or MATLAB-v4 binary."""
    with open(path, "rb") as f:
        head = f.read(32)
    if head.startswith(b"#Insight Transform File"):
        return _read_itk_text_transform(path)
    return _read_ants_mat4(path)


def write_itk_affine_tfm(path: str, matrix: np.ndarray, translation: np.ndarray,
                         center: np.ndarray) -> None:
    """Write an ITK AffineTransform text file (.tfm)."""
    params = np.concatenate([
        np.asarray(matrix, dtype=float).reshape(9, order="C"),
        np.asarray(translation, dtype=float).reshape(3),
    ])
    fixed = np.asarray(center, dtype=float).reshape(3)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write("Parameters: " + " ".join(f"{x:.17g}" for x in params) + "\n")
        f.write("FixedParameters: " + " ".join(f"{x:.17g}" for x in fixed) + "\n")


def _closest_rotation(A: np.ndarray) -> np.ndarray:
    """Extract the closest proper rotation matrix from A via SVD."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# ANTs command building & execution
# ---------------------------------------------------------------------------


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

    Decomposes the affine matrix via SVD to extract the closest proper
    rotation, discarding scaling and shear.

    Returns:
        Tuple of (rigid_transform_path, rigid_inverse_transform_path).
    """
    output_dir = Path(output_dir)
    rigid_path = str(output_dir / "rigid_acpc.tfm")
    inverse_path = str(output_dir / "rigid_acpc_inverse.tfm")

    tx = read_itk_transform(transform_file)
    A = tx["matrix"]
    t = tx["translation"]
    c = tx["center"]

    R = _closest_rotation(A)

    write_itk_affine_tfm(rigid_path, R, t, c)
    write_itk_affine_tfm(inverse_path, R.T, -R.T @ t, c)

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
        inverse_transform: Path to the *inverse* rigid ITK transform.
        output_image: Path for the output image with modified header.

    Returns:
        Path to the output image.
    """
    tx = read_itk_transform(inverse_transform)
    A = tx["matrix"]
    c = tx["center"]
    t = tx["translation"]

    # ITK applies: y = A*(x - c) + c + t  →  offset = c - A*c + t
    offset = c - A @ c + t

    lps_matrix = np.eye(4)
    lps_matrix[:3, :3] = A
    lps_matrix[:3, 3] = offset

    # Convert LPS → RAS: flip x and y axes
    lps2ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_matrix = lps2ras @ lps_matrix @ lps2ras

    # Compose with existing NIfTI affine
    img = nb.load(str(input_image))
    new_affine = ras_matrix @ img.affine

    # Preserve on-disk voxel storage and NIfTI scaling exactly.
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
