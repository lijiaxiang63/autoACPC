"""ITK transform I/O and rigid extraction for AC-PC alignment."""

import logging
from pathlib import Path

import nibabel as nb
import numpy as np

logger = logging.getLogger(__name__)


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
        k for k in variables if k.startswith(("AffineTransform_", "MatrixOffsetTransformBase_"))
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


def write_itk_affine_tfm(
    path: str, matrix: np.ndarray, translation: np.ndarray, center: np.ndarray
) -> None:
    """Write an ITK AffineTransform text file (.tfm)."""
    params = np.concatenate(
        [
            np.asarray(matrix, dtype=float).reshape(9, order="C"),
            np.asarray(translation, dtype=float).reshape(3),
        ]
    )
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
