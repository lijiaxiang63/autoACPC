"""Main AC-PC alignment pipeline."""

import logging
import tempfile
from pathlib import Path

import nibabel as nb

from .registration import (
    affine_to_rigid,
    apply_transform,
    apply_transform_to_header,
    run_registration,
)
from .template import DEFAULT_TEMPLATE, get_template

logger = logging.getLogger(__name__)


def acpc_align(
    input_image: str,
    output_image: str,
    template: str = DEFAULT_TEMPLATE,
    modality: str = "T1w",
    interpolation: str = "LanczosWindowedSinc",
    fast: bool = False,
    work_dir: str | None = None,
    save_transform: str | None = None,
    template_path: str | None = None,
    template_mask: str | None = None,
    header_only: bool = False,
) -> str:
    """Run the full AC-PC alignment pipeline.

    Steps:
        1. Fetch standard template from TemplateFlow
        2. Register input to template (Similarity + Affine via ANTs)
        3. Extract rigid (6-DOF) transform from the affine
        4. Apply the rigid transform to the output grid or NIfTI header

    Args:
        input_image: Path to input NIfTI image.
        output_image: Path for AC-PC aligned output image.
        template: TemplateFlow template name (default: MNI152NLin2009cAsym).
        modality: Template modality to match against (T1w, T2w).
        interpolation: Interpolation method for resampling.
        fast: Use fast (less accurate) registration settings.
        work_dir: Working directory for intermediate files. Uses a temp dir if None.
        save_transform: If set, save the rigid transform used for the output
            image to this path.
        template_path: Path to a local template image. Skips TemplateFlow when set.
        template_mask: Path to a local brain mask. Only used with template_path.
        header_only: If True, modify the NIfTI header affine instead of resampling.

    Returns:
        Path to the output AC-PC aligned image.
    """
    input_path = Path(input_image).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_path = Path(output_image).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate input is loadable
    img = nb.load(str(input_path))
    logger.info(
        "Input image: %s (shape=%s, voxel_size=%s)",
        input_path.name,
        img.shape,
        img.header.get_zooms()[:3],
    )

    cleanup = work_dir is None
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix="autoacpc_")
        work_dir = tmp
    else:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Get template (local path or TemplateFlow)
        if template_path is not None:
            tp = Path(template_path).resolve()
            if not tp.exists():
                raise FileNotFoundError(f"Template image not found: {tp}")
            template_path = str(tp)
            mask_path = None
            if template_mask is not None:
                mp = Path(template_mask).resolve()
                if not mp.exists():
                    raise FileNotFoundError(f"Template mask not found: {mp}")
                mask_path = str(mp)
            logger.info("Using local template: %s", template_path)
        else:
            logger.info("Fetching template: %s (%s)", template, modality)
            template_path, mask_path = get_template(
                output_dir=work_dir,
                template_name=template,
                modality=modality,
            )

        # Step 2: Register to template
        affine_path = run_registration(
            moving_image=str(input_path),
            reference_image=template_path,
            output_dir=work_dir,
            reference_mask=mask_path,
            fast=fast,
        )

        # Step 3: Extract rigid transform
        rigid_path, inverse_path = affine_to_rigid(affine_path, work_dir)

        if save_transform:
            import shutil

            transform_to_save = inverse_path if header_only else rigid_path
            shutil.copy(transform_to_save, save_transform)
            logger.info("Transform saved to: %s", save_transform)

        # Step 4: Apply rigid transform
        if header_only:
            apply_transform_to_header(
                input_image=str(input_path),
                inverse_transform=inverse_path,
                output_image=str(output_path),
            )
        else:
            apply_transform(
                input_image=str(input_path),
                reference_image=template_path,
                transform=rigid_path,
                output_image=str(output_path),
                interpolation=interpolation,
            )

        logger.info("AC-PC aligned image written to: %s", output_path)
        return str(output_path)

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
