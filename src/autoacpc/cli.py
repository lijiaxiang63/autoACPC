"""Command-line interface for autoacpc."""

import logging
from pathlib import Path

import click

from .pipeline import acpc_align
from .template import DEFAULT_TEMPLATE


@click.command()
@click.argument("input_image", type=click.Path(exists=True, path_type=Path))
@click.argument("output_image", type=click.Path(path_type=Path))
@click.option(
    "--template",
    default=DEFAULT_TEMPLATE,
    show_default=True,
    help="TemplateFlow template name for AC-PC reference.",
)
@click.option(
    "--modality",
    type=click.Choice(["T1w", "T2w"]),
    default="T1w",
    show_default=True,
    help="Template modality to match input contrast.",
)
@click.option(
    "--interpolation",
    type=click.Choice(["LanczosWindowedSinc", "Linear", "NearestNeighbor", "BSpline"]),
    default="LanczosWindowedSinc",
    show_default=True,
    help="Interpolation method for resampling.",
)
@click.option("--fast", is_flag=True, help="Use fast (less accurate) registration.")
@click.option(
    "--work-dir", type=click.Path(path_type=Path), help="Working directory for intermediate files."
)
@click.option(
    "--save-transform",
    type=click.Path(path_type=Path),
    help="Save the rigid transform used for the output image to this path.",
)
@click.option(
    "--template-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a local template image. Bypasses TemplateFlow download.",
)
@click.option(
    "--template-mask",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a local brain mask. Only used with --template-path.",
)
@click.option(
    "--header-only",
    is_flag=True,
    help="Modify NIfTI header affine instead of resampling. Preserves original voxel data.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(
    input_image: Path,
    output_image: Path,
    template: str,
    modality: str,
    interpolation: str,
    fast: bool,
    work_dir: Path | None,
    save_transform: Path | None,
    template_path: Path | None,
    template_mask: Path | None,
    header_only: bool,
    verbose: bool,
) -> None:
    """Set brain image origin to AC-PC alignment.

    Takes an INPUT_IMAGE (NIfTI) and writes the AC-PC aligned result to OUTPUT_IMAGE.

    The algorithm registers the input to a standard template using ANTs,
    extracts the rigid (6-DOF) component, and writes AC-PC aligned output
    by resampling or, with --header-only, by updating the NIfTI affine.

    Requires ANTs (antsRegistration, antsApplyTransforms) to be installed
    and available on PATH.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        acpc_align(
            input_image=input_image,
            output_image=output_image,
            template=template,
            modality=modality,
            interpolation=interpolation,
            fast=fast,
            work_dir=work_dir,
            save_transform=save_transform,
            template_path=template_path,
            template_mask=template_mask,
            header_only=header_only,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
