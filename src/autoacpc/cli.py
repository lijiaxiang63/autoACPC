"""Command-line interface for autoacpc."""

import logging
import sys

import click

from .pipeline import acpc_align
from .template import DEFAULT_TEMPLATE


@click.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.argument("output_image", type=click.Path())
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
@click.option("--work-dir", type=click.Path(), help="Working directory for intermediate files.")
@click.option("--save-transform", type=click.Path(), help="Save the rigid transform to this path.")
@click.option(
    "--template-path",
    type=click.Path(exists=True),
    help="Path to a local template image. Bypasses TemplateFlow download.",
)
@click.option(
    "--template-mask",
    type=click.Path(exists=True),
    help="Path to a local brain mask. Only used with --template-path.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(
    input_image: str,
    output_image: str,
    template: str,
    modality: str,
    interpolation: str,
    fast: bool,
    work_dir: str | None,
    save_transform: str | None,
    template_path: str | None,
    template_mask: str | None,
    verbose: bool,
) -> None:
    """Set brain image origin to AC-PC alignment.

    Takes an INPUT_IMAGE (NIfTI) and writes the AC-PC aligned result to OUTPUT_IMAGE.

    The algorithm registers the input to a standard template using ANTs,
    extracts the rigid (6-DOF) component, and resamples the input into
    AC-PC aligned space.

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
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
