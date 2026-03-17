"""Template fetching via TemplateFlow."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = "MNI152NLin2009cAsym"


def get_template(
    output_dir: str,
    template_name: str = DEFAULT_TEMPLATE,
    modality: str = "T1w",
) -> tuple[str, str]:
    """Fetch a template image and brain mask from TemplateFlow.

    Returns:
        Tuple of (template_path, mask_path).
    """
    from templateflow.api import get as get_tf

    cohort = None
    if "+" in template_name:
        template_name, cohort = template_name.split("+")

    template_path = get_tf(
        template_name,
        cohort=cohort,
        resolution="1",
        desc=None,
        suffix=modality,
        extension=".nii.gz",
    )
    mask_path = get_tf(
        template_name,
        cohort=cohort,
        resolution="1",
        desc="brain",
        suffix="mask",
        extension=".nii.gz",
    )

    output_dir = Path(output_dir)
    local_template = output_dir / Path(template_path).name
    local_mask = output_dir / Path(mask_path).name

    shutil.copy(template_path, local_template)
    shutil.copy(mask_path, local_mask)

    logger.info("Template: %s", local_template)
    logger.info("Mask: %s", local_mask)

    return str(local_template), str(local_mask)
