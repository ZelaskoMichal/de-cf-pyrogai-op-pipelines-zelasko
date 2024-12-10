"""Module with utilities for i/o."""

import logging

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step  # noqa: E402

logger = logging.getLogger(name=__name__)


def get_ioslot_name(step: Step, base_ioslot_name: str) -> str:
    """Get an AML uri or cloudfile IO slot name.

    If on AML, get the aml_uri.
    Otherwise, get the cloudfile.
    """
    if step.platform == Platform.AML:
        io_slot = base_ioslot_name
    else:
        io_slot = f"cf_{base_ioslot_name}"

    logger.debug(f"{step.platform}'s {base_ioslot_name} io_slot is {io_slot}")
    return io_slot
