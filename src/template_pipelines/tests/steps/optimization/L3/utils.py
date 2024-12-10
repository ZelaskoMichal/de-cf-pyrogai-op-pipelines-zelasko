"""L3 testing utilities, including workarounds for known cross platform inconsistencies."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def repair_dbr_ioslot_dir_path(ioslot_path: Path, ioslots_tempdir: Path) -> Path:
    """Repair local ioslot paths for ioslots of type directory and input.

    Args:
        ioslot_path (Path): path as returned by referring to ioslot, ie self.inputs["slot_name"]
        ioslots_tempdir (Path): path where ioslots should be downloaded

    Returns:
        Path: fixed path to where ioslot data is downloaded
    """
    # fix: dbr does not resolve path where ioslot folder is lazily copied to tempdir, manual prepend needed
    # issue: pathlib a / b will result in b if b is absolute, ie starts with /, thats why prefix is dropped
    # https://adb-6002052623675423.3.azuredatabricks.net/jobs/83214013260733/runs/1096547908951954?o=6002052623675423

    logger.info(f"repairing regular ioslot directory: {ioslot_path}")
    tempdir_path = ioslots_tempdir / str(ioslot_path).lstrip("/")
    logger.info(f"repaired ioslot directory path: {tempdir_path}")
    return tempdir_path


@contextmanager
def link_mlflow_server(mlflow_tracking_uri: str):
    """Temporarily set mlflow tracking uri.

    Cleans up env vars per pyrogai expectation of no mlflow env vars enabled after post_run.

    Args:
        mlflow_tracking_uri (str): tracking uri
    """
    # if tracking uri is already there (explicit set_tracking_uri or env var) - no need to set
    # it is not present on AML
    updated_tracking_uri = False
    if not mlflow.is_tracking_uri_set():
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        logger.debug(f"setting env var MLFlOW_TRACKING_URI = {mlflow_tracking_uri}")
        updated_tracking_uri = True
    try:
        yield
    finally:
        # always clean up for pyrogai step - assumes no mlflow env vars are set
        if updated_tracking_uri:
            try:
                logger.debug(
                    f'deleting env var MLFLOW_TRACKING_URI = {os.environ.get("MLFLOW_TRACKING_URI")}'
                )
                del os.environ["MLFLOW_TRACKING_URI"]
            except KeyError:
                pass
