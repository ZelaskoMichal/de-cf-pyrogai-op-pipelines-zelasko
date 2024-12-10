"""Unit tests for OutputWithSaveToIoslotsDqStep."""
import re
from pathlib import Path

import pytest

from template_pipelines.steps.optimization.output_dq_with_save_to_ioslots import (
    OutputWithSaveToIoslotsDqStep,
)
from template_pipelines.tests.steps.optimization.constants import (
    CONFIG_FILE,
    RUNTIME_PARAMETERS,
    TEST_CONFIG_MODULE,
    TEST_DATA_DIR,
)
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet
from template_pipelines.utils.optimization.setup_tests_utils import setup_semi_integrated_test


@pytest.fixture(scope="function")
def step_semi_integrated():
    """Fixture returns OutputWithSaveToIoslotsDqStep step for semi-integrated tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    setup_semi_integrated_test(
        config_module=TEST_CONFIG_MODULE,
        config_file_name=CONFIG_FILE,
        pipeline_name="optimization_semi_integrated_test",
        runtime_parameters=RUNTIME_PARAMETERS,
        step_name="output_dq_with_save_to_ioslots",
    )

    step = OutputWithSaveToIoslotsDqStep()

    # prepare data for step in icotx
    copy_data_to_parquet(
        file_paths=[
            TEST_DATA_DIR / "output_data" / "output.parquet",
        ],
        dest_path=Path(step.ioctx.get_output_fn(step.config["output_tmp_dir"])),
    )

    yield step


def test_process(step_semi_integrated):
    """Test that no errors occur, check if csv file exists, and if ge report exists and contains specific phrase."""
    # prepare
    ge_doc_path = step_semi_integrated.ioctx.get_output_fn(
        f"ge_dq_datadocs/{step_semi_integrated.step_name}/output/output/warning.html"
    )
    # regex pattern is used because searched file is a html
    pattern = r"values must be greater than or equal to .*10.* and less than or equal to .*100.*"

    # execute
    step_semi_integrated.process()

    # assert
    assert "output.csv" in step_semi_integrated.outputs
    assert "output_ge_warning.html" in step_semi_integrated.outputs
    with open(ge_doc_path, "r") as f:
        assert re.search(pattern, f.read())
