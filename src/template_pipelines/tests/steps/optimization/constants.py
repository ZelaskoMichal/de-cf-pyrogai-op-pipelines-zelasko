"""Constants for tests."""
from pathlib import Path
from types import MappingProxyType

# NOTE: Wrapping dictionary by MappingProxyType makes that dictionary behaves like a ordinary
# dictionary but in read-only mode. You can't add/delete items to it.
CONFIG = MappingProxyType(
    {
        "data_dir": "my_data_dir",
        "sdm_tmp_dir": "my_sdm_tmp_dir",
        "solution_tmp_dir": "my_solution_tmp_dir",
        "optimization": MappingProxyType({"optimization_name": "formulate_and_solve"}),
    }
)

RUNTIME_PARAMETERS = MappingProxyType(
    {
        "max_risky_stocks": "3",
        "max_risky_stocks_ratio": "0.25",
        "max_ratio_per_stock": "0.3",
        "min_ratio_per_stock": "0.01",
        "min_ratio_per_region": "0.2",
        "min_stocks_per_region": "2",
        "max_total_stocks": "8",
        "min_number_per_region_activation": "soft",
        "max_number_risky_sum_activation": "hard",
    }  # Runtime parameters are loaded as strings during config parsing by PyrogAI
)

DATA_PARAMETERS = MappingProxyType(
    {
        "penalty_max_number_risky": 0.1,
        "penalty_min_number_per_region": 0.1,
    }
)

CONFIG_FILE = "config.json"
TEST_CONFIG_MODULE = "template_pipelines.tests.steps.optimization.config"
TEST_DATA_DIR = Path("src/template_pipelines/tests/steps/optimization/test_data")
