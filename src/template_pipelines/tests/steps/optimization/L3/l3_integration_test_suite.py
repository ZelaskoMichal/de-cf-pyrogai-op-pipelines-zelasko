"""Base class for L3 integration test suites."""

import traceback
from abc import ABC, abstractmethod
from collections import namedtuple
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional

from aif.pyrogai.ioutils.iocontext import IoContext

logger = getLogger(__name__)


class L3IntegrationTestSuite(ABC):
    """Base class for L3 integration test suites."""

    @abstractmethod
    def run_tests(
        self,
        step_mlflow_run_id: str,
        pipeline_mlflow_run_id: str,
        config: dict,
        aux_data_path: Optional[Path],
        ioctx: IoContext,
    ) -> None:
        """Base class for L3 integration test suites.

        Should run all relevant tests encapsulated within the class, one per test suite.

        Args:
            step_mlflow_run_id (str): run id of step
            pipeline_mlflow_run_id (str): run id of pipeline
            config (dict): step config, the same as in regular step
            aux_data_path (Optional[Path]): path to auxiliary data folder (input ioslot)
            ioctx (IoContext): pyrogai iocontext reference
        """
        pass

    def _run_tests_with_late_failure(self, tests: list[Callable], *args, **kwargs):
        """Run all tests and report all results, raising exception if any test has failed.

        Args:
            tests (list[Callable]): list of test functions to call

        Raises:
            RuntimeError: raised if any test has failed
        """
        results = []
        test_log = namedtuple("test_log", ["name", "result", "exception"], defaults=[None])

        logger.info(
            f"\n\n====================== Running L3 test suite {self.__class__.__name__} \n"
        )

        for test in tests:
            logger.info(f"Running test: {test.__name__}")
            try:
                test(*args, **kwargs)
            except Exception as e:
                logger.error(f"Test {test.__name__} failed!")
                results.append(test_log(test.__name__, False, e))
                tb = traceback.format_exc()
                logger.error(tb)
            else:
                logger.info(f"Test {test.__name__} succeeded!")
                results.append(test_log(test.__name__, True))

        logger.info("------------------")
        logger.info("Test suite results: ")
        for result in results:
            if not result.result:
                # make it red
                logger.error(f"{result[:2]} ❌")
            else:
                logger.info(f"{result[:2]} ✅")
        logger.info("Test suite report finished")
        logger.info("------------------")

        logger.info(
            f"\n\n====================== Finished execution of L3 test suite {self.__class__.__name__} \n"
        )

        if any(test_log.result is False for test_log in results):
            raise RuntimeError(
                f"L3 Test suite {self.__class__.__name__} failed, see the full test log above for more details"
            )
