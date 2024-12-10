#!/usr/bin/env python

"""Run automated testing with code coverage.

Produces a report in HTML format, XML format, and text format. The produced report is ready to be
shipped to various tools.
"""

import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree
from subprocess import CalledProcessError, run


def _parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional optional arguments
    parser.add_argument("filter", help="Run tests matching filter", nargs="?", type=str, default="")

    # Optional arguments
    parser.add_argument("--include-slow", action="store_true", help="include long-running tests")

    # filter tests by marker
    parser.add_argument("--markers", help="filter tests by markers", type=str, default="")

    # Parse arguments
    args = parser.parse_args()

    return args


def _main():
    # Execute tests from the root directory
    root = Path(__file__).parent.parent.resolve()
    os.chdir(root)
    args = _parse_arguments()

    # Clean up old coverage reports
    condemned = [
        Path(root, "coverage.txt"),
        Path(root, "coverage.xml"),
        Path(root, "coverage.json"),
        Path(root, ".coverage"),
        Path(root, "htmlcov"),
    ] + list(Path(root).glob(".coverage.*"))
    for c in condemned:
        if c.is_file():
            c.unlink()
        elif c.is_dir():
            rmtree(c)
    run(["coverage", "erase"], check=True)

    filt = []
    xdist = ["-n4"]
    markers = []

    # Include long-running tests
    if args.include_slow:
        filt.append("--include-slow")

    # Run only tests that are decorated with proper markers
    if args.markers:
        markers.extend(["-m", args.markers])
    # Run only tests matching filter - no output capture (-s)
    if args.filter:
        filt.extend(["-k", args.filter, "-s"])
        xdist = ["-p", "no:xdist"]

    cmd = (
        ["pytest", "-s", "--log-cli-level=INFO", "--cov=src", "-vvv"]
        + filt
        + xdist
        + markers
        + [
            "--force-sugar",
            "-p",
            "no:cacheprovider",
            "--html=pytest_report/index.html",
            "--durations=10",
            "./src/template_pipelines",
            "--json-report",
            "--json-report-file=report.json",
        ]
    )

    try:
        print(f"Running: {' '.join(cmd)}")  # noqa
        run(cmd, check=True)
    except CalledProcessError:
        sys.exit(1)

    # Save coverage report as text
    with open("coverage.txt", "wb") as fp:
        run(["coverage", "report"], check=True, stdout=fp)

    # Produce HTML and XML reports too
    for mode in ("html", "xml", "json"):
        run(["coverage", mode], check=True)


if __name__ == "__main__":
    _main()
