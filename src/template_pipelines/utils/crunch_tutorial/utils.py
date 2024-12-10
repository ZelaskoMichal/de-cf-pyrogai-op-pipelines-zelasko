"""Utility functions for the CRUNCH Tutorial template pipeline."""
from pathlib import Path


def pretty_list_files(path):
    """Pretty prints directory tree."""
    path = Path(path)
    for item in path.iterdir():
        if item.is_dir():
            print(f"{item}/")  # noqa
            subindent = " " * 4
            for file in item.iterdir():
                print(f"{subindent}{file.name}")  # noqa
        elif item.is_file():
            print(item.name)  # noqa
