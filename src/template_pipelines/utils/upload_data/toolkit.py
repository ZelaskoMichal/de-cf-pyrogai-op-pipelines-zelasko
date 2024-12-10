"""Resuable methods and classes."""

import os
import shutil
import tempfile
from collections import Counter
from typing import Tuple


def count_files_and_dirs(directory: str) -> Tuple[int, int, Counter]:
    """Count files and dirs."""
    total_files = 0
    total_dirs = 0
    file_extensions: Counter = Counter()

    for root, dirs, files in os.walk(directory):
        total_dirs += len(dirs)
        total_files += len(files)
        file_extensions.update(os.path.splitext(f)[1].lower() for f in files)

    return total_files, total_dirs, file_extensions


def print_tree_recursive(directory: str, level: int, logger) -> None:
    """Print tree."""
    indent = "    " * level
    files_and_dirs = os.listdir(directory)

    for item in files_and_dirs:
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            logger.info(f"{indent}{item} [Folder]")
            print_tree_recursive(path, level + 1, logger)
        else:
            logger.info(f"{indent}{item} [File]")


def print_summary(
    logger, total_dirs: int, total_files: int, file_extensions: Counter, max_files: int
) -> None:
    """Print summary."""
    logger.info(f"[Directory structure contains more than {max_files} files. Summary:]")
    logger.info(f"Total directories: {total_dirs}")
    logger.info(f"Total files: {total_files}")
    logger.info("File types:")
    for ext, count in file_extensions.most_common():
        extension = ext if ext else "[no extension]"
        logger.info(f"    {count}x {extension}")


def print_tree(logger, root_dir: str, max_files: int = 25) -> None:
    """Print the structure of a folder with a summary for large directory structures."""
    total_files, total_dirs, file_extensions = count_files_and_dirs(root_dir)

    if total_files <= max_files:
        print_tree_recursive(root_dir, 0, logger)
    else:
        print_summary(logger, total_dirs, total_files, file_extensions, max_files)


def copy_directory_tree_to_temp(logger, source_dir):
    """Copies the entire directory tree from source_dir to a temporary directory."""
    try:
        temp_dir = tempfile.mkdtemp()
        shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
        return temp_dir
    except OSError as e:
        logger.error(f"Error copying directory {source_dir} to a temporary location: {e}")
        return None


def filter_files_in_directory(logger, root_dir, allowed_extensions):
    """Removes files with extensions not in the allowed list and deletes empty directories."""
    is_dir_empty = True
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            if not filter_files_in_directory(logger, path, allowed_extensions):
                _remove_directory_if_empty(logger, path)
                is_dir_empty = False
            else:
                is_dir_empty = False
        else:
            file_extension = os.path.splitext(item)[1].replace(".", "")
            if file_extension not in allowed_extensions:
                os.remove(path)
            else:
                is_dir_empty = False

    _remove_directory_if_empty(logger, root_dir)
    return is_dir_empty


def _remove_directory_if_empty(logger, dir_path):
    """Removes the directory if it is empty."""
    if not os.listdir(dir_path):
        try:
            os.rmdir(dir_path)
            return True
        except OSError:
            logger.warning(f"Cannot remove directory {dir_path}. Directory not empty.")
            return False
    return False


def copy_and_filter_files_in_temp(logger, source_dir, allowed_extensions):
    """Copies the directory tree to a temporary location and filters files in that temporary directory."""
    # Step 1: Copy the directory tree to a temporary location
    temp_dir = copy_directory_tree_to_temp(logger, source_dir)
    if temp_dir is None:
        logger.error("Failed to create a temporary directory for copying.")
        return None

    # Step 2: Filter files in the temporary directory
    filter_files_in_directory(logger, temp_dir, allowed_extensions)

    # Return the path to the temporary directory with the filtered files
    return temp_dir
