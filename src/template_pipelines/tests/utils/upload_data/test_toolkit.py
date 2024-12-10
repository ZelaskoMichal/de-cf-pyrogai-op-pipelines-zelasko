"""Unittests toolkit."""
from collections import Counter
from unittest.mock import MagicMock, call, patch

from template_pipelines.utils.upload_data.toolkit import (
    _remove_directory_if_empty,
    copy_and_filter_files_in_temp,
    copy_directory_tree_to_temp,
    count_files_and_dirs,
    filter_files_in_directory,
    print_summary,
    print_tree,
    print_tree_recursive,
)


def test_print_tree():
    """Test print tree."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.walk") as walk_mock,
    ):
        listdir_mock.side_effect = lambda x: ["folder1", "file1.txt"] if x == "root" else []
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: True if x == "root/folder1" else False
        walk_mock.side_effect = lambda x: [("root", ["folder1"], ["file1.txt"])]

        print_tree(logger_mock, "root")

        logger_mock.info.assert_any_call("folder1 [Folder]")
        logger_mock.info.assert_any_call("file1.txt [File]")


def test_print_tree_with_nested_directories():
    """Test print tree with nested directories."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.walk") as walk_mock,
    ):
        listdir_mock.side_effect = lambda x: (
            ["folder1", "file1.txt"]
            if x == "root"
            else ["folder2", "file2.txt"]
            if x == "root/folder1"
            else []
        )
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: x in ["root/folder1", "root/folder1/folder2"]
        walk_mock.side_effect = lambda x: [
            ("root", ["folder1"], ["file1.txt"]),
            ("root/folder1", ["folder2"], ["file2.txt"]),
            ("root/folder1/folder2", [], []),
        ]

        print_tree(logger_mock, "root")

        expected_calls = [
            call("folder1 [Folder]"),
            call("    folder2 [Folder]"),
            call("    file2.txt [File]"),
            call("file1.txt [File]"),
        ]
        logger_mock.info.assert_has_calls(expected_calls, any_order=False)
        assert logger_mock.info.call_count == 4


def test_print_tree_with_large_directory():
    """Test print tree with large directory structure."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.walk") as walk_mock,
    ):
        listdir_mock.side_effect = lambda x: ["folder1", "file1.txt"] if x == "root" else []
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: True if x == "root/folder1" else False
        walk_mock.side_effect = lambda x: [("root", ["folder1"], ["file1.txt"])] * 30

        print_tree(logger_mock, "root", max_files=10)

        logger_mock.info.assert_any_call(
            "[Directory structure contains more than 10 files. Summary:]"
        )
        logger_mock.info.assert_any_call("Total directories: 30")
        logger_mock.info.assert_any_call("Total files: 30")
        logger_mock.info.assert_any_call("File types:")
        logger_mock.info.assert_any_call("    30x .txt")


def test_filter_files_in_directory():
    """Test filter_files_in_directory."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.path.splitext") as splitext_mock,
        patch("os.remove") as remove_mock,
        patch("os.rmdir") as rmdir_mock,
    ):
        listdir_mock.side_effect = (
            lambda x: ["folder1", "file1.txt", "file2.jpg"] if x == "root" else []
        )
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: x == "root/folder1"
        splitext_mock.side_effect = lambda x: ("file1", ".txt") if "txt" in x else ("file2", ".jpg")

        filter_files_in_directory(logger_mock, "root", ["txt"])

        remove_mock.assert_called_once_with("root/file2.jpg")
        rmdir_mock.assert_called_once_with("root/folder1")


def test_filter_files_with_mixed_extensions_and_empty_directories():
    """Test filter_files with mixed extensions and empty directories."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.path.splitext") as splitext_mock,
        patch("os.remove") as remove_mock,
        patch("os.rmdir") as rmdir_mock,
    ):
        listdir_mock.side_effect = lambda x: (
            ["folder1", "file1.txt", "file2.jpg", "file3.txt"] if x == "root" else []
        )
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: x == "root/folder1"
        splitext_mock.side_effect = lambda x: ("file1", ".txt") if "txt" in x else ("file2", ".jpg")

        filter_files_in_directory(logger_mock, "root", ["txt"])

        remove_mock.assert_called_once_with("root/file2.jpg")
        rmdir_mock.assert_called_once_with("root/folder1")


def test_filter_files_with_no_files_left():
    """Test filter_files with no files left after filtering."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
        patch("os.path.splitext") as splitext_mock,
        patch("os.remove") as remove_mock,
        patch("os.rmdir"),
    ):
        listdir_mock.side_effect = lambda x: ["file1.jpg"] if x == "root" else []
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: False
        splitext_mock.side_effect = lambda x: ("file1", ".jpg")

        filter_files_in_directory(logger_mock, "root", ["txt"])

        remove_mock.assert_called_once_with("root/file1.jpg")


def test_count_files_and_dirs():
    """Test count_files_and_dirs function."""
    with patch("os.walk") as walk_mock:
        walk_mock.side_effect = lambda x: [
            ("root", ["folder1"], ["file1.txt", "file2.jpg"]),
            ("root/folder1", [], ["file3.txt"]),
        ]
        total_files, total_dirs, file_extensions = count_files_and_dirs("root")
        assert total_files == 3
        assert total_dirs == 1
        assert file_extensions == Counter({".txt": 2, ".jpg": 1})


def test_print_summary():
    """Test print_summary function."""
    logger_mock = MagicMock()
    file_extensions = Counter({".txt": 30})

    print_summary(logger_mock, 30, 30, file_extensions, 10)

    logger_mock.info.assert_any_call("[Directory structure contains more than 10 files. Summary:]")
    logger_mock.info.assert_any_call("Total directories: 30")
    logger_mock.info.assert_any_call("Total files: 30")
    logger_mock.info.assert_any_call("File types:")
    logger_mock.info.assert_any_call("    30x .txt")


def test_print_tree_recursive():
    """Test print_tree_recursive function."""
    logger_mock = MagicMock()
    with (
        patch("os.listdir") as listdir_mock,
        patch("os.path.join") as join_mock,
        patch("os.path.isdir") as isdir_mock,
    ):
        listdir_mock.side_effect = lambda x: ["folder1", "file1.txt"] if x == "root" else []
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: True if x == "root/folder1" else False

        print_tree_recursive("root", 0, logger_mock)

        logger_mock.info.assert_any_call("folder1 [Folder]")
        logger_mock.info.assert_any_call("file1.txt [File]")


def test_copy_directory_tree_to_temp():
    """Test copying directory tree to a temporary location."""
    logger_mock = MagicMock()
    with patch("shutil.copytree") as copytree_mock, patch("tempfile.mkdtemp") as mkdtemp_mock:
        # Simulate the temp directory creation
        mkdtemp_mock.return_value = "/tmp/some_temp_dir"

        # Test successful copy
        temp_dir = copy_directory_tree_to_temp(logger_mock, "root")
        assert temp_dir == "/tmp/some_temp_dir"
        copytree_mock.assert_called_once_with("root", "/tmp/some_temp_dir", dirs_exist_ok=True)

        # Test failure in copying
        copytree_mock.side_effect = OSError("Copy failed")
        temp_dir = copy_directory_tree_to_temp(logger_mock, "root")
        assert temp_dir is None
        logger_mock.error.assert_called_once_with(
            "Error copying directory root to a temporary location: Copy failed"
        )


def test_remove_directory_if_empty():
    """Test removing directory if it is empty."""
    logger_mock = MagicMock()
    with patch("os.listdir") as listdir_mock, patch("os.rmdir") as rmdir_mock:
        # Directory is empty
        listdir_mock.return_value = []
        result = _remove_directory_if_empty(logger_mock, "root")
        assert result is True
        rmdir_mock.assert_called_once_with("root")

        # Directory is not empty
        listdir_mock.return_value = ["file1.txt"]
        result = _remove_directory_if_empty(logger_mock, "root")
        assert result is False

        # Directory removal fails
        rmdir_mock.side_effect = OSError("Remove failed")
        listdir_mock.return_value = []
        result = _remove_directory_if_empty(logger_mock, "root")
        assert result is False
        logger_mock.warning.assert_called_once_with(
            "Cannot remove directory root. Directory not empty."
        )


def test_copy_and_filter_files_in_temp():
    """Test copying and filtering files in a temporary directory."""
    logger_mock = MagicMock()
    with patch("shutil.copytree") as copytree_mock, patch(
        "tempfile.mkdtemp"
    ) as mkdtemp_mock, patch("os.listdir") as listdir_mock, patch(
        "os.path.join"
    ) as join_mock, patch(
        "os.path.isdir"
    ) as isdir_mock, patch(
        "os.path.splitext"
    ) as splitext_mock, patch(
        "os.remove"
    ) as remove_mock, patch(
        "os.rmdir"
    ) as rmdir_mock:
        mkdtemp_mock.return_value = "/tmp/some_temp_dir"
        copytree_mock.return_value = "/tmp/some_temp_dir"

        listdir_mock.side_effect = (
            lambda x: ["folder1", "file1.txt", "file2.jpg"] if x == "/tmp/some_temp_dir" else []
        )
        join_mock.side_effect = lambda x, y: f"{x}/{y}"
        isdir_mock.side_effect = lambda x: x == "/tmp/some_temp_dir/folder1"
        splitext_mock.side_effect = lambda x: ("file1", ".txt") if "txt" in x else ("file2", ".jpg")

        # Test successful copy and filtering
        result = copy_and_filter_files_in_temp(logger_mock, "root", ["txt"])
        assert result == "/tmp/some_temp_dir"
        remove_mock.assert_called_once_with("/tmp/some_temp_dir/file2.jpg")
        rmdir_mock.assert_called_once_with("/tmp/some_temp_dir/folder1")

        # Test failure in copying
        copytree_mock.side_effect = OSError("Copy failed")
        result = copy_and_filter_files_in_temp(logger_mock, "root", ["txt"])
        assert result is None
        assert logger_mock.error.mock_calls == [
            call("Error copying directory root to a temporary location: Copy failed"),
            call("Failed to create a temporary directory for copying."),
        ]
