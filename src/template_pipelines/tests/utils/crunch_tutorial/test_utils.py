"""Tests for utils module."""
import tempfile


def test_pretty_list_files():
    """Test pretty_list_files function."""
    from pathlib import Path

    from template_pipelines.utils.crunch_tutorial.utils import pretty_list_files

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        for i in range(2):
            (path / f"dir_{i}").mkdir()
            for j in range(2):
                (path / f"dir_{i}" / f"file_{j}").touch()
            (path / f"file_{i}").touch()
        pretty_list_files(path)

    assert True
