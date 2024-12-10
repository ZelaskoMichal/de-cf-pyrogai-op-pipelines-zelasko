"""Unit tests for template_pipelines/steps/upload_data/upload_data.py."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    EnvironmentCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
)

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.dbr.dbrfile import DbrFile
from dnalib.azfile import AzFile
from template_pipelines.steps.upload_data.upload_data import UploadData

try:
    from aif.pyrogai.steps.kfp.gcpfile import GcpFile
except ImportError:
    from aif.pyrogai.steps.gcp.gcpfile import GcpFile


@pytest.fixture(scope="function")
def fixture_upload_data():
    """Fixture for upload_data step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        ud = UploadData()
        ud.source_dir = "source_dir_mock"
        ud.remote_dest_dir = "remote_dest_dir"
        ud.config = {
            "template": {"data_dir": "data_dir"},
            "upload_data": {"file_types": ["parquet"]},
        }
        ud.logger = MagicMock()
        ud.runtime_parameters = {}
        yield ud


def test_data_upload_run(fixture_upload_data):
    """Test run()."""
    fixture_upload_data.validate_and_get_runtime_params = Mock(
        return_value=["dest_platform", "remote_dest_dir", "source_dir"]
    )
    fixture_upload_data.upload = Mock()

    fixture_upload_data.platform = Platform.LOCAL

    fixture_upload_data.run()

    fixture_upload_data.logger.info.assert_called()
    fixture_upload_data.validate_and_get_runtime_params.assert_called_once()
    fixture_upload_data.upload.assert_called_once()
    assert fixture_upload_data.dest_platform == "dest_platform"


def test_data_upload_validate_and_get_runtime_params(fixture_upload_data):
    """Test validate_and_get_runtime_params()."""
    fixture_upload_data.runtime_parameters = {
        "dest_platform": "AML",
        "remote_dest_dir": "/some/directory/path",
        "source_dir": "./CouponData",
    }

    with patch("os.path.isdir") as isdir_mock:
        isdir_mock.side_effect = (
            lambda x: True if x == "CouponData" or x == "./CouponData" else False
        )
        (
            dest_platform,
            remote_dest_dir,
            source_dir,
        ) = fixture_upload_data.validate_and_get_runtime_params()

    assert dest_platform == "AML"
    assert remote_dest_dir == "/some/directory/path"
    assert source_dir == "CouponData"

    fixture_upload_data.logger.info.assert_called()


@pytest.mark.parametrize("params", [({"dest_platform": "NONEXIST"}), ({})])
def test_data_upload_validate_and_get_runtime_params_error(params, fixture_upload_data):
    """Test validate_and_get_runtime_params() for value error."""
    with pytest.raises(ValueError):
        fixture_upload_data.runtime_parameters = params
        fixture_upload_data.validate_and_get_runtime_params()
        fixture_upload_data.logger.fatal.assert_called()


@patch("template_pipelines.steps.upload_data.upload_data.copy_and_filter_files_in_temp")
@patch("template_pipelines.steps.upload_data.upload_data.print_tree")
def test_data_upload_cloud_platform(
    mock_print_tree, mock_filter_files_by_extension, fixture_upload_data
):
    """Test cloud platforms."""
    fixture_upload_data.all_providers = {"mock_provider": Mock(platform="AML")}
    fixture_upload_data.get_cloud_file_object = Mock()
    cloud_file_mock = Mock()
    fixture_upload_data.get_cloud_file_object.return_value = cloud_file_mock

    fixture_upload_data.dest_platform = "AML"
    fixture_upload_data.upload()

    fixture_upload_data.logger.info.assert_called()


@patch("template_pipelines.steps.upload_data.upload_data.copy_and_filter_files_in_temp")
@patch("template_pipelines.steps.upload_data.upload_data.print_tree")
def test_data_upload_local_platform(
    mock_print_tree, mock_filter_files_by_extension, fixture_upload_data
):
    """Test local platform."""
    mock_source_dir = "path/to/there"
    mock_remote_dest_dir = "mock_dir"

    fixture_upload_data.runtime_parameters = {"source_dir": mock_source_dir}
    fixture_upload_data.remote_dest_dir = mock_remote_dest_dir
    fixture_upload_data.config = {"upload_data": {"file_types": ["parquet"]}}

    mock_file = mock_source_dir + "/mock_file.parquet"

    with patch("shutil.copy2"), patch.object(Path, "rglob", return_value=[Path(mock_file)]):
        fixture_upload_data.dest_platform = "Local"
        fixture_upload_data.upload()

        fixture_upload_data.logger.info.assert_called()


@patch("template_pipelines.steps.upload_data.upload_data.copy_and_filter_files_in_temp")
@patch("template_pipelines.steps.upload_data.upload_data.print_tree")
def test_data_upload_upload_invalid_platform(
    mock_print_tree, mock_filter_files_by_extension, fixture_upload_data
):
    """Test invalid platform."""
    fixture_upload_data.dest_platform = "Invalid"

    with pytest.raises(ValueError):
        fixture_upload_data.upload()


def test_data_upload_get_cloud_file_object_aml(fixture_upload_data):
    """Test get_cloud_file_object_aml."""
    mock_provider = Mock(details=Mock(client_info=Mock(proxy="proxy")))
    fixture_upload_data.unset_proxy = Mock()
    fixture_upload_data.get_ml_client = Mock()
    fixture_upload_data.get_azure_url_path = Mock(return_value="url")
    mock_file = Mock(relative_to=Mock(return_value="relative_file"))

    with patch.object(AzFile, "from_string") as mock_az_file:
        fixture_upload_data.dest_platform = "AML"

        blob_file = fixture_upload_data.get_cloud_file_object(mock_provider, mock_file)

        mock_az_file.assert_called_once_with("url")
        fixture_upload_data.logger.info.assert_called_once()
        assert blob_file == mock_az_file.return_value


def test_data_upload_get_cloud_file_object_dbr(fixture_upload_data):
    """Test get_cloud_file_object_dbr."""
    mock_provider = Mock(details=Mock(host="http://host"))
    mock_file = Mock(relative_to=Mock(return_value="relative_file"))

    with patch.object(DbrFile, "from_string") as mock_dbr_file:
        fixture_upload_data.dest_platform = "DBR"
        fixture_upload_data.secrets = {"dbr_token": "dbr_token"}

        blob_file = fixture_upload_data.get_cloud_file_object(mock_provider, mock_file)

        mock_dbr_file.assert_called_once_with(
            full_dbfs_path="dbfs://host/remote_dest_dir/relative_file"
        )
        fixture_upload_data.logger.info.assert_called_once()
        assert blob_file == mock_dbr_file.return_value


def test_data_upload_get_cloud_file_object_gcp(fixture_upload_data):
    """Test get_cloud_file_object_dbr."""
    mock_provider = Mock(details=Mock(host="http://host", gcp_bucket="gcp_bucket"))
    mock_file = Mock(relative_to=Mock(return_value="relative_file"))

    with patch.object(GcpFile, "from_string") as mock_gcp_file:
        fixture_upload_data.dest_platform = "Vertex"

        blob_file = fixture_upload_data.get_cloud_file_object(mock_provider, mock_file)

        mock_gcp_file.assert_called_once_with("gs://gcp_bucket/remote_dest_dir/relative_file")
        fixture_upload_data.logger.info.assert_called_once()
        assert blob_file == mock_gcp_file.return_value


@patch.object(InteractiveBrowserCredential, "__init__", return_value=None)
@patch.object(AzureCliCredential, "__init__", return_value=None)
@patch.object(EnvironmentCredential, "__init__", return_value=None)
@patch.object(ManagedIdentityCredential, "__init__", return_value=None)
@patch.object(ChainedTokenCredential, "__init__", return_value=None)
def test_azure_credential(
    mock_chained_token_credential,
    mock_managed_identity_credential,
    mock_environment_credential,
    mock_azure_cli_credential,
    mock_interactive_browser_credential,
    fixture_upload_data,
):
    """Test azure_credential()."""
    fixture_upload_data.azure_credential()

    mocks = [a for a in locals().values() if type(a) is MagicMock]
    for mock in mocks:
        mock.assert_called_once()


@patch.object(MLClient, "__init__", return_value=None)
def test_get_ml_client(mock_mlclient, fixture_upload_data):
    """Test get_ml_client()."""
    mock_provider = MagicMock()
    fixture_upload_data.azure_credential = Mock(return_value="creds")

    fixture_upload_data.get_ml_client(mock_provider)

    mock_mlclient.assert_called_once()


@patch("dnalib.azfile.azure_init", return_value=None)
def test_data_upload_get_azure_url_path(mock_azure_init, fixture_upload_data):
    """Test get_azure_url_path()."""
    mock_ml_client = MagicMock()
    mock_ml_client.datastores.get_default.return_value = MagicMock(
        account_name="account_name", container_name="container_name"
    )

    res = fixture_upload_data.get_azure_url_path(
        mock_ml_client, Path("remote/file/path"), use_china_config=False
    )

    mock_azure_init.assert_called_once()
    assert res == "account_name:container_name:remote/file/path"


def test_data_upload_to_cloud(fixture_upload_data):
    """Test upload_to_cloud."""
    fixture_upload_data.get_cloud_file_object = Mock()
    cloud_file_mock = Mock()
    fixture_upload_data.get_cloud_file_object.return_value = cloud_file_mock

    fixture_upload_data.dest_platform = "AML"
    provider = Mock()
    path = Mock()
    fixture_upload_data._upload_to_cloud(provider, path)


def test_data_upload_copy_to_local(fixture_upload_data):
    """Test copy to local."""
    file_type = "parquet"
    fixture_upload_data.source_dir = "source_dir_mock"
    fixture_upload_data.remote_dest_dir = "remote_dest_dir"
    mock_file = Path("source_dir_mock/mock_file.parquet")

    with patch("shutil.copy2"), patch.object(Path, "rglob", return_value=[mock_file]):
        fixture_upload_data._copy_to_local(file_type)

        fixture_upload_data.logger.info.assert_called()


def test_data_upload_filter_and_log_files(fixture_upload_data):
    """Test filter and log files."""
    with patch(
        "template_pipelines.steps.upload_data.upload_data.copy_and_filter_files_in_temp"
    ) as mock_filter, patch(
        "template_pipelines.steps.upload_data.upload_data.print_tree"
    ) as mock_print_tree:
        fixture_upload_data._filter_and_log_files()

        mock_filter.assert_called_once_with(
            fixture_upload_data.logger,
            fixture_upload_data.source_dir,
            fixture_upload_data.config["upload_data"]["file_types"],
        )
        mock_print_tree.assert_called_once_with(
            fixture_upload_data.logger, fixture_upload_data.source_dir
        )
        fixture_upload_data.logger.info.assert_called()


def test_data_upload_get_file_types(fixture_upload_data):
    """Test get file types."""
    file_types = fixture_upload_data._get_file_types()
    assert file_types == ["parquet"]

    fixture_upload_data.config["upload_data"]["file_types"] = []
    with pytest.raises(ValueError):
        fixture_upload_data._get_file_types()


def test_data_upload_get_provider_for_platform(fixture_upload_data):
    """Test get provider for platform."""
    mock_provider = Mock(platform="AML")
    fixture_upload_data.all_providers = {"mock_provider": mock_provider}
    fixture_upload_data.dest_platform = "AML"
    provider = fixture_upload_data._get_provider_for_platform()
    assert provider == mock_provider

    fixture_upload_data.all_providers = {}
    with pytest.raises(ValueError):
        fixture_upload_data._get_provider_for_platform()
