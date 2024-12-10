"""Data Ingestion step class."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PosixPath
from urllib.parse import urlparse

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step
from template_pipelines.utils.upload_data.toolkit import (  # noqa I202
    copy_and_filter_files_in_temp,
    print_tree,
)


# Define DataIngestion class and inherit properties from pyrogai Step class
class UploadData(Step):
    """Data Ingestion step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run data upload step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running upload data...")

        (
            self.dest_platform,
            self.remote_dest_dir,
            self.source_dir,
        ) = self.validate_and_get_runtime_params()

        # Check provider
        self.logger.info(f"The pipeline is running on {self.platform} platform.")

        # Download data to proper provider
        if self.platform in Platform.LOCAL:
            self.logger.info(f"Destination platform: {self.dest_platform}")
            self.upload()

        self.logger.info("Data upload is done.")

    def validate_and_get_runtime_params(self):
        """Validata input parameters content and types.

        Expects 2 parameters:
        - dest_platform (str): the platform to which data are uploaded
        - remote_dest_dir (str): the directory on the remote platform

        Raises:
            ValueError: If parameters are missing or invalid

        Returns:
            str: dest_platform
            str: remote_dest_dir
        """
        # Check and setup runtime parameters
        dest_platform = None
        remote_dest_dir = None
        source_dir = None

        # Get and parse run time params
        for param_name, param_value in self.runtime_parameters.items():
            self.logger.info(f"{param_name}, {param_value}")
            if param_name == "dest_platform":
                dest_platform = param_value
            elif param_name == "remote_dest_dir":
                remote_dest_dir = param_value
            elif param_name == "source_dir":
                source_dir = Path(param_value).as_posix()

        # Check if all required params are there
        runtime_params = [dest_platform, remote_dest_dir, source_dir]
        for var in runtime_params:
            if var is None:
                comment = f"The run time parameter '{var}' is required."
                self.logger.fatal(comment)
                raise ValueError(comment)

        valid_platforms = ["Local", "AML", "Vertex", "DBR"]
        if dest_platform not in valid_platforms:
            comment = f"dest platform must be one of {valid_platforms}"
            self.logger.fatal(comment)
            raise ValueError(comment)

        # Verify that the souce_dir is indeed a dir
        if not os.path.isdir(source_dir):
            self.logger.fatal(f"{source_dir} is not a folder")
            raise ValueError

        self.logger.info(
            f"Runtime parameters: dest_platform: '{dest_platform}', "
            f"remote_dest_dir: '{remote_dest_dir}',"
            f"source_dir: '{source_dir}'"
        )

        return dest_platform, remote_dest_dir, source_dir

    def upload(self):
        """Upload data from tmp_data to the dest_platform using multithreading."""
        self._filter_and_log_files()

        file_types = self._get_file_types()

        if self.dest_platform in ["AML", "DBR", "Vertex"]:
            provider = self._get_provider_for_platform()
            self._upload_files_to_cloud(provider, file_types)
        elif self.dest_platform == "Local":
            self._copy_files_to_local(file_types)
        else:
            raise ValueError(f"Invalid platform: {self.dest_platform}")

    def _filter_and_log_files(self):
        copy_and_filter_files_in_temp(
            self.logger, self.source_dir, self.config["upload_data"]["file_types"]
        )
        self.logger.info("------STRUCTURE AFTER FILTERING------")
        print_tree(self.logger, self.source_dir)
        self.logger.info("-------------------------------------")

    def _get_file_types(self):
        file_types = self.config["upload_data"]["file_types"]
        if not file_types:
            raise ValueError(
                "No file types have been provided in config_upload-data.json. "
                "Please define what types need to be uploaded (eg. ['csv', 'parquet'])"
            )
        return file_types

    def _get_provider_for_platform(self):
        provider_list = [p for p in self.all_providers.values() if p.platform == self.dest_platform]
        try:
            return provider_list.pop()
        except IndexError:
            raise ValueError(f"No matching provider for platform: {self.dest_platform}")

    def _upload_files_to_cloud(self, provider, file_types):
        self.logger.info(f"File types that will be uploaded: {file_types}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(self._upload_to_cloud, provider, path): path
                for file_type in file_types
                for path in PosixPath(self.source_dir).rglob(f"*.{file_type}")
            }

            for future in as_completed(future_to_file):
                _file = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f"{_file} generated an exception: {exc}")
                else:
                    self.logger.info(f"{_file} upload completed")

    def _copy_files_to_local(self, file_types):
        for file_type in file_types:
            self._copy_to_local(file_type)

    def get_cloud_file_object(self, provider, file):
        """Creates a "cloud file" object (either AzFile, GcpFile or DbrFile).

        The type of file object is based on the contents of provider and on dest_platform.

        Args:
            provider (AIF Provider): A provider object
            file (str): Filename to upload

        Returns:
            _type_: _description_
        """
        remote_file_path = Path(self.remote_dest_dir) / file.relative_to(self.source_dir)

        if self.dest_platform == "AML":
            from dnalib.azfile import AzFile

            ml_client = self.get_ml_client(provider=provider)
            url = self.get_azure_url_path(
                ml_client=ml_client,
                remote_file_path=remote_file_path,
                use_china_config=provider.use_china_config,
            )
            blob_file = AzFile.from_string(url)

        elif self.dest_platform == "Vertex":
            # import depends on pyrogai version (gcp 1.3.1 >)
            try:
                from aif.pyrogai.steps.kfp.gcpfile import GcpFile
            except ImportError:
                from aif.pyrogai.steps.gcp.gcpfile import GcpFile

            url = f"gs://{provider.details.gcp_bucket}/{PosixPath(remote_file_path)}"
            blob_file = GcpFile.from_string(url)

        elif self.dest_platform == "DBR":
            from aif.pyrogai.steps.dbr.dbrfile import DbrFile

            os.environ["DATABRICKS_TOKEN"] = self.secrets["dbr_token"]
            host = urlparse(provider.details.host).netloc
            url = f"dbfs://{host}/{PosixPath(remote_file_path)}"
            blob_file = DbrFile.from_string(full_dbfs_path=url)

        else:
            assert False, f"unsupported provider: {self.dest_platform}, we should not be here"

        self.logger.info(f"Uploading {file} -> {url}")
        return blob_file

    def azure_credential(self):
        """Return Azure credentials."""
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            EnvironmentCredential,
            InteractiveBrowserCredential,
            ManagedIdentityCredential,
        )

        return ChainedTokenCredential(
            ManagedIdentityCredential(),
            EnvironmentCredential(),
            AzureCliCredential(),
            InteractiveBrowserCredential(),
        )

    def get_ml_client(self, provider):
        """Gets the ML client object."""
        from azure.ai.ml import MLClient

        aml_client_info = provider.details.client_info
        sub_id = aml_client_info.subscription_id
        rg = aml_client_info.resource_group
        ws_name = aml_client_info.workspace_name
        cloud = "AzureChinaCloud" if provider.use_china_config else "AzureCloud"
        ml_client = MLClient(
            credential=self.azure_credential(),
            subscription_id=sub_id,
            resource_group_name=rg,
            workspace_name=ws_name,
            cloud=cloud,
        )
        return ml_client

    def get_azure_url_path(self, ml_client, remote_file_path, use_china_config):
        """Sets the azure init environment variables."""
        from dnalib.azfile import azure_init

        datastore = ml_client.datastores.get_default()
        credential = ml_client.workspaces.get_keys().user_storage_key
        end_point_suffix = "core.chinacloudapi.cn" if use_china_config else "core.windows.net"
        azure_init(
            azure_storage_secrets={
                datastore.account_name: {
                    "azure_storage_connection_string": f"DefaultEndpointsProtocol=https;"
                    f"AccountName={datastore.account_name};"
                    f"AccountKey={credential};"
                    f"EndpointSuffix={end_point_suffix}"
                }
            },
            update=True,
        )
        return (
            f"{datastore.account_name}:"
            f"{datastore.container_name}:"
            f"{PosixPath(remote_file_path)}"
        )

    def _upload_to_cloud(self, provider, path):
        """Helper method for uploading a single file to cloud."""
        cloud_file = self.get_cloud_file_object(provider, path)
        cloud_file.upload(str(path))
        self.logger.info(f"Uploaded {path} to {provider.platform}")

    def _copy_to_local(self, file_type):
        source_dir_local = PosixPath(self.source_dir)
        dest_dir_local = Path(self.remote_dest_dir)
        dest_dir_local.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Copying data {source_dir_local} -> {dest_dir_local}")
        pathlist = source_dir_local.rglob(f"*.{file_type}")
        for path in pathlist:
            if path.is_file():
                shutil.copy2(path, dest_dir_local)
                self.logger.info(f"File '{path}' copied to {dest_dir_local}")
