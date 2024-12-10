# Blob Data – Template Pipeline

The `blob_data` template pipeline serves as the starting point of your pipeline, providing a quick way to retrieve data from CDL. Please note that the data access is not managed by us. We assume that the data is stored on blob storage. If the data is too large to be processed using Parquet files, alternative data storage and processing methods should be considered.

Please also remember that data on blob storage data is kept on storage with `_ex` suffix. More info about how to use data from `_ex` you can find here [How to use data from _ex Azure Blob Storage](https://developerportal.pg.com/docs/default/component/ai_factory_general/pipelines/howtos/how-to-use-storage-accounts/)

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_blob_data.yml`: Creates a typical pipeline with the following steps:
  - `loading_blob_data_step.py`: step for loading data from DBR or Azure Blob Storage
-  `config/config_blob_data.json`: configuration
- `requirements_blob_data_aml.txt`: Additional requirements for AML
- `requirements_blob_data_dbr.txt`: Additional requirements for DBR
- `requirements_blob_data_local.txt`: Additional requirements for local
- `utils/`: A folder containing multiple helper packages. 
  
## Usage

This is just one step designed to relieve you of the initial work of downloading files from DBR or blob storage. You can then add your subsequent steps, such as preprocessing, etc. :)

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,blob_data_local]" --use-pep517
```

**Note:**
> There are three requirements for the CDL data pipeline, each connected to the platform on which you plan to run it. For instance, if you aim to run it on DBR, then `provider_dbr` will use `requirements_blob_data_dbr.txt`. This distinction is important because DBR does not require PySpark installation, and AML does not support PySpark sessions. As a result, we use PyArrow.

### Parameters

When configuring parameters for the CDL pipeline, it's essential to adjust them based on your execution environment. If you're running the pipeline on DBR, you need to set up `dbr_table`. For instance:
```yaml
params:
  dbr_table: default.test_table
```
However, if you intend to run it on AML, you should focus on configuring either `folder_path` or `file_path` depending on your data organization. For example:
```yaml
params:
  folder_path: # Specify the folder path here
  file_path: Files/file.parquet
```
This ensures that the pipeline accesses the correct data source based on the platform it's executed on.

### Unit Testing in Your Project
When you pull your project's pipeline, you'll find the tests under the `src/<your project name>/tests/` directory. There are two main approaches to creating your own unit tests:
- Using Pyrogai Mock from the Pyrogai library
- Creating your own mock

**Approach 1: Using Pyrogai Mock**

This is the recommended method for most cases.
For documentation, refer to: [Pyrogai Mock Documentation](https://developerportal.pg.com/docs/default/Component/PyrogAI/test_mock_step/)
To see implementation examples, you can pull the `ml_iris` or `ml_skeleton` pipelines.

**Approach 2: Creating Your Own Mock**

If you prefer this method, refer to the test examples that came with your pipeline.

To get started, pull your project's pipeline and navigate to the tests directory. Choose the approach that best suits your needs and refer to the provided examples for guidance.

### How to run example

```bash
# Run locally
aif pipeline run --pipelines blob_data --environment dev

# Run on DBR platform (use similar commands for the other platforms)
aif pipeline run --pipelines blob_data --platform DBR --environment dev
```
