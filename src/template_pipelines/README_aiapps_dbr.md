# Iris AIApps Databricks –- Template Pipeline

The `aiapps_dbr` pipeline is the “hello world” example customized for the Config-YAML app and designed for execution on Databricks. It is based on the example from `ml_iris` pipeline, it uses similar structure; generates fake data and uses scikit-learn to solve the problem.

## Files

This template pipeline adds the following files to your repository:

- `src/project_name/config/pipeline_aiapps_dbr.yml`: Pipeline configuration with the following steps:
  - `generate_data`: (`src/project_name/steps/aiapps_base/generate_data.py`)
  - `preprocess_data`: (`src/project_name/steps/aiapps_base/preprocess_data.py`)
  - `train_model`: (`src/project_name/steps/aiapps_base/train_model.py`)
  - `score_data`: (`src/project_name/steps/aiapps_dbr/score_data.py`)

- `requirements_aiapps_dbr.txt`: additional requirements needed by the aiapps_dbr pipeline.
- `src/project_name/config/config_aiapps-dbr.json`: configuration

## Configuration

### DBR Authentication

In order to authenticate with Databricks, you need to generate a token in your Databricks user profile. See [how-to setup your access token](https://developerportal.pg.com/docs/default/component/ai_factory_pyrogai/dbr-specific/how-tos/run-in-dbr-from-vscode/)

The generated token should be saved in `src/project_name/config/secrets.json`.

```json
{
    "dbr_token": "your-generated-token"
}
```

### Blob Storage Authentication

Pipeline saves output files to the Blob Storage account, to be accesible by Config-YAML app. It is reccomended to use default blob storage account. You can find its name together with default container name under `provider_dbr.yml` or `provider_aml.yml` files. 

Secret should be saved in `src/project_name/config/secrets.json`.

```json
{
    "dbr_token": "your-generated-token",
    "AZFILE-STORAGE-SECRETS" : {
        "storage123": {
         "azure_storage_connection_string":
             "xxx"
        }
    }
}
```

Replace `storage123` placeholder with your actual storage account name.

**To run on the cloud**, the same secret should be uploaded to the Azure keyvault.

For more information go to the [documentation](https://developerportal.pg.com/docs/default/component/pyrogai/dbr-specific/how-tos/howto-copy-from-dbr-to-blob/)

### `config_aiapps-dbr.json` Parameters

`config_ml-aiapps-dbr.json` file consists of two dictionaries: `blobstore`, and `aiapps_output_files`. Below, we outline the structure and purpose of each dictionary:

1. `blobstore`

The `blobstore` dictionary is used to specify the configuration for accessing Blob Storage. It has two keys: `storage_account` and `account_name`. 
You need to fill in values for these keys with the data of the Blob Storage, which will be used to read and save files during the pipeline execution. It is recommended to use the default Blob Storage set for the project.
```
"blobstore": {
    "storage_account": "your_storage_account_name",
    "account_key": "your_storage_account_key"
  }
```

> **Note:** Replace Current Values
Please note that the current values provided ("mlwpyrogaicse4e8fyksa" for storage account and "azureml-blobstore-5a4bef7e-1bf4-4448-913c-e5b27ab9db0a" for the account key) are only for test purposes and will not work for you. Replace them with the actual values for your storage account.

> **Tip:** You can find values for the default Blob Storage under `provider_dbr.yml` or `provider_aml.yml` files.

2. `aiapps_output_files`

The `aiapps_output_files` dictionary is used to configure the files that will be created by the pipeline as output. Users can set values for these configurations, which can be later utilized within the steps' code.

## Usage

### Install requirements

Follow the instructions you received after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,aiapps_dbr]" --use-pep517
```

### Runtime Parameters

The pipeline includes four runtime parameters:

1. config_text_field
2. run_text_field
3. run_file_1
4. config_file_1

Parameters **1** and **2** can be configured through the Config-YAML application interface.

Parameters **3** and **4** are designed as references to files uploaded via the application to the blob storage. These parameters are utilized within input slots to construct URLs pointing to specific files, enabling access to uploaded content in the pipeline's steps.

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

### How-to run example

```bash
# Run locally
aif pipeline run --pipelines aiapps_dbr

# Run on DBR platform
aif pipeline run --pipelines aiapps_dbr --platform DBR
```

## More information

See [how-to run ml_iris pipeline](https://developerportal.pg.com/docs/default/component/ai_factory_pyrogai/general-information/how-tos/pyrogai/template-pipelines/add-and-run-ml-iris-pipeline/)
