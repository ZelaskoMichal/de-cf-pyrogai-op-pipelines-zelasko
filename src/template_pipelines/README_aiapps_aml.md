# Iris AIApps Azure ML –- Template Pipeline

The `aiapps_aml` pipeline is the “hello world” example customized for the Config-YAML app and designed for execution on Azure Machine Learning. It is based on the example from `ml_iris` pipeline, it uses similar structure; generates fake data and uses scikit-learn to solve the problem.

## Files

This template pipeline adds the following files to your repository:

- `src/project_name/config/pipeline_aiapps_aml.yml`: Pipeline configuration with the following steps:
  - `generate_data`: (`src/project_name/steps/aiapps_base/generate_data.py`)
  - `preprocess_data`: (`src/project_name/steps/aiapps_base/preprocess_data.py`)
  - `train_model`: (`src/project_name/steps/aiapps_base/train_model.py`)
  - `score_data`: (`src/project_name/steps/aiapps_aml/score_data.py`)

- `requirements_aiapps_aml.txt`: additional requirements needed by the aiapps_dbr pipeline.
- `src/project_name/config/config_aiapps-aml.json`: configuration

## Configuration


### `config_aiapps-aml.json` Parameters

`config_ml-aiapps-dbr.json` file consists of a single dictionary: `aiapps_output_files`. Below, we outline the structure and purpose of this dictionary:

The `aiapps_output_files` dictionary is used to configure the files that will be created by the pipeline as output. Users can set values for these configurations, which can be later utilized within the steps' code.

## Usage

### Install requirements

Follow the instructions you received after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,aiapps_aml]" --use-pep517
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
aif pipeline run --pipelines aiapps_aml

# Run on AML platform
aif pipeline run --pipelines aiapps_aml --platform AML
```

## More information

See [how-to run ml_iris pipeline](https://developerportal.pg.com/docs/default/component/ai_factory_pyrogai/general-information/how-tos/pyrogai/template-pipelines/add-and-run-ml-iris-pipeline/)
