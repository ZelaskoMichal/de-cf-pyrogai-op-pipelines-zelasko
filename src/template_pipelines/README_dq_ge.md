# DQ & GE – Template Pipeline

The `dq_ge` pipeline is a guide for using data quality tools, specifically Great Expectations, in pyrogAI. This mini pipeline includes a brief tutorial showing two ways to work with GE. The first method uses the pyrogAI's DQ class, which doesn't need any extra files in your pipeline. The second method involves coding your DQ step directly into the pipeline. To choose your preferred method, go to `config/pipeline_dq_ge.yml` and remove the steps you don't need.

## Files

This `dq_ge` pipeline adds the following files to your repository:

- `config/pipeline_dq_ge.yml`: Creates a 4 pipeline steps:
  - `data_loading` - `steps/dq_ge/dq_ge_1_data_loading.py`
  - `initial_data_validation` - `steps/dq_ge/dq_ge_2_initial_data_validation.py`
    -  `initial_data_validation_dq` - step in pipeline_dq_ge.yml which contain pyrogais' DQ class 
  - `data_processing` - `steps/dq_ge/dq_ge_3_data_processing.py`
  - `post_processing_data_validation` - `steps/dq_ge/dq_ge_4_post_processing_data_validation.py`
    - `post_processing_data_validation_dq` - step in pipeline_dq_ge.yml which contain pyrogais' DQ class

- `reqs/requirements_dq_ge.txt`: additional requirements needed by the dq_ge pipeline.
- `config/config_dq-ge.json`: configuration

## Configuration Parameters

### DBR Authentication

In order to authenticate with Databricks, you need to generate a token in your Databricks user profile. See [how-to setup your access token](https://developerportal.pg.com/docs/default/component/pyrogai/dbr-specific/how-tos/generate-dbr-token/)

The generated token should be saved in `src/template_pipelines/data/secrets.json`.

```json
{
    "dbr_token": "your-generated-token"
}
```

### `config_dq-ge.json` Parameters

All parameters for this pipeline are explained inside `config/config_dq-ge.json`

## Usage

### Install requirements

Follow the instructions you received after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,dq_ge]"
```
### Parameters

This pipeline does not have any runtime parameters.

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
aif pipeline run --pipelines dq_ge

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines dq_ge --envirnoment env --platform AML
```

## More information

See [Data Quality (Great Expectations) Step](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.base_dq_step/)