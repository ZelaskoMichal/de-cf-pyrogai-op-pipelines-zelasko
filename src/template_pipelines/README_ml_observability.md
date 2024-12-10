# Observability –- Opinionated Pipeline

The `ml_observability` pipeline is a very basic example of Observability usage. It relies on the `ml_iris` pipeline that generates fake data conforming with the IRIS dataset and uses a dummy classifier to solve the problem.

## Files

This Observability pipeline adds the following files to your repository:

- `config/pipeline_ml_observability.yml`: Creates 7 pipeline steps:
  - `generate_data` - `steps/ml_observability/iris_1_generate_data.py`
  - `standardize_data` - `steps/ml_observability/iris_2_standarize_data.py`
  - `fix_data` - `steps/ml_observability/iris_3_fix_data.py`
  - `split_data` - `steps/ml_observability/iris_4_split_data.py`
  - `train_model` - `steps/ml_observability/iris_5_train_model.py`
  - `score_data` - `steps/ml_observability/iris_6_score_data.py`
  - `observability_step` - `steps/ml_observability/iris_observability_step.py`

- `reqs/requirements_ml_observability.txt`: additional requirements needed by the ml_observability pipeline.
- `config/config_ml-observability.json`: configuration 

## Configuration Parameters

### DBR Authentication

In order to authenticate with Databricks, you need to generate a token in your Databricks user profile. See [how-to setup your access token](https://developerportal.pg.com/docs/default/component/pyrogai/dbr-specific/how-tos/generate-dbr-token/)

The generated token should be saved in `src/opinionated_pipelines/data/secrets.json`.

```json
{
    "dbr_token": "your-generated-token"
}
```

### `config_ml-observability.json` Parameters

The  `config_ml-observability.json` contains `ml_observability` dictionary with the following keys:

- `target`: name of predicted column,
- `features`: list of feature column name,
- `random_state`: seed,
- `train_size`: part of the data intended for training data

Apart from that there is also the `observability_models` key that stores information about your models:

```json
{
    ...
    "observability_models": {
        "Iris_Classifier": {
            "version": "1",
            "type": "categorical"
        }
    }
}
```
where:
- `Iris_Classifier` - model name. It can be any name you choose.
- `version` - version of the model.
- `type` - type of the task. Either categorical or numeric.

For more information visit: [Observability - Pyrogai - Observability step](https://developerportal.pg.com/docs/default/component/ai_factory_observability/observability-specific/explainers/observability-pyrogai-observability_step/)

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

## Usage

### Install requirements

Follow the instructions you received after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,ml_observability]"
```

### Parameters

This pipeline does not have any runtime parameters.

### How-to run example

```bash
# Run locally
aif pipeline run --pipelines ml_observability

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines ml_observability --environment dev --platform AML
```
