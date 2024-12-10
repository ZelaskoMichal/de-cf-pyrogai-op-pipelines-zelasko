# ML Skeleton – Template Pipeline

The `ml_skeleton` template pipeline implements the the steps you will need in a typical Machine Learning training pipeline. 

It is based on the example from `ml_training` pipeline, it uses the same structure (same steps in the same sequence), but the steps are empty so that you can start to implement your model right away.

Use the `ml_skeleton` pipeline if you are already familiary with AI Accelerators and don't want to start from a blank canvas.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_ml_skeleton.yml`: Creates a typical ML skeleton pipeline with the following steps:
  - `sk_data_preprocessing`: (`steps/ml_skeleton/sk_data_preprocessing.py`): Template to read data from blob storage, performs basic transformation/imputation, and writes the results to the work directory. Note that this step uses [ioslots](https://developerportal.pg.com/docs/default/Component/PyrogAI/io_overview/).
  - `sk_feature_creation`: (`steps/ml_skeleton/sk_feature_creation.py`): Template to read output of previous steps, do feature engineering, and save a single file containing the engineered features
  - `data_validation_after_feature_creation` (standard `DQStep`): Data validation with Great Expectations
  - `sk_imputation_scaling` (`steps/ml_skeleton/sk_imputation_scaling.py`): Template for imputation of missing values in the features file
  - `data_validation_after_imputation`  (standard `DQStep`): Data validation with Great Expectations
  - `sk_model_training` (`steps/ml_skeleton/sk_model_training.py`): Tremplate for model training
  - `sk_model_evaluation` (`steps/ml_skeleton/sk_model_evaluation.py`): Template for model loading and computing evaluation metrics
- `config/config_ml-skeleton.json`: configuration
- `reqs/requirements_ml_skeleton.txt`: Additional requirements
- `utils/`: A folder containing multiple helper packages of the template pipelines.

## Configuration Parameters

The `config_ml-skeleton.json` file contains the following keys:

- `ml_skeleton`, with the following subkeys:
  - `parameter`: example param
- `great_expectations`: contains the expectations used by the various steps in the pipeline i.e. data quality
  - Please note that the example contains two example expectations on a dummy column "col1" (to provide you with an example). You will have to **modify or remove** these example for the pipeline to run correctly with your data.

## Usage

**Important** Open all the steps and files created with the template, and modify them based on your needs, before you attempt to run the pipeline.

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,ml_skeleton]"
```

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

### Parameters

This pipeline does not have any runtime parameters. To add runtime parameters, see [pipeline creation](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.pipelines.models.pipeline/) and [run time parameters in steps](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.step/#aif.pyrogai.steps.step--runtime-parameters)

### How to run example

```bash
# Run locally
aif pipeline run --pipelines ml_skeleton

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines ml_skeleton  --environment dev --platform AML
```
