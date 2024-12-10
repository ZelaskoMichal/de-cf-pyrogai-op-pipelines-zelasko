# Quickstart – Template Pipeline

The `quickstart` template pipeline is a simple example of a four step pipeline, that trains a random forest on randomly-geenerated data.
It illustrates the main features of pyrogAI and is used for our quickstart tutorial:

- Input/Outpt Slots
- I/O Context
- MLFlow integration
- Pipeline flow
- Great expectation steps.

It is not meant to be used as a starting point in real projects.

The data was generated with the `make_dataset` function in `sklearn`:

```python
# Generate Data
from sklearn.datasets import make_classification
X1, Y1 = make_classification(
    n_features=5, n_redundant=0, n_informative=4, n_clusters_per_class=2, n_classes=2, n_samples=10000
)
df = pd.DataFrame(X1, columns=["feature1", "feature2", "feature3", "feature4", "feature5"]) 
df['target'] = Y1
df.to_parquet("dummy.parquet")
```

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_quickstart.yml`: Creates a typical simple:
  - `data_preprocessing`: (`steps/quickstart/data_preprocessing.py`): Reads data from blob or local storage, and writes the results to the working directory. Note that this step uses [ioslots](https://developerportal.pg.com/docs/default/Component/PyrogAI/io_overview/).
  - `data_validation_after_processing` (standard `DQStep`): Data validation with Great Expectations
  - `model_training` (`steps/quickstart/sk_model_training.py`): Trains a random forest model and save it to MLFlow
  - `sk_model_evaluation` (`steps/quickstart/sk_model_evaluation.py`): Computes several performance metrics and plots, and logs them to MLFlow
- `config/config_quickstart.json`: configuration
- `reqs/requirements_quickstart.txt`: Additional requirements

## Configuration Parameters

The `config_quickstart.json` file contains the following keys:

- `quickstart`, with the following subkeys:
  - `data_dir`, folder that should contain the `dummy.parquet` file
  - `features`, list of column names to be used as features
  - `n_estimator`, number of estimators in the random forest
  - `random_state`, random state, for reproducibility.

- `great_expectations`: contains the expectations used by the various steps in the pipeline i.e. data quality
  - Please note that the example contains two example expectations on a dummy column "col1" (to provide you with an example). You will have to **modify or remove** these example for the pipeline to run correctly with your data.

## Usage

**Important** Open all the steps and files created with the template, and modify them based on your needs, before you attempt to run the pipeline.

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,quickstart]"
```

### Parameters

This pipeline does not have any runtime parameters. To add runtime parameters, see [pipeline creation](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.pipelines.models.pipeline/) and [run time parameters in steps](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.step/#aif.pyrogai.steps.step--runtime-parameters)

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

### Datasets setup

Before running this pipeline, download the parquet files from
[the sharepoint site](https://pgone.sharepoint.com/:f:/r/sites/AIFUserFiles/Tutorial%20Data/Quickstart?csf=1&web=1&e=tDHHFg), and place it in the `data` folder of your repository.

### How to run example

```bash
# Run locally
aif pipeline run --pipelines quickstart
```
