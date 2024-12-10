# Time series – Template Pipeline

The `time_series` template pipeline demonstrates a typical time series data use case. This pipeline shows how to uses time series CO2 data to forecast future emissions over time. The data came from statsmodels API. It also shows some mlflow capabilities.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_timeseries_data.yml`: Creates a typical time series data pipeline with the following steps:
  - `preprocess_data`: (`steps/time_series/preprocess_data.py`): Loads the data from statsmodel.api and save it for furture steps
  - `train_model` (`steps/time_series/train_model.py`): Reads output of previous steps, train model using the ARIMA model (Autoregressive Integrated Moving Average), and saves it for furture steps
  - `model_evaluation` (`steps/time_series/model_evaluation.py`): Generates the model metrics, diagnostics graph, prediction vs actual graph and saves it to artifacts
  - `prediction` (`steps/time_series/prediction.py`): Generates forecast graph for 200 periods and saves it to artifacts
- `config/config_time-series.json`: Basic config file
- `reqs/requirements_time_series.txt`: Package requirements for the pipeline


### Install requirements

Follow the instructions below to install all the necessary requirements after importing this pipeline:


```sh
pip install -e ".[devel,time_series]"
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

This pipeline does not have any runtime parameters.

### How to run example

```bash
# Run locally
aif pipeline run --environment dev --pipelines time_series

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --environment dev --pipelines time_series --platform AML
```
