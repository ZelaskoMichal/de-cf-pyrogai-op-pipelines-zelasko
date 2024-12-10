# AML Sweep Step

This template pipeline provides an example of using the AzureML Sweep Step in pyrogAI (`AmlSweepStep`) for hyperparameter tuning- which is documented [here](https://developerportal.pg.com/docs/default/Component/PyrogAI/aif.pyrogai.steps.amlsweep.base_amlsweep_step/).

## Overview

**Hyperparameters** are external configuration variables that data scientists manually set before training a model.
**Hyperparameter tuning** is the process of selecting and optimizing the set of hyperparameters for a machine learning model.

The AML Sweep Step is an AzureML-specific pyrogAI step type that **automates hyperparameter tuning** by running the model multiple times with different sets of parameter values. The best run is selected based on the mlflow model metric specified in the config.  Microsoft has documented the underlying sweep step [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?view=azureml-api-2).

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_aml_sweep.yml`: Creates an example pipeline with the following steps:
  - `steps/aml_sweep/generate_data.py` generates the data and splits it into train and test sets.
  - `steps/aml_sweep/sweep_step.py` is the `AmlSweepStep` which runs trials with different hyperparameters and selects the best.
  - `steps/aml_sweep/consumer.py` shows how to read and work with the outut of the AML Sweep Step.
- `config/config_aml_sweep.json`: configuration parameters for the pipeline
- `reqs/requirements_aml_sweep.txt`: package requirements for the pipeline
- `README_aml_sweep.md`: this documentation

It makes the following changes to existing files:

- `pyproject.toml`: adds aml_sweep package as an install option

## Usage

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements for all of the pipelines in your project:

```sh
pip install -e ".[devel,aml_sweep]"
```

### Data

Mocked data is generated within the generate_data step and saved to io context so that all of the trials receive the same data.

### Configuration Parameters

The `configuration_aml-sweep.json` file contains the following keys that set up the pipeline:

- `sweep`, required for AmlSweepStep, with the following subkeys:
  - `my_sweep`: the name of a sweep step
    - `sampling_algorithm`: the method for selecting trials, see the AzureML sweep step documentation linked below for the list of acceptable algoirhtms. Subkeys will depend on the selection.
    - `search_space`: defines the range for each hyperparameter you want to vary.  There is a subkey for each hyperparameter and the subkeys of them will depend on the method chosen, see the AzureML sweep step documentation linked below. The manipulated hyperparameters must be runtime parameters of the pipeline and they need to impact the training _within_ the AMLSweepStep.
    - `objective`: defines the metric to use to select the best hyperparameters and the goal.  The AML Sweep Step must log the metric to mlFlow under the name provided here.  By default, the AML Sweep Step uses `mlflow.autolog()` and the ML algorithm will determine the metrics loged - you may also log any other metric of your choosing within the AML Sweep Step and use it here.
    - `limits`: define how the trials will be conducted. See the AzureML sweep step documentation linked below for options.
    - `early_termination`: defines when the planned trials will be aborted. See the AzureML sweep step documentation linked below for options.

### Runtime Parameters

The `pipeline_aml_sweep.yml` must contain all parameters specified in the config.json under `search_space` (there may be additional runtime parameters just like in any pipeline):

- searched hyperparameters:
  - `learning_rate`: the rate of learning for the boosting
  - `boosting_type`: the boosting algorithm, dart, gbdt or rf
- other runtime parameters:
  - `metric`: metric(s) to be evaluated on the evaluation set(s) - this is used by the lightgbm algorithm as the metric for training the model, and is not necessarily the same as the metric used by AML Sweep Step for hyperparameter tuning
  - `num_iterations`: the number of boosting iterations to perform
  - `max_leaf_nodes`: maximum number of leaves in one tree
  - `random_seed`: the random number generator seed
  - `verbose`: the verbosity of the lightgbm logging

All of these parameters are defined by lightgbm, [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

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

Please note, the the pipeline will run locally however the AMLSweepStep will not perform a hyperparameter optimization because the sweep step that it runs is specific to AML.  When the pipeline runs locally, the cloudfile IO slots are used to communicate between the sweep_step and the consumer.  When the pipeline is run on AML, the uri_file IO slots are used instead.

```bash
# Run locally
aif pipeline run --pipelines aml_sweep

# Run on AML platform 
aif pipeline run --pipelines aml_sweep  --environment dev --platform AML
```

## More information

Check Developer Portal documentation: [AML Sweep Step](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.amlsweep.base_amlsweep_step/)
