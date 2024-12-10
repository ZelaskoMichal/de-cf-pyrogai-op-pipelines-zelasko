# Optimization – Template Pipeline

The `optimization` pipeline demonstrates a typical optimization use case. Specifically, it formulates and solves a simple stock portfolio optimization from FICO's examples.

While the steps' content in this example is specific to the use case at hand, the pipeline's structure, use of data quality steps, and logging of metrics with MLFlow are generically re-applicable to most optimization use cases. We recommend using this example as a skeleton for your optimization use cases by replacing the code in each step and the data quality expectations defined in your configuration file - [How to - Customize optimization pipeline](https://developerportal.pg.com/docs/default/Component/PyrogAI/general-information/how-tos/pyrogai/template-pipelines/customize-optimization-pipeline/) will walk you through this.

Please note that this example assumes your I/O is stored in file storage in the platform you are running on. In many realistic use cases, you will read data from or write data to other locations such as the Spark metastore (CDL data) or from BigQuery (consumer data lake).

???+ note "optimization-models package integration with template pipelines"
  The optimization template pipeline leverages the optimization-models package from dnalib, which offers plenty
  helpers and structure for building and testing optimization models. Due to technical limitations, the package
  is installed through the installation of pyrogai, where a single version is hardcoded, thus the version of the
  package cannot be changed in your template pipeline. Check Pyrogai's release notes to find the optimization-models
  version installed with Pyrogai.

???+ warning "optimization-models-v1.x.x is not an stable release, expect breaking changes"
  This package is under development and it has not reached a stable version, meaning that we expect breaking changes in
  minor releases. We expect the release optimization-models-v2.x.x to be a stable version.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_optimization.yml`: Creates a typical optimization pipeline with the following steps:
  - `steps/optimization/copy_input_to_ioctx.py`: Copies input files from IOslots to the IOctx working directory, storing them in parquet format. In the process, it ensures that CSV and Excel files read the string "NA" as just that and not a code for null values and that Excel files are separated into individual dataframes for each tab.  These pre-actions are required for the pyrogAI Data Quality step to be able to work with the input files for this example; they may or may not be needed for your input files.
  - `steps/optimization/preprocess_data.py`: Joins input data files into the standard data model that the formulation expects.
    - `steps/optimization/preprocessing/sdm.py`: Defines the logic used in the preprocessing step to create the standard data model.
    - `steps/optimization/preprocessing/transformers.py`: Example of further modularizing the logic for building the standard data model via "transformers".
  - `steps/optimization/preprocessed_critical_custom_dq.py`: Custom data validation checks of the standard data model which, if failed, should fail the pipeline.
  - `steps/optimization/preprocessed_warning_custom_dq.py`: Custom data validation checks of the standard data model which, if failed, should NOT fail the pipeline. These generate information for an operations team to use when looking for the root cause of a failure.
  - `steps/optimization/formulate_and_solve.py`: Formulates and solves the optimization. MLflow is used throughout to capture key information about this model run. The optimal solution is stored in the working directory for the next step.
    - `steps/optimization/formulation/stock_portfolio_optimization.py`: The specific formulation (set, variables, constraints, and objective function) for this example.
  - `steps/optimization/solution_custom_dq.py`: Custom data validation checks of the optimal solution.
  - `steps/optimization/postprocess_data.py`: Logic to convert the optimal solution with raw and SDM data into the pipeline outputs.
  - `steps/optimization/output_dq_with_save_to_ioslots.py`: Customized DataQuality step that saves the output to IOslots after running GreatExpectations.
- `config/config_optimization.json`: configuration parameters for the pipeline
- `utils/optimization/io_utils.py`: utilities for reading and saving files common accross multiple steps
- `reqs/requirements_optimization.txt`: package requirements for the pipeline
- `README_optimization.md`: this documentation

It makes the following changes to existing files:

- `pyproject.toml`: adds optimization package as an install option

**NOTE**: upon running the pipeline some steps aside of ones listed above will be present - notably data validation steps using Great Expectations framework. These do not require custom files and you can inspect them in `config/pipeline_optimization.yml`.

## Usage

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,optimization]"
```

### Data

Example data files can be found in the [StockPortfolio](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx) data set. It is necessary to put the data files in the configured `data_dir`.

### Configuration Parameters

The `config_optimization.json` file contains the following keys that set up the pipeline:

- `data_dir`: folder which contains the input data, relative to the base path implied by the provider and any additional path configuration in the IOslot definition in `config_optimization.json`
- `input_tmp_dir`: folder in the working directory to where the input files are stored (after copying)
- `sdm_tmp_dir`: folder in the working directory to where the standard data model is stored
- `solution_tmp_dir`: folder in the working directory to where the optimal solution is stored
- `output_tmp_dir`: folder in the working directory to where the output is stored for final data validation
- `output_dir`: folder which contains the output data, relative to the base path implied by the provider and any additional path configuration in the IOslot definition in `config_optimization.json`
- `optimization`, required for XpressStep, with the following subkeys:
  - `formulate_and_solve`: Xpress license settings for the formulate_and_solve step
    - `fall_back_to_community_license`: true = use the community license unless another license is provided
- `great_expectations`: contains the expectation suites used by the various data quality steps in the pipeline and configures which expecations suites each step runs

### Runtime Parameters

The `pipeline_optimization.yml` file contains parameter used to build the optimization problem:

- `parameters`: inputs for formulating the optimization, with the following subkeys:
  - `max_risky_stocks`: The maximum number of risky stocks in the portfolio
  - `max_risky_stocks_ratio`: The maximum fraction of risky stocks in the portfolio
  - `max_ratio_per_stock`: The maximum fraction of the portfolio invested in a single stock
  - `min_ratio_per_stock`: If the portfolio invests in a stock, it must be at least this much of the portfolio
  - `min_ratio_per_region`: Each region must have at least this fraction of the portfolio
  - `min_stocks_per_region`: Each region must have at least this number of stocks in the portfolio
  - `max_total_stocks`: The maximum number of stocks in the portfolio
  - `max_number_risky_sum_activation`: Toggle the activation of the max_number_risky_sum constraint between hard, soft or off.
  - `min_number_per_region_activation`: Toggle the activation of the min_number_per_region constraint between hard, soft or off.

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
aif pipeline run --pipelines optimization

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines optimization  --environment dev --platform AML
```

## More information

See [How to - Add and run optimization pipeline](https://developerportal.pg.com/docs/default/Component/PyrogAI/general-information/how-tos/pyrogai/template-pipelines/add-and-run-optimization-pipeline/)
