# ML Training – Template Pipeline

The `ml_training` template pipeline demonstrates a typical ML training use case. Specifically, it trains an autoencoder to perform anomaly detection on a coupon redemption dataset and identify users that will redeem a coupon. The majority of coupons are unredeemed making the dataset imbalanced. Instead of upsampling/downsampling the dataset, the autoencoder is trained on unredeemed coupons to reconstruct the input data and to estimate the distribution of training losses for the majority class. Once appropriate mean and std values are found to describe the distribution, they can be used to determine the coupon redemption status. The current autoencoder architecture is implemented using PyTorch. Though the tensorflow implementation of the autoencoder is deprecated, it is still accessible in previous releases.

![](../../docs/docs/assets/ml_training_pipeline.png)

While the steps content in this example is specific to the use case at hand, the pipeline's structure, use of data quality steps, and logging of metrics with MLFlow are generically re-applicable to most ML use cases. We recommend using this example as a skeleton for your ML use cases by replacing the code in each step and the data quality expectations defined in your configuration file.

Please note that this example assumes your input data is stored in file storage. In many realistic use cases, you will read data from the Spark metastore (CDL data) or from BigQuery (consumer data lake). In this case, you should change the preprocessing step or add another step to extract data. The data extraction from these sources will be demonstrated in future examples.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_ml_training.yml`: Creates a typical ML training pipeline with the following steps:
  - `data_preprocessing`: (`steps/ml_training/data_preprocessing.py`): Reads data from blob storage, performs basic transformation/imputation, and writes the results to the work directory. Note that this step uses ioslots. See [Input and Output slots](https://developerportal.pg.com/docs/default/component/pyrogai/io_overview/)
  - `feature_creation` (`steps/ml_training/data_creation.py`): Reads output of previous steps, does feature engineering, and saves a single file containing the engineered features
  - `data_validation_after_feature_creation` (`steps/ml_training/dv_after_feature_creation.py`) Validates transformed data from feature_creation based on expectations defined in the config file and then visualizes validation results (see note below)
  - `imputation_scaling` (`steps/ml_training/imputation_scaling.py`): Imputes missing values in newly created features, normalizes and splits data into training and test sets
  - `data_validation_after_imputation` (`steps/ml_training/dv_after_imputation.py`): Validates transformed data from imputation based on expectations defined in the config file and then visualizes validation results (see note below)
  - `model_training` (`steps/ml_training/model_training.py`): Trains the autoencoder with PyTorch and logs the trained model in MLFlow
  - `model_evaluation` (`steps/ml_training/model_evaluation.py`): Loads the trained model previously logged in MLFlow and computes several evaluation metrics
- `config/config_ml-training.json`: configuration parameters for the pipeline
- `reqs/requirements_ml_training.txt`:  package requirements for the pipeline
- `utils/ml_training`: folder containing helper functions and classes

**NOTE:** You can also use the great expectactions functionality as a no code solution with the out-of-the box "DqStep" in pyrogai. This custom version improves the visualization of data docs in GCP and AML. 

## Configuration Parameters

The `config_ml-training.json` file contains the following keys:

- `ml_training`, with the following subkeys:
  - `random_state`: The random seed control parameter
  - `target`: The response variable
  - `learning_rate`: The hyperparameter to control the pace at which model weights are updated
  - `stop_learning`: The patience parameter that determines when the model training stops if there are no further improvements. It is the number of epochs to wait after the last improvement
  - `epochs`: The number of complete passes through the entire training dataset.
  - `batch_size`: The hyperparameter that refers to the number of training examples utilized in one iteration of the training process
  - `min_tp`: The minimum true positive threshold used to set a model working point
  - `data_dir`: The folder which contains the input data relative to the base path implied by the provider
- `great_expectations`: contains the expectations used by the various data quality steps in the pipeline

## Usage

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,ml_training]"
```

### Parameters

This pipeline does not have any runtime parameters.

### Datasets setup
Download the parquet files from
[the sharepoint site](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx?id=%2Fsites%2FAIFUserFiles%2FTutorial%20Data%2FCouponData&viewid=5510c1e4%2D1bc7%2D4f0f%2D8bee%2D93bc5b9ba294).


- For local pipeline runs, save the parquet files in the root project directory in the folder called `CouponData`.
- For cloud pipeline runs, upload the `CouponData` folder containing the downloaded data to the appropriate platform storage defined in provider_[platform].yml in your config module.

If the `CouponData` folder does not exist, create one. Make sure the folder name matches the value of `data_dir` in `config_ml-training.json`.

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
aif pipeline run --pipelines ml_training

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines ml_training  --environment dev --platform AML
```

## What's next?
- See [how-to run ml_training pipeline](https://developerportal.pg.com/docs/default/component/pyrogai/general-information/how-tos/pyrogai/template-pipelines/add-and-run-ml_training-pipeline/) for more information.
- After the autoencoder model has been trained and evaluated, the ml_inference pipeline can be triggered to generate predictions. See this [instruction](README_ml_inference.md) on how to set up and run the ml_inference pipeline.