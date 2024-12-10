# ML Inference – Template Pipeline (Beta)

The `ml_inference` template pipeline implements the steps for a typical Machine Learning inference pipeline. Specifically,
it reads in an autoencoder model trained in the [ml_training](README_ml_training.md) pipeline, gets model training losses to estimate the loss distribution and then generates predictions based on that distribution. The `ml_inference` performs batch processing, so there is a significant amount of preprocessing that occurs before getting the predictions.

![](../../docs/docs/assets/ml_inference_pipeline.png)

**Running the ml_training pipeline beforehand or providing the trained model path/URI is a prerequiste for being able to run inference.** See this [instruction](README_ml_training.md) on how to set up and run the ml_training pipeline.

While the steps content in this example is specific to the use case at hand, the pipeline's structure, use of data quality steps, and model loading with MLFlow are generically re-applicable to most ML use cases. We recommend using this example as a skeleton for your ML use cases by replacing the code in each step and the data quality expectations defined in your configuration file.

Please note that this example assumes your input data is stored in file storage. In many realistic use cases, you will read data from the Spark metastore (CDL data) or from BigQuery (consumer data lake). In this case, you should change the preprocessing step or add another step to extract data. The data extraction from these sources will be demonstrated in future examples.

## Files

This template pipeline adds the following files to your repository:
- `config/pipeline_ml_inference.yml`: Creates a typical ML inference pipeline with the following steps:
  - `data_preprocessing`: (`steps/ml_training/data_preprocessing.py`): Reads data from blob storage, performs basic transformation/imputation, and writes the results to the work directory. Note that this step uses ioslots. See [Input and Output slots](https://developerportal.pg.com/docs/default/component/pyrogai/io_overview/)
  - `feature_creation` (`steps/ml_training/data_creation.py`): Reads output of previous steps, does feature engineering, and saves a single file containing the engineered features
  - `data_validation_after_feature_creation` (`steps/ml_training/dv_after_feature_creation.py`) Validates transformed data from feature_creation based on expectations defined in the config file and then visualizes validation results (see note below)
  - `run_inference`: Reads in the trained model from registry and run inference on the transformed dataset
- `config/config_ml-inference.json`: configuration parameters for the pipeline
- `reqs/requirements_ml_inference.txt`: package requirements for the pipeline
- `utils/ml_inference`: folder containing helper functions and classes

**NOTE:** You can also use the great expectactions functionality as a no code solution with the out-of-the box "DqStep" in pyrogai. This custom version improves the visualization of data docs in GCP and AML. 

## Configuration Parameters

The `config_ml-inference.json` file contains the following keys:

- `ml_inference`, with the following subkeys:
  - `target`: The response variable
  - `data_dir`: The folder which contains the input data relative to the base path implied by the provider
  - `model_dir`: The folder which contains trained models relative to the base path implied by the provider (local/AML/Vertex) 
  - `dbr_model_uri`: The databricks mlflow model uri. This is an alternative to `model_dir` if databrics was used to train a model and then mlflow was used to log the trained model
  - `output_file`: The output file that contains model predictions
- `great_expectations`: contains the expectations used by the various steps in the pipeline i.e. data quality

## Usage

### Get model (DBR)

Before running inference, you need to train a model using the [ml_training](README_ml_training.md) pipeline. Once the model has been trained, find the following log message in the model training step run of the ml_training pipeline.

```
The model has been trained and saved to: runs:/<RUN>/anomaly_detector
```

Then, update the model URI in `config_ml-inference.json` in order to load the up-to-date trained model during inferencing.

```json
"ml_inference": {
    "dbr_model_uri": "runs:/<RUN>/anomaly_detector"
}
```

### Get model (local, AML, Vertex)

For the remaining providers, the `ml_inference` pipeline loads the trained model from the default `model_dir`, which is created during the ml_training pipeline run.

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,ml_inference]"
```

### Parameters

This pipeline does not have any runtime parameters. To add runtime parameters, see [pipeline creation](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.pipelines.models.pipeline/) and [run time parameters in steps](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.step/#aif.pyrogai.steps.step--runtime-parameters).

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
aif pipeline run --pipelines ml_inference

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines ml_inference --environment dev --platform AML
```