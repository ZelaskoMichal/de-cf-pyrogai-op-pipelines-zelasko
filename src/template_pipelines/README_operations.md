# Operations – Template Pipeline

The `operations` template pipeline implements the the steps you will need from operations team standpoint and example with mlflow logging or pyrogai logging 

The end user for this pipeline is an Ops team, who doesn’t know how to debug the code but just knows how to run the code. In a real life scenario the Ops team will run a pipeline, if it fails there will be an email sent to them with the logs. This team cannot debug the code but they will send the logs and the failure message to the data scientists and developers to debug the code.


## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_operations.yml`: Creates a typical operations pipeline with the following steps:
  - `logging`: (`steps/operations/logging.py`): 
  - `mlflow`: (`steps/operations/mlflow.py`): 
  - `notification`: (`steps/operations/notification.py`): 
- `src/template_pipelines/utils/operations/logging_utils.py`: utils for logging
- `config/config_operations.json`: configuration
- `requirements_operations.txt`: Addition requirements
- 
## Configuration Parameters

The `config_operations.json` file contains the following keys:

- `operations`, with the following subkeys:
  - `parameter`: example param
- `notification_handlers`: notification configuration. For full guide refer to: [Notifications in PyrogAI](https://developerportal.pg.com/docs/default/component/pyrogai/notifications/#email-notification)

## Usage

**Important**
By default this is expected to be running inside Github workflows, as it needs `SMTP-PASSWORD` from secrets.

**Important**
Mlflow isn't supported by Vertex for now. Use `vertex_meta` pipeline instead.


### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,operations]"
```

### Parameters

This pipeline does not have any runtime parameters. To add runtime parameters, see [pipeline creation](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.pipelines.models.pipeline/) and [run time parameters in steps](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.step/#aif.pyrogai.steps.step--runtime-parameters)

### Mlflow
Example of mlflow usage and more detailed description you can find in mlflow step of that pipeline

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

The notification feature will only work in Github Actions workflows where environment contain `SMTP-PASSWORD` secret.

```bash
# Run locally
aif pipeline run --pipelines operations --environment dev

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines operations --platform AML --environment dev
```

to see how mlflow step works with examples, we recommend to run that pipeline in this way:

```bash
aif pipeline run --environment dev  --pipelines operations --experiment-name california_flats

# and after successful pipeline run you need to start mlflow UI
mlflow ui --backend-store-uri ./pyrogai_run_mlflow/operations/mlruns
```

setting up `--experiment-name california_flats` helps you with your mlflow UI for easier finding, you can change this name

`./pyrogai_run_mlflow/operations/mlruns` - This is the path that likely leads you to your MLflow folder where all the logs are saved.
