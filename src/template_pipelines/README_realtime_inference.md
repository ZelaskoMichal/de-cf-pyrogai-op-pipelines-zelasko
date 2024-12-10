# Realtime inference - Template Pipeline

Real time inference endpoints allow you to upload your trained ML models and enable an HTTP endpoint for inferencing. Inference is the process of applying new input data to the machine learning model or pipeline to generate outputs. While these outputs are typically referred to as "predictions," inferencing can be used to generate outputs for other machine learning tasks, such as classification and clustering.

## Files structure

This realtime pipeline adds the following files to your repository:

- `config/pipeline_realtime_inference.yml`: Creates one pipeline step:
  - `log_model` - `steps/realtime_inference/log_model.py` - example with logging your model
- `config/config_realtime-inference.json` - configuration file.
- `config/model_backend_realtime_inference.yml` - components for describing and validating model backend
- `config/model_configuration_realtime_inference.yml` - components for describing and validating the model configuration data
- `config/model_set_realtime_inference.yml` - Components for describing and validating the model set data
- `config/realtime_sample_request.json` - sample request, needed for you backend config
- `utils/realtime_inference/process_data.py` - a script that will run after the endpoint is triggered
- `utils/realtime_inference/realtime_scoring.py` - a script required for real time endpoint deployments to perform the inferencing/prediction

## Configuration Parameters

To authenticate with AML, you need to have your PAT token

Once generated, the token should be stored in `src/<project_name>/config/secrets.json`:

```json
{
    "gh_token": "your-generated-token"
}
```

You have two options to create and store this `.json` file:

**Manually create** the `.json` file and add the secret within it.

After adding the secret, upload it to the cloud.

## Usage

### Install requirements

After successfully downloading the pipeline, the next step is to install all required dependencies. To avoid any issues with dependencies, it is recommended to use `miniforge` and set up a fresh environment for this purpose. Install the dependencies with the command:

```sh
pip install -e ".[devel,realtime_inference]"
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

### Execute the pipeline

```bash
# Run locally
aif pipeline run --pipelines realtime_inference --environment dev

# Run on AML platform
aif pipeline run --pipelines realtime_inference --environment dev --platform AML
```

after finished pipeline you should see model in your AML->models 

then you can deploy your endpoint based on created model

```bash
# Deploying endpoint
aif model deploy --provider-platform aml --environment dev --config-module <your_config_folder>.config --model-config realtime_inference_configuration --provider-environment dev --provider-name '<your AML provider>'
```

## More information

See [AIF model deploy overview](https://developerportal.pg.com/docs/default/Component/PyrogAI/aif.pyrogai.models.overview/)