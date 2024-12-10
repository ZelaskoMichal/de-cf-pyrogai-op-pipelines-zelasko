# MDF IIOT - Template Pipeline - BETA
## General Information
A general demonstration use-case of the MDF (Manufacturing Data Flow) IIOT Simulator using the PyrogAI framework. The whole documentation about MDF and what adapters are can found under [MDF TUTORIAL](https://github.com/procter-gamble/iiot-platform-demo-project).

## Prerequisites
Before you run PyrogAI with MDF we kindly request you complete the whole tutorial linked above and read about adapters, its logic, usage etc. In the process of completing the tutorial, you will learn how to request all access and install the required tools for your local environment based on your machine.

Packaging the model (creation of an adapter), running the adapter (packaged model) and simulator locally requires access to the MDF's internal ACR. Access is granted by membership in the security group: GBSG-IDS-Engineering-MDF-IOT-DEVELOPERS-READ`. You can check your group memberships [here](https://identitycentral.pg.com/groups). You may request access to the security group via [itAccess](https://itaccess.pg.com/identityiq/accessRequest/accessRequest.jsf#/accessRequest/selectUser?quickLink=Request%20Access).

## Files
This template pipeline adds the following files to your repository:
- `config/pipeline_mdf_iiot.yml`: Configuration of the pipeline logic:
  - `log_model`: (`src/template_pipelines/steps/mdf_iiot/log_model.py`): Single pipeline step to create a simple model and save via MLflow
- `config/config_mdf-iiot.json`: configuration parameters for the pipeline
- `config/model_backend_mdf.yml`: ACR backend 
- `config/model_configuration_mdf.yml`: ACR configuration
- `config/model_set_mdf.yml`: Model specifications
- `requirements_mdf_iiot.txt`: package requirements for the pipeline
- `utils/mdf_iiot/adapter`: folder containing adapter from MDF tutorial

## Usage

After downloading MDF IIOT pipeline, you can run the pipeline locally and in the Azure cloud. Running the pipeline in the Azure cloud is required prior to running the adapter and simulator locally.

### Pipeline Local Run

Update your local environment by running:

```bash
pip install -e ".[mdf_iiot]"
```

Run the md_iiot pipeline locally by running:

```bash
aif pipeline run --pipelines mdf_iiot --environment local
```

### Pipeline Azure Cloud Run

Run the mdf_iiot pipeline on Azure ML by running:

```bash
aif pipeline run --pipelines mdf_iiot --environment dev --platform AML
```

### Package Model

Login to Azure CLI and MDF Dev ACR:

```bash
az login
az acr login -n acrci3415437dev01
```

Pull the required Docker images:

```bash
docker pull acrci3415437dev01.azurecr.io/base/python:3.10
docker pull acrci3415437dev01.azurecr.io/platform/simulator:beta
```

After running the pipeline in Azure ML, go to the AML model registry to review the available models for packaging. Once you have decided on a model version, you can specify the model version in the `model_set_mdf.yml` file. To package the model and deliver to the specified ACR in the `model_backend_mdf.yml` file, run:

```bash
aif model package --provider-platform aml --environment local --model-config mdf_configuration --provider-name 'AML Provider' --provider-environment dev --model-backend mdf_backend --config-module your.config_module
```

### Run the Adapter (Packaged Model) and Simulator Locally

Update the `dockerfile_path` and `ADAPTER_PATH` variables in the `model_backend_mdf.yml` file to point to the correct paths in your project.

Change directory to the `adapter` folder and run:

```bash
docker compose build --no-cache
docker compose up
```

You should be able to see the adapter and simulator interacting with each other locally.

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
