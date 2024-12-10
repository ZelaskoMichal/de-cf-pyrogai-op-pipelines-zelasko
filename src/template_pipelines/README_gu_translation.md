# Generalized Utilities Translation – Template Pipeline - Beta version

The `gu_translation` template pipeline demonstrates a simplified Generative AI application for text translation. Specifically, it focuses on localizing a product marketing slogan by translating it into multiple target languages for effective communication across diverse markets.

While the steps' content in this example is specific to the use case at hand, the pipeline's structure and the use of the Generalized Utilities translation service from the GenAI platform are generically re-applicable to most GenAI use cases. We recommend using this example as a skeleton for your GenAI use cases by replacing the code in each step and the parameters defined in your configuration file. **Currently, the `gu_translation` pipeline is limited to local execution and AML. This capability will be later extended to remaining cloud providers.**

Please note that this pipeline is in its beta version.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_gu_translation.yml`: Creates a typical Translation pipeline with the following steps:
  - `translation` (`steps/gu_translation/translation.py`): Reads text from specified pipeline parameters and, leveraging the GenAI platform, translates it into multiple target languages defined within those pipeline parameters
- `config/config_gu-translation.json`: configuration parameters for the pipeline
- `reqs/requirements_gu_translation.txt`: package requirements for the pipeline
- `utils/gu_translation/base_genai_step.py`: a customized base step for using the Generalized Utilities translation service from the GenAI platform and Pyrogai functionalities

## Configuration Parameters

The `config_gu-translation.json` file contains the following keys:

- `gu_translation`, with the following subkeys:
  - `genai_url`: GenAI platform url
  - `service_endpoint`: Generalized Utilities translation service endpoint
  - `cognitive_services`: Set of OpenAI or GenAI platform APIs to interact with
  - `headers`: Additional information to be sent along with a request to the GenAI platform
    - `userid`: @pg.com email address of a person utilizing the GenAI platform
    - `project-name`: Project name should match one from the itaccess group created for your project
    - `accept`: Format in which the client (your application) would like to receive the response
  - `thread_num`: Number of threads used to send requests to the GenAI platform

## Secrets
If you want to run the `gu_translation` template pipeline locally using your pg account, skip this section and jump to the next one on how to grant [access to the GenAI platform](#access-to-the-genai-platform).

For running the `gu_translation` template pipeline locally or on your AML platform using service principal credentials, some additional secret configuration is required. If you have created your project using AI Provisioner, you should have your service principal credentials (AML-APP-SP-ID and AML-APP-SP-SECRET) stored in an appropriate Azure keyvault. If not, upload the service principal credentials to the keyvault.

To be able to run the `gu_translation` template pipeline locally using the service principal, the following secrets need to be added to secrets.json in your config module.
```sh
{
  "AML-APP-SP-ID": "service principal id",
  "AML-APP-SP-SECRET": "service principal secret",
  "tenant-id": "azure tenant id"
}
```
To be able to run the `gu_translation` template pipeline on AML, upload the values in secrets.json to the Azure keyvault.

## Access to the GenAI Platform

In order to take full advantage of Generative AI capabilities and services, a required access to the GenAI platform is needed. See [how-to get your access to GenAI](https://developerportal.pg.com/docs/default/Component/genAI-Platform/access/).

## Usage

**Important:** Open all the steps and files created with the template, and modify them based on your needs, before you attempt to run the pipeline.

### Install requirements

After importing this gu_translation pipeline, find the relative path to requirements_gu_translation.txt and install all the necessary dependencies:

```sh
pip install -r <RELATIVE_PATH>/requirements_gu_translation.txt
```

### Parameters

- `text`: Source text to be translated
- `original_language`: Language code of the source `text`
- `target_languages`: Comma-separated list of target language codes specifying the languages to which the `text` should be translated

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
aif pipeline run --pipelines gu_translation --environment dev
```

```bash
# Run on AML platform
aif pipeline run --pipelines gu_translation --environment dev --platform AML
```

### Extra configuration (Optional)
If the `gu_translation` template pipeline is not able to access the GenAI platform from AML, some private endpoint between AML and AKS (in which the GenAI platform is deployed) needs to be created. Reach out to the Azure platform team for support on this matter.