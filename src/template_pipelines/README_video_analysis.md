# Video Analysis – Template Pipeline - Beta version

The `video_analysis` template pipeline demonstrates a simplistic Generative AI use case for video processing, leveraging the Gemini model. Specifically, it analyzes videos posted by influencers promoting specific products and provides insights on how to enhance the marketing of those products within the videos.

While the steps' content in this example is specific to the use case at hand, the pipeline's structure, use of the Gemini model, and prompt definition are generically re-applicable to most GenAI use cases. We recommend using this example as a skeleton for your GenAI use cases by replacing the code in each step and the parameters defined in your configuration file. **Currently, the `video_analysis` pipeline is limited to local execution and AML. This capability will be later extended to remaining cloud providers.**

Please note that this pipeline is in its beta version.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_video_analysis.yml`: Creates a typical Video Analysis pipeline with the following steps:
  - `video_processing` (`steps/video_analysis/video_processing.py`): Reads data from blob storage, analyzes videos and provides insights into the product's marketing strategy. Note that this step uses [ioslots](https://developerportal.pg.com/docs/default/Component/PyrogAI/io_overview/).
  - `result_aggregation` (`steps/video_analysis/result_aggregation.py`): Aggregates the processed video results and merges them with the original data
- `config/config_video-analysis.json`: configuration parameters for the pipeline
- `reqs/requirements_video_analysis.txt`: package requirements for the pipeline
- `utils/video_analysis/base_genai_step.py`: a customized base step for using the GenAI Gemini model and Pyrogai functionalities
- `utils/video_analysis/toolkit.py`: utilities for working with GenAI

## Configuration Parameters

The `config_video-analysis.json` file contains the following keys:

- `video_analysis`, with the following subkeys:
  - `data_dir`: Folder which contains the input data, relative to the base path implied by the provider
  - `genai_url`: GenAI platform url for accessing Vertex AI functionalities
  - `model_endpoint`: Gemini model endpoint to be used for the video analysis
  - `cognitive_services`: Set of OpenAI or GenAI platform APIs to interact with
  - `headers`: Additional information to be sent along with a request to the GenAI platform
    - `userid`: @pg.com email address of a person utilizing the GenAI platform
    - `project-name`: Project name should match one from the itaccess group created for your project
    - `accept`: Format in which the client (your application) would like to receive the response
  - `thread_num`: Number of threads used to send requests to the GenAI platform

## Secrets
If you want to run the `video_analysis` template pipeline locally using your pg account, skip this section and jump to the next one on how to grant [access to the GenAI platform](#access-to-the-genai-platform).

For running the `video_analysis` template pipeline locally or on your AML platform using service principal credentials, some additional secret configuration is required. If you have created your project using AI Provisioner, you should have your service principal credentials (AML-APP-SP-ID and AML-APP-SP-SECRET) stored in an appropriate Azure keyvault. If not, upload the service principal credentials to the keyvault.

To be able to run the `video_analysis` template pipeline locally using the service principal, the following secrets need to be added to secrets.json in your config module.
```sh
{
  "AML-APP-SP-ID": "service principal id",
  "AML-APP-SP-SECRET": "service principal secret",
  "tenant-id": "azure tenant id"
}
```
To be able to run the `video_analysis` template pipeline on AML, upload the values in secrets.json to the Azure keyvault.

## Access to the GenAI Platform

In order to take full advantage of Generative AI capabilities and services, a required access to the GenAI platform is needed. See [how-to get your access to GenAI](https://developerportal.pg.com/docs/default/Component/genAI-Platform/access/).

## Usage

**Important:** Open all the steps and files created with the template, and modify them based on your needs, before you attempt to run the pipeline.

### Install requirements

After importing this video_analysis pipeline, find the relative path to requirements_video_analysis.txt and install all the necessary dependencies:

```sh
pip install -r <RELATIVE_PATH>/requirements_video_analysis.txt
```

### Parameters

This pipeline does not have any runtime parameters. To add runtime parameters, see [pipeline creation](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.pipelines.models.pipeline/) and [run time parameters in steps](https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.step/#aif.pyrogai.steps.step--runtime-parameters).


### Datasets setup
If you have access and can use the same userid and project-name as defined in config_video-analysis.json, then download video_data.csv from [the sharepoint site](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx?id=%2Fsites%2FAIFUserFiles%2FTutorial%20Data%2FVideoData&viewid=5510c1e4%2D1bc7%2D4f0f%2D8bee%2D93bc5b9ba294).
The video_data.csv file includes a column named "genai_filepath" which contains video file URIs. These URIs correspond to videos that have been uploaded to the GenAI platform. 

If not, then download the videos from the same [sharepoint site](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx?id=%2Fsites%2FAIFUserFiles%2FTutorial%20Data%2FVideoData&viewid=5510c1e4%2D1bc7%2D4f0f%2D8bee%2D93bc5b9ba294) and follow this [guide](https://developerportal.pg.com/docs/default/Component/genAI-Platform/gu_upload_files/) to upload them to the GenAI platform. After uploading, replace the existing URIs in the "genai_filepath" column of video_data.csv with the new URIs. It is important to use the same userid and project-name for both uploading and reading the videos from the GenAI platform.

- For local pipeline runs, save the csv file in the root project directory in the folder called `video_data`.
- For AML pipeline runs, upload the `video_data` folder containing the downloaded data to Azure Blob Storage defined in provider_aml.yml in your config module.

If the `video_data` folder does not exist, create one. Make sure the folder name matches the value of `data_dir` in `config_video-analysis.json`.

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
aif pipeline run --pipelines video_analysis --environment dev
```

```bash
# Run on AML platform
aif pipeline run --pipelines video_analysis --environment dev --platform AML
```

### Extra configuration (Optional)
If the `video_analysis` template pipeline is not able to access the GenAI platform from AML, some private endpoint between AML and AKS (in which the GenAI platform is deployed) needs to be created. Reach out to the Azure platform team for support on this matter.