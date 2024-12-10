# Orchestration – Template Pipeline

Pipeline orchestration is the process of managing and automating the execution of multiple data pipelines using RHEA, ensuring they run in the correct order and scope

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_orchestration.yml`: Creates a typical orchestration pipeline with the following steps:
  - `trigger_pipelines.py`: (`steps/orchestration/trigger_pipelines.py`): triggering pipelines
- `config/config_orchestration.json`: configuration
- `config/notification_orchestration_email_template.html`: email template
- `requirements_orchestration.txt`: Addition requirements


## Configuration Parameters

The `config_orchestration.json` file contains the following keys:

- `config_orchestration` - configuration of orchestration
- `config_scopes` - configuration of scopes
- `config_dbr` - configuration of dbr
- `config_aml` - configuration of aml
- `config_vertex` - configuration of vertex
- `notification_handlers`: notification configuration. For full guide refer to: [Notifications in PyrogAI](https://developerportal.pg.com/docs/default/component/pyrogai/notifications/#email-notification)

## Secrets

```json
{
  "gh_token": "xxx",
  "AML-APP-SP-ID": "xxx",
  "AML-APP-SP-SECRET": "xxx",
  "dbr_token": "xxx"
}
```

## Runtime params

- `dest_platform` - when you run this pipeline locally, you need to set up on which cloud you want to trigger pipelines

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

## Usage

```bash
aif pipeline run --pipelines orchestration -p dest_platform=aml/dbr/vertex
```

## Additional info
1. Pipeline save summary on mlflow
2. Mlflow isnt supported on VertexAI (yet)
3. Pipeline is able to send email notification but there is no possibility to put variables from step there
4. To be able to trigger pipeline on DBR you need to first submit it manually. RHEA can not trigger pipelines which don't exists
5. To trigger pipeline on AML you need to first create batch endpoint with pipeline manually or via aif pipeline publish
6. Vertex part in this template is still under developing (WIP)

## RHEA
More info about rhea you can find in [official docs](https://developerportal.pg.com/docs/default/Component/PyrogAI/aif.rhea/)
