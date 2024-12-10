# BQ load – Template Pipeline

The `bq_io` template pipeline demonstrates how to load data from BigQuery.
It only works on Local and KFP.

## Configuration Parameters

```yaml
input_output_slots:
-   name: bq_slot
    type: bigquery
```
In the same file bq_io steps will use bigquery as inputs:
```yaml
steps:
  - name: bq_io
    class: template_pipelines.steps.bq_io:BQIo
    inputs:
    - bq_slot
```

Add configuration for your own dataset, project id and bucket name in `config_bq-io.json`.
```json
{
    "bq_io":{
        "bq_dataset": "<FILL_YOUR_DATASET>",
        "gcp_project": "<FILL_YOUR_PROJECT_ID>",
        "bucket_name": "<FILL_YOUR_BUCKET>"
    }
}
```


## Usage

### Data prerequisites

In order to run current pipeline you need to have campaign table in your dataset. You have 2 main ways of adding that data.

a) You can copy it from [sharepoint](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx) and upload to BigQuery manually.  

b) You can also copy it from `dna-aif-dev-dev-ebbb.common_kubeflow` dataset in BigQuery using UI or command below.
```
CREATE OR REPLACE TABLE
    `[PROJECT_NAME].[DATASET_NAME].campaign` AS
SELECT
    *
FROM
    `dna-aif-dev-dev-ebbb.common_kubeflow.campaign`
```

### Access rights

Find service account name and confirm it exists in IAM.  
It follows the pattern: *[use-case-name]*@*[gcp_project_id]*.iam.gserviceaccount.com. You can find the _use-case-name_ in _provider_vertex.yml_ under _service_account_, project id 
  can also be found in the same file.  

  You must have the permission to add roles on the BigQuery dataset. Create a new dataset if one does not exist.
  1. On the dataset level add `BigQuery Data Viewer` role for service account.  
  2. Click the dataset, then **Sharing** -> **Permissions**.  
  3. **Add Principal** and fill out the full service account name and the role.  


### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install --upgrade pip  
pip install -e ".[devel,bq_io]"
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

### How to run example


####  Run pipeline

```bash
## Run locally
aif pipeline run --pipelines bq_io --scope bq-io

## Run on Vertex platform - run from Cloud Workstations
aif pipeline run --pipelines bq_io --platform Vertex --environment dev

```

## More information
It uses PGFlow implementation of BigQuery helper class for common operations.  
Refer to [pyrogai](https://github.com/procter-gamble/de-cf-pyrogai/blob/main/src/aif/pyrogai/steps/ioslots/bigquery.py) for available methods.
