# Logging metadata

This pipeline is meant to present the way of saving, loading and initializing metadata for Vertex AI platform.

## Parameters and metrics

Logging these values is quite straightforward - using `aiplatform` package from GCP's python SDK, you can log them in a way similar to the one known from mlflow:

```python
from google.cloud import aiplatform
aiplatform.log_metrics(metrics)
aiplatform.log_params(exp_params)
```

Where `metrics` and `exp_params` are dictionaries in form: {"param1": value1} etc.

## Artifacts

With `aiplatform.Artifact.create`, we're able to create and log the Artifact object to Vertex AI run:

```python
# Create an artifact
artifact = aiplatform.Artifact.create(
    schema_title="your_schema_title",
    uri="gs://your_bucket_name/dataframe.csv",
    resource_id="your_resource_id",
    display_name="your_display_name",
    schema_version="your_schema_version",
    description="your_description",
    metadata={"key": "value"},
    project="your_project_id",
    location="your_location"
)

# Log the artifact to the run
self.aiplatform.log_artifact(artifact)
```

All the data is available in provider file (except schema, but shouldn't be a problem)

## Models

To save the model to Vertex AI, we need to do the following:

```python
def save_model_sample(
    project: str,
    location: str,
    model: Union["sklearn.base.BaseEstimator", "xgb.Booster", "tf.Module"],
    artifact_id: Optional[str] = None,
    uri: Optional[str] = None,
    input_example: Optional[Union[list, dict, "pd.DataFrame", "np.ndarray"]] = None,
    display_name: Optional[str] = None,
) -> None:
    aiplatform.init(project=project, location=location) # This line shouldn't be needed - to be checked
    aiplatform.save_model(
        model=model,
        artifact_id=artifact_id,
        uri=uri,
        input_example=input_example,
        display_name=display_name,
    )
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

## Setting run

Real challenge emerges when it comes to getting the right setup - this is to be determined. For now, one thing that works is:
```python
aiplatform.init(experiment='template-meta')
aiplatform.start_run(self.mlflow.active_run().info.run_id)
aiplatform.log_metrics(metrics)
aiplatform.end_run()
```
The new run is created in the experiment and the metrics are logged inside. We want to have them in our pipeline's run, but that's trickier. Yet to determine a way to do this. As of now, logging things works fine, but the run is a separate, entirely new one so it's a bit hard to find it in Vertex's UI. I think that's enough to start development with hope that we'll figure it out.




