# EDA Pipeline – Template Pipeline

The `eda_pipeline` template pipeline is an example pipeline that performs Exploratory Data Analysis (EDA) on the Iris dataset using various EDA tools. It demonstrates how to use different EDA tools within a PyrogAI pipeline and serves as a starting point for projects requiring data analysis and visualization.

This pipeline includes:

- **Loading data**: Fetches the Iris dataset from `sklearn.datasets`.
- **Performing EDA**: Uses a selected EDA tool to analyze the data.
- **Generating reports**: Creates and saves EDA reports based on the analysis.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_eda_pipeline.yml`: Defines the pipeline and its steps.
- `config/config_eda_pipeline.json`: Configuration file with parameters.
- `reqs/requirements_eda_pipeline.txt`: Additional requirements for the pipeline.
- `steps/eda_pipeline/eda_pipeline.py`: Contains the `EdaExample` step that loads data and performs EDA.
- `utils/eda_pipeline/eda_toolkit.py`: Utility class `EDAToolkit` that provides methods to generate reports using various EDA tools.

## Pipeline Steps

The pipeline consists of the following step:

- **`eda_pipeline`** (`steps/eda_pipeline/eda_pipeline.py`): Performs EDA on the Iris dataset using the specified tool.

### EdaExample Step

- **`load_iris_data`**: Loads the Iris dataset from `sklearn.datasets` and converts it to a pandas DataFrame.
- **`perform_eda`**: Uses the `EDAToolkit` to perform EDA using the specified tool.
- **`run`**: Executes the step by loading data and performing EDA.

### EDAToolkit

The `EDAToolkit` class in `steps/eda_pipeline/eda_toolkit.py` provides methods to generate EDA reports using different tools.

Supported EDA tools:

- **Sweetviz**: Generates a detailed EDA report and saves it as an HTML file.
- **Klib**: Provides data cleaning and visualization functions.
- **Dabl**: Offers automatic data visualization.
- **Missingno**: Visualizes missing data patterns.

## Configuration Parameters

The `config_eda_pipeline.json` file contains the following keys:

  - **`eda_tool`**: The EDA tool to use. Supported tools are:
    - `sweetviz`
    - `klib`
    - `dabl`
    - `missingno`

## Usage

**Important**: Before running the pipeline, ensure that all the required packages are installed, and review the steps and files to understand how they work.

### Install Requirements

Install all the necessary requirements:

```sh
pip install -e ".[devel,eda]"
```

### How to Run the Example

To run the pipeline locally with the default EDA tool (e.g., `sweetviz`):

```bash
# Run locally
aif pipeline run --pipelines eda_pipeline
```

Replace `sweetviz` with the desired EDA tool from the supported list.

### Unit Testing in Your Project

When you pull your project's pipeline, you'll find the tests under the `src/<your project name>/tests/` directory. There are two main approaches to creating your own unit tests:

- **Using PyrogAI Mock**: This is the recommended method for most cases.
  - Refer to the [PyrogAI Mock Documentation](https://developerportal.pg.com/docs/default/Component/PyrogAI/test_mock_step/).
  - Implementation examples can be found in the `ml_iris` or `ml_skeleton` pipelines.
- **Creating Your Own Mock**: If you prefer this method, refer to the test examples that came with your pipeline.

To get started, navigate to the tests directory in your project and choose the approach that best suits your needs.

## Customization

You can modify the pipeline to work with your own dataset or to include additional EDA tools.

- **Using Your Data**: Replace the `load_iris_data` method with code to load your dataset.
- **Adding Tools**: Extend the `EDAToolkit` class to include other EDA tools.

## Dataset Setup

No additional dataset setup is required as the pipeline uses the Iris dataset from `sklearn.datasets`. If you want to use your own data, ensure that the data is accessible and modify the `load_data` method accordingly.

## Additional Notes

- Ensure that you have installed all the required packages listed in `requirements_eda_pipeline.txt`.
- Some EDA tools may require additional configuration or have dependencies that need to be met.
- The output reports will be saved in the working directory, such as `sweetviz_report.html` for Sweetviz.

## Limitations

- The pipeline currently supports a limited set of EDA tools. You can extend it to include more tools as needed.
- The output of some tools may require manual review or adjustments in the code to save the reports properly.

## Conclusion

This `eda_pipeline` pipeline provides a flexible framework for performing EDA using different tools. It's designed to be easily extendable and can serve as a solid foundation for your data analysis projects.