# RL Advertising – Template Pipeline

The `rl_advertising` template pipeline demonstrates a simplistic programmatic advertising use case utilizing Reinforcement Learning (RL). In programmatic advertising, advertisers bid in real-time to display their ads. RL Contextual Multi-Armed Bandits can improve in determining the optimial bid for an ad slot by estimating the expected click-through rate of displaying an ad to a particular user. Using the information, the contextual bandit agent can estimate action values through a Bayesian Neural Network architecture for each ad, select the most relevant ad and adaptively learn which ads perform best for various user segments based on real-time performance data.

Traditional A/B testing displays different ads to users (often randomly) and determines the best performing ad after a significant amount of data is collected. Contextual Multi-Armed Bandits can optimize this ad display process by adaptively allocating more ad impressions to better performing ads while still exploring less shown ads.

Advantages of the Contextual Multi-Armed Bandit system:
- Efficiency: it quickly adapts to changing user preferences
- Personalization: it recommends ads to individual users based on their context information
- Balanced trade-off between exploration and exploitation: it displays high-performing ads while trying new ads

**Scenario**: There's an ad server that gathers information about users. The ad server has one ad each that targets users with different levels of education. If the target audience of the ad which is displayed to a user matches the user's actual education level, there is a high chance of a click. If not, the chance of a click decreases gradually by some constant as the gap between the target and user education levels increases.\
**Dataset:** 1994 U.S. Census data is used to get user information and apply to this advertising scenario.\
**Goal:** Display personalized ads to users based on their profile to maximize click-through rates.


While the steps content in this example is specific to the use case at hand, the pipeline's structure, use of data quality steps, and logging of metrics with MLFlow are generically re-applicable to most RL use cases. We recommend using this example as a skeleton for your RL use cases by replacing the code in each step.

Please note that this example assumes your input data is stored in file storage. In many realistic use cases, you will read data from the Spark metastore (CDL data) or from BigQuery (consumer data lake). In this case, you should change the preprocessing step or add another step to extract data. The data extraction from these sources will be demonstrated in future examples.

## Files

This template pipeline adds the following files to your repository:

- `config/pipeline_rl_advertising.yml`: Creates a typical RL advertising pipeline with the following steps:
  - `data_preprocessing`: (`steps/rl_advertising/data_preprocessing.py`): Reads data from blob storage, performs basic transformation, and writes the results to the work directory. Note that this step uses ioslots. See [Input and Output slots](https://developerportal.pg.com/docs/default/component/pyrogai/io_overview/)
  - `bandit_simulation` (`step/rl_advertising/bandit_simulation.py`): Runs the advertising simulation and trains the contextual multi-armed bandit agent
  - `bandit_evaluation` (`step/rl_advertising/bandit_evaluation.py`): Evaluates the performance of the trained contextual multi-armed bandit agent

- `config/config_rl-advertising.json`: configuration parameters for the pipeline
- `reqs/requirements_rl_advertising.txt`:  package requirements for the pipeline
- `utils/rl_advertising`: folder containing helper functions and classes

## Configuration Parameters

The `config_rl-advertising.json` file contains the following keys:

- `rl_advertising`, with the following subkeys:
  - `data_dir`: The folder which contains the input data relative to the base path implied by the provider
  - `train_size`: The parameter that determines the proportion of the dataset to include in the training and test sets. The train_size values range from 0 to 1
  - `dropout_levels`: The list of dropouts for testing different versions of the Bayesian NN model. The dropout values range from 0 to 1
  - `simulation_size`: The number of simulations
  - `model_update_freq`: The frequency of updating the Bayesian NN model

## Usage

### Install requirements

Follow the instructions you also got after importing this pipeline to install all the necessary requirements:

```sh
pip install -e ".[devel,rl_advertising]"
```

### Parameters

This pipeline does not have any runtime parameters.

### Datasets setup
Download us_census_data.parquet from
[the sharepoint site](https://pgone.sharepoint.com/sites/AIFUserFiles/Tutorial%20Data/Forms/AllItems.aspx?id=%2Fsites%2FAIFUserFiles%2FTutorial%20Data%2FUSCensusData&viewid=5510c1e4%2D1bc7%2D4f0f%2D8bee%2D93bc5b9ba294).


- For local pipeline runs, save the parquet file in the root project directory in the folder called `us_census_data`.
- For cloud pipeline runs, upload the `us_census_data` folder containing the downloaded data to the appropriate platform storage defined in provider_[platform].yml in your config module.

If the `us_census_data` folder does not exist, create one. Make sure the folder name matches the value of `data_dir` in `config_rl-advertising.json`.

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
aif pipeline run --pipelines rl_advertising

# Run on AML platform (use similar commands for the other platforms)
aif pipeline run --pipelines rl_advertising  --environment dev --platform AML
```
