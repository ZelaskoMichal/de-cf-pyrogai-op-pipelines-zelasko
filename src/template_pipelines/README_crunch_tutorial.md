# CRUNCH Tutorial – Template Pipeline

The `CRUNCH Tutorial` pipeline demonstrates how to use a CRUNCH step in PyrogAI pipeline. 

The pipeline comprises three steps: generate data, run parallel processing in CRUNCH, and validate the output.

## What is CRUNCH?
CRUNCH is an external platform integrated into PyrogAI through a dedicated CRUNCH step.

CRUNCH enables the execution of multiple parallel jobs within a single PyrogAI step.

Read [Explainer: What is CRUNCH in AIF](https://developerportal.pg.com/docs/default/Component/AI_Factory_General/pipelines/explainers/what-is-crunch-in-aif) to learn more.


## How CRUNCH step works
- For each CRUNCH step class used in the pipeline YAML file, a Docker image is built by PyrogAI.
The Docker image includes the code implemented in the `crunch_run()` method of the CRUNCH step class.
  
- During the pipeline run, the CRUNCH step lists the provided input directory and generates a batch of jobs.
The number of jobs in the batch is equal to the number of subdirectories in the input directory.
  
- The CRUNCH step sends configuration of the batch of jobs to the CRUNCH API. CRUNCH creates a pod in CRUNCH AKS cluster for each job.
Each pod automatically downloads the input data, runs the code, and uploads the output data for each job.

- The CRUNCH step actively monitors the CRUNCH API for the status of the batch and terminates when the batch is completed


## Access to CRUNCH
To run CRUNCH Tutorial on any of the supported platforms you need to get access to CRUNCH.

Read [how to get your access to CRUNCH in AI Factory](https://jira-pg-ds.atlassian.net/wiki/spaces/DSUG/pages/4213572157/How+to+get+access+to+CRUNCH).

## Step-by-step guide to run CRUNCH Tutorial pipeline

> [!TIP]
> If you encounter any issue with CRUNCH in AIF check the [Issues & Workarounds page](https://jira-pg-ds.atlassian.net/wiki/x/G4LMKQE) for possible solution.

### Step 1: Download the crunch_tutorial pipeline

```sh
  aif pipeline from-template --pipe-name crunch_tutorial
```

### Files
This template pipeline adds the following files to your repository:

- `config/pipeline_crunch_tutorial.yml`: A simple pipeline with three steps: generate data, run parallel processing in CRUNCH, and validate the output.
   - `steps/crunch-tutorial/generate_crunch_inputs.py`: Generates input data files for the CRUNCH step and writes them to ADLS.
   - `steps/crunch-tutorial/tutorial_crunch_step.py`: CRUNCH step that runs parallel processing on the input data files and writes the output to ADLS.
   - `steps/crunch-tutorial/validate_crunch_outputs.py`: Validates the output data files from the CRUNCH step.
- `config/config_crunch-tutorial.json`: configuration parameters for the pipeline
- `utils/crunch_tutorial/utils.py`: helper functions
- `utils/crunch-tutorial/example_logic.py`: Example logic for the CRUNCH step.
- `utils/crunch-tutorial/generate_data.py`: Logic to generate input data for the CRUNCH step.
- `utils/update_crunch_acr_secret.py`: Script to update CRUNCH-ACR secret in your secrets.yaml file'
- `requirements-crunch_tutorial.txt`: requirements for the CRUNCH step
- `README_crunch_tutorial.md`: this documentation

It makes the following changes to existing files:

- `pyproject.toml`: adds CRUNCH tutorial requirements as an installation option


### Step 2: Install requirements

```sh
  pip install -e ".[devel,crunch-tutorial]"
```

`requirements-crunch_tutorial.txt` contains requirements for both the CRUNCH tutorial pipeline running on local, AML or DBR platform; and for the CRUNCH jobs executed by TutorialCrunchStep in CRUNCH AKS cluster.

> [!TIP]
> Note that having single requirements file is a simplification for the purposes of this tutorial. 
> It is recommended for your further development to provide separate requirements, for example:
> - `requirements_<pipeline-name>.txt` for the pipeline running on local, AML or DBR platform
> - `requirements_<step-name>.txt`, `requirements_<step-name2>.txt` for each of the CRUNCH steps running in CRUNCH AKS cluster
> 
> The CRUNCH jobs triggered by your CRUNCH step are intended to run in a lightweight environment with minimal dependencies, PyrogAI itself is not needed.


### Step 3: Define secrets

To run the CRUNCH Tutorial pipeline locally you have to add the following secrets to `secrets.json` in your config module:

> [!IMPORTANT]
> Note that you first need to get access to CRUNCH to make CRUNCH-URL and CRUNCH-APP-USER-NAME available in your keyvault.

```json
{
    "CRUNCH-APP-USER-SECRET": "<use value of AML-APP-SP-SECRET from your keyvault>",
    "secretA": "42", 
    "CRUNCH-URL": "<look for CRUNCH-URL in your keyvault>",
    "CRUNCH-APP-USER-NAME": "<look for CRUNCH-APP-USER-NAME in your keyvault>",
    "ADLSFILE-STORAGE-SECRETS": [
      {
        "type": "conn_str",
        "conn_str": "<use your ADLS storage account, i.e. the one suffixed with -ex, go to Azure portal -> Security + Networking -> Access keys and copy&paste the connection string"
      }
    ],
    "CRUNCH-ACR": "follow instructions below"
}
```
Find `update_crunch_acr_secret.py` file in `src/<project_name>/utils/crunch_tutorial` and run it to update CRUNCH-ACR secret in your secrets.yaml file:
    
```sh
cd <path to your utils/crunch_tutorial folder>
python update_crunch_acr_secret.py
```

> [!TIP] 
> Set up proxy. If you encounter issues running the script, try exporting a proxy variable in your terminal:
> ```export https_proxy='http://localhost:9000'```

You will see the secrets.json file updated similarly to this example:
```json
{
    "CRUNCH-ACR": {
      "registry": "acrregistryname.azurecr.io",
      "token": {
        "name": "smith-j",
        "password": "abcdefgh12345",
        "creation_time": "2023-09-07T18:53:19.188509+00:00",
        "expiry": "2023-10-07T18:53:18+00:00"
      }
    }
}
```

> [!NOTE]
> Before running your pipeline on AML or DBR, remember to create the secrets in your Azure keyvault.

> [!TIP]
> To learn how to customize the secrets for your project read [how to configure CRUNCH step](https://developerportal.pg.com/docs/default/Component/AI_Factory_General/pipelines/howtos/crunch/how-to-use-crunch-in-aif/#how-to-configure-crunch-step-in-the-pipeline-config-file).


### Step 4: Modify the configuration file

The `config_crunch-tutorial.json` file contains the following requirements:

```
{
  "crunch": { 
    "step_classes": {
      "TutorialCrunchStep": { -> matches the name of your CRUNCH step class, if not provided, 'requirements.txt` is used
        "requirements": "requirements-crunch_tutorial.txt"
      }
    },
    "step_instances": {
      "run_crunch_tutorial": { -> matches the name of your CRUNCH step from pipeline YAML, if not provided, default configuration is used
        "infra_config": {
          "memory": "200Mi", -> memory limit for the CRUNCH step, do not change for the tutorial
          "node_pool": "aif_standard_d8as_v5_regular" -> node pool for the CRUNCH step in CRUNCH AKS cluster, do not change for the tutorial
        },
        "secrets": [
          "secretA" -> a subset of secrets from your secrets.yaml (LOCAL) file or keyvault (AML, DBR) that needs to be propagated to CRUNCH, do not change for the tutorial
        ]
      }
    }
  },
  "my_config_param": 42 -> example custom config parameter available in CRUNCH jobs
  "crunch_tutorial_storage_account": "mlw...ex" -> ADLS storage account from your AIF Resource Group, not the one suffixed with -sa
}
```

The config file requires one modification - set your ADLS storage account in `crunch_tutorial_storage_account`, where this tutorial will write data.
Use the storage account suffixed with `-ex`. The blob storage account suffixed with `-sa` is not supported by CRUNCH.

> [!TIP]
> To learn how to customize the configuration for your project read [how to configure CRUNCH step](https://developerportal.pg.com/docs/default/Component/AI_Factory_General/pipelines/howtos/crunch/how-to-use-crunch-in-aif/#how-to-configure-crunch-step-in-the-pipeline-config-file).


### Step 5: Run the pipeline

Run locally
```bash
aif pipeline run --pipelines crunch_tutorial
```

Run on AML
```bash
aif pipeline run --pipelines crunch_tutorial --environment dev --platform AML
```

Run on DBR
```bash
aif pipeline run --pipelines crunch_tutorial --environment dev --platform DBR
```

> [!IMPORTANT]
> Note that other platforms are not supported.


## Need more help with CRUNCH in PyrogAI?
You can find all the information about CRUNCH in AI Factory [in general documentation](https://developerportal.pg.com/docs/default/Component/AI_Factory_General/pipelines/howtos/crunch/how-to-use-crunch-in-aif).
