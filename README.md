# Topo_LLM

## Overview

This repository contains code for analyzing the representations produced by contextual language models from a topological perspective.

## Installation

### Prerequisites

- Python 3.12
- `uv` package manager

#### MacOS specific instructions

On MacOS, you can install `uv` with Homebrew:

- `brew install uv`.

#### Ubuntu/Debian instructions

Install `pipx` via `apt` and then install `uv` via `pipx`:

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath

pipx install uv
```

### Installation instructions with poetry

1. Install python version with `uv`:

```bash
uv install 3.12
```


### Project-specific setup

1. Set the correct environment variables used in the project config.
Edit the script `topollm/scripts/setup_environment.sh` with the correct paths and run it once.

```bash
./topollm/scripts/setup_environment.sh
```

1. If required, e.g. when running jobs on the HHU Hilbert cluster, set the correct environment variables in the `.env` file in the project root directory.

1. For setting up the repository to support job submissions to the HHU Hilbert HPC, follow the instructions here: [https://gitlab.cs.uni-duesseldorf.de/dsml/HydraHPCLauncher].
Submission scripts are located in the `topollm/scripts/submission_scripts` directory.

1. Download the files necessary for `nltk`: Start a python interpreter and run the following:

```python
>>> import nltk
>>> nltk.download('punkt_tab')
>>> nltk.download('averaged_perceptron_tagger_eng')
```

## Project Structure

### Config file management

- We use Hydra for the config managment:
  [https://hydra.cc/docs/patterns/configuring_experiments/]

- Overwrite config variable:
  `python run.py run.seed=42`

- Multirun example:
  `python run.py --multirun run.seed=1,2,3,4`

- See the instructions here for the HHU Hilbert HPC launcher:
  [https://gitlab.cs.uni-duesseldorf.de/dsml/HydraHPCLauncher]

### Data directory

The data directory is set in most of the python scripts via the Hydra config (see the script `topollm/config_classes/get_data_dir.py` for a common function to access the data directory path).
We additionally set the path to the local data directory in the `.env` file in the project root directory, in the variable `LOCAL_TOPO_LLM_DATA_DIR`.
Most of the shell scripts use this variable to set the data directory path.

For compatibility, please make sure that these paths are set correctly and point to the same directory.

## Datasets

- Dialogue data
  - MultiWOZ
  - SGD:
    [https://github.com/google-research-datasets/dstc8-schema-guided-dialogue]

## Usage

```bash
uv run pipeline_local_estimates # This runs the full pipeline embedding -> embeddings_data_prep -> compute local estimates
uv run compute_perplexity
uv run finetune_language_model
```

### Specific instructions for HHU Hilbert HPC Cluster

On HHU Hilbert HPC, you might need to pin a torch version in the `pyproject.toml` file, to make the installation of torch and a compatible CUDA version work.
For example, it currently appears to work when you set the torch version to `2.3.*`:

```toml
torch = "2.3.*"
```

### Run tests

We provide a python script that can be called via a poetry run command to run the tests.

```bash
uv run tests
```
