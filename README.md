# Less is More: Local Intrinsic Dimensions of Contextual Language Models (Topo_LLM)

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

### Installation instructions with uv

1. Install python version with `uv`:

```bash
uv python install 3.12
```

1. You can check the installed python versions with:

```bash
uv python list
```

1. Lock the dependencies with `uv` and sync the environment:

```bash
uv lock
uv sync
```

1. Start a python interpreter with the local environment:

```bash
uv run python3
```

### Specific instructions for HPC Cluster

On some HPC clusters, you might need to pin a torch version in the `pyproject.toml` file, to make the installation of torch and a compatible CUDA version work.
For example, on our HPC cluster, it currently appears to work when you set the torch version to `2.3.*`:

```toml
torch = "2.3.*"
```

### Project-specific setup

1. Set the correct environment variables used in the project config.
This step can be achieve by running the setup script in the `topollm/setup/` directory once.

```bash
./topollm/setup/setup_environment.sh
```

1. If required, e.g. when running jobs on a HPC cluster, set the correct environment variables in the `.env` file in the project root directory.

2. For setting up the repository to support job submissions to the a HPC cluster using our custom Hydra launcher, follow the instructions here: [https://github.com/carelvniekerk/Hydra-HPC-Launcher]. Additional submission scripts are located in the `topollm/scripts/submission_scripts` directory.

3. Download the files necessary for `nltk`: Start a python interpreter and run the following:

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

### General instructions to run the pipeline

TODO: Add instructions for how to use and configure the uv run commands.

```bash
uv run pipeline_local_estimates # This runs the full pipeline embedding -> embeddings_data_prep -> compute local estimates
```

### Experiments: Fine-Tuning Induces Dataset-Specific Shifts in Heterogeneous Local Dimensions

TODO: Explain how to run these experiments.
TODO: Add instructions for finetuning the language model.
TODO: Explain how to compute local estimates for the finetuned models.
TODO: Explain how to call the violin plot creation script.

```bash
uv run finetune_language_model
```

### Experiments: Local Dimensions Predict Grokking

Refer to the separate `grokking` repository for instructions on how to run these experiments.

### Experiments: Local Dimensions Detect Exhaustion of Training Capabilities

TODO: Explain how to train the Trippy-R dialogue state tracking models.
TODO: Explain how to compute local estimates for the Trippy-R models.
TODO: Explain how to create the plots comparing local dimensions and task performance for the Trippy-R models.

### Experiments: Local Dimensions Detect Overfitting

TODO: Explain how to train the ERC models.
TODO: Explain how to compute local estimates for the ERC models.
TODO: Explain how to create the plots comparing local dimensions and task performance for the ERC models.

### Run tests

We provide a python script that can be called via a poetry run command to run the tests.

```bash
uv run tests
```
