# Topo_LLM

## Overview

This repository contains code for analyzing the representations produced by contextual language models from a topological perspective.

## Installation

### Prerequisites

- Python 3.12
- `pyenv` for managing Python versions
- `pipx` for managing Python packages
- `poetry` package manager

#### MacOS specific instructions

On MacOS, you can install `pyenv` and `pipx` with Homebrew: 
- `brew install pyenv`
- `brew install pipx`.

You can install `poetry` with `pipx`: 
- `pipx install poetry`.

#### Ubuntu/Debian instructions

Install `pipx` via `apt` and then install `poetry` via `pipx`:

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath

pipx install poetry
```

### Installation instructions with poetry

1. Install python version with `pyenv` and set local python version for the project.
You can install `pyenv` via Homebrew on MacOS and UNIX systems.

```bash
pyenv install 3.12
pyenv local 3.12
```

1. Tell poetry to use the local python version.

```bash
# Optional: Tell poetry to create a virtual environment for the project inside the project directory.
poetry config virtualenvs.in-project true

poetry env use 3.12
```

You can manage the poetry environments with the following commands:

```bash
poetry env list --full-path # List all the environments
poetry env remove '<path>' # Remove an environment
poetry env remove --all # Remove all the environments
```

1. Install the project with dependencies.
Select the appropriate dependency groups for your system.

```bash
# For GPU
poetry install --with gpu,dev --without cpu
```

```bash
# For CPU
poetry install --with cpu,dev --without gpu
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

## Datasets

- Dialogue data
  - MultiWOZ
  - SGD:
    [https://github.com/google-research-datasets/dstc8-schema-guided-dialogue]

## Usage

If you run the commands from the command line, make sure to activate the poetry environment:

```bash
poetry shell
```

You can also check the `[tool.poetry.scripts]` block in the `pyproject.toml` file for available entry points. For example, the following commands give access to the main entry points:

```bash
poetry run pipeline_local_estimates # This runs the full pipeline embedding -> embeddings_data_prep -> compute local estimates
poetry run compute_perplexity
poetry run finetune_language_model
```

### Specific instructions for HHU Hilbert Cluster

On HHH Hilbert, you might need to pin a torch version in the `pyproject.toml` file, to make the installation of torch and a compatible CUDA version work.
For example, it currently appears to work when you set the torch version to `2.3.*`:

```toml
torch = "2.3.*"
```

### Run tests

We provide a python script that can be called via a poetry run command to run the tests.

```bash
poetry run tests
```
