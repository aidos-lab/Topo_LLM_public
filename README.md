# Topo_LLM

## Installation instructions with poetry

1. Preparing the environment.

Required Python version: 3.10
On MacOS, you can install `pyenv` with homebrew: `brew install pyenv`.
You can install `poetry` with `pipx` for example: `pipx install poetry`.

1. Install python version with `pyenv` and set local python version for the project.

```bash
pyenv install 3.10
pyenv local 3.10
```

1. Tell poetry to use the local python version.

```bash
# Optional: Tell poetry to create a virtual environment for the project
# inside the project directory.
poetry config virtualenvs.in-project true

poetry env use 3.10
```

You can manage the poetry environments with the following commands:

```bash
poetry env list --full-path # List all the environments
poetry env remove <path> # Remove an environment
poetry env remove --all # Remove all the environments
```

1. Install the project with dependencies.
Select the appropriate dependency groups for your system.

```bash
poetry install --with gpu,dev --without cpu # For GPU
```

```bash
poetry install --with cpu,dev --without gpu # For CPU
```

1. Set the correct environment variables used in the project config.
Edit the script `setup_environment.sh` with the correct paths and run it once.

```bash
./setup_environment.sh
```

## Project Structure

### Config file management

- We want to use Hydra for the config managment:
  [https://hydra.cc/docs/patterns/configuring_experiments/]

- Overwrite config variable:
  `python run.py run.seed=42`

- Multirun example:
  `python run.py --multirun run.seed=1,2,3,4`

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

### Computing and storing embeddings

In the directory `topollm/compute_embeddings`, call the embedding script:

```bash
python3 compute_embeddings.py
```
