# Topo_LLM

## Installation instructions

Required Python version: 3.10

1. Install python version with `pyenv` (install pyenv via brew on MacOS) and set local python version for the project:

```bash
pyenv install 3.10
pyenv local 3.10
```

1. Tell poetry (install poetry for example via pipx) to use the local python version:

```bash
# Optional: Tell poetry to create a virtual environment for the project
poetry config virtualenvs.in-project true

poetry env use 3.10
```

1. Install the project with dependencies:

```bash
poetry install
```

## Project Structure

### Config file management

- We want to use Hydra for the config managment:
  https://hydra.cc/docs/patterns/configuring_experiments/

- Overwrite config variable:
  `python run.py run.seed=42`

- Multirun example:
  `python run.py --multirun run.seed=1,2,3,4`

## Datasets

- Dialogue data
  - MultiWOZ
  - SGD:
    https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

## Usage

### Computing and storing embeddings

In the directory `topollm/compute_embeddings`, call the embedding script:

```bash
python3 compute_embeddings.py
```