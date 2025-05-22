# Less is More: Local Intrinsic Dimensions of Contextual Language Models (Topo_LLM)

## Overview

This repository contains code for analyzing the representations produced by contextual language models from a topological perspective.

## Installation

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager

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
This step can be achieved by running the setup script in the `topollm/setup/` directory once.

```bash
./topollm/setup/setup_environment.sh
```

1. If required, e.g. when running jobs on an HPC cluster, set the correct environment variables in the `.env` file in the project root directory.

2. For setting up the repository to support job submissions to the a HPC cluster using our custom Hydra launcher, follow the instructions here: [https://github.com/carelvniekerk/Hydra-HPC-Launcher]. Additional submission scripts are located in the `topollm/scripts/submission_scripts` directory.

3. Download the files necessary for `nltk`: Start a python interpreter and run the following:

```python
>>> import nltk
>>> nltk.download('punkt_tab')
>>> nltk.download('averaged_perceptron_tagger_eng')
```

## Project Structure

### Config file management

We use [Hydra](https://hydra.cc/docs/patterns/configuring_experiments/) for the config managment.
Please see the documentation and the experiments below for examples on how to use the config file and command line overrides.

### Data directory

The data directory is set in most of the python scripts via the Hydra config (see the script `topollm/config_classes/get_data_dir.py` for a common function to access the data directory path).
We additionally set the path to the local data directory in the `.env` file in the project root directory, in the variable `LOCAL_TOPO_LLM_DATA_DIR`.
Most of the shell scripts use this variable to set the data directory path.

For compatibility, please make sure that these paths are set correctly and point to the same directory.

### Datasets

The following datasets can be used via their config file:

- `multiwoz21.yaml`:
  MultiWOZ2.1; [HuggingFace](https://huggingface.co/datasets/ConvLab/multiwoz21)
- `ertod_emowoz.yaml`:
  EmoWOZ; [HuggingFace](https://huggingface.co/datasets/hhu-dsml/emowoz)
- `trippy_r_dataloaders_processed.yaml`:
  TripPy-R training, validation, test data; [GitLab](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-r-public)
- `sgd.yaml`:
  SGD; [GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), [HuggingFace](https://huggingface.co/datasets/google-research-datasets/schema_guided_dstc8)
- `wikitext-103-v1.yaml`:
  Wikipedia; [HuggingFace](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1)
- `one-year-of-tsla-on-reddit.yaml`:
  Reddit; [HuggingFace](https://huggingface.co/datasets/SocialGrep/one-year-of-tsla-on-reddit)

## Usage

### General instructions to run the pipeline

We provide `uv run` commands in the `pyproject.toml` file for the most important entry points of the module.
Most importantly, the following command runs the full pipeline:

- from computing embeddings,
- to the embedding data preparation (collecting token embeddings, removing padding token embedding),
- and finally computing the local estimates (for example, the TwoNN-based local dimension).

```bash
uv run pipeline_local_estimates
```

Usually, you will want to select a specific dataset and model to run the pipeline on, and change additional hyperparameters.
You can do this by passing the parameters as command line arguments to the `uv run` command:

```bash
uv run pipeline_local_estimates \
  data="wikitext-103-v1" \
  data.data_subsampling.number_of_samples=512 \
  data.data_subsampling.sampling_mode="random" \
  data.data_subsampling.split="validation" \
  data.data_subsampling.sampling_seed=778 \
  language_model="roberta-base" \
  embeddings.embedding_data_handler.mode="regular" \
  embeddings_data_prep.sampling.num_samples=3000 \
  local_estimates=twonn \
  local_estimates.filtering.num_samples=500 \
  local_estimates.pointwise.absolute_n_neighbors=128
```

The parameters in the command line override the default parameters in the config file.
Here, we explain the parameters in the example command:

- `data="wikitext-103-v1"`: Compute local estimates for the Wikipedia dataset.
- `data.data_subsampling.number_of_samples=512`: Sample 512 sequences from the dataset (i.e., set `M=512` as size of the text corpus sequence sub-sample).
- `data.data_subsampling.sampling_mode="random"`: Randomly sample the sequences from the dataset (other option is `take_first` for taking the first `M` sequences).
- `data.data_subsampling.split="validation"`: Use the validation split of the dataset (other options are `train` and `test` or `dev`, depending on the dataset).
- `data.data_subsampling.sampling_seed=778`: Set the random seed for the random sequences sampling.
- `language_model="roberta-base"`: Use the RoBERTa base model for embeddings.
- `embeddings.embedding_data_handler.mode="regular"`: Use the regular mode for the embeddings (other option is `masked_token` for masked language models).
- `embeddings_data_prep.sampling.num_samples=3000`: This many non-padding tokens are sampled from the sequences in the embeddings data preparation step.
- `local_estimates=twonn`: Compute the TwoNN-based local estimates (other options are `lpca` for local PCA based dimension estimates).
- `local_estimates.filtering.num_samples=500`: This many non-padding tokens are sampled from the tokens (i.e., set `N=500` as size of the token sub-sample).
- `local_estimates.pointwise.absolute_n_neighbors=128`: Use 128 neighbors for the pointwise local estimates (i.e., set `L=128` as the local neighborhood size).

The results will be saved in the `data_dir` specified in the config file, and the file path will contain the information about the model and the dataset used, together with additional hyperparameter choices.
For example, the results for the command above will be saved in the following directory:

```bash
data/analysis/local_estimates/data=wikitext-103-v1_strip-True_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=validation_samples=512_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=3000/desc=twonn_samples=500_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/
```

After the computation, this directory contains the following files:

```bash
.
├── additional_distance_computations_results.json
├── array_for_estimator.npy # <-- The array of vectors used for the local estimates computation (optional)
├── global_estimate.npy # <-- The global estimate (optional)
└── n-neighbors-mode=absolute_size_n-neighbors=128
    ├── additional_pointwise_results_statistics.json # <-- The statistics of the pointwise results
    ├── local_estimates_pointwise_array.npy # <-- The vector of pointwise local estimate results
    └── local_estimates_pointwise_meta.pkl # <-- The metadata of the vector of the pointwise computation
```

As a reference, you can also take a look at the file `.vscode/launch.json`, which contains various configurations for running the pipeline in different ways.
In the following sections, we will explain how to set up the experiments that we present in the paper.

### Experiments: Fine-Tuning Induces Dataset-Specific Shifts in Heterogeneous Local Dimensions

#### Fine-tuning the language model

TODO: Explain how to run these experiments.
TODO: Add instructions for finetuning the language model.

```bash
uv run finetune_language_model
```

#### Local estimates computation for the finetuned models

TODO: Explain how to compute local estimates for the finetuned models.
TODO: Explain how to call the violin plot creation script.

### Experiments: Local Dimensions Predict Grokking

Refer to the separate `grokking` repository for instructions on how to run these experiments.

### Experiments: Local Dimensions Detect Exhaustion of Training Capabilities

#### Train the Trippy-R dialogue state tracking models

TODO: Explain how to train the Trippy-R dialogue state tracking models.

To train the dialogue state tracking models for which we compute the local estimates, use the official [TripPy-R](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-r-public) codebase.


#### Local estimates computation for the Trippy-R models

- Update the model file paths in the config file `configs/language_model/roberta-base-trippy_r_multiwoz21_short_runs.yaml` to the location where you place the model files.

TODO: Explain how to compute local estimates for the Trippy-R models.
TODO: Explain how to create the plots comparing local dimensions and task performance for the Trippy-R models.

### Experiments: Local Dimensions Detect Overfitting

#### Train the ERC models

[ConvLab-3](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/dst/emodst/modeling)
TODO: Explain how to train the ERC models.

#### Local estimates computation for the ERC models


TODO: Explain how to compute local estimates for the ERC models.
TODO: Explain how to create the plots comparing local dimensions and task performance for the ERC models.

### Run tests

We provide a python script that can be called via a poetry run command to run the tests.

```bash
uv run tests
```
