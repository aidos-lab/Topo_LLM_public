# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for setting global variables for the config files."""

import logging
import os

from dotenv import load_dotenv

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

load_dotenv()

# # # #
# The following options are inspired by the hydra framework
# for customizing the output directory.
# https://hydra.cc/docs/configure_hydra/workdir/
#
# key-value separator for paths
KV_SEP: str = "="
# item separator for paths
ITEM_SEP: str = "_"

# # # #
# This dictionary of prefixes allows us to
# easily change the prefixes for file paths and names
# in one place without modifying the functions itself,
# enhancing the maintainability of the code.
#
# Note: This should NOT come with key-value separators,
# which we keep separate to make them configurable.
NAME_PREFIXES: dict[
    str,
    str,
] = {
    "array_truncation_size": "a-tr-s",
    "add_prefix_space": "add-prefix-space",
    "aggregation": "agg",
    "attention_probs_dropout_prob": "attn-dr",
    "batch_size": "bs",
    "batch_size_eval": "bs-eval",
    "batch_size_test": "bs-test",
    "batch_size_train": "bs-train",
    "center": "center",
    "classifier_dropout": "clf-dr",
    "context": "ctxt",
    "data": "data",
    "dataloader_desc": "dataloader",
    "data_splitting_mode": "spl-mode",
    "data_filtering_remove_empty_sequences": "rm-empty",
    "deduplication_mode": "dedup",
    "description": "desc",
    "dropout_mode": "dr",
    "embedding_data_handler_mode": "edh-mode",
    "epoch": "ep",
    "feature_column_name": "feat-col",
    "FinetuningMode": "ftm",
    "global_step": "gs",
    "GradientModifierMode": "gradmod",
    "hidden_dropout_prob": "h-dr",
    "label_map_description": "labelmap",
    "layer": "layer",
    "learning_rate": "lr",
    "lr_scheduler_type": "lr-scheduler-type",
    "level": "lvl",
    "local_estimates_noise_artificial_noise_mode": "noise",
    "local_estimates_noise_distortion_parameter": "distor",
    "local_estimates_noise_seed": "seed",
    "lora_alpha": "alpha",
    "lora_dropout": "lora-dr",
    "lora_r": "r",
    "lora_target_modules": "lora-target",
    "metric": "metric",
    "model": "model",
    "model_parameters": "mparam",
    "masking_mode": "mask",
    "max_length": "max-len",
    "normalization": "norm",
    "number_of_samples": "samples",
    "num_samples": "samples",
    "n_neighbors": "n-neighbors",
    "n_neighbors_mode": "n-neighbors-mode",
    "query": "query",
    "sampling_mode": "sampling",
    "sampling_seed": "sampling-seed",
    "seed": "seed",
    "split": "split",
    "split_shuffle": "spl-shuf",
    "split_seed": "spl-seed",
    "target_modules_to_freeze": "target-freeze",
    "task_type": "task",
    "test_short": "te",
    "transformation": "trans",
    "train_short": "tr",
    "use_canonical_values_from_dataset": "use-canonical-val",
    "use_rslora": "rslora",
    "validation_short": "va",
    "weight_decay": "wd",
    "zero_vector_handling_mode": "zerovec",
}

# The values in this dictionary can be used to uniquely identify
# the full descriptions of the augmented descriptions,
# for example, for the checkpoints they also contain the prefix
# "model_" to make them unique.
# They will usually be used as column headers in the analysis dataframes.
NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS: dict[
    str,
    str,
] = {
    # Data parameters
    "rm-empty": "data_filtering_remove_empty_sequences",
    # Model parameters
    "model_ckpt": "model_checkpoint",
    "model_seed": "model_seed",
    # Dropout parameters
    "dr": "model_dropout_mode",
    "h-dr": "model_hidden_dropout_prob",
    "attn-dr": "model_attention_probs_dropout_prob",
    "clf-dr": "model_classifier_dropout",
    # Noise parameters
    "local_estimates_dedup": "local_estimates_deduplication",
    "local_estimates_desc": "local_estimates_description",
    "local_estimates_distor": "local_estimates_noise_distortion",
    "local_estimates_noise": "local_estimates_noise_artificial_noise_mode",
    "local_estimates_seed": "local_estimates_noise_seed",
    "local_estimates_samples": "local_estimates_samples",
    "local_estimates_zerovec": "local_estimates_zero_vector_handling_mode",
}

# # # #
# Limit for length of file names
FILE_NAME_TRUNCATION_LENGTH: int = 200

EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH: str = os.getenv(
    key="EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH",
    default="/Volumes/ruppik_external/research_data/Topo_LLM",
)
TOPO_LLM_REPOSITORY_BASE_PATH: str = os.path.expandvars(
    path=os.getenv(
        key="TOPO_LLM_REPOSITORY_BASE_PATH",
        default="${HOME}/git-source/Topo_LLM",
    ),
)
# Note that the 'data/' directory can be set in the hydra config
ZIM_TOPO_LLM_REPOSITORY_BASE_PATH: str = os.path.expandvars(
    path=os.getenv(
        key="ZIM_TOPO_LLM_REPOSITORY_BASE_PATH",
        default="/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM",
    ),
)

HYDRA_CONFIGS_BASE_PATH: str = f"{TOPO_LLM_REPOSITORY_BASE_PATH}/configs"
