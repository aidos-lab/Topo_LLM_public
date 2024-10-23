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

import os

from dotenv import load_dotenv

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

load_dotenv()

# # # #
# The following options are inspired by the hydra framework
# for customizing the output directory.
# https://hydra.cc/docs/configure_hydra/workdir/
#
# key-value separator for paths
KV_SEP: str = "-"
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
    "aggregation": "agg",
    "add_prefix_space": "add-prefix-space",
    "batch_size": "bs",
    "batch_size_eval": "bs-eval",
    "batch_size_test": "bs-test",
    "batch_size_train": "bs-train",
    "center": "center",
    "context": "ctxt",
    "data": "data",
    "dataloader_desc": "dataloader",
    "deduplication_mode": "dedup",
    "description": "desc",
    "epoch": "ep",
    "feature_column_name": "feat-col",
    "FinetuningMode": "ftm",
    "global_step": "gs",
    "GradientModifierMode": "gradmod",
    "label_map_description": "labelmap",
    "layer": "layer",
    "learning_rate": "lr",
    "lr_scheduler_type": "lr_scheduler_type",
    "level": "lvl",
    "lora_alpha": "alpha",
    "lora_dropout": "lora-dropout",
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
    "seed": "seed",
    "split": "split",
    "target_modules_to_freeze": "target-freeze",
    "task_type": "task",
    "transformation": "trans",
    "use_canonical_values_from_dataset": "use-canonical-val",
    "use_rslora": "rslora",
    "weight_decay": "wd",
    "zero_vector_handling_mode": "zerovec",
}

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

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
