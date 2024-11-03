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

"""Functions for parsing information contained in file paths."""

import pathlib
import re


def parse_path_info_full(
    path: str | pathlib.Path,
) -> dict[str, str | int]:
    """Parse the information from the path.

    Example path:
    /Users/USER_NAME/git-source/Topo_LLM/
    data/
    analysis/
    twonn/
    data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags/
    lvl-token/
    add-prefix-space-True_max-len-512/
    model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm/
    layer--1_agg-mean/
    norm-None/
    sampling-take_first_seed-42_samples-30000/
    desc-twonn_samples-2500_zerovec-keep_dedup-array_deduplicator/
    n-neighbors-mode-absolute_size_n-neighbors-128/
    local_estimates_pointwise.npy
    """
    # Convert the path to a string
    path_str = str(path)

    # Initialize an empty dictionary to hold parsed values
    parsed_info: dict = {}

    # Extract sampling information
    sampling_match = re.search(
        pattern=r"sampling-(\w+)_seed-(\d+)_samples-(\d+)",
        string=path_str,
    )
    if sampling_match:
        parsed_info["data_prep_sampling_method"] = sampling_match.group(1)
        parsed_info["data_prep_sampling_seed"] = int(sampling_match.group(2))
        parsed_info["data_prep_sampling_samples"] = int(sampling_match.group(3))

    # Extract local estimates information
    # Comments on Regex:
    # Capture `zerovec` completely and correctly handle `dedup` as an optional field
    desc_match = re.search(
        pattern=r"desc-(\w+)_samples-(\d+)_zerovec-([a-zA-Z0-9]+)(?:_dedup-([a-zA-Z0-9_]+))?",
        string=path_str,
    )
    if desc_match:
        parsed_info["local_estimates_description"] = desc_match.group(1)
        parsed_info["local_estimates_samples"] = int(desc_match.group(2))
        parsed_info["zerovec"] = desc_match.group(3)
        parsed_info["deduplication"] = desc_match.group(4) if desc_match.group(4) else None

    # Extract neighbors information
    neighbors_match = re.search(
        pattern=r"n-neighbors-mode-(\w+)_size_n-neighbors-(\d+)",
        string=path_str,
    )
    if neighbors_match:
        parsed_info["neighbors_mode"] = neighbors_match.group(1)
        parsed_info["n_neighbors"] = int(neighbors_match.group(2))

    # Extract model information
    model_match = re.search(
        pattern=r"model-(\w+-[\w-]+)_seed-(\d+)_ckpt-(\d+)",
        string=path_str,
    )
    if model_match:
        parsed_info["model_name"] = model_match.group(1)
        parsed_info["model_seed"] = int(model_match.group(2))
        parsed_info["checkpoint"] = int(model_match.group(3))

    # Extract layer and aggregation information
    layer_match = re.search(
        pattern=r"layer-(\w+)_agg-(\w+)/norm-(\w+)",
        string=path_str,
    )
    if layer_match:
        parsed_info["model_layer"] = int(layer_match.group(1))
        parsed_info["aggregation"] = layer_match.group(2)
        parsed_info["normalization"] = layer_match.group(3)

    return parsed_info
