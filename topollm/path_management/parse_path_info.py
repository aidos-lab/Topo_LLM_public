# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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
    """Parse the information from the path."""
    # Convert the path to a string
    path_str = str(
        object=path,
    )

    # Extract sampling information
    # Matches sampling method, seed, and number of samples, e.g.,
    # "sampling-random_seed-44_samples-20000"
    # - (\w+): Match one or more word characters for the sampling method.
    # - (\d+): Match one or more digits for the seed.
    # - (\d+): Match one or more digits for the number of samples.
    sampling_info: dict = {}
    sampling_match: re.Match[str] | None = re.search(
        pattern=r"sampling=(\w+)_seed=(\d+)_samples=(\d+)",
        string=path_str,
    )
    if sampling_match:
        sampling_info["data_prep_sampling_method"] = sampling_match.group(1)
        sampling_info["data_prep_sampling_seed"] = int(sampling_match.group(2))
        sampling_info["data_prep_sampling_samples"] = int(sampling_match.group(3))

    # Extract local estimates information
    # Matches description, samples, zerovec, and optional deduplication, e.g.,
    # "desc-twonn_samples-2500_zerovec-keep_dedup-array_deduplicator"
    # - (\w+): Match one or more word characters for the description.
    # - (\d+): Match one or more digits for the number of samples.
    # - ([a-zA-Z0-9]+): Match one or more alphanumeric characters for the zerovec.
    # - (?:_dedup-([a-zA-Z0-9_]+))?: Optionally match "_dedup-"
    #   followed by one or more alphanumeric or underscore characters for deduplication.
    local_estimates_info: dict = {}
    desc_match: re.Match[str] | None = re.search(
        pattern=r"desc=(\w+)_samples=(\d+)_zerovec=([a-zA-Z0-9]+)(?:_dedup=([a-zA-Z0-9_]+))?",
        string=path_str,
    )
    if desc_match:
        local_estimates_info["local_estimates_desc_full"] = desc_match.group(0)  # The full matched description string
        local_estimates_info["local_estimates_description"] = desc_match.group(1)
        local_estimates_info["local_estimates_samples"] = int(desc_match.group(2))
        local_estimates_info["local_estimates_zerovec"] = desc_match.group(3)
        local_estimates_info["local_estimates_deduplication"] = desc_match.group(4) if desc_match.group(4) else None

    # Extract neighbors information
    # Matches neighbors mode and number of neighbors, e.g.,
    # "n-neighbors-mode-absolute_size_n-neighbors-256"
    # - ([a-zA-Z0-9]+): Match one or more alphanumeric characters for the neighbors mode.
    # - (\d+): Match one or more digits for the number of neighbors.
    neighbors_info: dict = {}
    neighbors_match: re.Match[str] | None = re.search(
        pattern=r"n-neighbors-mode=([a-zA-Z0-9_]+)_n-neighbors=(\d+)",
        string=path_str,
    )
    if neighbors_match:
        neighbors_info["n_neighbors_mode"] = neighbors_match.group(1)
        neighbors_info["n_neighbors"] = int(neighbors_match.group(2))

    # Extract model information
    model_info: dict = parse_model_info(
        path=path_str,
    )

    # Extract layer and aggregation information
    # Matches layer index, aggregation type, and normalization, e.g.,
    # "layer--1_agg-mean/norm-None"
    # - (-?\d+): Match an optional negative sign followed by one or more digits for the layer index.
    # - ([\w-]+): Match one or more word characters or hyphens for the aggregation type.
    # - ([\w-]+): Match one or more word characters or hyphens for the normalization type.
    layer_info: dict = {}
    layer_match: re.Match[str] | None = re.search(
        pattern=r"layer=(-?\d+)_agg=([\w-]+)/norm=([\w-]+)",
        string=path_str,
    )
    if layer_match:
        layer_info["model_layer"] = int(layer_match.group(1))
        layer_info["aggregation"] = layer_match.group(2)
        layer_info["normalization"] = str(object=layer_match.group(3))

    # Extract data information
    data_info: dict = parse_data_info(
        path_str=path_str,
    )

    # Extract data subsampling information
    data_subsampling_info: dict = parse_data_subsampling_info(
        path_str=path_str,
    )

    # Assemble the parsed information
    parsed_info: dict = {
        **sampling_info,
        **local_estimates_info,
        **neighbors_info,
        **model_info,
        **layer_info,
        **data_info,
        **data_subsampling_info,
    }

    return parsed_info


def parse_data_info(
    path_str: str,
) -> dict[str, str | int]:
    """Parse the data information from the given path."""
    parsed_info = {}

    # Matches and parses data information including dataset name, split, context, number of samples, and feature column,
    # e.g., "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags"
    # - data=(\w+): Match the dataset name.
    # - spl-mode=([a-zA-Z0-9_]+): Match the data splitting mode.
    # - ctxt=([a-zA-Z0-9_]+): Match the context type (e.g., "dataset_entry").
    # - feat-col=([a-zA-Z0-9_]+): Match the feature column.
    data_match: re.Match[str] | None = re.search(
        pattern=r"data=([\w-]+)_spl-mode=([a-zA-Z0-9_]+)_ctxt=([a-zA-Z0-9_]+)_feat-col=([a-zA-Z0-9_]+)",
        string=path_str,
    )
    if data_match:
        parsed_info["data_full"] = data_match.group(0)  # The full matched data string
        parsed_info["data_dataset_name"] = data_match.group(1)
        parsed_info["data_splitting_mode"] = data_match.group(2)
        # TODO: This does not include additional data splitting information yet
        parsed_info["data_context"] = data_match.group(3)
        parsed_info["data_feature_column"] = data_match.group(4)

    return parsed_info


def parse_data_subsampling_info(
    path_str: str,
) -> dict:
    """Parse the data subsampling information from the given path."""
    parsed_info = {}

    # e.g.
    # > "split=test_samples=2000_sampling=take_first"
    # > "split=test_samples=10000_sampling=random_sampling-seed=777"

    subsampling_match = re.search(
        pattern=r"split=(\w+)_samples=(\d+)_sampling=(take_first|random)(?:_sampling-seed=(\d+))?",
        string=path_str,
    )
    if subsampling_match:
        parsed_info["data_subsampling_full"] = subsampling_match.group(0)  # The full matched string
        parsed_info["data_subsampling_split"] = subsampling_match.group(1)
        parsed_info["data_subsampling_number_of_samples"] = int(subsampling_match.group(2))
        parsed_info["data_subsampling_sampling_mode"] = subsampling_match.group(3)
        parsed_info["data_subsampling_sampling_seed"] = (
            int(subsampling_match.group(4)) if subsampling_match.group(4) else None
        )

    return parsed_info


def parse_model_info(
    path: str | pathlib.Path,
) -> dict[
    str,
    str | int | None,
]:
    """Parse the model information from the given path.

    Extracts the model name, optional seed, optional checkpoint, and the full model section.

    Args:
        path: Path string from which to extract model information.

    Returns:
        Dictionary containing the parsed model information.

    """
    # Convert the path to a string
    path_str = str(object=path)

    # Initialize an empty dictionary to hold parsed values
    parsed_info: dict[str, str | int | None] = {}

    # Use pathlib to handle different path delimiters
    path_parts = pathlib.PurePath(path_str).parts

    # Find the segment containing the model information
    model_segment = None
    for segment in path_parts:
        if segment.startswith("model="):
            model_segment = segment
            break

    if model_segment:
        # Store the full model section
        parsed_info["model_full"] = model_segment

        # Start from the end and remove optional components (task, checkpoint, seed)
        model_name = model_segment

        # Remove task if present (start from the end)
        task_match = re.search(r"_task=([\w-]+)$", model_name)
        if task_match:
            parsed_info["model_task"] = task_match.group(1)
            model_name = model_name[: task_match.start()]
        else:
            parsed_info["model_task"] = None

        # Note: For the checkpoint and seed
        # we still use the old '-' key-value separator,
        # because this still appears in the model names.

        # Remove checkpoint if present (after removing task)
        ckpt_match = re.search(r"_ckpt-(\d+)$", model_name)
        if ckpt_match:
            parsed_info["model_checkpoint"] = int(ckpt_match.group(1))
            model_name = model_name[: ckpt_match.start()]
        else:
            parsed_info["model_checkpoint"] = None

        # Remove seed if present (after removing checkpoint)
        seed_match = re.search(r"_seed-(\d+)$", model_name)
        if seed_match:
            parsed_info["model_seed"] = int(seed_match.group(1))
            model_name = model_name[: seed_match.start()]
        else:
            parsed_info["model_seed"] = None

        # Store the final model name
        parsed_info["model_partial_name"] = model_name

    return parsed_info
