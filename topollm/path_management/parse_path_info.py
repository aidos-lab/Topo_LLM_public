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
from collections.abc import Callable
from dataclasses import dataclass

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS


@dataclass
class ParsingRule:
    """Encapsulate information required for parsing a specific component from the path part string."""

    pattern: str
    key: str
    cast_fn: Callable[
        [str],
        str | int | None,
    ]


def extract_and_update_in_place_and_truncate(
    rule: ParsingRule,
    target: str,
    parsed_info: dict[str, str | int | None],
) -> str:
    """Extract a value from the target string using a ParsingRule.

    Updates the parsed_info dictionary in place, and returns the truncated string.

    Args:
        rule:
            An instance of ParsingRule defining the pattern, key, and cast function.
        target:
            String to search and truncate.
        parsed_info:
            Dictionary to store extracted values.

    Returns:
        Truncated string with the matched part removed.

    """
    match: re.Match[str] | None = re.search(
        pattern=rule.pattern,
        string=target,
    )
    if match:
        # Extract the value and store it in the parsed_info dictionary
        parsed_info[rule.key] = rule.cast_fn(
            match.group(1),
        )
        # Return the truncated string without the matched part
        return target[: match.start()]

    # If no match was found, set the value to None
    parsed_info[rule.key] = None
    return target


def parse_path_info_full(
    path: str | pathlib.Path,
) -> dict[str, str | int]:
    """Parse the information from the path."""
    # Convert the path to a string
    path_str = str(
        object=path,
    )

    # Extract data information
    data_info: dict = parse_data_info(
        path=path_str,
    )

    # Extract data subsampling information
    data_subsampling_info: dict = parse_data_subsampling_info(
        path=path_str,
    )

    # Extract tokenizer information
    tokenizer_info: dict = parse_tokenizer_info(
        path_str=path_str,
    )

    # Extract embedding data handler information
    embedding_data_handler_info: dict = parse_embedding_data_handler_info(
        path_str=path_str,
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

    # Extract local estimates information
    local_estimates_info: dict = parse_local_estimates_info(
        path=path_str,
    )

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

    # Assemble the parsed information
    parsed_info: dict = {
        **data_info,
        **data_subsampling_info,
        **tokenizer_info,
        **embedding_data_handler_info,
        **sampling_info,
        **local_estimates_info,
        **neighbors_info,
        **model_info,
        **layer_info,
    }

    return parsed_info


def parse_local_estimates_info(
    path: str | pathlib.Path,
) -> dict[str, str | int]:
    """Parse the local estimates information from the given path."""
    path_str = str(
        object=path,
    )

    # Extract local estimates information
    # Matches description, samples, zerovec, and optional deduplication, e.g.,
    # "desc-twonn_samples-2500_zerovec-keep_dedup-array_deduplicator"
    # - (\w+): Match one or more word characters for the description.
    # - (\d+): Match one or more digits for the number of samples.
    # - ([a-zA-Z0-9]+): Match one or more alphanumeric characters for the zerovec.
    # - (?:_dedup-([a-zA-Z0-9_]+))?: Optionally match "_dedup-"
    #   followed by one or more alphanumeric or underscore characters for deduplication.
    local_estimates_info: dict = {}

    local_estimates_info_pattern = (
        r"desc=(\w+)_samples=(\d+)"
        r"_zerovec=([a-zA-Z0-9]+)"
        r"(?:_dedup=(do_nothing|array_deduplicator))?"
        r"(?:_noise=(do_nothing|gaussian))?"
        r"(?:_distor=([\d.]+))?"
        r"(?:_seed=(\d+))?"
    )
    desc_match: re.Match[str] | None = re.search(
        pattern=local_estimates_info_pattern,
        string=path_str,
    )
    if desc_match:
        local_estimates_info["local_estimates_desc_full"] = desc_match.group(0)  # The full matched description string
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_desc"]] = desc_match.group(1)
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_samples"]] = int(
            desc_match.group(2),
        )
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_zerovec"]] = (
            desc_match.group(3)
        )
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]] = (
            desc_match.group(4) if desc_match.group(4) else None
        )
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_noise"]] = (
            str(desc_match.group(5)) if desc_match.group(5) else None
        )
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_distor"]] = (
            float(desc_match.group(6)) if desc_match.group(6) else None
        )
        local_estimates_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_seed"]] = (
            int(desc_match.group(7)) if desc_match.group(7) else None
        )

    return local_estimates_info


def parse_data_info(
    path: str | pathlib.Path,
) -> dict[str, str | int]:
    """Parse the data information from the given path."""
    path_str = str(
        object=path,
    )

    parsed_info: dict = {}

    # Matches and parses data information including dataset name, split, context, number of samples, and feature column,
    # e.g., "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags"
    # - data=(\w+): Match the dataset name.
    # - spl-mode=([a-zA-Z0-9_]+): Match the data splitting mode.
    # - ctxt=([a-zA-Z0-9_]+): Match the context type (e.g., "dataset_entry").
    # - feat-col=([a-zA-Z0-9_]+): Match the feature column.
    regex = (
        r"data=([\w-]+)"  # Dataset name
        r"(?:_rm-empty=([a-zA-Z0-9_]+))?"  # Optional remove empty
        r"_spl-mode=([a-zA-Z0-9_]+)"  # Splitting mode
        r"(?:_spl-[a-zA-Z0-9_=.-]+)*"  # Optional splitting parameters
        r"_ctxt=([a-zA-Z0-9_]+)"  # Context
        r"_feat-col=([a-zA-Z0-9_]+)"  # Feature column
    )
    data_match: re.Match[str] | None = re.search(
        pattern=regex,
        string=path_str,
    )
    if data_match:
        parsed_info["data_full"] = data_match.group(0)  # Full matched string
        parsed_info["data_dataset_name"] = data_match.group(1)
        parsed_info[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["rm-empty"]] = data_match.group(2)
        parsed_info["data_splitting_mode"] = data_match.group(3)
        parsed_info["data_context"] = data_match.group(4)
        parsed_info["data_feature_column"] = data_match.group(5)
    return parsed_info


def parse_data_subsampling_info(
    path: str | pathlib.Path,
) -> dict:
    """Parse the data subsampling information from the given path."""
    path_str = str(
        object=path,
    )

    parsed_info: dict = {}

    # e.g.
    # > "split=test_samples=2000_sampling=take_first"
    # > "split=test_samples=10000_sampling=random_sampling-seed=777"

    subsampling_match: re.Match[str] | None = re.search(
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


def parse_tokenizer_info(
    path_str: str,
) -> dict:
    """Parse the tokenizer information from the given path."""
    parsed_info: dict = {}

    # e.g.
    # > "add-prefix-space=False_max-len=512"
    # > "add-prefix-space=True_max-len=512
    #
    matched_info: re.Match[str] | None = re.search(
        pattern=r"add-prefix-space=(False|True)_max-len=(\d+)",
        string=path_str,
    )
    if matched_info:
        parsed_info["tokenizer_full"] = matched_info.group(0)
        parsed_info["tokenizer_add_prefix_space"] = matched_info.group(1)
        parsed_info["tokenizer_max_len"] = int(matched_info.group(2))

    return parsed_info


def parse_embedding_data_handler_info(
    path_str: str,
) -> dict:
    """Parse the embedding data handler information from the given path."""
    parsed_info: dict = {}

    # e.g.
    # > "edh-mode=masked_token_lvl=token"
    # > "edh-mode=regular_lvl=token"

    match: re.Match[str] | None = re.search(
        pattern=r"edh-mode=(masked_token|regular)_lvl=(token|word)",
        string=path_str,
    )
    if match:
        parsed_info["embedding_data_handler_full"] = match.group(0)  # The full matched string
        parsed_info["embedding_data_handler_mode"] = match.group(1)
        parsed_info["embedding_data_handler_level"] = match.group(2)

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
    path_str = str(
        object=path,
    )

    # Initialize an empty dictionary to hold parsed values
    parsed_info: dict[
        str,
        str | int | None,
    ] = {}

    # Use pathlib to handle different path delimiters
    path_parts: tuple = pathlib.PurePath(path_str).parts

    # Find the segment containing the model information
    model_segment = None
    for segment in path_parts:
        if segment.startswith(
            "model=",
        ):
            model_segment = segment
            break

    if model_segment is None:
        # Early return if no model segment was found
        return parsed_info

    # If model segment was found, parse the model information

    # Store the full model section
    parsed_info["model_full"] = model_segment

    # # # #
    # Start from the end and remove optional components,
    # i.e., dropout parameters, task, checkpoint, seed, ...
    model_name = model_segment

    # Define parsing rules
    rules: list[ParsingRule] = [
        # > Classification dropout rate
        ParsingRule(
            pattern=r"_clf-dr=([\wd.-]+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["clf-dr"],
            cast_fn=str,
        ),
        # > Attention dropout rate
        # The '.' is necessary to match the decimal point in the dropout rate.
        ParsingRule(
            pattern=r"_attn-dr=([\wd.-]+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["attn-dr"],
            cast_fn=str,
        ),
        # > Hidden dropout rate
        # The '.' is necessary to match the decimal point in the dropout rate.
        ParsingRule(
            pattern=r"_h-dr=([\wd.-]+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["h-dr"],
            cast_fn=str,
        ),
        # > Dropout mode
        ParsingRule(
            pattern=r"_dr=([\w-]+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["dr"],
            cast_fn=str,
        ),
        # > Task description
        ParsingRule(
            pattern=r"_task=([\w-]+)$",
            key="model_task",
            cast_fn=str,
        ),
        # Note: For the checkpoint and seed
        # we use the short_description_separator '-',
        # because this appears in the model names.
        # We avoid the '=' sign here because of clashes with the hydra overrides.
        #
        # > Checkpoint number
        ParsingRule(
            pattern=r"_ckpt-(\d+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["model_ckpt"],
            cast_fn=int,
        ),
        # > Model seed
        ParsingRule(
            pattern=r"_seed-(\d+)$",
            key=NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["model_seed"],
            cast_fn=int,
        ),
    ]

    # Iterate over rules to extract and truncate.
    # Note that parsed_info is updated in-place.
    for rule in rules:
        model_name: str = extract_and_update_in_place_and_truncate(
            rule=rule,
            target=model_name,
            parsed_info=parsed_info,
        )

    # Store the final truncated model name
    parsed_info["model_partial_name"] = model_name

    return parsed_info
