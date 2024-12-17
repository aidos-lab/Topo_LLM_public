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

"""Tests for the path management module."""

import logging
import pathlib
import pprint

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS
from topollm.path_management.parse_path_info import (
    parse_data_info,
    parse_local_estimates_info,
    parse_model_info,
    parse_path_info_full,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compare_result_for_example_path_and_expected_result(
    example_path: str,
    expected_result: dict,
    logger: logging.Logger = default_logger,
) -> None:
    """Compare the result of parse_path_info_full with the expected result."""
    logger.info(
        msg=f"{example_path = }",  # noqa: G004 - low overhead
    )

    result: dict = parse_path_info_full(
        path=example_path,
    )

    compare_result_and_expected_result(
        result=result,
        expected_result=expected_result,
        logger=logger,
    )


def compare_result_and_expected_result(
    result: dict,
    expected_result: dict | None = None,
    logger: logging.Logger = default_logger,
) -> None:
    """Compare the result of parse_path_info_full with the expected result."""
    logger.info(
        msg=f"result:\n{pprint.pformat(object=result)}",  # noqa: G004 - low overhead
    )

    # If expected result is not provided, return without further checks
    if expected_result is None:
        return

    # Check that result is a valid dictionary
    assert isinstance(  # noqa: S101 - pytest assertion
        result,
        dict,
    )

    # Assert that the result matches the expected result
    assert result == expected_result, (  # noqa: S101 - pytest assertion
        f"Parsing failed.\n"
        f"Expected:\n{pprint.pformat(object=expected_result)}\n"
        f"Got:\n{pprint.pformat(object=result)}"
    )


def test_parse_path_info_full_sampling_random(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_path_info_full function."""
    example_path_base_model_str: str = (
        "/Users/USER_NAME/git-source/Topo_LLM/"
        "data/analysis/twonn/"
        "data=one-year-of-tsla-on-reddit_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/"
        "split=test_samples=10000_sampling=random_sampling-seed=777/"
        "lvl=token/"
        "add-prefix-space=True_max-len=512/"
        "model=roberta-base_task=masked_lm/"
        "layer=-1_agg=mean/"
        "norm=None/"
        "sampling=random_seed=44_samples=100000/"
        "desc=twonn_samples=5000_zerovec=keep_dedup=array_deduplicator/"
        "n-neighbors-mode=absolute_size_n-neighbors=128/"
        "local_estimates_pointwise.npy"
    )

    expected_result: dict = {
        "aggregation": "mean",
        "data_context": "dataset_entry",
        "data_dataset_name": "one-year-of-tsla-on-reddit",
        "data_feature_column": "ner_tags",
        "data_full": "data=one-year-of-tsla-on-reddit_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        "data_filtering_remove_empty_sequences": None,
        "data_prep_sampling_method": "random",
        "data_prep_sampling_samples": 100000,
        "data_prep_sampling_seed": 44,
        "data_subsampling_split": "test",
        "data_splitting_mode": "proportions",
        "data_subsampling_full": "split=test_samples=10000_sampling=random_sampling-seed=777",
        "data_subsampling_number_of_samples": 10000,
        "data_subsampling_sampling_mode": "random",
        "data_subsampling_sampling_seed": 777,
        NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]: "array_deduplicator",
        "local_estimates_desc_full": "desc=twonn_samples=5000_zerovec=keep_dedup=array_deduplicator",
        "local_estimates_description": "twonn",
        "local_estimates_noise_artificial_noise_mode": None,
        "local_estimates_noise_distortion": None,
        "local_estimates_noise_seed": None,
        "local_estimates_zero_vector_handling_mode": "keep",
        "local_estimates_samples": 5000,
        NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["model_ckpt"]: None,
        "model_full": "model=roberta-base_task=masked_lm",
        "model_layer": -1,
        "model_partial_name": "model=roberta-base",
        "model_seed": None,
        "model_task": "masked_lm",
        "model_attention_probs_dropout_prob": None,
        "model_classifier_dropout": None,
        "model_dropout_mode": None,
        "model_hidden_dropout_prob": None,
        "n_neighbors": 128,
        "n_neighbors_mode": "absolute_size",
        "normalization": "None",
    }

    compare_result_for_example_path_and_expected_result(
        example_path=example_path_base_model_str,
        expected_result=expected_result,
        logger=logger_fixture,
    )


def test_parse_model_info(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_model_info function."""
    # # # # # # # #
    # Test case 1:
    # Default dropout rate parameters
    example_path_str = pathlib.Path(
        "example_prefix",
        "model=model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-13200_task=masked_lm_dr=defaults",
        "example_suffix",
    )

    expected_result: dict = {
        "model_attention_probs_dropout_prob": None,
        "model_checkpoint": 13200,
        "model_classifier_dropout": None,
        "model_dropout_mode": "defaults",
        "model_full": "model=model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-13200_task=masked_lm_dr=defaults",
        "model_hidden_dropout_prob": None,
        "model_partial_name": "model=model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
        "model_seed": 1234,
        "model_task": "masked_lm",
    }

    model_info: dict = parse_model_info(
        path=example_path_str,
    )

    logger_fixture.info(
        msg=f"example_path_str:\n{example_path_str}",  # noqa: G004 - low overhead
    )
    logger_fixture.info(
        msg=f"model_info:\n{pprint.pformat(object=model_info)}",  # noqa: G004 - low overhead
    )

    # Check that result is a valid dictionary
    assert isinstance(  # noqa: S101 - pytest assertion
        model_info,
        dict,
    )

    # Assert that the result matches the expected result
    assert model_info == expected_result, (  # noqa: S101 - pytest assertion
        f"Parsing failed for {example_path_str = }\n"
        f"Expected:\n{pprint.pformat(object=expected_result)}\n"
        f"Got:\n{pprint.pformat(object=model_info)}"
    )

    # # # # # # # #
    # Test case 2:
    # Modified dropout rate parameters

    example_path_str = pathlib.Path(
        "example_prefix",
        "model=roberta-base-masked_lm-0.05-0.05-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50_seed-1234_ckpt-8800_task=masked_lm_dr=modify_roberta_dropout_parameters_h-dr=0.05_attn-dr=0.05_clf-dr=None",
        "example_suffix",
    )

    expected_result: dict = {
        "model_attention_probs_dropout_prob": "0.05",
        "model_checkpoint": 8800,
        "model_classifier_dropout": "None",
        "model_dropout_mode": "modify_roberta_dropout_parameters",
        "model_full": "model=roberta-base-masked_lm-0.05-0.05-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50_seed-1234_ckpt-8800_task=masked_lm_dr=modify_roberta_dropout_parameters_h-dr=0.05_attn-dr=0.05_clf-dr=None",
        "model_hidden_dropout_prob": "0.05",
        "model_partial_name": "model=roberta-base-masked_lm-0.05-0.05-None_multiwoz21-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-constant-0.01-50",
        "model_seed": 1234,
        "model_task": "masked_lm",
    }

    model_info: dict = parse_model_info(
        path=example_path_str,
    )

    logger_fixture.info(
        msg=f"example_path_str:\n{example_path_str}",  # noqa: G004 - low overhead
    )
    logger_fixture.info(
        msg=f"model_info:\n{pprint.pformat(object=model_info)}",  # noqa: G004 - low overhead
    )

    # Check that result is a valid dictionary
    assert isinstance(  # noqa: S101 - pytest assertion
        model_info,
        dict,
    )

    # Assert that the result matches the expected result
    assert model_info == expected_result, (  # noqa: S101 - pytest assertion
        f"Parsing failed for {example_path_str = }\n"
        f"Expected:\n{pprint.pformat(object=expected_result)}\n"
        f"Got:\n{pprint.pformat(object=model_info)}"
    )


def test_parse_data_info(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_data_info function."""
    # # # # # # # #
    # Test case 1:
    # With data filtering description
    example_path_str = pathlib.Path(
        "example_prefix",
        "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        "split=validation_samples=10000_sampling=random_sampling-seed=780",
        "example_suffix",
    )

    expected_result: dict = {
        "data_context": "dataset_entry",
        "data_dataset_name": "wikitext-103-v1",
        "data_feature_column": "ner_tags",
        "data_filtering_remove_empty_sequences": "True",
        "data_full": "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        "data_splitting_mode": "proportions",
    }

    data_info: dict = parse_data_info(
        path=example_path_str,
    )

    logger_fixture.info(
        msg=f"example_path_str:\n{example_path_str}",  # noqa: G004 - low overhead
    )
    logger_fixture.info(
        msg=f"data_info:\n{pprint.pformat(object=data_info)}",  # noqa: G004 - low overhead
    )

    # Check that result is a valid dictionary
    assert isinstance(  # noqa: S101 - pytest assertion
        data_info,
        dict,
    )

    # Assert that the result matches the expected result
    assert data_info == expected_result, (  # noqa: S101 - pytest assertion
        f"Parsing failed for {example_path_str = }\n"
        f"Expected:\n{pprint.pformat(object=expected_result)}\n"
        f"Got:\n{pprint.pformat(object=data_info)}"
    )


def test_parse_local_estimates_info(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of the parse_local_estimates_info function."""
    instances_and_expected_results: list[tuple] = [
        # # # # # # # #
        # Test case 1
        (
            pathlib.Path(
                "example_prefix",
                "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing",
                "example_suffix",
            ),
            {
                NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]: "array_deduplicator",
                "local_estimates_desc_full": "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing",
                "local_estimates_description": "twonn",
                "local_estimates_noise_artificial_noise_mode": "do_nothing",
                "local_estimates_noise_distortion": None,
                "local_estimates_noise_seed": None,
                "local_estimates_samples": 60000,
                "local_estimates_zero_vector_handling_mode": "keep",
            },
        ),
        # # # # # # # #
        # Test case 2:
        (
            pathlib.Path(
                "example_prefix",
                "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=gaussian_distor=0.002_seed=6",
                "example_suffix",
            ),
            {
                "local_estimates_deduplication": "array_deduplicator",
                "local_estimates_desc_full": "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=gaussian_distor=0.002_seed=6",
                "local_estimates_description": "twonn",
                "local_estimates_noise_artificial_noise_mode": "gaussian",
                "local_estimates_noise_distortion": 0.002,
                "local_estimates_noise_seed": 6,
                "local_estimates_samples": 60000,
                "local_estimates_zero_vector_handling_mode": "keep",
            },
        ),
    ]

    for example_path_str, expected_result in instances_and_expected_results:
        local_estimates_info: dict = parse_local_estimates_info(
            path=example_path_str,
        )

        compare_result_and_expected_result(
            result=local_estimates_info,
            expected_result=expected_result,
            logger=logger_fixture,
        )
