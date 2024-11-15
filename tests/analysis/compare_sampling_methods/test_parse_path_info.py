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

import logging
import pprint

from topollm.path_management.parse_path_info import parse_path_info_full

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def compare_example_path_and_expected_result(
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

    logger.info(
        msg=f"result:\n{pprint.pformat(object=result)}",  # noqa: G004 - low overhead
    )

    # Check that result is a valid dictionary
    assert isinstance(  # noqa: S101 - pytest assertion
        result,
        dict,
    )

    # Assert that the result matches the expected result
    assert result == expected_result, (  # noqa: S101 - pytest assertion
        f"Parsing failed for {example_path = }\n"
        f"Expected:\n{pprint.pformat(object=expected_result)}\n"
        f"Got:\n{pprint.pformat(object=result)}"
    )


def test_parse_path_info_full_sampling_take_first(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_path_info_full function."""
    example_path_base_model_str: str = (
        "/Users/USER_NAME/git-source/Topo_LLM/"
        "data/analysis/twonn/"
        "data=multiwoz21_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/"
        "split=test_samples=2000_sampling=take_first/"
        "lvl=token/add-prefix-space=True_max-len=512/"
        "model=roberta-base_task=masked_lm/"
        "layer=-1_agg=mean/norm=None/"
        "sampling=random_seed=42_samples=100000/"
        "desc=twonn_samples=2500_zerovec=keep_dedup=array_deduplicator/"
        "n-neighbors-mode=absolute_size_n-neighbors=128/"
        "local_estimates_pointwise.npy"
    )

    expected_result: dict = {
        "aggregation": "mean",
        "data_context": "dataset_entry",
        "data_full": "data=multiwoz21_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data_dataset_name": "multiwoz21",
        "data_feature_column": "ner_tags",
        "data_prep_sampling_method": "random",
        "data_prep_sampling_samples": 100000,
        "data_prep_sampling_seed": 42,
        "data_subsampling_split": "test",
        "data_splitting_mode": "do_nothing",
        "data_subsampling_full": "split=test_samples=2000_sampling=take_first",
        "data_subsampling_number_of_samples": 2000,
        "data_subsampling_sampling_mode": "take_first",
        "data_subsampling_sampling_seed": None,
        "local_estimates_deduplication": "array_deduplicator",
        "local_estimates_desc_full": "desc=twonn_samples=2500_zerovec=keep_dedup=array_deduplicator",
        "local_estimates_description": "twonn",
        "local_estimates_samples": 2500,
        "local_estimates_zerovec": "keep",
        "model_checkpoint": None,
        "model_full": "model=roberta-base_task=masked_lm",
        "model_layer": -1,
        "model_partial_name": "model=roberta-base",
        "model_seed": None,
        "model_task": "masked_lm",
        "n_neighbors": 128,
        "n_neighbors_mode": "absolute_size",
        "normalization": "None",
    }

    compare_example_path_and_expected_result(
        example_path=example_path_base_model_str,
        expected_result=expected_result,
        logger=logger_fixture,
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
        "lvl=token/add-prefix-space=True_max-len=512/"
        "model=roberta-base_task=masked_lm/"
        "layer=-1_agg=mean/norm=None/"
        "sampling=random_seed=44_samples=100000/"
        "desc=twonn_samples=5000_zerovec=keep_dedup=array_deduplicator/"
        "n-neighbors-mode=absolute_size_n-neighbors=128/"
        "local_estimates_pointwise.npy"
    )

    expected_result: dict = {
        "aggregation": "mean",
        "data_context": "dataset_entry",
        "data_dataset_name": "one-year-of-tsla-on-reddit",  # TODO: This parsing currently does not work
        "data_feature_column": "ner_tags",
        "data_full": "data=one-year-of-tsla-on-reddit_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",  # TODO this parsing currently does not work
        "data_prep_sampling_method": "random",
        "data_prep_sampling_samples": 100000,
        "data_prep_sampling_seed": 44,
        "data_subsampling_split": "test",
        "data_splitting_mode": "proportions",  # TODO: This parsing currently does not work
        "data_subsampling_full": "split=test_samples=10000_sampling=random_sampling-seed=777",
        "data_subsampling_number_of_samples": 10000,
        "data_subsampling_sampling_mode": "random",
        "data_subsampling_sampling_seed": 777,
        "local_estimates_deduplication": "array_deduplicator",
        "local_estimates_desc_full": "desc=twonn_samples=5000_zerovec=keep_dedup=array_deduplicator",
        "local_estimates_description": "twonn",
        "local_estimates_samples": 5000,
        "model_checkpoint": None,
        "model_full": "model=roberta-base_task=masked_lm",
        "model_layer": -1,
        "model_partial_name": "model=roberta-base",
        "model_seed": None,
        "model_task": "masked_lm",
        "n_neighbors": 128,
        "n_neighbors_mode": "absolute_size",
        "normalization": "None",
        "local_estimates_zerovec": "keep",
    }

    compare_example_path_and_expected_result(
        example_path=example_path_base_model_str,
        expected_result=expected_result,
        logger=logger_fixture,
    )


def test_parse_path_info_full_for_finetuned_model(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_path_info_full function for finetuned models."""
    example_path_finetuned_model_str: str = (
        "/Users/USER_NAME/git-source/Topo_LLM/"
        "data/analysis/twonn/"
        "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags/"
        "lvl-token/add-prefix-space-True_max-len-512/"
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm/"
        "layer--1_agg-mean/norm-None/"
        "sampling-random_seed-47_samples-100000/"
        "desc-twonn_samples-2500_zerovec-keep_dedup-array_deduplicator/"
        "n-neighbors-mode-absolute_size_n-neighbors-256/"
        "local_estimates_pointwise.npy"
    )

    expected_result = {
        "aggregation": "mean",
        "data_context": "dataset_entry",
        "data_full": "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data_prep_sampling_method": "random",
        "data_prep_sampling_samples": 100000,
        "data_prep_sampling_seed": 47,
        "data_dataset_name": "one-year-of-tsla-on-reddit",
        "local_estimates_deduplication": "array_deduplicator",
        "data_feature_column": "ner_tags",
        "local_estimates_desc_full": "desc-twonn_samples-2500_zerovec-keep_dedup-array_deduplicator",
        "local_estimates_description": "twonn",
        "local_estimates_samples": 2500,
        "model_checkpoint": 400,
        "model_full": "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
        "model_layer": -1,
        "model_partial_name": "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50",
        "model_seed": 1234,
        "model_task": "masked_lm",
        "n_neighbors": 256,
        "neighbors_mode": "absolute",
        "normalization": "None",
        "samples": 10000,
        "split": "train",
        "local_estimates_zerovec": "keep",
    }

    # NOTE: This test has not been updated and implemented to the new format.
