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


def test_parse_path_info_full(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of parse_path_info_full function."""
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

    example_path_str_list: list[str] = [
        example_path_finetuned_model_str,
    ]

    # TODO: Implement the test
