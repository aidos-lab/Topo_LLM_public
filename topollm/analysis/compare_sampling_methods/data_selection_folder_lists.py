# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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

"""Variables to describe the different data selections."""


def get_data_folder_list() -> list[str]:
    data_folder_list: list[str] = [
        "data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-test_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
    ]

    return data_folder_list


def get_model_folder_list() -> list[str]:
    model_folder_list: list[str] = [
        "model-roberta-base_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
    ]

    return model_folder_list
