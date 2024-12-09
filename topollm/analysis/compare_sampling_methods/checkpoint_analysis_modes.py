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

from dataclasses import dataclass, field
from itertools import product

import pandas as pd


@dataclass
class CheckpointAnalysisCombination:
    """Configuration for a single combination of checkpoint analysis."""

    data_full: str
    data_subsampling_split: str
    data_subsampling_sampling_mode: str
    embedding_data_handler_mode: str
    model_partial_name: str
    language_model_seed: int


@dataclass
class CheckpointAnalysisModes:
    """Configuration for different modes of checkpoint analysis."""

    data_full_list: list[str] = field(
        default_factory=lambda: [
            "data=multiwoz21_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
            "data=one-year-of-tsla-on-reddit_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        ],
    )
    data_subsampling_split_list: list[str] = field(
        default_factory=lambda: [
            "train",
            "validation",
            "test",
        ],
    )
    data_subsampling_sampling_mode_list: list[str] = field(
        default_factory=lambda: [
            "random",
            "take_first",
        ],
    )
    embedding_data_handler_mode_list: list[str] = field(
        default_factory=list,
    )
    model_partial_name_list: list[str] = field(
        default_factory=list,
    )
    # Note: The "model_seed" column contains type integer values
    language_model_seed_list: list[int] = field(
        default_factory=list,
    )

    def from_concatenated_df(
        self,
        concatenated_df: pd.DataFrame,
    ) -> None:
        """Populate fields that depend on concatenated_df."""
        self.embedding_data_handler_mode_list = concatenated_df["embedding_data_handler_mode"].unique().tolist()
        self.model_partial_name_list = concatenated_df["model_partial_name"].unique().tolist()
        self.language_model_seed_list = concatenated_df["model_seed"].unique().tolist()

    def all_combinations(
        self,
    ) -> list[CheckpointAnalysisCombination]:
        return [
            CheckpointAnalysisCombination(*combo)
            for combo in product(
                self.data_full_list,
                self.data_subsampling_split_list,
                self.data_subsampling_sampling_mode_list,
                self.embedding_data_handler_mode_list,
                self.model_partial_name_list,
                self.language_model_seed_list,
            )
        ]
