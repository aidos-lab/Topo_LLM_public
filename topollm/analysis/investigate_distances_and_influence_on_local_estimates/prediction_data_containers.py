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

import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class LMHeadPredictionResults:
    """Container for the results of the LM head predictions."""

    output_logits_softmax_np: np.ndarray | None
    top_k_tokens: list[str]
    top_k_probabilities: list[float]
    loss: float | None

    actual_token_id: int | None = None
    actual_token_name: str | None = None


@dataclass
class LocalEstimatesAndPredictionsSavePathCollection:
    """Dataclass to hold the paths for saving and loading the results."""

    # This is the base directory under which all the other files will be placed
    distances_and_influence_on_local_estimates_dir_absolute_path: pathlib.Path

    descriptive_statistics_dict_save_path: pathlib.Path
    correlations_df_save_path: pathlib.Path

    @staticmethod
    def from_base_directory(
        distances_and_influence_on_local_estimates_dir_absolute_path: pathlib.Path,
    ) -> "LocalEstimatesAndPredictionsSavePathCollection":
        """Create a new instance of the dataclass from the base directory."""
        descriptive_statistics_dict_save_path: pathlib.Path = pathlib.Path(
            distances_and_influence_on_local_estimates_dir_absolute_path,
            "descriptive_statistics_dict.json",
        )

        correlations_df_save_path: pathlib.Path = pathlib.Path(
            distances_and_influence_on_local_estimates_dir_absolute_path,
            "correlations_df.csv",
        )

        return LocalEstimatesAndPredictionsSavePathCollection(
            distances_and_influence_on_local_estimates_dir_absolute_path=distances_and_influence_on_local_estimates_dir_absolute_path,
            correlations_df_save_path=correlations_df_save_path,
            descriptive_statistics_dict_save_path=descriptive_statistics_dict_save_path,
        )

    def setup_directories(
        self,
    ) -> None:
        for path in [
            self.distances_and_influence_on_local_estimates_dir_absolute_path,
            self.descriptive_statistics_dict_save_path,
        ]:
            # Create the directories if they do not exist
            if not path.parent.exists():
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )


@dataclass
class LocalEstimateAndPrediction:
    """Container for the local estimate and the corresponding prediction results."""

    vector_index: int

    extracted_local_estimate: float
    lm_head_prediction_results: LMHeadPredictionResults

    extracted_metadata_reduced: dict | None = None
    original_prepared_data_index: int | None = None
