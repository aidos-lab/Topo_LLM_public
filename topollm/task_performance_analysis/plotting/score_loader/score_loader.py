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

"""Score loader for model performance metrics."""

import pathlib
from typing import Protocol

import pandas as pd

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


class ScoreLoader(Protocol):
    """Protocol for loading model performance metrics."""

    def get_scores(self) -> pd.DataFrame:
        """Return loaded scores as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns for epochs, dataset splits, and associated metrics.

        """
        ...  # pragma: no cover

    def get_columns_to_plot(self) -> list[str]:
        """Return the columns to plot.

        Returns:
            list[str]: List of column names to plot.

        """
        ...  # pragma: no cover


class EmotionClassificationScoreLoader:
    """Concrete implementation loading emotion classification model scores from a JSON file."""

    def __init__(
        self,
        filepath: pathlib.Path,
    ) -> None:
        """Initialize the loader with a file path."""
        self.filepath: pathlib.Path = filepath

    def get_scores(
        self,
    ) -> pd.DataFrame:
        with self.filepath.open(
            mode="r",
        ) as file:
            loaded_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=file)

        post_processed_loaded_df = self.post_process_loaded_df(
            loaded_df=loaded_df,
        )

        return post_processed_loaded_df

    def get_columns_to_plot(
        self,
    ) -> list[str]:
        # The dataframe should contain the following columns:
        # > [
        # > 'model_checkpoint',
        # > 'data_subsampling_split',
        # > 'train_loss',
        # > 'Micro F1 (w/o Neutral)',
        # > 'Macro F1 (w/o Neutral)',
        # > 'Weighted F1 (w/o Neutral)',
        # > 'Micro F1 (with Neutral)',
        # > 'Macro F1 (with Neutral)',
        # > 'Weighted F1 (with Neutral)',
        # > 'model_seed'
        # >]

        columns_to_plot: list[str] = [
            "train_loss",
            "Macro F1 (w/o Neutral)",
            "Weighted F1 (w/o Neutral)",
        ]

        return columns_to_plot

    def post_process_loaded_df(
        self,
        loaded_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Rename the 'epoch' column to 'model_checkpoint'
        loaded_df = loaded_df.rename(
            columns={
                "epoch": "model_checkpoint",
            },
        )

        # Subtract 1 from the 'model_checkpoint' column to make it zero-based
        loaded_df["model_checkpoint"] = loaded_df["model_checkpoint"].astype(int) - 1

        return loaded_df


class HardcodedScoreLoader:
    """Concrete implementation loading scores directly from hardcoded data."""

    def __init__(self, data: dict):
        self.data = data

    def get_scores(self) -> pd.DataFrame:
        records = []
        for epoch, splits in self.data.items():
            epoch_num = int(epoch)
            for split, scores in splits.items():
                record: dict = {
                    "epoch": epoch_num,
                    "split": split,
                }
                metrics = [
                    "accuracy",
                    "loss",
                    "f1_macro",
                    "precision_macro",
                    "recall_macro",
                    "f1_micro",
                ]
                for metric, score in zip(
                    metrics,
                    scores,
                    strict=True,
                ):
                    record[metric] = score
                records.append(record)

        return pd.DataFrame(data=records)

    def get_columns_to_plot(self) -> list[str]:
        return [
            "accuracy",
            "loss",
        ]


# Example Usage
if __name__ == "__main__":
    file_loader = EmotionClassificationScoreLoader(
        filepath=pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data/models/EmoLoop/output_dir/debug=-1/use_context=False/ep=5/seed=50/parsed_data/raw_data/",
            "parsed_data.csv",
        ),
    )
    hardcoded_loader = HardcodedScoreLoader(
        data={
            "1": {"test": [0.8, 0.5, 0.75, 0.9, 0.6, 0.85]},
        },
    )

    print(  # noqa: T201 - We want this test function to print
        file_loader.get_scores().head(),
    )
    print(  # noqa: T201 - We want this test function to print
        hardcoded_loader.get_scores().head(),
    )
