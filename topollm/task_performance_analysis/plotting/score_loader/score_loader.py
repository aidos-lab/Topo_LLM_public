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

import json
import logging
import pathlib
import re
import subprocess
from typing import Protocol

import pandas as pd

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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


class TrippyRScoreLoader:
    """Concrete implementation loading TrippyR model scores from the output directory."""

    def __init__(
        self,
        results_folder_for_given_seed_path: pathlib.Path,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the loader."""
        self.results_folder_for_given_seed_path: pathlib.Path = results_folder_for_given_seed_path

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def get_scores(
        self,
    ) -> pd.DataFrame:
        scores_df_list = []

        for split in [
            "dev",
            "test",
        ]:
            # # # #
            # Load the evaluation results from the JSON file
            eval_res_json_path = pathlib.Path(
                self.results_folder_for_given_seed_path,
                f"eval_res.{split}.json",
            )

            if not eval_res_json_path.exists():
                self.logger.warning(
                    msg=f"Evaluation results file for {split = } does not exist: {eval_res_json_path = }",  # noqa: G004 - low overhead
                )
                continue

            with eval_res_json_path.open(
                mode="r",
            ) as file:
                eval_res_json_loaded = json.load(
                    fp=file,
                )

                eval_res_df = pd.DataFrame(
                    data=eval_res_json_loaded,
                )

                # Tag the current split
                eval_res_df["data_subsampling_split"] = split

                # The entries in the "checkpoint_name" column have the form: "checkpoint-10650".
                # We need to extract the checkpoint number from the string and add it as a new column "model_checkpoint".

                checkpoint_name_df = eval_res_df["checkpoint_name"].str.extract(r"checkpoint-(\d+)")
                eval_res_df["model_checkpoint"] = checkpoint_name_df[0].astype(int)

                # Sort the DataFrame by the "model_checkpoint" column
                eval_res_df: pd.DataFrame = eval_res_df.sort_values(
                    by="model_checkpoint",
                )

            # # # #
            # Load the joint goal accuracy from the log file
            eval_res_log_path = pathlib.Path(
                self.results_folder_for_given_seed_path,
                f"eval_pred_{split}.1.0.log",
            )

            jga_lines: list[str] = self.extract_jga_lines_grep(
                filename=str(object=eval_res_log_path),
            )

            jga_df: pd.DataFrame = self.parse_log_lines_to_df(
                log_lines=jga_lines,
            )

            # Merge the JGA DataFrame with the evaluation results DataFrame on the "model_checkpoint" column
            merged_df = eval_res_df.merge(
                right=jga_df,
                how="left",
                on="model_checkpoint",
            )

            # # # #
            # Append the DataFrame to the list
            scores_df_list.append(
                merged_df,
            )

        # Concatenate all DataFrames in the list into a single DataFrame
        scores_df = pd.concat(
            objs=scores_df_list,
            axis=0,
            ignore_index=True,
        )

        if self.verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=scores_df,
                df_name="scores_df",
                logger=self.logger,
            )

        return scores_df

    def extract_jga_lines_grep(
        self,
        filename: str,
    ) -> list[str]:
        """Extract lines starting with 'Joint goal acc:' using grep.

        Args:
            filename: The path to the log file.

        Returns:
            A list of lines matching the pattern.

        """
        result = None
        try:
            result = subprocess.run(
                args=[
                    "grep",
                    "Joint goal acc:",
                    filename,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.splitlines()
        except subprocess.CalledProcessError:
            # Log an error if grep fails
            self.logger.exception(
                msg=f"Error running grep on file: {filename}",  # noqa: G004
            )
            self.logger.warning(
                msg=f"subprocess.run result: {result}",  # noqa: G004
            )
            self.logger.warning(
                msg="Will return empty list.",
            )

            lines = []
        return lines

    def parse_log_lines_to_df(
        self,
        log_lines: list[str],
    ) -> pd.DataFrame:
        """Parse log lines to extract checkpoint numbers and JGA scores.

        Args:
            log_lines: A sequence of log lines, each containing a JGA score and checkpoint path.

        Returns:
            A pandas DataFrame with 'checkpoint' and 'jga' columns, sorted by checkpoint.

        """
        data: list[tuple[int, float]] = []

        pattern = re.compile(
            pattern=r"Joint goal acc:\s*([\d.]+),.*?checkpoint-(\d+)\.json",
        )

        for line in log_lines:
            match = pattern.search(line)
            if match:
                jga = float(match.group(1))
                checkpoint = int(match.group(2))
                data.append((checkpoint, jga))
            else:
                msg: str = f"Line does not match expected format: {line}"
                raise ValueError(msg)

        df = pd.DataFrame(
            data=data,
            columns=[
                "model_checkpoint",
                "jga",
            ],
        )
        df_sorted = df.sort_values(by="model_checkpoint").reset_index(drop=True)

        return df_sorted

    def get_columns_to_plot(
        self,
    ) -> list[str]:
        columns_to_plot: list[str] = [
            "loss",
            "jga",
        ]

        return columns_to_plot


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
            loaded_df: pd.DataFrame = pd.read_csv(
                filepath_or_buffer=file,
            )

        post_processed_loaded_df: pd.DataFrame = self.post_process_loaded_df(
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
