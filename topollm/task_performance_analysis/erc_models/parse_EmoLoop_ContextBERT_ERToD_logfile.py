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

"""Parse the log file of EmoLoop's ContextBERT ERToD model training run and plot F1 scores."""

import json
import logging
import os
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.typing.enums import Verbosity

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

# Define metric names corresponding to the six scores in each f1_scores list.
METRIC_NAMES: list[str] = [
    "Micro F1 (w/o Neutral)",
    "Macro F1 (w/o Neutral)",
    "Weighted F1 (w/o Neutral)",
    "Micro F1 (with Neutral)",
    "Macro F1 (with Neutral)",
    "Weighted F1 (with Neutral)",
]

EPOCH_COLUMN_NAME: str = "epoch"


def main() -> None:
    """Run main function."""
    logger: logging.Logger = global_logger
    verbosity: Verbosity = Verbosity.NORMAL

    do_create_plots: bool = True

    logger.info(
        msg="Running script ...",
    )

    for use_context in [
        True,
        False,
    ]:
        # Define the file paths.
        model_training_runs_output_dir: pathlib.Path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data/models/EmoLoop/output_dir/debug=-1/",
            f"use_context={use_context}",
            "ep=5",
        )

        match use_context:
            case True:
                seed_range = range(42, 47)
            case False:
                seed_range = range(50, 54)
            case _:
                msg = f"Invalid value: {use_context = }"
                raise ValueError(
                    msg,
                )

        for seed in seed_range:
            single_training_run_parent_dir = pathlib.Path(
                model_training_runs_output_dir,
                f"seed={seed}",
            )

            process_log_file_of_single_model_training_run(
                single_training_run_parent_dir=single_training_run_parent_dir,
                do_create_plots=do_create_plots,
                verbosity=verbosity,
                logger=logger,
            )

    logger.info(
        msg="Running script DONE",
    )


def process_log_file_of_single_model_training_run(
    single_training_run_parent_dir: os.PathLike,
    *,
    do_create_plots: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Process the log file of a single model training run."""
    log_file_path: pathlib.Path = pathlib.Path(
        single_training_run_parent_dir,
        "log.txt",
    )

    json_output_path: pathlib.Path = pathlib.Path(
        single_training_run_parent_dir,
        "scores.json",
    )

    plots_output_dir: pathlib.Path = pathlib.Path(
        single_training_run_parent_dir,
        "plots",
    )
    plots_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # # # #
    # Parse the log file from disk.
    subset_column_name: str = "data_subsampling_split"

    parsed_data: dict = parse_log(
        file_path=log_file_path,
    )

    print("Parsed F1 scores per epoch:")
    for epoch, scores in parsed_data.items():
        print(f"Epoch {epoch}: {scores}")

    # Save the extracted scores to a JSON file.
    save_parsed_data_to_json(
        parsed_data=parsed_data,
        output_file=json_output_path,
    )

    parsed_data_df: pd.DataFrame = convert_parsed_data_dict_to_df(
        parsed_data=parsed_data,
        subset_column_name=subset_column_name,
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=parsed_data_df,
            df_name="parsed_data_df",
            logger=logger,
        )

    # # # # # # # #
    # Plotting

    if do_create_plots:
        for subset_value in [
            "train",
            "validation",
            "test",
        ]:
            plot_scores(
                df=parsed_data_df,
                subset_column_name=subset_column_name,
                subset_value=subset_value,
                plots_output_dir=plots_output_dir,
                mark_best=True,
                show_plot=False,
            )


def parse_log(
    file_path: os.PathLike,
) -> dict[int, dict[str, dict[str, float | list[float] | None] | None]]:
    """Parse the log file to extract training losses and aggregated F1 scores per epoch.

    Args:
        file_path: Path to the log file.

    Returns:
        A dictionary where keys are epoch numbers (int) and values are dictionaries
        containing "train_loss", and "validation" and "test" dictionaries with named F1 scores.

    """
    epoch_data: dict[int, dict[str, dict[str, float | list[float] | None] | None]] = {}
    current_epoch: int = -100

    score_labels: list[str] = METRIC_NAMES

    with open(file_path, "r") as f:
        for line in f:
            epoch_match = re.search(r"Training for epoch (\d+) out of", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epoch_data[current_epoch] = {
                    "train_loss": None,
                    "validation": {},
                    "test": {},
                }
                continue

            loss_match = re.search(r"Training loss:\s*([\d\.]+)", line)
            if loss_match and current_epoch is not None:
                epoch_data[current_epoch]["train_loss"] = float(  # type: ignore - problem with dict value typing here
                    loss_match.group(1),
                )
                continue

            val_match = re.search(r"Validation F1 scores:\s*\[(.*?)\]", line)
            if val_match and current_epoch is not None:
                val_scores = [float(s.strip()) for s in val_match.group(1).split(",")]
                epoch_data[current_epoch]["validation"] = dict(
                    zip(
                        score_labels,
                        val_scores,
                        strict=True,
                    ),
                )
                continue

            test_match = re.search(r"Test F1 scores:\s*\[(.*?)\]", line)
            if test_match and current_epoch is not None:
                test_scores = [float(s.strip()) for s in test_match.group(1).split(",")]
                epoch_data[current_epoch]["test"] = dict(
                    zip(
                        score_labels,
                        test_scores,
                        strict=True,
                    ),
                )
                continue

    return epoch_data


def convert_parsed_data_dict_to_df(
    parsed_data: dict[int, dict[str, dict[str, float | list[float]]]],
    subset_column_name: str = "data_subsampling_split",
) -> pd.DataFrame:
    rows: list = []
    for epoch, data in parsed_data.items():
        # Train data
        rows.append(
            {
                EPOCH_COLUMN_NAME: epoch,
                subset_column_name: "train",
                "train_loss": data["train_loss"],
                **dict.fromkeys(data["validation"]),
            },
        )
        # Validation and Test data
        for split in ["validation", "test"]:
            rows.append(
                {
                    EPOCH_COLUMN_NAME: epoch,
                    subset_column_name: split,
                    "train_loss": None,
                    **data[split],
                },
            )

    return pd.DataFrame(rows)


def save_parsed_data_to_json(
    parsed_data: dict[int, dict[str, list[float]]],
    output_file: os.PathLike,
) -> None:
    """Save the extracted scores to a JSON file.

    Note: JSON keys are converted to strings.

    Args:
        scores: The dictionary containing F1 scores per epoch.
        output_file: Path for the output JSON file.
    """
    print(
        f"Scores saved to {output_file = }",
    )

    with open(output_file, "w") as f:
        json.dump(parsed_data, f, indent=4)


def plot_scores(
    df: pd.DataFrame,
    subset_column_name: str = "data_subsampling_split",
    subset_value: str = "validation",
    plots_output_dir: pathlib.Path | None = None,
    y_axis_range: tuple[float, float] = (0.3, 1.0),
    *,
    mark_best: bool = False,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot the scores per epoch."""
    if df.empty:
        print("No data to plot. Skipping this plot creation.")
        return

    df_subset: pd.DataFrame = df[df[subset_column_name] == subset_value]
    plt.figure(
        figsize=(10, 6),
    )

    plot_columns = df_subset.dropna(
        axis=1,
        how="all",
    ).columns.drop(
        labels=[EPOCH_COLUMN_NAME, subset_column_name],
    )

    for col in plot_columns:
        plt.plot(
            df_subset[EPOCH_COLUMN_NAME],
            df_subset[col],
            marker="o",
            label=col,
        )

        # # Mark best value if required.
        # if mark_best and scores:
        #     best_value = max(scores)
        #     best_epoch = epochs[scores.index(best_value)]
        #     plt.scatter(best_epoch, best_value, marker="*", color="gold", s=150, zorder=5)
        #     plt.annotate(f"{best_value:.3f}", (best_epoch, best_value), textcoords="offset points", xytext=(5, 5))

    plt.xlabel(EPOCH_COLUMN_NAME)
    plt.ylabel("Score")
    plt.title(f"{subset_value.capitalize()} Scores per Epoch")
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim(y_axis_range)
    plt.tight_layout()

    if show_plot:
        plt.show()

    if plots_output_dir:
        plot_file_path: pathlib.Path = plots_output_dir / f"{subset_value}_f1_scores.pdf"

        plot_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_file_path = } ...",  # noqa: G004
            )
        plt.savefig(
            plot_file_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_file_path = } DONE",  # noqa: G004
            )

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
