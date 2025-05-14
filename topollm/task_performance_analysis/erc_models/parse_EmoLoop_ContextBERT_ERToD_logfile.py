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

import itertools
import json
import logging
import os
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.logging.create_and_configure_global_logger import create_and_configure_global_logger
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

# Logger for this file
global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
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

# Define a fixed colormap for consistent coloring
COLORMAP: dict[str, str] = {
    "train_loss": "green",
    "validation_loss": "green",
    "test_loss": "green",
    "Micro F1 (w/o Neutral)": "orange",
    "Macro F1 (w/o Neutral)": "purple",
    "Weighted F1 (w/o Neutral)": "brown",
    "Micro F1 (with Neutral)": "red",
    "Macro F1 (with Neutral)": "blue",
    "Weighted F1 (with Neutral)": "black",
}


def get_color_from_colormap(
    metric_name: str,
) -> str:
    """Get color from colormap for a given metric name."""
    return COLORMAP.get(
        metric_name,
        "black",  # Default color if not found
    )


EPOCH_COLUMN_NAME: str = "epoch"

SPLITS_TO_PROCESS = [
    "train",
    "validation",
    "test",
]


def main() -> None:
    """Run main function."""
    logger: logging.Logger = global_logger
    verbosity: Verbosity = Verbosity.NORMAL

    do_create_plots: bool = True

    logger.info(
        msg="Running script ...",
    )

    # Define which options we want to iterate over to process the log files.
    debug_truncation_size_options: list[int] = [
        -1,
        60,
    ]
    use_context_options: list[bool] = [
        True,
        False,
    ]
    num_train_epochs_options: list[int] = [
        5,
        50,
    ]

    combinations = itertools.product(
        debug_truncation_size_options,
        use_context_options,
        num_train_epochs_options,
    )

    for (
        debug_truncation_size,
        use_context,
        num_train_epochs,
    ) in combinations:
        # Define the file paths.
        model_training_runs_output_dir: pathlib.Path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
            "models",
            "EmoLoop",
            "output_dir",
            f"debug={debug_truncation_size}",
            f"use_context={use_context}",
            f"ep={num_train_epochs}",
        )

        match use_context:
            case True:
                seed_range = range(42, 47)
            case False:
                seed_range = range(49, 55)
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

            if not single_training_run_parent_dir.exists():
                logger.warning(
                    msg=f"Path {single_training_run_parent_dir = } does not exist. Skipping this training run.",  # noqa: G004 - low overhead
                )
                continue

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

    parsed_data_output_dir: pathlib.Path = pathlib.Path(
        single_training_run_parent_dir,
        "parsed_data",
    )

    parsed_data_raw_data_output_dir: pathlib.Path = pathlib.Path(
        parsed_data_output_dir,
        "raw_data",
    )

    parsed_data_json_output_path: pathlib.Path = pathlib.Path(
        parsed_data_raw_data_output_dir,
        "parsed_data.json",
    )
    parsed_data_df_output_path: pathlib.Path = pathlib.Path(
        parsed_data_raw_data_output_dir,
        "parsed_data.csv",
    )

    plots_output_dir: pathlib.Path = pathlib.Path(
        parsed_data_output_dir,
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

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Parsed scores per epoch:",
        )
        for epoch, scores in parsed_data.items():
            logger.info(
                msg=f"{epoch = }: {scores}",  # noqa: G004 - low overhead
            )

    # # # #
    # Save the extracted scores
    save_parsed_data_to_json(
        parsed_data=parsed_data,
        output_file=parsed_data_json_output_path,
        verbosity=verbosity,
        logger=logger,
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

    save_parsed_data_df_to_csv(
        df=parsed_data_df,
        output_file=parsed_data_df_output_path,
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # #
    # Plotting

    if do_create_plots:
        for subset_value in SPLITS_TO_PROCESS:
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

    with pathlib.Path(
        file_path,
    ).open(
        mode="r",
    ) as f:
        for line in f:
            epoch_match = re.search(
                pattern=r"Training for epoch (\d+) out of",
                string=line,
            )
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epoch_data[current_epoch] = {
                    "train_loss": None,
                    "validation_loss": None,
                    "test_loss": None,
                    "train": {},
                    "validation": {},
                    "test": {},
                }
                continue

            for split in SPLITS_TO_PROCESS:
                split_name_in_log = "training" if split == "train" else split

                loss_match = re.search(rf"{split_name_in_log.capitalize()} loss:\s*([\d\.]+)", line)
                if loss_match and current_epoch is not None:
                    epoch_data[current_epoch][f"{split}_loss"] = float(  # type: ignore - problem with dict value typing here
                        loss_match.group(1),
                    )
                    continue

            for split in SPLITS_TO_PROCESS:
                split_name_in_log = "training" if split == "train" else split

                match = re.search(
                    pattern=rf"{split_name_in_log.capitalize()} F1 scores:\s*\[(.*?)\]",
                    string=line,
                )
                if match and current_epoch is not None:
                    scores = [float(s.strip()) for s in match.group(1).split(",")]
                    epoch_data[current_epoch][split] = dict(
                        zip(
                            score_labels,
                            scores,
                            strict=True,
                        ),
                    )
                    continue

    return epoch_data


def convert_parsed_data_dict_to_df(
    parsed_data: dict[
        int,
        dict[
            str,
            dict[str, float | list[float]],
        ],
    ],
    subset_column_name: str = "data_subsampling_split",
) -> pd.DataFrame:
    """Convert the parsed data dictionary to a DataFrame."""
    rows: list = []
    for epoch, data in parsed_data.items():
        for split in SPLITS_TO_PROCESS:
            new_entry: dict = {
                EPOCH_COLUMN_NAME: epoch,
                subset_column_name: split,
                f"{split}_loss": data[f"{split}_loss"],
                **data[split],
            }
            rows.append(new_entry)

    return pd.DataFrame(
        data=rows,
    )


def save_parsed_data_to_json(
    parsed_data: dict[int, dict[str, list[float]]],
    output_file: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the extracted scores to a JSON file.

    Note: JSON keys are converted to strings.
    """
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving parsed data to {output_file = } ...",  # noqa: G004 - low overhead
        )

    output_file = pathlib.Path(
        output_file,
    )
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        mode="w",
    ) as f:
        json.dump(
            obj=parsed_data,
            fp=f,
            indent=4,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving parsed data to {output_file = } DONE",  # noqa: G004 - low overhead
        )


def save_parsed_data_df_to_csv(
    df: pd.DataFrame,
    output_file: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the extracted scores to a CSV file."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving parsed data to {output_file = } ...",  # noqa: G004 - low overhead
        )

    output_file = pathlib.Path(
        output_file,
    )
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path_or_buf=output_file,
        index=False,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving parsed data to {output_file = } DONE",  # noqa: G004 - low overhead
        )


def plot_scores(
    df: pd.DataFrame,
    subset_column_name: str = "data_subsampling_split",
    subset_value: str = "validation",
    plots_output_dir: pathlib.Path | None = None,
    y_axis_range: tuple[float, float] = (0.1, 1.0),
    *,
    mark_best: bool = False,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot the scores per epoch."""
    if df.empty:
        logger.info(
            msg="No data to plot. Skipping this plot creation.",
        )
        return

    # Figure dimensions
    plt.figure(
        figsize=(10, 6),
    )

    df_subset: pd.DataFrame = df[df[subset_column_name] == subset_value]

    plot_columns = df_subset.dropna(
        axis=1,
        how="all",
    ).columns.drop(
        labels=[
            EPOCH_COLUMN_NAME,
            subset_column_name,
        ],
    )

    for col in plot_columns:
        plt.plot(
            df_subset[EPOCH_COLUMN_NAME],
            df_subset[col],
            marker="o",
            label=col,
            color=get_color_from_colormap(
                metric_name=col,
            ),
        )

        # # Mark best value if required.
        # if mark_best and scores:
        #     best_value = max(scores)
        #     best_epoch = epochs[scores.index(best_value)]
        #     plt.scatter(best_epoch, best_value, marker="*", color="gold", s=150, zorder=5)
        #     plt.annotate(f"{best_value:.3f}", (best_epoch, best_value), textcoords="offset points", xytext=(5, 5))

    plt.xlabel(xlabel=EPOCH_COLUMN_NAME)
    plt.ylabel(ylabel="Score")
    plt.title(label=f"{subset_value.capitalize()} Scores per Epoch")
    plt.legend(loc="best")
    plt.grid(visible=True)
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
