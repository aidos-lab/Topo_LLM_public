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
import os
import pathlib
import re

import matplotlib.pyplot as plt

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH

# Define metric names corresponding to the six scores in each f1_scores list.
METRIC_NAMES: list[str] = [
    "Micro F1 (w/o Neutral)",
    "Macro F1 (w/o Neutral)",
    "Weighted F1 (w/o Neutral)",
    "Micro F1 (with Neutral)",
    "Macro F1 (with Neutral)",
    "Weighted F1 (with Neutral)",
]


def parse_log(
    file_path: os.PathLike,
) -> dict[int, dict[str, list[float]]]:
    """Parse the log file to extract aggregated F1 scores per epoch.

    The log file is expected to contain lines that mark the start of an epoch,
    along with lines that contain 'Validation F1 scores:' and 'Test F1 scores:'.

    Args:
        file_path: Path to the log file.

    Returns:
        A dictionary where keys are epoch numbers (int) and values are dictionaries
        with keys "validation" and "test", each mapping to a list of six F1 scores.
    """
    epoch_data: dict[int, dict[str, list[float]]] = {}
    current_epoch: int | None = None

    with open(file_path, "r") as f:
        for line in f:
            # Check for the start of a new epoch.
            epoch_match = re.search(r"Training for epoch (\d+) out of", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epoch_data[current_epoch] = {}
                continue

            # Extract validation F1 scores.
            val_match = re.search(r"Validation F1 scores:\s*\[(.*?)\]", line)
            if val_match and current_epoch is not None:
                scores_str = val_match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(",")]
                epoch_data[current_epoch]["validation"] = scores
                continue

            # Extract test F1 scores.
            test_match = re.search(r"Test F1 scores:\s*\[(.*?)\]", line)
            if test_match and current_epoch is not None:
                scores_str = test_match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(",")]
                epoch_data[current_epoch]["test"] = scores
                continue

    return epoch_data


def save_scores_to_json(
    scores: dict[int, dict[str, list[float]]],
    output_file: os.PathLike,
) -> None:
    """Save the extracted scores to a JSON file.

    Note: JSON keys are converted to strings.

    Args:
        scores: The dictionary containing F1 scores per epoch.
        output_file: Path for the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=4)


def compute_combined(scores: list[float], start: int, end: int) -> float:
    """Compute the arithmetic mean of a sub-range of scores.

    Args:
        scores: list of F1 scores.
        start: Starting index (inclusive).
        end: Ending index (exclusive).

    Returns:
        The arithmetic mean of the specified sub-list.
    """
    subset = scores[start:end]
    return sum(subset) / len(subset) if subset else 0.0


def plot_f1_scores(
    epoch_data: dict[int, dict[str, list[float]]],
    set_type: str = "validation",
    plots_output_dir: pathlib.Path | None = None,
    y_axis_range: tuple[float, float] = (0.5, 1.0),
    *,
    plot_combined: bool = True,
    mark_best: bool = False,
    show_plot: bool = False,
) -> None:
    """Plot the aggregated F1 scores over epochs for each metric and the combined scores.

    Args:
        epoch_data: dictionary mapping epoch numbers to F1 scores.
        set_type: Which scores to plot ('validation' or 'test').
        plot_combined: If True, computes and plots combined scores:
            - Combined (w/o Neutral): Average of indices 0-2.
            - Combined (with Neutral): Average of indices 3-6.
    """
    if not epoch_data:
        print("No data to plot.")
        return

    epochs = sorted(epoch_data.keys())
    num_metrics = len(epoch_data[epochs[0]][set_type])

    plt.figure(figsize=(10, 6))
    # Plot each individual metric.
    for i in range(num_metrics):
        label = METRIC_NAMES[i] if i < len(METRIC_NAMES) else f"Metric {i}"
        scores = [epoch_data[epoch][set_type][i] for epoch in epochs if set_type in epoch_data[epoch]]
        plt.plot(epochs, scores, marker="o", label=label)

        # Mark best value if required.
        if mark_best and scores:
            best_value = max(scores)
            best_epoch = epochs[scores.index(best_value)]
            plt.scatter(best_epoch, best_value, marker="*", color="gold", s=150, zorder=5)
            plt.annotate(f"{best_value:.3f}", (best_epoch, best_value), textcoords="offset points", xytext=(5, 5))

    # Plot the combined scores.
    if plot_combined:
        combined_without = [
            compute_combined(epoch_data[epoch][set_type], 0, 3) for epoch in epochs if set_type in epoch_data[epoch]
        ]
        combined_with = [
            compute_combined(epoch_data[epoch][set_type], 3, 6) for epoch in epochs if set_type in epoch_data[epoch]
        ]
        plt.plot(epochs, combined_without, marker="x", linestyle="--", color="black", label="Combined (w/o Neutral)")
        plt.plot(epochs, combined_with, marker="x", linestyle="--", color="red", label="Combined (with Neutral)")

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")

    plt.ylim(y_axis_range)
    plt.xticks(epochs)

    plt.title(f"{set_type.capitalize()} Aggregated F1 Scores over Epochs")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.grid(True)
    plt.tight_layout()

    if plots_output_dir:
        plots_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        plot_file_path = plots_output_dir / f"{set_type}_f1_scores.pdf"
        plt.savefig(
            plot_file_path,
        )
        print(f"Plot saved to {plot_file_path = }")

    if show_plot:
        plt.show()


def main() -> None:
    # # # #
    # Define the file paths.
    model_training_runs_output_dir: pathlib.Path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/models/EmoLoop/output_dir",
    )

    for seed in range(42, 47):
        single_training_run_parent_dir = pathlib.Path(
            model_training_runs_output_dir,
            f"ep=5/seed={seed}",
        )

        process_log_file_of_single_model_training_run(
            single_training_run_parent_dir=single_training_run_parent_dir,
        )


def process_log_file_of_single_model_training_run(
    single_training_run_parent_dir: os.PathLike,
) -> None:
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
    scores_data: dict[int, dict[str, list[float]]] = parse_log(
        file_path=log_file_path,
    )
    print("Parsed F1 scores per epoch:")
    for epoch, scores in scores_data.items():
        print(f"Epoch {epoch}: {scores}")

    # Save the extracted scores to a JSON file.
    save_scores_to_json(
        scores=scores_data,
        output_file=json_output_path,
    )

    print(
        f"Scores saved to {json_output_path = }",
    )

    # Compute and print combined validation F1 scores per epoch.
    print("Combined validation F1 scores per epoch:")
    for epoch in sorted(scores_data.keys()):
        combined_without = compute_combined(scores_data[epoch]["validation"], 0, 3)
        combined_with = compute_combined(scores_data[epoch]["validation"], 3, 6)
        print(
            f"Epoch {epoch}: Combined (w/o Neutral): {combined_without:.4f}, Combined (with Neutral): {combined_with:.4f}"
        )

    # Plot the validation F1 scores.
    plot_f1_scores(
        epoch_data=scores_data,
        set_type="validation",
        plots_output_dir=plots_output_dir,
        mark_best=True,
        plot_combined=True,
        show_plot=False,
    )

    # Plot the test F1 scores.
    plot_f1_scores(
        epoch_data=scores_data,
        set_type="test",
        plots_output_dir=plots_output_dir,
        mark_best=True,
        plot_combined=True,
        show_plot=False,
    )


if __name__ == "__main__":
    main()
