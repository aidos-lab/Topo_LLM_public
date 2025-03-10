import json
import os
import pathlib
import re

import matplotlib.pyplot as plt

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH

# Define metric names corresponding to the six scores in each f1_scores list.
METRIC_NAMES = [
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


def read_scores_from_json(file_path: str) -> dict[int, dict[str, list[float]]]:
    """Read the scores from a JSON file and converts keys back to int.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A dictionary mapping epoch numbers (as ints) to F1 score data.
    """
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    return {int(epoch): data for epoch, data in raw_data.items()}


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
    *,
    plot_combined: bool = True,
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
    plt.title(f"{set_type.capitalize()} Aggregated F1 Scores over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # # # #
    # Define the file paths.
    model_training_run_parent_dir = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/models/EmoLoop/output_dir/ep=5/seed=42",
    )

    log_file_path: pathlib.Path = pathlib.Path(
        model_training_run_parent_dir,
        "log.txt",
    )

    json_output_path: pathlib.Path = pathlib.Path(
        model_training_run_parent_dir,
        "scores.json",
    )

    # Parse the log file from disk.
    scores_data: dict[int, dict[str, list[float]]] = parse_log(
        file_path=log_file_path,
    )
    print("Parsed F1 scores per epoch:")
    for epoch, scores in scores_data.items():
        print(f"Epoch {epoch}: {scores}")

    # Save the extracted scores to a JSON file.
    save_scores_to_json(
        scores_data,
        json_output_path,
    )

    print(f"Scores saved to {json_output_path}")

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
        plot_combined=True,
    )

    # Plot the test F1 scores.
    plot_f1_scores(
        epoch_data=scores_data,
        set_type="test",
        plot_combined=True,
    )


if __name__ == "__main__":
    main()
