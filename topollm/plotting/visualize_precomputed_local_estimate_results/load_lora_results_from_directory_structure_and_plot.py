# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Script for loading results from the directory structure and plotting them."""

import logging
import os
import pathlib
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH

default_logger = logging.getLogger(__name__)


def parse_model_config(
    model_config: str,
) -> tuple[
    int,
    int,
]:
    """Extract lora_r parameter and checkpoint number from model configuration string."""
    lora_r_match = re.search(r"ftm-lora_r-(\d+)", model_config)
    checkpoint_match = re.search(r"ckpt-(\d+)", model_config)

    if lora_r_match and checkpoint_match:
        lora_r = int(lora_r_match.group(1))
        checkpoint = int(checkpoint_match.group(1))
        return lora_r, checkpoint

    msg = f"Invalid model configuration string: {model_config}"
    raise ValueError(msg)


def load_numpy_arrays(
    base_dir: pathlib.Path,
) -> defaultdict:
    """Recursively load numpy arrays into a nested dictionary structure."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, _, files in tqdm(
        os.walk(base_dir),
        desc="Loading numpy arrays",
    ):
        for file in files:
            if file.endswith(".npy"):
                file_path = pathlib.Path(
                    root,
                    file,
                )
                relative_path = os.path.relpath(
                    file_path,
                    base_dir,
                )
                parts = relative_path.split(os.sep)

                model_config = parts[0]
                layer = parts[1]

                try:
                    lora_r, checkpoint = parse_model_config(model_config)
                except ValueError as e:
                    msg = f"Error parsing model configuration: {model_config}"
                    print(msg)
                    continue

                arr = np.load(file_path)
                data[lora_r][checkpoint][layer].append(arr)

    return data


def apply_accumulation_function(
    data: defaultdict,
    func: Callable[[np.ndarray], float],
) -> defaultdict:
    """Recursively apply an accumulation function to all numpy arrays in the data structure."""

    def recursive_apply(current_data: Any) -> Any:
        if isinstance(
            current_data,
            list,
        ):
            return [func(np.array(arr)) for arr in current_data]
        elif isinstance(
            current_data,
            dict,
        ):
            return {key: recursive_apply(value) for key, value in current_data.items()}
        else:
            msg = "Unexpected data type in the nested structure."
            raise TypeError(msg)

    return recursive_apply(data)


default_output_dir = pathlib.Path(
    TOPO_LLM_REPOSITORY_BASE_PATH,
    "data",
    "saved_plots",
    "mean_estimates_over_different_checkpoints",
)


def extract_layer_index(
    layer_name: str,
):
    """Extract the numerical index from the layer name for sorting."""
    match = re.search(
        r"layer--(\d+)",
        layer_name,
    )
    if match:
        return int(match.group(1))
    return layer_name


def plot_data(
    accumulated_data: defaultdict,
    output_dir: pathlib.Path = default_output_dir,
    *,
    save_individual_plots: bool = True,
    save_grid_plot: bool = True,
) -> None:
    """Plot the accumulated data."""
    plot_files = []
    all_layers = set()
    for checkpoints in accumulated_data.values():
        for layers in checkpoints.values():
            all_layers.update(layers.keys())

    all_layers = sorted(all_layers, key=extract_layer_index)
    num_layers = len(all_layers)
    num_lora_r = len(accumulated_data)

    for lora_r, checkpoints in accumulated_data.items():
        layers_dict = defaultdict(list)
        for checkpoint, layers in sorted(checkpoints.items()):
            for layer, value in layers.items():
                layers_dict[layer].append((checkpoint, value))

        for layer, checkpoint_values in layers_dict.items():
            checkpoint_values.sort()
            checkpoints_x, values = zip(*checkpoint_values)

            plt.figure()
            plt.plot(checkpoints_x, values, marker="o")
            plt.xlabel("Checkpoint")
            plt.ylabel("Accumulated Value")
            plt.title(f"Accumulated Values for Lora_r: {lora_r}, Layer: {layer}")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot as a PDF file
            if save_individual_plots:
                file_name = f"lora_r-{lora_r}_layer-{layer}.pdf"
                output_dir_path = pathlib.Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_file = pathlib.Path(output_dir_path, file_name)
                plt.savefig(output_file, format="pdf")
                plot_files.append(output_file)
            plt.close()

    if save_grid_plot:
        sorted_lora_r_values = sorted(accumulated_data.keys())
        fig, axes = plt.subplots(num_lora_r, num_layers, figsize=(num_layers * 5, num_lora_r * 5), squeeze=False)

        for i, lora_r in enumerate(sorted_lora_r_values):
            checkpoints = accumulated_data[lora_r]
            for j, layer in enumerate(all_layers):
                ax = axes[i][j]
                checkpoint_values = []
                for checkpoint, layers in sorted(checkpoints.items()):
                    if layer in layers:
                        checkpoint_values.append((checkpoint, layers[layer]))

                if checkpoint_values:
                    checkpoints_x, values = zip(*checkpoint_values)
                    ax.plot(checkpoints_x, values, marker="o")
                    ax.set_xlabel("Checkpoint")
                    ax.set_ylabel("Accumulated Value")
                    ax.set_title(f"Lora_r: {lora_r}, Layer: {layer}")
                    ax.tick_params(axis="x", rotation=45)
                else:
                    ax.set_visible(False)

        plt.tight_layout()
        grid_output_file = pathlib.Path(output_dir, "grid_plot.pdf")
        plt.savefig(grid_output_file, format="pdf")
        plt.close()


def main() -> None:
    """Load and plot the data."""
    logger = default_logger
    logger.info(
        "Starting main() function ...",
    )

    # Base directory
    base_dir = pathlib.Path(
        "/Volumes/ruppik_external/research_data/",
        "Topo_LLM",
        "data",
        "analysis",
        "twonn/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000/lvl-token/add-prefix-space-False_max-len-512",
    )

    # Load and plot data
    data = load_numpy_arrays(
        base_dir=base_dir,
    )
    # Apply an accumulation function (e.g., np.mean)
    accumulated_data = apply_accumulation_function(
        data,
        np.mean,
    )

    plot_data(
        accumulated_data=accumulated_data,
    )

    logger.info(
        "Starting main() function DONE",
    )


if __name__ == "__main__":
    main()
