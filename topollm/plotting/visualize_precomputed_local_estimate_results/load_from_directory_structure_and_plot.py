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
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

default_logger = logging.getLogger(__name__)


def load_numpy_arrays(
    base_dir: pathlib.Path,
) -> defaultdict:
    """Recursively load numpy arrays into a nested dictionary structure."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, _, files in tqdm(
        os.walk(
            base_dir,
        ),
        desc="Loading arrays",
    ):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_dir)
                parts = relative_path.split(os.sep)

                model_config = parts[0]
                layer = parts[1]
                checkpoint = parts[2]

                arr = np.load(file_path)
                data[model_config][layer][checkpoint].append(arr)

    return data


def apply_accumulation_function(data, func: Callable[[np.ndarray], float]) -> defaultdict:
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


def plot_data(accumulated_data):
    """Plot the accumulated data."""
    for model_config, layers in accumulated_data.items():
        for layer, checkpoints in layers.items():
            sorted_checkpoints = sorted(checkpoints.items())
            checkpoint_labels, values = zip(*sorted_checkpoints)

            plt.figure()
            plt.plot(checkpoint_labels, values, marker="o")
            plt.xlabel("Checkpoint")
            plt.ylabel("Accumulated Value")
            plt.title(f"Accumulated Values for {model_config} - {layer}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def main() -> None:
    logger = default_logger
    logger.info(
        "Starting main() function ...",
    )

    # Base directory
    base_dir = pathlib.Path(
        "/Volumes/ruppik_external/research_data/",
        "Topo_LLM/data/analysis/twonn/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000/lvl-token/add-prefix-space-False_max-len-512",
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
