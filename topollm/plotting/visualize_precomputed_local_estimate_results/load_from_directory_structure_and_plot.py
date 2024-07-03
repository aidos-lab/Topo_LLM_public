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

import matplotlib.pyplot as plt
import numpy as np

default_logger = logging.getLogger(__name__)


# Function to recursively load numpy arrays
def load_numpy_arrays(
    base_dir: pathlib.Path,
    logger: logging.Logger = default_logger,
) -> defaultdict:
    data = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(
        base_dir,
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
                mean_value = np.mean(arr)

                data[model_config][layer].append((checkpoint, mean_value))

    return data


# Function to plot data
def plot_data(data):
    for model_config, layers in data.items():
        for layer, values in layers.items():
            values.sort()  # Sort values by checkpoint
            checkpoints, means = zip(*values)

            plt.figure()
            plt.plot(checkpoints, means, marker="o")
            plt.xlabel("Checkpoint")
            plt.ylabel("Mean Value")
            plt.title(f"Mean Values for {model_config} - {layer}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def main() -> None:
    # Base directory
    base_dir = pathlib.Path(
        "/Volumes/ruppik_external/research_data/",
        "Topo_LLM/data/analysis/twonn/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000/lvl-token/add-prefix-space-False_max-len-512",
    )

    # Load and plot data
    data = load_numpy_arrays(
        base_dir,
    )
    plot_data(
        data,
    )


if __name__ == "__main__":
    main()
