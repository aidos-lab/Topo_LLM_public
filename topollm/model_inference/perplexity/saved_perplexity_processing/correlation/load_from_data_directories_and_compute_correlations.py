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

import logging
import os
import pathlib
from dataclasses import dataclass, field

import hydra
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


setup_omega_conf()


@dataclass
class AlignedDFMetadata:
    """
    A dataclass that holds metadata extracted from the file path of an aligned_df.csv file.
    """

    dataset: str | None = None
    model: str | None = None
    checkpoint: str | None = None

    @staticmethod
    def from_path(
        file_path: os.PathLike,
    ) -> "AlignedDFMetadata":
        """
        Extracts metadata from the given file path.

        Args:
            file_path (str): The full path to an aligned_df.csv file.

        Returns:
            AlignedDFMetadata: An instance of AlignedDFMetadata with extracted metadata.
        """
        file_path_str = str(file_path)

        components = file_path_str.split(os.sep)
        dataset, model, checkpoint = None, None, None

        for component in components:
            if "data-" in component:
                dataset = component.split("data-")[-1]
            if "model-" in component:
                model = component.split("model-")[-1]
            if "ckpt-" in component:
                checkpoint = component.split("ckpt-")[-1].split("_")[0]

        return AlignedDFMetadata(dataset=dataset, model=model, checkpoint=checkpoint)


@dataclass
class AlignedDF:
    """Dataclass that holds the path to an aligned_df.csv file, the associated DataFrame,
    and metadata extracted from the file path.
    """

    file_path: pathlib.Path
    dataframe: pd.DataFrame = field(init=False)
    metadata: AlignedDFMetadata = field(init=False)

    def __post_init__(self) -> None:
        self.dataframe = pd.read_csv(self.file_path)
        self.metadata = AlignedDFMetadata.from_path(self.file_path)


@dataclass
class AlignedDFCollection:
    """Dataclass to manage a collection of AlignedDF objects for analysis."""

    aligned_dfs: list[AlignedDF] = field(default_factory=list)

    def add_aligned_df(self, aligned_df: AlignedDF) -> None:
        self.aligned_dfs.append(aligned_df)

    def filter_by_model(self, model_name: str) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.model == model_name]

    def filter_by_dataset(self, dataset_name: str) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.dataset == dataset_name]

    def filter_by_checkpoint(self, checkpoint: str) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.checkpoint == checkpoint]

    def get_aggregated_statistics(self) -> pd.DataFrame:
        """Aggregate the statistics from all loaded DataFrames.

        Returns:
            pd.DataFrame: Aggregated statistics for each checkpoint.
        """
        aggregated_data = []
        for df in self.aligned_dfs:
            stats = df.dataframe.describe().transpose()
            stats["Dataset"] = df.metadata.dataset
            stats["Model"] = df.metadata.model
            stats["Checkpoint"] = df.metadata.checkpoint
            aggregated_data.append(stats)

        return pd.concat(aggregated_data, axis=0)


def find_aligned_dfs(
    root_dir: os.PathLike,
    dataset: str,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> AlignedDFCollection:
    """Recursively find all aligned_df.csv files in the given directory that match the dataset pattern.

    Args:
    ----
        root_dir: Root directory to start the search.
        dataset: The dataset identifier to filter the models.

    Returns:
    -------
        AlignedDFCollection: A collection of AlignedDF objects.

    """
    aligned_df_collection = AlignedDFCollection()

    for dirpath, _, filenames in tqdm(
        os.walk(
            root_dir,
        ),
    ):
        if "aligned_df.csv" in filenames:
            file_path = pathlib.Path(
                dirpath,
                "aligned_df.csv",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    f"Found aligned_df.csv file: {file_path = }",  # noqa: G004 - low overhead
                )

            aligned_df_object = AlignedDF(
                file_path=file_path,
            )
            if aligned_df_object.metadata.dataset == dataset:
                aligned_df_collection.add_aligned_df(
                    aligned_df=aligned_df_object,
                )

    return aligned_df_collection


def plot_statistics_comparison(
    aggregated_stats: pd.DataFrame,
    stat: str,
    output_dir: os.PathLike,
) -> None:
    """Plot statistics comparison over different checkpoints.

    Args:
    ----
        aggregated_stats (pd.DataFrame): DataFrame with aggregated statistics.
        stat (str): Statistic to compare (e.g., 'mean', 'std', etc.).
        output_dir (str): Directory to save the plots.

    """
    plt.figure(figsize=(10, 6))
    for column in aggregated_stats.columns[:-3]:  # Exclude 'Dataset', 'Model', and 'Checkpoint' columns
        plt.plot(
            aggregated_stats["Checkpoint"],
            aggregated_stats.loc[aggregated_stats.index == stat, column],
            marker="o",
            label=column,
        )

    plt.title(f"Comparison of {stat} across Checkpoints")
    plt.xlabel("Checkpoint")
    plt.ylabel(stat)
    plt.legend(title="Feature")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"{stat}_comparison.png"))
    plt.show()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    root_directory = pathlib.Path(
        embeddings_path_manager.data_dir,
        "analysis/aligned_and_analyzed/twonn",
    )
    dataset_name = "multiwoz21"
    # TODO: This currently does not match the correct dataset name

    output_plot_directory = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "correlation_analyis",
    )

    aligned_df_collection: AlignedDFCollection = find_aligned_dfs(
        root_dir=root_directory,
        dataset=dataset_name,
    )

    # TODO: Currently, we cannot filter for differnt checkpoints of the same model
    aggregated_statistics = aligned_df_collection.get_aggregated_statistics()

    # Example: plot the mean comparison across checkpoints
    plot_statistics_comparison(
        aggregated_statistics,
        "mean",
        output_plot_directory,
    )


if __name__ == "__main__":
    main()
