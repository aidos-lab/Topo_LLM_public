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

"""Run script to compare computed Hausdorff distances with the local estimates."""

import json
import logging
import os
import pathlib
import pprint
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure

    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


def parse_noise_info(
    folder_name: str,
) -> dict:
    """Extract noise magnitude and seed information from the folder name.

    Args:
        folder_name: Name of the folder containing noise information.

    Returns:
        A dictionary with extracted noise details.
    """
    parts = folder_name.split("_")
    noise_info: dict = {
        "noise_magnitude": None,
        "seed": None,
    }
    for part in parts:
        if part.startswith("gaussian_distor="):
            noise_info["noise_magnitude"] = float(part.split("=")[1])
        elif part.startswith("seed="):
            noise_info["seed"] = int(part.split("=")[1])
    return noise_info


def load_experiment_data(
    experiment_dir: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict | None:
    """Load experiment data from a given directory.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        A dictionary containing the data, or None if an error occurs.

    """

    # TODO: Load this via the local estimates saving manager
    try:
        # # # #
        # Load JSON with distances
        distances_data_path = pathlib.Path(
            experiment_dir,
            "additional_distance_computations_results.json",
        )

        with distances_data_path.open(
            mode="r",
        ) as f:
            distances_data = json.load(
                fp=f,
            )

        # Extract specific distance value (replace 'some_distance_key' with actual key)
        distance = distances_data.get(
            "approximate_hausdorff_via_kdtree",
            None,
        )
        if distance is None:
            msg = "Distance key not found in JSON."
            raise ValueError(msg)

        # # # #
        # Load global estimate
        global_estimates_path = pathlib.Path(
            experiment_dir,
            "global_estimate.npy",
        )
        global_estimate = np.load(
            file=global_estimates_path,
        ).item()

        # # # #
        # Load local estimates

        # TODO: Load the local estimates information

        # # # #
        # Parse noise information
        noise_info = parse_noise_info(
            folder_name=experiment_dir.name,
        )

        output = {
            "experiment": experiment_dir.name,
            "distance": distance,
            "global_estimate": global_estimate,
            **noise_info,
        }
    except Exception as e:
        logger.warning(
            msg=f"Error loading data from {experiment_dir}: {e}",
        )
        return None

    return output


def iterate_and_collect_data(
    base_path: pathlib.Path,
    subdirectory_to_match: str = "n-neighbors-mode=absolute_size_n-neighbors=128",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Iterate over experiment directories and collect data into a DataFrame.

    Args:
        base_path:
            Path to the directory containing experiment subfolders.

    Returns:
        A pandas DataFrame with the collected data.

    """
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Iterating over experiments in {base_path = } ...",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Matching {subdirectory_to_match = }",  # noqa: G004 - low overhead
        )

        base_path_iterdir_list = list(base_path.iterdir())
        logger.info(
            msg=f"{len(base_path_iterdir_list) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"base_path_iterdir_list:\n{pprint.pformat(base_path_iterdir_list)}",  # noqa: G004 - low overhead
        )

    # Container for the collected data
    data: list = []

    experiment_dir_count = 0
    subdirectory_count = 0
    matched_subdirectory_count = 0

    for experiment_dir in tqdm(
        iterable=base_path.iterdir(),
        desc="Iterating over experiments ...",
    ):
        if experiment_dir.is_dir():
            experiment_dir_count += 1

            # Iterate over the subdirectories
            for subdirectory in experiment_dir.iterdir():
                if subdirectory.is_dir():
                    subdirectory_count += 1
                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg=f"{subdirectory = }",  # noqa: G004 - low overhead
                        )

                if subdirectory.is_dir() and subdirectory.name == subdirectory_to_match:
                    matched_subdirectory_count += 1

                    local_estimates_saving_manager: LocalEstimatesSavingManager = (
                        LocalEstimatesSavingManager.from_local_estimates_pointwise_dir_absolute_path(
                            local_estimates_pointwise_dir_absolute_path=subdirectory,
                            verbosity=verbosity,
                            logger=logger,
                        )
                    )
                    local_estimates_container: LocalEstimatesContainer = (
                        local_estimates_saving_manager.load_local_estimates()
                    )

                    pass  # TODO: This is here for setting breakpoints

                    # TODO: This is the old code
                    #
                    # experiment_data = load_experiment_data(
                    #     experiment_dir=subdirectory,
                    #     verbosity=verbosity,
                    #     logger=logger,
                    # )

                    # if experiment_data:
                    #     data.append(experiment_data)

                    # TODO: Separate the loading and the extraction of the relevant values

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Finished iterating over experiments.",
        )
        logger.info(
            msg=f"{experiment_dir_count = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{subdirectory_count = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{matched_subdirectory_count = }",  # noqa: G004 - low overhead
        )

    collected_data_df = pd.DataFrame(
        data=data,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Collected data for {len(collected_data_df)} experiments.",  # noqa: G004 - low overhead
        )
    return collected_data_df


def save_dataframe(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a DataFrame to disk in CSV format.

    Args:
        df: The DataFrame to save.
        output_path: Path to save the DataFrame.

    """
    output_file = pathlib.Path(output_path)
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
            msg=f"DataFrame saved to {output_file = }",  # noqa: G004 - low overhead
        )


def create_scatter_plot(
    df: pd.DataFrame,
    output_file: pathlib.Path | None = None,
    *,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create an interactive scatter plot using Plotly.

    Args:
        df: DataFrame containing the data to plot.
        output_file: Optional; path to save the HTML plot.

    """
    fig: Figure = px.scatter(
        data_frame=df,
        x="distance",
        y="global_estimate",
        color="noise_magnitude",
        hover_data=["experiment", "seed", "noise_magnitude"],
        title="Distance vs Global Estimate (Interactive)",
        labels={
            "distance": "Distance",
            "global_estimate": "Global Estimate",
            "noise_magnitude": "Noise Magnitude",
        },
    )
    fig.update_traces(
        marker={
            "size": 10,
            "opacity": 0.7,
        },
    )

    if show_plot:
        fig.show()

    if output_file:
        output_file = pathlib.Path(
            output_file,
        )
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file} ...",  # noqa: G004 - low overhead
            )
        fig.write_html(
            output_file,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file} DONE",  # noqa: G004 - low overhead
            )


def iterate_over_different_local_estimates_directories(
    base_dir: os.PathLike,
    output_directory: os.PathLike,
    subdirectory_to_match: str = "n-neighbors-mode=absolute_size_n-neighbors=128",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Iterate over experiments, save collected data, and create a scatter plot.

    Args:
        base_dir: Path to the directory containing experiment subfolders.
        output_directory: Root path to save data.

    """
    # Convert to pathlib.Path
    base_dir = pathlib.Path(base_dir)
    output_directory = pathlib.Path(output_directory)

    # Collect data
    df: pd.DataFrame = iterate_and_collect_data(
        base_path=base_dir,
        subdirectory_to_match=subdirectory_to_match,
        verbosity=verbosity,
        logger=logger,
    )

    # Save raw data
    dataframe_output_path = pathlib.Path(
        output_directory,
        "raw_data.csv",
    )
    save_dataframe(
        df=df,
        output_path=dataframe_output_path,
        verbosity=verbosity,
        logger=logger,
    )

    # Check if the DataFrame is empty
    if df.empty:
        logger.warning(
            msg="No data collected. Skipping scatter plot creation.",
        )
        return

    # Create scatter plot
    plot_output_path = pathlib.Path(
        output_directory,
        "scatter_plot.html",
    )
    create_scatter_plot(
        df=df,
        output_file=plot_output_path,
        verbosity=verbosity,
        logger=logger,
    )


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    #
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # ================================================== #
    #
    # ================================================== #

    local_estimates_dir_absolute_path: pathlib.Path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    root_iteration_dir: pathlib.Path = local_estimates_dir_absolute_path.parent

    logger.info(
        msg=f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{root_iteration_dir = }",  # noqa: G004 - low overhead
    )

    # Get the local estimates config description,
    # our directory iteration will match only these directories.
    local_estimates_pointwise_config_description: str = (
        embeddings_path_manager.get_local_estimates_pointwise_config_description()
    )

    output_directory = pathlib.Path(
        embeddings_path_manager.analysis_dir,
        "debug_output_data",  # TODO: Find a good name for the output path
        f"subdirectory_to_match={local_estimates_pointwise_config_description}",
        embeddings_path_manager.get_local_estimates_subfolder_path().parent,
    )

    iterate_over_different_local_estimates_directories(
        base_dir=root_iteration_dir,
        output_directory=output_directory,
        subdirectory_to_match=local_estimates_pointwise_config_description,
        verbosity=verbosity,
        logger=logger,
    )

    # TODO(Ben): Implement iteration over different noise levels and noise seeds, to make a plot of Hausdorff distances vs. local estimates for each noise level and seed.
    # TODO(Ben): Plot of Hausdorff distances vs. global estimates.

    # ================================================== #
    #
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
