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

"""Save and load local estimates to and from disk."""

import json
import logging
import pathlib
import pprint
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class LocalEstimatesSavePathCollection:
    """Dataclass to hold the paths for saving and loading local estimates."""

    local_estimates_dir_absolute_path: pathlib.Path
    global_estimates_save_path: pathlib.Path
    local_estimates_pointwise_array_save_path: pathlib.Path
    local_estimates_pointwise_meta_save_path: pathlib.Path
    additional_distance_computations_results_save_path: pathlib.Path

    def setup_directories(
        self,
    ) -> None:
        # Make sure the save path exists
        pathlib.Path(self.local_estimates_dir_absolute_path).mkdir(
            parents=True,
            exist_ok=True,
        )
        pathlib.Path(self.global_estimates_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pathlib.Path(self.local_estimates_pointwise_array_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pathlib.Path(self.local_estimates_pointwise_meta_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pathlib.Path(self.additional_distance_computations_results_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )


@dataclass
class LocalEstimatesSavingManager:
    """Dataclass to hold the paths for saving and loading local estimates."""

    def __init__(
        self,
        embeddings_path_manager: EmbeddingsPathManager,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the local estimates saving manager from an embeddings path manager."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        local_estimates_dir_absolute_path: pathlib.Path = (
            embeddings_path_manager.get_local_estimates_dir_absolute_path()
        )
        global_estimates_save_path: pathlib.Path = embeddings_path_manager.get_global_estimate_save_path()
        local_estimates_pointwise_array_save_path: pathlib.Path = (
            embeddings_path_manager.get_local_estimates_pointwise_array_save_path()
        )
        local_estimates_pointwise_meta_save_path: pathlib.Path = (
            embeddings_path_manager.get_local_estimates_pointwise_meta_save_path()
        )
        additional_distance_computations_results_save_path: pathlib.Path = (
            embeddings_path_manager.get_additional_distance_computations_results_save_path()
        )

        self.save_path_collection = LocalEstimatesSavePathCollection(
            local_estimates_dir_absolute_path=local_estimates_dir_absolute_path,
            global_estimates_save_path=global_estimates_save_path,
            local_estimates_pointwise_array_save_path=local_estimates_pointwise_array_save_path,
            local_estimates_pointwise_meta_save_path=local_estimates_pointwise_meta_save_path,
            additional_distance_computations_results_save_path=additional_distance_computations_results_save_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"save_path_collection:\n{pprint.pformat(object=self.save_path_collection)}",  # noqa: G004 - low overhead
            )

    def save_local_estimates(
        self,
        local_estimates_container: LocalEstimatesContainer,
    ) -> None:
        """Save the local estimates array to disk."""
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling save_local_estimates ...",
            )

        # This creates the directories if they do not exist
        self.save_path_collection.setup_directories()

        # # # #
        # Save the local estimates

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Saving pointwise_results_array_np array to "  # noqa: G004 - low overhead
                f"{self.save_path_collection.local_estimates_pointwise_array_save_path = } ...",
            )

        np.save(
            file=self.save_path_collection.local_estimates_pointwise_array_save_path,
            arr=local_estimates_container.pointwise_results_array_np,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Saving pointwise_results_array_np array to "  # noqa: G004 - low overhead
                f"{self.save_path_collection.local_estimates_pointwise_array_save_path = } DONE",
            )

        # # # #
        # Save the meta data

        if local_estimates_container.pointwise_results_meta_frame is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving local estimates meta to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.local_estimates_pointwise_meta_save_path = } ...",
                )

            local_estimates_container.pointwise_results_meta_frame.to_pickle(
                path=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving local estimates meta to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.local_estimates_pointwise_meta_save_path = } DONE",
                )
        else:
            self.logger.info(
                msg="No meta data to save.",
            )

        # # # #
        # Save the global estimate

        if local_estimates_container.global_estimate_array_np is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving global estimate array to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.global_estimates_save_path = } ...",
                )

            np.save(
                file=self.save_path_collection.global_estimates_save_path,
                arr=local_estimates_container.global_estimate_array_np,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving global estimate array to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.global_estimates_save_path = } DONE",
                )
        else:
            self.logger.info(
                msg="No global estimate to save.",
            )

        # # # #
        # Save the additional additional_distance_computations_results dictionary as json file
        if local_estimates_container.additional_distance_computations_results is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving additional_distance_computations_results to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.additional_distance_computations_results_save_path = } ...",
                )

            # Save dictionary as json file
            with self.save_path_collection.additional_distance_computations_results_save_path.open(
                mode="w",
            ) as fp:
                json.dump(
                    obj=local_estimates_container.additional_distance_computations_results,
                    fp=fp,
                    sort_keys=True,
                    indent=4,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving additional_distance_computations_results to "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.additional_distance_computations_results_save_path = } DONE",
                )
        else:
            self.logger.info(
                msg="No additional_distance_computations_results to save.",
            )

        # TODO Implement saving of the additional local estimates statistics

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling save_local_estimates DONE",
            )

    def load_local_estimates(
        self,
    ) -> LocalEstimatesContainer:
        """Load the local estimates from disk."""
        # # # #
        # Local estimates array
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Loading local estimates array from "  # noqa: G004 - low overhead
                f"{self.save_path_collection.local_estimates_pointwise_array_save_path = } ...",
            )

        try:
            pointwise_results_array_np = np.load(
                file=self.save_path_collection.local_estimates_pointwise_array_save_path,
            )
        except FileNotFoundError as e:
            msg: str = f"FileNotFoundError: {e}"
            self.logger.exception(
                msg=msg,
            )
            raise

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Loading local estimates array from "  # noqa: G004 - low overhead
                f"{self.save_path_collection.local_estimates_pointwise_array_save_path = } DONE",
            )

        # # # #
        # Check if the meta data exists
        if not pathlib.Path(
            self.save_path_collection.local_estimates_pointwise_meta_save_path,
        ).exists():
            self.logger.warning(
                msg="No meta data found.",
            )
            pointwise_results_meta_frame = None
        else:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading local estimates meta from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.local_estimates_pointwise_meta_save_path = } ...",
                )

            # Load the meta data
            pointwise_results_meta_frame = pd.read_pickle(  # noqa: S301 - we trust our own data
                filepath_or_buffer=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading local estimates meta from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.local_estimates_pointwise_meta_save_path = } DONE",
                )

        # # # #
        # Check if the global estimate exists
        if not pathlib.Path(
            self.save_path_collection.global_estimates_save_path,
        ).exists():
            self.logger.warning(
                msg="No global estimate found.",
            )
            global_estimate_array_np = None
        else:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading global estimate array from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.global_estimates_save_path = } ...",
                )

            # Load the global estimate
            global_estimate_array_np = np.load(
                file=self.save_path_collection.global_estimates_save_path,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading global estimate array from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.global_estimates_save_path = } DONE",
                )

        # # # #
        # Load the additional additional_distance_computations_results

        if not pathlib.Path(
            self.save_path_collection.additional_distance_computations_results_save_path,
        ).exists():
            self.logger.warning(
                msg="No additional_distance_computations_results found.",
            )
            additional_distance_computations_results = None
        else:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading additional_distance_computations_results from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.additional_distance_computations_results_save_path = } ...",
                )

            # Load the additional_distance_computations_results
            with self.save_path_collection.additional_distance_computations_results_save_path.open(
                mode="r",
            ) as fp:
                additional_distance_computations_results = json.load(
                    fp=fp,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading additional_distance_computations_results from "  # noqa: G004 - low overhead
                    f"{self.save_path_collection.additional_distance_computations_results_save_path = } DONE",
                )

        # TODO Implement loading of the additional statistics

        # # # #
        # Create the local estimates container
        local_estimates_container = LocalEstimatesContainer(
            pointwise_results_array_np=pointwise_results_array_np,
            pointwise_results_meta_frame=pointwise_results_meta_frame,
            global_estimate_array_np=global_estimate_array_np,
            additional_distance_computations_results=additional_distance_computations_results,
        )

        return local_estimates_container
