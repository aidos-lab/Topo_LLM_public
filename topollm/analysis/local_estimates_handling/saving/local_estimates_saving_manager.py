# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

import logging
import pathlib
import pprint
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.storage.saving_and_loading_functions.saving_and_loading import (
    load_dataframe_from_pickle,
    load_numpy_array_from_npy,
    load_python_dict_from_json,
    save_numpy_array_as_npy,
    save_python_dict_as_json,
    save_python_object_as_pickle,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import pandas as pd

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

    array_for_estimator_save_path: pathlib.Path

    additional_distance_computations_results_save_path: pathlib.Path
    additional_pointwise_results_statistics_save_path: pathlib.Path

    def setup_directories(
        self,
    ) -> None:
        for path in [
            self.local_estimates_dir_absolute_path,
            self.global_estimates_save_path,
            self.local_estimates_pointwise_array_save_path,
            self.local_estimates_pointwise_meta_save_path,
            self.array_for_estimator_save_path,
            self.additional_distance_computations_results_save_path,
            self.additional_pointwise_results_statistics_save_path,
        ]:
            # Create the directories if they do not exist
            if not path.parent.exists():
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )


@dataclass
class LocalEstimatesSavingManager:
    """Dataclass to hold the paths for saving and loading local estimates."""

    def __init__(
        self,
        save_path_collection: LocalEstimatesSavePathCollection,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the local estimates saving manager from an embeddings path manager."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        self.save_path_collection: LocalEstimatesSavePathCollection = save_path_collection

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"save_path_collection:\n{pprint.pformat(object=self.save_path_collection)}",  # noqa: G004 - low overhead
            )

    @staticmethod
    def from_embeddings_path_manager(
        embeddings_path_manager: EmbeddingsPathManager,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> "LocalEstimatesSavingManager":
        """Initialize the local estimates saving manager from an embeddings path manager."""
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
        additional_pointwise_results_statistics_save_path: pathlib.Path = (
            embeddings_path_manager.get_additional_pointwise_results_statistics_save_path()
        )
        array_for_estimator_save_path: pathlib.Path = embeddings_path_manager.get_array_for_estimator_save_path()

        # Note: Logging of the save_path_collection is done in the __init__ method
        save_path_collection = LocalEstimatesSavePathCollection(
            local_estimates_dir_absolute_path=local_estimates_dir_absolute_path,
            global_estimates_save_path=global_estimates_save_path,
            local_estimates_pointwise_array_save_path=local_estimates_pointwise_array_save_path,
            local_estimates_pointwise_meta_save_path=local_estimates_pointwise_meta_save_path,
            additional_distance_computations_results_save_path=additional_distance_computations_results_save_path,
            additional_pointwise_results_statistics_save_path=additional_pointwise_results_statistics_save_path,
            array_for_estimator_save_path=array_for_estimator_save_path,
        )

        instance = LocalEstimatesSavingManager(
            save_path_collection=save_path_collection,
            verbosity=verbosity,
            logger=logger,
        )

        return instance

    @staticmethod
    def from_local_estimates_pointwise_dir_absolute_path(
        local_estimates_pointwise_dir_absolute_path: pathlib.Path,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> "LocalEstimatesSavingManager":
        """Initialize the local estimates saving manager from a local estimates pointwise directory."""
        # Parent directory of the local estimates pointwise directory
        local_estimates_dir_absolute_path: pathlib.Path = local_estimates_pointwise_dir_absolute_path.parent

        # Files located in the parent directory
        global_estimates_save_path: pathlib.Path = pathlib.Path(
            local_estimates_dir_absolute_path,
            "global_estimate.npy",
        )
        additional_distance_computations_results_save_path: pathlib.Path = pathlib.Path(
            local_estimates_dir_absolute_path,
            "additional_distance_computations_results.json",
        )
        array_for_estimator_save_path: pathlib.Path = pathlib.Path(
            local_estimates_dir_absolute_path,
            "array_for_estimator.npy",
        )

        # Files located in the local estimates pointwise directory
        local_estimates_pointwise_array_save_path: pathlib.Path = pathlib.Path(
            local_estimates_pointwise_dir_absolute_path,
            "local_estimates_pointwise_array.npy",
        )
        local_estimates_pointwise_meta_save_path: pathlib.Path = pathlib.Path(
            local_estimates_pointwise_dir_absolute_path,
            "local_estimates_pointwise_meta.pkl",
        )
        additional_pointwise_results_statistics_save_path: pathlib.Path = pathlib.Path(
            local_estimates_pointwise_dir_absolute_path,
            "additional_pointwise_results_statistics.json",
        )

        # Note: Logging of the save_path_collection is done in the __init__ method
        save_path_collection = LocalEstimatesSavePathCollection(
            local_estimates_dir_absolute_path=local_estimates_dir_absolute_path,
            global_estimates_save_path=global_estimates_save_path,
            local_estimates_pointwise_array_save_path=local_estimates_pointwise_array_save_path,
            local_estimates_pointwise_meta_save_path=local_estimates_pointwise_meta_save_path,
            additional_distance_computations_results_save_path=additional_distance_computations_results_save_path,
            additional_pointwise_results_statistics_save_path=additional_pointwise_results_statistics_save_path,
            array_for_estimator_save_path=array_for_estimator_save_path,
        )

        instance = LocalEstimatesSavingManager(
            save_path_collection=save_path_collection,
            verbosity=verbosity,
            logger=logger,
        )

        return instance

    def __repr__(
        self,
    ) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}(save_path_collection={pprint.pformat(object=self.save_path_collection)})"

    def __str__(
        self,
    ) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}(save_path_collection={pprint.pformat(object=self.save_path_collection)})"

    def save_local_estimates(
        self,
        local_estimates_container: LocalEstimatesContainer,
    ) -> None:
        """Save the local estimates array to disk.

        Note that if one of these objects is None in the LocalEstimatesContainer, it will not be saved.
        This logic is implemented in the individual data type saving functions.
        """
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling save_local_estimates ...",
            )

        # This creates the directories if they do not exist
        self.save_path_collection.setup_directories()

        # # # #
        # Save the local estimates
        save_numpy_array_as_npy(
            array_np=local_estimates_container.pointwise_results_array_np,
            save_path=self.save_path_collection.local_estimates_pointwise_array_save_path,
            array_name_for_logging="pointwise_results_array_np",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Save the meta data
        save_python_object_as_pickle(
            python_object=local_estimates_container.pointwise_results_meta_frame,
            save_path=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            python_object_name_for_logging="pointwise_results_meta_frame",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Save the global estimate
        save_numpy_array_as_npy(
            array_np=local_estimates_container.global_estimate_array_np,
            save_path=self.save_path_collection.global_estimates_save_path,
            array_name_for_logging="global_estimate_array_np",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Save the additional additional_distance_computations_results dictionary as json file
        save_python_dict_as_json(
            python_dict=local_estimates_container.additional_distance_computations_results,
            save_path=self.save_path_collection.additional_distance_computations_results_save_path,
            python_dict_name_for_logging="additional_distance_computations_results",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Save the additional statistics dictionary as json file
        save_python_dict_as_json(
            python_dict=local_estimates_container.additional_pointwise_results_statistics,
            save_path=self.save_path_collection.additional_pointwise_results_statistics_save_path,
            python_dict_name_for_logging="additional_pointwise_results_statistics",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Save the array for the estimator
        save_numpy_array_as_npy(
            array_np=local_estimates_container.array_for_estimator_np,
            save_path=self.save_path_collection.array_for_estimator_save_path,
            array_name_for_logging="array_for_estimator",
            verbosity=self.verbosity,
            logger=self.logger,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling save_local_estimates DONE",
            )

    def load_local_estimates(
        self,
    ) -> LocalEstimatesContainer:
        """Load the local estimates from disk."""
        # # # #
        # Local estimates array, which is required
        pointwise_results_array_np: np.ndarray | None = load_numpy_array_from_npy(
            save_path=self.save_path_collection.local_estimates_pointwise_array_save_path,
            array_name_for_logging="pointwise_results_array_np",
            required=True,
            verbosity=self.verbosity,
            logger=self.logger,
        )
        if not isinstance(
            pointwise_results_array_np,
            np.ndarray,
        ):
            msg = (
                f"Expected pointwise_results_array_np to be of type np.ndarray, "
                f"but got {type(pointwise_results_array_np) = }."
            )
            raise TypeError(
                msg,
            )

        # # # #
        # Check if the meta data exists
        pointwise_results_meta_frame: pd.DataFrame | None = load_dataframe_from_pickle(
            save_path=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            dataframe_name_for_logging="pointwise_results_meta_frame",
            required=True,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Check if the global estimate exists
        global_estimate_array_np: np.ndarray | None = load_numpy_array_from_npy(
            save_path=self.save_path_collection.global_estimates_save_path,
            array_name_for_logging="global_estimate_array_np",
            required=False,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Load the additional dictionaries
        additional_distance_computations_results: dict | None = load_python_dict_from_json(
            save_path=self.save_path_collection.additional_distance_computations_results_save_path,
            python_dict_name_for_logging="additional_distance_computations_results",
            required=False,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        additional_pointwise_results_statistics: dict | None = load_python_dict_from_json(
            save_path=self.save_path_collection.additional_pointwise_results_statistics_save_path,
            python_dict_name_for_logging="additional_pointwise_results_statistics",
            required=False,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Load the array for the estimator
        array_for_estimator_np: np.ndarray | None = load_numpy_array_from_npy(
            save_path=self.save_path_collection.array_for_estimator_save_path,
            array_name_for_logging="array_for_estimator_np",
            required=False,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        # # # #
        # Create the local estimates container
        local_estimates_container = LocalEstimatesContainer(
            pointwise_results_array_np=pointwise_results_array_np,
            pointwise_results_meta_frame=pointwise_results_meta_frame,
            global_estimate_array_np=global_estimate_array_np,
            additional_distance_computations_results=additional_distance_computations_results,
            additional_pointwise_results_statistics=additional_pointwise_results_statistics,
            array_for_estimator_np=array_for_estimator_np,
        )

        return local_estimates_container
