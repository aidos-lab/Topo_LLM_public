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
        additional_pointwise_results_statistics_save_path: pathlib.Path = (
            embeddings_path_manager.get_additional_pointwise_results_statistics_save_path()
        )
        array_for_estimator_save_path: pathlib.Path = embeddings_path_manager.get_array_for_estimator_save_path()

        self.save_path_collection = LocalEstimatesSavePathCollection(
            local_estimates_dir_absolute_path=local_estimates_dir_absolute_path,
            global_estimates_save_path=global_estimates_save_path,
            local_estimates_pointwise_array_save_path=local_estimates_pointwise_array_save_path,
            local_estimates_pointwise_meta_save_path=local_estimates_pointwise_meta_save_path,
            additional_distance_computations_results_save_path=additional_distance_computations_results_save_path,
            additional_pointwise_results_statistics_save_path=additional_pointwise_results_statistics_save_path,
            array_for_estimator_save_path=array_for_estimator_save_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"save_path_collection:\n{pprint.pformat(object=self.save_path_collection)}",  # noqa: G004 - low overhead
            )

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
        self.save_numpy_array_as_npy(
            array_np=local_estimates_container.pointwise_results_array_np,
            save_path=self.save_path_collection.local_estimates_pointwise_array_save_path,
            array_name_for_logging="pointwise_results_array_np",
        )

        # # # #
        # Save the meta data
        self.save_python_object_as_pickle(
            python_object=local_estimates_container.pointwise_results_meta_frame,
            save_path=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            python_object_name_for_logging="pointwise_results_meta_frame",
        )

        # # # #
        # Save the global estimate
        self.save_numpy_array_as_npy(
            array_np=local_estimates_container.global_estimate_array_np,
            save_path=self.save_path_collection.global_estimates_save_path,
            array_name_for_logging="global_estimate_array_np",
        )

        # # # #
        # Save the additional additional_distance_computations_results dictionary as json file
        self.save_python_dict_as_json(
            python_dict=local_estimates_container.additional_distance_computations_results,
            save_path=self.save_path_collection.additional_distance_computations_results_save_path,
            python_dict_name_for_logging="additional_distance_computations_results",
        )

        # # # #
        # Save the additional statistics dictionary as json file
        self.save_python_dict_as_json(
            python_dict=local_estimates_container.additional_pointwise_results_statistics,
            save_path=self.save_path_collection.additional_pointwise_results_statistics_save_path,
            python_dict_name_for_logging="additional_pointwise_results_statistics",
        )

        # # # #
        # Save the array for the estimator
        self.save_numpy_array_as_npy(
            array_np=local_estimates_container.array_for_estimator_np,
            save_path=self.save_path_collection.array_for_estimator_save_path,
            array_name_for_logging="array_for_estimator",
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Calling save_local_estimates DONE",
            )

    def save_numpy_array_as_npy(
        self,
        array_np: np.ndarray | None,
        save_path: pathlib.Path,
        array_name_for_logging: str = "array_np",
    ) -> None:
        if array_np is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {array_name_for_logging} array to "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            if not isinstance(
                array_np,
                np.ndarray,
            ):
                msg = f"Expected {array_name_for_logging} to be of type np.ndarray, but got {type(array_np)}."
                raise ValueError(
                    msg,
                )

            np.save(
                file=save_path,
                arr=array_np,
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {array_name_for_logging} array to "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        else:
            self.logger.info(
                msg=f"No {array_name_for_logging} to save.",  # noqa: G004 - low overhead
            )

    def load_numpy_array_from_npy(
        self,
        save_path: pathlib.Path,
        array_name_for_logging: str = "array_np",
        *,
        required: bool = True,
    ) -> np.ndarray | None:
        if save_path.exists():
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading array {array_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            try:
                array_np = np.load(
                    file=save_path,
                )
            except FileNotFoundError as e:
                msg: str = f"FileNotFoundError: {e}"
                self.logger.exception(
                    msg=msg,
                )
                raise

            if not isinstance(
                array_np,
                np.ndarray,
            ):
                msg = f"Expected array {array_name_for_logging} to be of type np.ndarray, but got {type(array_np) = }."
                raise ValueError(
                    msg,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading array {array_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        elif required:
            msg: str = f"Required file for {array_name_for_logging} not found: {save_path = }."
            raise FileNotFoundError(
                msg,
            )
        else:
            array_np = None

        return array_np

    def save_python_object_as_pickle(
        self,
        python_object: object | None,
        save_path: pathlib.Path,
        python_object_name_for_logging: str = "python_object",
    ) -> None:
        """Save a python object as pickle file.

        Note:
        This function only supports pandas DataFrames.

        """
        if python_object is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            # Save object as pickle file
            if isinstance(
                python_object,
                pd.DataFrame,
            ):
                python_object.to_pickle(
                    path=save_path,
                )
            else:
                msg: str = f"Unsupported type for {python_object_name_for_logging}: {type(python_object) = }."
                raise ValueError(
                    msg,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        else:
            self.logger.info(
                msg=f"No {python_object_name_for_logging} to save.",  # noqa: G004 - low overhead
            )

    def load_dataframe_from_pickle(
        self,
        save_path: pathlib.Path,
        dataframe_name_for_logging: str = "dataframe",
        *,
        required: bool = True,
    ) -> pd.DataFrame | None:
        if save_path.exists():
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading {dataframe_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            try:
                dataframe = pd.read_pickle(  # noqa: S301 - we trust our own data
                    filepath_or_buffer=save_path,
                )
            except FileNotFoundError as e:
                msg: str = f"FileNotFoundError: {e}"
                self.logger.exception(
                    msg=msg,
                )
                raise

            if not isinstance(
                dataframe,
                pd.DataFrame,
            ):
                msg = f"Expected {dataframe_name_for_logging} to be of type pd.DataFrame, but got {type(dataframe) = }."
                raise ValueError(
                    msg,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading {dataframe_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        elif required:
            msg: str = f"Required file for {dataframe_name_for_logging} not found: {save_path = }."
            raise FileNotFoundError(
                msg,
            )
        else:
            dataframe = None

        return dataframe

    def save_python_dict_as_json(
        self,
        python_dict: dict | None,
        save_path: pathlib.Path,
        python_dict_name_for_logging: str = "python_dict",
    ) -> None:
        if python_dict is not None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {python_dict_name_for_logging} to "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            # Save dictionary as json file
            with save_path.open(
                mode="w",
            ) as fp:
                json.dump(
                    obj=python_dict,
                    fp=fp,
                    sort_keys=True,
                    indent=4,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Saving {python_dict_name_for_logging} to "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        else:
            self.logger.info(
                msg=f"No {python_dict} to save.",  # noqa: G004 - low overhead
            )

    def load_python_dict_from_json(
        self,
        save_path: pathlib.Path,
        python_dict_name_for_logging: str = "python_dict",
        *,
        required: bool = True,
    ) -> dict | None:
        if save_path.exists():
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading {python_dict_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } ...",
                )

            with save_path.open(
                mode="r",
            ) as fp:
                python_dict = json.load(
                    fp=fp,
                )

            if not isinstance(
                python_dict,
                dict,
            ):
                msg = f"Expected {python_dict_name_for_logging} to be of type dict, but got {type(python_dict) = }."
                raise ValueError(
                    msg,
                )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Loading {python_dict_name_for_logging} from "  # noqa: G004 - low overhead
                    f"{save_path = } DONE",
                )
        elif required:
            msg: str = f"Required file for {python_dict_name_for_logging} not found: {save_path = }."
            raise FileNotFoundError(
                msg,
            )
        else:
            python_dict = None

        return python_dict

    def load_local_estimates(
        self,
    ) -> LocalEstimatesContainer:
        """Load the local estimates from disk."""
        # # # #
        # Local estimates array, which is required
        pointwise_results_array_np = self.load_numpy_array_from_npy(
            save_path=self.save_path_collection.local_estimates_pointwise_array_save_path,
            array_name_for_logging="pointwise_results_array_np",
            required=True,
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
        pointwise_results_meta_frame: pd.DataFrame | None = self.load_dataframe_from_pickle(
            save_path=self.save_path_collection.local_estimates_pointwise_meta_save_path,
            dataframe_name_for_logging="pointwise_results_meta_frame",
            required=True,
        )

        # # # #
        # Check if the global estimate exists
        global_estimate_array_np: np.ndarray | None = self.load_numpy_array_from_npy(
            save_path=self.save_path_collection.global_estimates_save_path,
            array_name_for_logging="global_estimate_array_np",
            required=False,
        )

        # # # #
        # Load the additional dictionaries
        additional_distance_computations_results: dict | None = self.load_python_dict_from_json(
            save_path=self.save_path_collection.additional_distance_computations_results_save_path,
            python_dict_name_for_logging="additional_distance_computations_results",
            required=False,
        )

        additional_pointwise_results_statistics: dict | None = self.load_python_dict_from_json(
            save_path=self.save_path_collection.additional_pointwise_results_statistics_save_path,
            python_dict_name_for_logging="additional_pointwise_results_statistics",
            required=False,
        )

        # # # #
        # Load the array for the estimator
        array_for_estimator_np: np.ndarray | None = self.load_numpy_array_from_npy(
            save_path=self.save_path_collection.array_for_estimator_save_path,
            array_name_for_logging="array_for_estimator_np",
            required=False,
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
