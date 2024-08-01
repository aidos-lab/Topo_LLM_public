# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

import logging
import pathlib

import numpy as np
import pandas as pd

from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def save_local_estimates(
    embeddings_path_manager: EmbeddingsPathManager,
    local_estimates_container: LocalEstimatesContainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the local estimates array to disk."""
    local_estimates_dir_absolute_path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    local_estimates_array_save_path = embeddings_path_manager.get_local_estimates_array_save_path()
    local_estimates_meta_save_path = embeddings_path_manager.get_local_estimates_meta_save_path()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{local_estimates_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{local_estimates_meta_save_path = }",  # noqa: G004 - low overhead
        )

    # Make sure the save path exists
    pathlib.Path(local_estimates_dir_absolute_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Saving local estimates array ...")

    np.save(
        file=local_estimates_array_save_path,
        arr=local_estimates_container.results_array_np,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Saving local estimates array DONE")

    if local_estimates_container.results_meta_frame is None:
        logger.info("No meta data to save.")
        return

    if verbosity >= Verbosity.NORMAL:
        logger.info("Saving local estimates meta ...")

    local_estimates_container.results_meta_frame.to_pickle(
        path=local_estimates_meta_save_path,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Saving local estimates meta DONE")


def load_local_estimates(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesContainer:
    """Load the local estimates from disk."""
    local_estimates_dir_absolute_path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    local_estimates_array_save_path = embeddings_path_manager.get_local_estimates_array_save_path()
    local_estimates_meta_save_path = embeddings_path_manager.get_local_estimates_meta_save_path()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{local_estimates_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{local_estimates_meta_save_path = }",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Loading local estimates array ...")

    local_estimates_array = np.load(
        file=local_estimates_array_save_path,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Loading local estimates array DONE")

    # Check if the meta data exists
    if not pathlib.Path(
        local_estimates_meta_save_path,
    ).exists():
        logger.warning(
            "No meta data found.",
        )
        return LocalEstimatesContainer(
            results_array_np=local_estimates_array,
            results_meta_frame=None,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Loading local estimates meta ...")

    # Load the meta data
    local_estimates_meta_frame: pd.DataFrame = pd.read_pickle(  # noqa: S301 - we trust our own data
        filepath_or_buffer=local_estimates_meta_save_path,
    )

    local_estimates_container = LocalEstimatesContainer(
        results_array_np=local_estimates_array,
        results_meta_frame=local_estimates_meta_frame,
    )

    return local_estimates_container
