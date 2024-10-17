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

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def save_local_estimates(
    embeddings_path_manager: EmbeddingsPathManager,
    local_estimates_container: LocalEstimatesContainer,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the local estimates array to disk."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling save_local_estimates ...",
        )

    global_estimates_save_path, local_estimates_pointwise_array_save_path, local_estimates_pointwise_meta_save_path = (
        setup_local_estimate_directories(
            embeddings_path_manager=embeddings_path_manager,
            verbosity=verbosity,
            logger=logger,
        )
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Saving pointwise_results_array_np array ...",
        )

    np.save(
        file=local_estimates_pointwise_array_save_path,
        arr=local_estimates_container.pointwise_results_array_np,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Saving pointwise_results_array_np array DONE",
        )

    if local_estimates_container.pointwise_results_meta_frame is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Saving local estimates meta ...",
            )

        local_estimates_container.pointwise_results_meta_frame.to_pickle(
            path=local_estimates_pointwise_meta_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Saving local estimates meta DONE",
            )
    else:
        logger.info(
            msg="No meta data to save.",
        )

    if local_estimates_container.global_estimate_array_np is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Saving global estimate array ...",
            )

        np.save(
            file=global_estimates_save_path,
            arr=local_estimates_container.global_estimate_array_np,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Saving global estimate array DONE",
            )
    else:
        logger.info(
            msg="No global estimate to save.",
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling save_local_estimates DONE",
        )


def setup_local_estimate_directories(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    local_estimates_dir_absolute_path: pathlib.Path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    global_estimates_save_path: pathlib.Path = embeddings_path_manager.get_global_estimate_save_path()
    local_estimates_pointwise_array_save_path: pathlib.Path = (
        embeddings_path_manager.get_local_estimates_pointwise_array_save_path()
    )
    local_estimates_pointwise_meta_save_path: pathlib.Path = (
        embeddings_path_manager.get_local_estimates_pointwise_meta_save_path()
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{global_estimates_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{local_estimates_pointwise_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{local_estimates_pointwise_meta_save_path = }",  # noqa: G004 - low overhead
        )

    # Make sure the save path exists
    pathlib.Path(local_estimates_dir_absolute_path).mkdir(
        parents=True,
        exist_ok=True,
    )
    pathlib.Path(global_estimates_save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pathlib.Path(local_estimates_pointwise_array_save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pathlib.Path(local_estimates_pointwise_meta_save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        global_estimates_save_path,
        local_estimates_pointwise_array_save_path,
        local_estimates_pointwise_meta_save_path,
    )


def load_local_estimates(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesContainer:
    """Load the local estimates from disk."""
    local_estimates_dir_absolute_path: pathlib.Path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    global_estimates_save_path: pathlib.Path = embeddings_path_manager.get_global_estimate_save_path()
    local_estimates_array_save_path: pathlib.Path = (
        embeddings_path_manager.get_local_estimates_pointwise_array_save_path()
    )
    local_estimates_meta_save_path: pathlib.Path = (
        embeddings_path_manager.get_local_estimates_pointwise_meta_save_path()
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{global_estimates_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{local_estimates_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{local_estimates_meta_save_path = }",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Loading local estimates array ...",
        )

    try:
        local_estimates_array = np.load(
            file=local_estimates_array_save_path,
        )
    except FileNotFoundError as e:
        msg: str = f"FileNotFoundError: {e}"
        logger.exception(
            msg=msg,
        )
        raise

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Loading local estimates array DONE",
        )

    # Check if the meta data exists
    if not pathlib.Path(
        local_estimates_meta_save_path,
    ).exists():
        logger.warning(
            msg="No meta data found.",
        )
        pointwise_results_meta_frame = None
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Loading local estimates meta ...",
            )

        # Load the meta data
        pointwise_results_meta_frame = pd.read_pickle(  # noqa: S301 - we trust our own data
            filepath_or_buffer=local_estimates_meta_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Loading local estimates meta DONE",
            )

    # Check if the global estimate exists
    if not pathlib.Path(
        global_estimates_save_path,
    ).exists():
        logger.warning(
            msg="No global estimate found.",
        )
        global_estimate_array = None
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Loading global estimate array ...",
            )

        # Load the global estimate
        global_estimate_array = np.load(
            file=global_estimates_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Loading global estimate array DONE",
            )

    local_estimates_container = LocalEstimatesContainer(
        pointwise_results_array_np=local_estimates_array,
        pointwise_results_meta_frame=pointwise_results_meta_frame,
        global_estimate_array_np=global_estimate_array,
    )

    return local_estimates_container
