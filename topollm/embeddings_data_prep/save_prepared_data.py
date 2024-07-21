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

from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def save_prepared_data(
    embeddings_path_manager: EmbeddingsPathManager,
    arr_no_pad: np.ndarray,
    meta_frame: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the prepared data."""
    prepared_data_dir_absolute_path = embeddings_path_manager.prepared_data_dir_absolute_path

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "prepared_data_dir_absolute_path:%s",
            prepared_data_dir_absolute_path,
        )

    # Make sure the save directory exists
    pathlib.Path(prepared_data_dir_absolute_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    # # # #
    # Save the prepared data
    np.save(
        file=embeddings_path_manager.get_prepared_data_array_save_path(),
        arr=arr_no_pad,
    )

    meta_frame.to_pickle(
        path=embeddings_path_manager.get_prepared_data_meta_save_path(),
    )


# TODO: Add function for loading prepared data (so that we have this in one place)
