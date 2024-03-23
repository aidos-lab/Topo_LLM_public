# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import pprint
import warnings

# Third party imports
import numpy as np
import pandas as pd

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def log_dataframe_info(
    df: pd.DataFrame,
    df_name: str,
    max_log_rows: int = 20,
    check_for_nan: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Logs information about a pandas DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame to log information about.
        df_name (str):
            The name of the DataFrame.
        max_log_rows (int, optional):
            The maximum number of rows to log for the head and tail of the DataFrame.
            Defaults to 20.
        check_for_nan (bool, optional):
            Whether to check for NaN values in the DataFrame.
            Defaults to True.
        logger (logging.Logger, optional):
            The logger to log information to.
            Defaults to logging.getLogger(__name__).

    Returns:
        None

    Side effects:
        Logs information about the DataFrame to the logger.
    """

    logger.info(f"{df_name}.shape:\n" f"{df.shape}")
    logger.info(f"{df_name}.info():\n" f"{df.info()}")
    logger.info(
        f"{df_name}.head({max_log_rows}):\n" f"{df.head(max_log_rows).to_string()}"
    )
    logger.info(
        f"{df_name}.tail({max_log_rows}):\n" f"{df.tail(max_log_rows).to_string()}"
    )

    if check_for_nan:
        # Check if the dataframe contains NaN values
        has_nan = df.isna().any().any()
        logger.info(f"has_nan:\n" f"{has_nan}")

        if has_nan:
            logger.warning(f"{df_name}.isna().sum():\n" f"{df.isna().sum()}")
            warnings.warn(
                f"The dataframe contains NaN values. "
                f"Please make sure that this is intended.",
            )

    return


def log_list_info(
    list_: list,
    list_name: str,
    max_log_elements: int = 20,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Logs information about a list.

    Args:
        list_ (list):
            The list to log information about.
        list_name (str):
            The name of the list.
        max_log_elements (int, optional):
            The maximum number of elements to log for the head and tail of the list.
            Defaults to 20.
        logger (logging.Logger, optional):
            The logger to log information to.
            Defaults to logging.getLogger(__name__).

    Returns:
        None

    Side effects:
        Logs information about the list to the logger.
    """

    logger.info(f"len({list_name}):\n" f"{len(list_)}")
    logger.info(
        f"{list_name}[:{max_log_elements}]:\n"
        f"{pprint.pformat(list_[:max_log_elements])}"
    )
    logger.info(
        f"{list_name}[-{max_log_elements}:]:\n"
        f"{pprint.pformat(list_[-max_log_elements:])}"
    )

    return


def log_array_info(
    array_: np.ndarray,
    array_name: str,
    slice_size_to_log: int = 20,
    log_array_size: bool = False,
    log_row_l2_norms: bool = False,
    log_chunks: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    logger.info(f"type({array_name}):\n" f"{type(array_)}")

    logger.info(f"{array_name}.shape:\n" f"{array_.shape}")
    logger.info(f"{array_name}.dtype:\n" f"{array_.dtype}")
    logger.info(
        f"{array_name}[:{slice_size_to_log}]:\n"
        f"{pprint.pformat(array_[:slice_size_to_log])}"
    )
    logger.info(
        f"{array_name}[-{slice_size_to_log}:]:\n"
        f"{pprint.pformat(array_[-slice_size_to_log:])}"
    )

    if log_array_size:
        # Estimate the size of the .npy file in MB
        logger.info(f"{array_name}.nbytes:\n" f"{array_.nbytes}")
        array_file_size_MB = array_.nbytes / 1024**2
        logger.info(f"{array_name} size in MB:\n" f"{array_file_size_MB:.3f} MB")

    if log_chunks:
        # If array_ has a chunks attribute,
        # for instance if it is a zarr.core.Array,
        # log the chunks attribute
        if hasattr(
            array_,
            "chunks",
        ):
            logger.info(f"{array_name}.chunks:\n" f"{array_.chunks}")  # type: ignore
        else:
            logger.info(f"{array_name} has no chunks attribute.")

    if log_row_l2_norms:
        # Log the L2-norms of the first and last 10 rows of features_np
        logger.info(
            f"np.linalg.norm({array_name}[:{slice_size_to_log}], axis=1):\n"
            f"{np.linalg.norm(array_[:slice_size_to_log], axis=1)}"
        )
        logger.info(
            f"np.linalg.norm({array_name}[-{slice_size_to_log}:], axis=1):\n"
            f"{np.linalg.norm(array_[-slice_size_to_log:], axis=1)}"
        )

    return