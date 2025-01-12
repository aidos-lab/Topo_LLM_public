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

"""Saving and loading functions for python dictionaries, pandas dataframes, and numpy arrays."""

import json
import logging
import pathlib

import numpy as np
import pandas as pd

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saving and loading functions for python dictionaries
# and lists of python dictionaries


def save_python_dict_as_json(
    python_dict: dict | None,
    save_path: pathlib.Path,
    python_dict_name_for_logging: str = "python_dict",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a python dictionary as a json file."""
    if python_dict is None:
        logger.info(
            msg=f"No {python_dict_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
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

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_dict_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )


def load_python_dict_from_json(
    save_path: pathlib.Path,
    python_dict_name_for_logging: str = "python_dict",
    *,
    required: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict | None:
    """Load a python dictionary from a json file."""
    if save_path.exists():
        if verbosity >= Verbosity.NORMAL:
            logger.info(
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

        if verbosity >= Verbosity.NORMAL:
            logger.info(
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


def save_list_of_python_dicts_as_jsonl(
    list_of_python_dicts: list[dict] | None,
    save_path: pathlib.Path,
    python_object_name_for_logging: str = "list_of_python_dicts",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a list of python dictionaries as a jsonl file."""
    if list_of_python_dicts is None:
        logger.info(
            msg=f"No {python_object_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } ...",
        )

    with save_path.open(
        mode="w",
    ) as fp:
        for python_dict in list_of_python_dicts:
            # Indendation is set to None, so that each dictionary is written on a single line.
            # `indent=0` would print the dictionaries on multiple lines, but with no indentation.
            json.dump(
                obj=python_dict,
                fp=fp,
                sort_keys=True,
                indent=None,  # No indentation for jsonl
            )
            fp.write("\n")

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )


def save_list_of_python_dicts_as_indented_text(
    list_of_python_dicts: list[dict] | None,
    save_path: pathlib.Path,
    python_object_name_for_logging: str = "list_of_python_dicts",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a list of python dictionaries as a jsonl file."""
    if list_of_python_dicts is None:
        logger.info(
            msg=f"No {python_object_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } ...",
        )

    with save_path.open(
        mode="w",
    ) as fp:
        for python_dict in list_of_python_dicts:
            # Note that the indentation is set to 4 spaces,
            # thus each dictionary will be written on multiple lines
            json.dump(
                obj=python_dict,
                fp=fp,
                sort_keys=True,
                indent=4,  # Indentation to make it human-readable
            )
            fp.write(
                "," + "\n",  # Add a comma brefore the line break to separate the dictionaries
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saving and loading functions for pandas dataframes


def save_dataframe_as_csv(
    dataframe: pd.DataFrame | None,
    save_path: pathlib.Path,
    dataframe_name_for_logging: str = "dataframe",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a pandas dataframe as a csv file."""
    if dataframe is None:
        logger.info(
            msg=f"No {dataframe_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {dataframe_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } ...",
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        msg: str = f"Expected {dataframe_name_for_logging} to be of type pd.DataFrame, but got {type(dataframe) = }."
        raise TypeError(
            msg,
        )

    dataframe.to_csv(
        path_or_buf=save_path,
        index=False,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {dataframe_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )


def load_dataframe_from_pickle(
    save_path: pathlib.Path,
    dataframe_name_for_logging: str = "dataframe",
    *,
    required: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame | None:
    """Load a pandas dataframe from a pickle file."""
    if save_path.exists():
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading {dataframe_name_for_logging} from "  # noqa: G004 - low overhead
                f"{save_path = } ...",
            )

        try:
            dataframe = pd.read_pickle(  # noqa: S301 - we trust our own data
                filepath_or_buffer=save_path,
            )
        except FileNotFoundError as e:
            msg: str = f"FileNotFoundError: {e}"
            logger.exception(
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

        if verbosity >= Verbosity.NORMAL:
            logger.info(
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saving and loading functions for numpy arrays


def save_numpy_array_as_npy(
    array_np: np.ndarray | None,
    save_path: pathlib.Path,
    array_name_for_logging: str = "array_np",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a numpy array as a npy file."""
    if array_np is None:
        logger.info(
            msg=f"No {array_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {array_name_for_logging} array to "  # noqa: G004 - low overhead
            f"{save_path = } ...",
        )

    if not isinstance(
        array_np,
        np.ndarray,
    ):
        msg: str = f"Expected {array_name_for_logging} to be of type np.ndarray, but got {type(array_np)}."
        raise TypeError(
            msg,
        )

    np.save(
        file=save_path,
        arr=array_np,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {array_name_for_logging} array to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )


def load_numpy_array_from_npy(
    save_path: pathlib.Path,
    array_name_for_logging: str = "array_np",
    *,
    required: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> np.ndarray | None:
    """Load a numpy array from a npy file."""
    if save_path.exists():
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading array {array_name_for_logging} from "  # noqa: G004 - low overhead
                f"{save_path = } ...",
            )

        try:
            array_np = np.load(
                file=save_path,
            )
        except FileNotFoundError as e:
            msg: str = f"FileNotFoundError: {e}"
            logger.exception(
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

        if verbosity >= Verbosity.NORMAL:
            logger.info(
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saving and loading functions for python objects as pickle files


def save_python_object_as_pickle(
    python_object: object | None,
    save_path: pathlib.Path,
    python_object_name_for_logging: str = "python_object",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save a python object as pickle file.

    Note:
    This function only supports pandas DataFrames.

    """
    if python_object is None:
        logger.info(
            msg=f"No {python_object_name_for_logging} to save.",  # noqa: G004 - low overhead
        )
        return  # Early return if no data to save

    if verbosity >= Verbosity.NORMAL:
        logger.info(
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
        raise TypeError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving {python_object_name_for_logging} to "  # noqa: G004 - low overhead
            f"{save_path = } DONE",
        )
