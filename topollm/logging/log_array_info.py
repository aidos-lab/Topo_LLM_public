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

"""Log information about an array."""

import logging
import pprint
from typing import Any, TypeAlias

import numpy as np
import torch
import zarr

ArrayLike: TypeAlias = np.ndarray | zarr.Array
DType = Any

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_array_info(
    array_: ArrayLike,
    array_name: str,
    slice_size_to_log: int = 20,
    *,
    log_array_size: bool = False,
    log_row_l2_norms: bool = False,
    log_chunks: bool = False,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the array."""
    logger.info(
        msg=f"type({array_name}):\n{type(array_)}",  # noqa: G004 - low overhead
    )

    logger.info(
        msg=f"{array_name}.shape:\n{array_.shape}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{array_name}.dtype:\n{array_.dtype}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{array_name}[:{slice_size_to_log}]:\n{pprint.pformat(array_[:slice_size_to_log])}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{array_name}[-{slice_size_to_log}:]:\n{pprint.pformat(array_[-slice_size_to_log:])}",  # noqa: G004 - low overhead
    )

    if log_array_size:
        # Estimate the size of the .npy file in MB
        logger.info(
            msg=f"{array_name}.nbytes:\n{array_.nbytes}",  # noqa: G004 - low overhead
        )
        array_file_size_in_mb = array_.nbytes / 1024**2
        logger.info(
            msg=f"{array_name} size in MB:\n{array_file_size_in_mb:.3f} MB",  # noqa: G004 - low overhead
        )

    if log_chunks:
        # If array_ has a chunks attribute,
        # for instance if it is a zarr.Array,
        # log the chunks attribute
        if hasattr(
            array_,
            "chunks",
        ):
            logger.info(
                msg=f"{array_name}.chunks:\n{array_.chunks}",  # type: ignore - problem with zarr.Array; # noqa: G004 - low overhead
            )
        else:
            logger.info(
                msg=f"{array_name} has no chunks attribute.",  # noqa: G004 - low overhead
            )

    if log_row_l2_norms:
        # Log the L2-norms of the first and last 10 rows of features_np
        try:
            logger.info(
                f"np.linalg.norm({array_name}[:{slice_size_to_log}], axis=1):\n%s",  # noqa: G004 - low overhead
                np.linalg.norm(
                    array_[:slice_size_to_log],
                    axis=1,
                ),
            )
            logger.info(
                f"np.linalg.norm({array_name}[-{slice_size_to_log}:], axis=1):\n%s",  # noqa: G004 - low overhead
                np.linalg.norm(
                    array_[-slice_size_to_log:],
                    axis=1,
                ),
            )
        except np.exceptions.AxisError as e:
            # For example, we get
            # `numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1`
            # if we try to calculate the L2-norms of a 1D array.
            logger.exception(
                msg=f"Error when trying to calculate L2-norms of {array_name}: {e}",  # noqa: G004 - low overhead
            )


def log_tensor_info(
    tensor: torch.Tensor,
    tensor_name: str,
    slice_size_to_log: int = 1,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about the tensor."""
    logger.info(
        msg=f"type({tensor_name}):\n{type(tensor)}",  # noqa: G004 - low overhead
    )

    logger.info(
        msg=f"{tensor_name}.shape:\n{tensor.shape}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{tensor_name}.dtype:\n{tensor.dtype}",  # noqa: G004 - low overhead
    )

    # Log the first and last slice_size_to_log elements of the tensor
    logger.info(
        msg=f"{tensor_name}[:{slice_size_to_log}]:\n{tensor[:slice_size_to_log]}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{tensor_name}[-{slice_size_to_log}:]:\n{tensor[-slice_size_to_log:]}",  # noqa: G004 - low overhead
    )
