# Copyright 2024
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

import logging
import pathlib

import numpy as np
import zarr

# Note: In zarr 3, one should not import zarr.core since it is part of the private API
from topollm.logging.log_array_info import log_array_info

logger: logging.Logger = logging.getLogger(
    name=__name__,
)
logger.setLevel(level=logging.INFO)
# Add stdout handler
logger.addHandler(hdlr=logging.StreamHandler())


def compare_zarr_arrays(
    zarr1: zarr.Array,
    zarr2: zarr.Array,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """Compare two Zarr arrays for equality within a tolerance.

    Parameters
    ----------
    zarr1: First Zarr array.
    zarr2: Second Zarr array.
    rtol: Relative tolerance parameter for np.allclose().
    atol: Absolute tolerance parameter for np.allclose().

    Returns
    -------
    bool: True if arrays are equal within the given tolerance, False otherwise.

    """
    # First, compare the shapes of the arrays
    if zarr1.shape != zarr2.shape:
        return False

    # Convert Zarr arrays to NumPy arrays
    np_array1 = zarr1[:]
    np_array2 = zarr2[:]

    # Use numpy.allclose() to compare the arrays within a tolerance
    return np.allclose(
        np_array1,
        np_array2,
        rtol=rtol,
        atol=atol,
    )


def main() -> None:
    repository_base_path = pathlib.Path(
        "/home/benjamin_ruppik/git-source/Topo_LLM/",
    )

    # Path to base model embeddings
    path_1 = pathlib.Path(
        repository_base_path,
        "data/embeddings/arrays/data-multiwoz21_split-train_ctxt-dataset_entry_samples-100/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_mask-no_masking/layer-[11]_agg-mean/norm-None/array_dir",
    )
    # Path to the finetuned model embeddings
    path_2 = pathlib.Path(
        repository_base_path,
        "data/embeddings/arrays/data-multiwoz21_split-train_ctxt-dataset_entry_samples-100/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_finetuned-on-multiwoz21_ftm-lora_mask-no_masking/layer-[11]_agg-mean/norm-None/array_dir",
    )

    logger.info(f"{path_1 = }")
    logger.info(f"{path_2 = }")

    array_1 = zarr.open(
        str(path_1),
        mode="r",
    )
    array_2 = zarr.open(
        str(path_2),
        mode="r",
    )

    for array_name, array in zip(
        ["array_1", "array_2"],
        [array_1, array_2],
        strict=True,
    ):
        if not isinstance(
            array,
            zarr.Array,
        ):
            msg: str = f"{array_name = } is not a zarr.Array"
            raise TypeError(
                msg,
            )

        log_array_info(
            array_=array,
            array_name=array_name,
            slice_size_to_log=20,
            log_array_size=True,
            log_row_l2_norms=True,
            log_chunks=True,
            logger=logger,
        )

    # # # #
    # Check if the zarr arrays have the same shape

    if array_1.shape != array_2.shape:
        logger.error(
            msg=f"{array_1.shape = } != {array_2.shape = }",  # noqa: G004 - low overhead
        )
        return

    # # # #
    # Check if the zarr arrays contain the same values

    result: bool = compare_zarr_arrays(
        zarr1=array_1,  # type: ignore - problem with zarr typing
        zarr2=array_2,  # type: ignore - problem with zarr typing
    )
    logger.info(
        "result:\n%s",
        result,
    )

    return


if __name__ == "__main__":
    main()
