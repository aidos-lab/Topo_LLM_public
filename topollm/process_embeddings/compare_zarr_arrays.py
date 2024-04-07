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

import logging
import pathlib

import zarr
import zarr.core

from topollm.logging.log_array_info import log_array_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add stdout handler
logger.addHandler(logging.StreamHandler())

def main() -> None:
    repository_base_path = pathlib.Path("/home/benjamin_ruppik/git-source/Topo_LLM/",)

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
    ):
        if not isinstance(
            array,
            zarr.core.Array,
        ):
            raise ValueError(f"{array_name = } " f"is not a zarr.core.Array")

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
            f"{array_1.shape = } != " f"{array_2.shape = }"
        )
        return None
    
    # # # #
    # Check if the zarr arrays contain the same values

    if not (array_1 == array_2).all():
        logger.error("Arrays are not equal.")
        return None

    return None


if __name__ == "__main__":
    main()
