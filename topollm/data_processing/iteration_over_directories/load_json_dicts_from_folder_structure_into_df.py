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

"""Tools for iterating over directories and loading json files containing dictionaries into a DataFrame."""

import json
import logging
import os
import pathlib
import pprint

import pandas as pd
from tqdm import tqdm

from topollm.data_processing.dictionary_handling import flatten_dict
from topollm.path_management.parse_path_info import parse_path_info_full
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_json_dicts_from_folder_structure_into_df(
    iteration_root_dir: os.PathLike,
    pattern: str = "**/*.json",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load json files containing dictionaries from the folder structure and organize them in a DataFrame.

    For example:
    This can be used to iterate over directories
    containing the descriptive statistics of estimates for different datasets and models.
    """
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{iteration_root_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{pattern = }",  # noqa: G004 - low overhead
        )

    # Iterate over the different dataset folders in the 'iteration_root_dir' directory
    rootdir: pathlib.Path = pathlib.Path(
        iteration_root_dir,
    )
    # Only match the files compatible with the pattern
    file_path_list: list[pathlib.Path] = [
        f
        for f in rootdir.resolve().glob(
            pattern=pattern,
        )
        if f.is_file()
    ]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{len(file_path_list) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"file_list:\n{pprint.pformat(object=file_path_list)}",  # noqa: G004 - low overhead
        )

    # > Full path example:
    #
    # data/analysis/distances_and_influence_on_losses_and_local_estimates/a-tr-s=60000/twonn/
    # data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/
    # split=validation_samples=10000_sampling=random_sampling-seed=777/
    # edh-mode=masked_token_lvl=token/
    # add-prefix-space=True_max-len=512/
    # model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-800_task=masked_lm_dr=defaults/
    # layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/
    # desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/
    # descriptive_statistics_dict.json
    #
    # > Example dataset folder:
    #
    # data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags

    loaded_data_list: list[dict] = []

    for file_path in tqdm(
        iterable=file_path_list,
        desc="Loading data from files",
    ):
        with file_path.open(
            mode="r",
        ) as file:
            file_data: dict = json.load(
                fp=file,
            )

            flattened_file_data: dict = flatten_dict(
                d=file_data,
                separator="_",
            )

            path_info: dict = parse_path_info_full(
                path=file_path,
            )

            # Combine the path information with the flattened file data
            combined_data: dict = {
                **path_info,
                **flattened_file_data,
                "file_path": str(object=file_path),
            }

            loaded_data_list.append(
                combined_data,
            )

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(
        data=loaded_data_list,
    )

    return result_df
