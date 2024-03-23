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

"""
Convert the split ICLR 2024 submissions data
into huggingface datasets format.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import json
import logging
import os
import pathlib

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
import pandas as pd
from tqdm import tqdm

# Local imports
import convlab  # type: ignore
import topollm.data_processing.DialogueUtteranceDataset as DialogueUtteranceDataset
from topollm.config_classes.MainConfig import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.logging.log_dataset_info import log_torch_dataset_info
from topollm.logging.log_dataframe_info import log_dataframe_info

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# torch.set_num_threads(1)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""

    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    data_dir = main_config.paths.data_dir
    global_logger.info(f"{data_dir = }")

    process_dataset(
        data_dir=data_dir,
        logger=global_logger,
    )

    return None


def process_dataset(
    data_dir: os.PathLike,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    dataset_load_dir = pathlib.Path(
        data_dir,
        "datasets",
        "iclr_2024_submissions",
        "csv_format",
    )

    # Folder into which we save the dataset files
    dataset_save_dir = pathlib.Path(
        data_dir,
        "datasets",
        "iclr_2024_submissions",
        "jsonl_format",
    )
    # Create the folder if it does not exist
    dataset_save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(f"{dataset_load_dir = }")
    logger.info(f"{dataset_save_dir = }")

    split_list = [
        "train",
        "validation",
        "test",
    ]

    for split in tqdm(
        split_list,
        desc="Iterating over splits",
    ):
        csv_file_path = pathlib.Path(
            dataset_load_dir,
            get_csv_file_name(split=split,),
        )
        
        # TODO: Continue here
        logger.info(
            f"Loading data from:\n" f"{csv_file_path = }\n..."
        )
        
        # TODO load the dataset

        global_logger.info(
            f"Loading convlab dataset:\n" f"{csv_file_path = }\nDONE"
        )
        global_logger.info(f"{convlab_dataset_dict.keys() = }")

        # TODO: Make an extra column with concatenated title and abstract

        write_single_split_to_file(
            save_dir=dataset_save_dir,
            split_dataframe=convlab_dataset_dict,
            split=split,
        )

    return None

def get_csv_file_name(split: str,) -> str:
    return f"ICLR_{split}" f".csv"


def write_single_split_to_file(
    save_dir: pathlib.Path,
    split_dataframe: pd.DataFrame,
    split: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    
    # ! TODO Update this

    log_dataframe_info(
        dataset=split_dataframe,
        dataset_name=split,
        num_samples_to_log=5,
        logger=logger,
    )

    # We want to write the dataset entries to a file in JSONlines format.
    # Open the file for writing
    save_file_path = pathlib.Path(
        save_dir,
        f"{split}.jsonl",
    )
    global_logger.info(f"Writing the dataset to file:\n" f"{save_file_path = }\n...")

    with open(
        save_file_path,
        "w",
    ) as file:
        for idx in tqdm(
            range(
                len(split_dataset),
            )
        ):
            sample: dict = split_dataset[idx]
            json.dump(
                obj=sample,
                fp=file,
            )
            file.write("\n")

    global_logger.info(f"Writing the dataset to file:\n" f"{save_file_path = }\nDONE")

    return None


if __name__ == "__main__":
    main()

    global_logger.info("Script Done.")
