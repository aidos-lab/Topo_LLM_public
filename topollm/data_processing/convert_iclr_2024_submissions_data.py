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

"""Convert the split ICLR 2024 submissions data into huggingface datasets format."""

import json
import logging
import os
import pathlib
from typing import IO, TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_logger = logging.getLogger(__name__)
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


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

    global_logger.info("Running script DONE")


def process_dataset(
    data_dir: os.PathLike,
    logger: logging.Logger = default_logger,
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
        logger.info(f"Processing {split = } ...")

        csv_file_path = pathlib.Path(
            dataset_load_dir,
            get_csv_file_name(
                split=split,
            ),
        )

        logger.info(f"Loading data from:\n" f"{csv_file_path = }\n...")
        # The argument
        # `keep_default_na=False`
        # makes sure that string 'nan' is not interpreted as NaN.
        # Optional: Specify the data types of the columns via `dtype=dtypes`.
        dataframe = pd.read_csv(
            filepath_or_buffer=csv_file_path,
            keep_default_na=False,
            low_memory=False,
        )
        logger.info(f"Loading data from:\n" f"{csv_file_path = }\nDONE")

        # Add additional columns.
        # Here: Combining title and abstract into additional column.
        dataframe_augmented = add_additional_columns(
            dataframe=dataframe,
        )

        write_single_split_dataframe_to_file(
            save_dir=dataset_save_dir,
            split_dataframe=dataframe_augmented,
            split=split,
        )

        logger.info(f"Processing {split = } DONE")


def get_csv_file_name(
    split: str,
) -> str:
    return f"ICLR_{split}.csv"


def add_additional_columns(
    dataframe: pd.DataFrame,
    separator: str = ". ",
) -> pd.DataFrame:
    """Add additional columns to the dataframe.

    We combine the title and abstract into a new column.
    """
    # Make a copy of the dataframe
    dataframe_augmented = dataframe.copy()

    # Add additional columns
    dataframe_augmented["text"] = dataframe_augmented["title"] + separator + dataframe_augmented["abstract"]

    return dataframe_augmented


def write_single_split_dataframe_to_file(
    save_dir: pathlib.Path,
    split_dataframe: pd.DataFrame,
    split: str,
    logger: logging.Logger = default_logger,
) -> None:
    log_dataframe_info(
        df=split_dataframe,
        df_name=split,
        max_log_rows=20,
        check_for_nan=True,
        logger=logger,
    )

    # We want to write the dataset entries to a file in JSONlines format.
    # Open the file for writing
    save_file_path = pathlib.Path(
        save_dir,
        f"{split}.jsonl",
    )
    logger.info(f"Writing the dataset to file:\n{save_file_path = }\n...")  # noqa: G004 - low overhead

    with open(
        save_file_path,
        mode="w",
        encoding="utf-8",
    ) as file:
        iterate_over_dataframe(
            df=split_dataframe,
            file=file,
            logger=logger,
        )

    logger.info(f"Writing the dataset to file:\n{save_file_path = }\nDONE")  # noqa: G004 - low overhead


def iterate_over_dataframe(
    df: pd.DataFrame,
    file: IO,
    logger: logging.Logger = default_logger,
) -> None:
    # Convert the DataFrame to a list of dictionaries
    records: list[dict] = df.to_dict(
        orient="records",
    )

    log_list_info(
        list_=records,
        list_name="records",
        max_log_elements=20,
        logger=logger,
    )

    # Write each record as a JSON-formatted string
    for record in tqdm(records):
        json_record = json.dumps(
            obj=record,
        )
        file.write(
            json_record + "\n",
        )


if __name__ == "__main__":
    main()
