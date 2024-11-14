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

"""Load dialogue data from the convlab unified dataset format and save it in the huggingface datasets format."""

import json
import logging
import pathlib
from typing import TYPE_CHECKING

import convlab  # type: ignore - we do not add convlab to the requirements
import hydra
import hydra.core.hydra_config
import omegaconf
from tqdm import tqdm

from topollm.config_classes.get_data_dir import get_data_dir
from topollm.data_processing import DialogueUtteranceDataset
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataset_info import log_torch_dataset_info
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

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

    data_dir: pathlib.Path = get_data_dir(
        main_config=main_config,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    convlab_dataset_identifier_list = [
        "multiwoz21",
        "sgd",
    ]

    split_list = [
        "train",
        "validation",
        "test",
    ]

    for convlab_dataset_identifier in tqdm(
        convlab_dataset_identifier_list,
        desc="convlab datasets",
    ):
        # Folder into which we save the dataset files
        save_dir = pathlib.Path(
            data_dir,
            "datasets",
            "dialogue_datasets",
            convlab_dataset_identifier,
        )
        # Create the folder if it does not exist
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        global_logger.info(f"Loading convlab dataset:\n" f"{convlab_dataset_identifier = }\n...")
        convlab_dataset_dict = convlab.util.load_dataset(
            dataset_name=convlab_dataset_identifier,
        )
        global_logger.info(f"Loading convlab dataset:\n" f"{convlab_dataset_identifier = }\nDONE")
        global_logger.info(f"{convlab_dataset_dict.keys() = }")

        for split in tqdm(
            split_list,
            desc="splits",
        ):
            write_single_split_to_file(
                save_dir,
                convlab_dataset_dict,
                split,
            )

    global_logger.info("Running script DONE")


def write_single_split_to_file(
    save_dir: pathlib.Path,
    convlab_dataset_dict: dict[str, list[dict]],
    split: str,
) -> None:
    split_data = convlab_dataset_dict[split]

    split_dataset = DialogueUtteranceDataset.DialogueUtteranceDataset(
        dialogues=split_data,
        split=split,
    )

    log_torch_dataset_info(
        dataset=split_dataset,
        dataset_name=split,
        num_samples_to_log=5,
        logger=global_logger,
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
            ),
        ):
            sample: dict = split_dataset[idx]
            json.dump(
                obj=sample,
                fp=file,
            )
            file.write("\n")

    global_logger.info(f"Writing the dataset to file:\n" f"{save_file_path = }\nDONE")


if __name__ == "__main__":
    main()
