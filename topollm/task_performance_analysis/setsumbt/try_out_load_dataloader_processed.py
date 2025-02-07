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

"""Example script to load a dataloader processed file and check the content."""

import logging
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch
import transformers

from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_list_info import log_list_info
from topollm.logging.log_recursive_dict_info import log_recursive_dict_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_handling.loaded_model_container import LoadedModelContainer
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # The embeddings path manager will be used to get the data directory
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # # # # # # # # # #
    # Load the dataloaders

    dataloaders_processed_root_directory: pathlib.Path = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models",
        "setsumbt_checkpoints",
        "multiwoz21",
        "dataloaders_processed",
    )

    # selected_dataloader_processed_name = "train_0.data"
    selected_dataloader_processed_name = "train_1.data"

    selected_dataloader_processed_path = pathlib.Path(
        dataloaders_processed_root_directory,
        selected_dataloader_processed_name,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading from {selected_dataloader_processed_path = } ...",  # noqa: G004 - low overhead
        )

    dataloader_processed = torch.load(
        f=selected_dataloader_processed_path,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading from {selected_dataloader_processed_path = } DONE",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{type(dataloader_processed) = }",  # noqa: G004 - low overhead
        )

        if isinstance(
            dataloader_processed,
            dict,
        ):
            logger.info(
                msg="Logging additional information about the dictionary.",
            )

            log_recursive_dict_info(
                dictionary=dataloader_processed,
                dictionary_name="dataloader_processed",
                logger=logger,
            )

    # # # # # # # #
    # Decode some of the input_ids to text to check the content

    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )
    tokenizer = loaded_model_container.tokenizer

    # Example for 'train_0.data':
    # > dataloader_processed["input_ids"].shape = torch.Size([8438, 12, 64])
    #
    # Select the first dialogue:
    # > dataloader_processed['input_ids'][0].shape = torch.Size([12, 64])
    # > dataloader_processed['attention_mask'][0].shape = torch.Size([12, 64])
    # In each dialogue, some attention masks at the end for selected utterances are completely zero.
    # We can select only those utterances where the attention mask is non-zero.
    #
    # Example: Select the first sequence of the first dialogue:
    # > dataloader_processed["input_ids"][0][0]

    for sequence_to_decode in dataloader_processed["input_ids"][0]:
        decode_and_log_sequence(
            sequence_to_decode=sequence_to_decode,
            tokenizer=tokenizer,
            verbosity=verbosity,
            logger=logger,
        )

    # # # #
    # Try out the function to concatenate the tensors of the individual dialogues and filter fully padded utterances

    dataloader_stacked: dict = stack_tensors_from_dialogues_and_filter_fully_padded_utterances(
        dataloader_processed=dataloader_processed,
    )

    max_index: int = 100
    decoded_sequences: list[str] = []
    for sequence_to_decode, attention_sequence, dialogue_id in zip(
        dataloader_stacked["input_ids"][:max_index],
        dataloader_stacked["attention_mask"][:max_index],
        dataloader_stacked["dialogue_ids"][:max_index],
        strict=True,
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=20 * "-",
            )
            logger.info(
                msg=f"{dialogue_id = }",  # noqa: G004 - low overhead
            )

        decoded_sequence = decode_and_log_sequence(
            sequence_to_decode=sequence_to_decode,
            tokenizer=tokenizer,
            verbosity=verbosity,
            logger=logger,
        )
        decoded_sequences.append(
            decoded_sequence,
        )

        # Check that the attention_sequence is compatible with the sequence_to_decode
        if attention_sequence.shape != sequence_to_decode.shape:
            logger.warning(
                msg=f"Attention sequence shape {attention_sequence.shape = } "  # noqa: G004 - low overhead
                f"does not match sequence_to_decode shape {sequence_to_decode.shape = }",
            )
        # Check that the non-zero attention positions correspond to the non-padded positions in the sequence
        # TODO: Implement this check

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=20 * "-",
            )

    log_list_info(
        list_=decoded_sequences,
        list_name="decoded_sequences",
        logger=logger,
    )

    logger.info(
        msg="Running script DONE",
    )


def decode_and_log_sequence(
    sequence_to_decode: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> str:
    """Decode a sequence and log the result."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{sequence_to_decode.shape = }",  # noqa: G004 - low overhead
        )

    # Decode the sequence
    decoded_sequence: str = tokenizer.decode(
        token_ids=sequence_to_decode,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{sequence_to_decode = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{decoded_sequence = }",  # noqa: G004 - low overhead
        )

    return decoded_sequence


def stack_tensors_from_dialogues_and_filter_fully_padded_utterances(
    dataloader_processed: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Concatenate the tensors of the individual dialogues and filter fully padded utterances.

    This function also replicates the dialogue-ids so that each utterance has a dialogue-id from which it originates.
    """
    input_ids_to_stack = []
    attention_masks_to_stack = []
    dialogue_ids_to_concatenate = []
    # Iterate over the input_ids and attention_mask tensors for the dialogues and select only those utterances where the attention mask is non-zero
    #
    # Note the shapes:
    # > dataloader_processed["input_ids"].shape = torch.Size([8438, 12, 64])
    # We want to iterate over the first dimension (8438),
    # then for each dialogue over the second dimension (12) with the turns,
    # and select only those utterances where the attention mask is non-zero.
    for index, (
        dialogue_id,
        input_ids_dialogue,
        attention_mask,
    ) in enumerate(
        iterable=zip(
            dataloader_processed["dialogue_ids"],
            dataloader_processed["input_ids"],
            dataloader_processed["attention_mask"],
            strict=True,
        ),
    ):
        for turn_index, (
            input_ids_turn,
            attention_mask_turn,
        ) in enumerate(
            iterable=zip(
                input_ids_dialogue,
                attention_mask,
                strict=True,
            ),
        ):
            if attention_mask_turn.sum() > 0:
                input_ids_to_stack.append(
                    input_ids_turn,
                )
                attention_masks_to_stack.append(
                    attention_mask_turn,
                )
                dialogue_ids_to_concatenate.append(
                    dialogue_id,
                )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{len(input_ids_to_stack) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{len(attention_masks_to_stack) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{len(dialogue_ids_to_concatenate) = }",  # noqa: G004 - low overhead
        )

    # Stack the tensors of the individual turns
    input_ids_concatenated: torch.Tensor = torch.stack(
        tensors=input_ids_to_stack,
        dim=0,
    )
    attention_masks_concatenated: torch.Tensor = torch.stack(
        tensors=attention_masks_to_stack,
        dim=0,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{input_ids_concatenated.shape = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{attention_masks_concatenated.shape = }",  # noqa: G004 - low overhead
        )

    result = {
        "input_ids": input_ids_concatenated,
        "attention_mask": attention_masks_concatenated,
        "dialogue_ids": dialogue_ids_to_concatenate,
    }

    return result


if __name__ == "__main__":
    setup_omega_conf()

    main()
