"""Example script to load a dataloader processed file and check the content."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch
import transformers

from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.setup_omega_conf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_list_info import log_list_info
from topollm.logging.log_recursive_dict_info import log_recursive_dict_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.task_performance_analysis.setsumbt.stack_tensors_from_dialogues_and_filter_fully_padded_utterances import (
    stack_tensors_from_dialogues_and_filter_fully_padded_utterances,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_handling.loaded_model_container import LoadedModelContainer
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


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
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = (
        loaded_model_container.tokenizer
    )
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(
        pad_token_id,
        int,
    ):
        msg = "pad_token_id is not an integer"
        raise ValueError(  # noqa: TRY004
            msg,
        )

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

    max_index: int = 1_000
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

        check_if_sequence_is_compatible_with_attention_via_mask(
            sequence_to_decode=attention_sequence,
            attention_sequence=sequence_to_decode,
            pad_token_id=pad_token_id,
            verbosity=verbosity,
            logger=logger,
        )

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


def check_if_sequence_is_compatible_with_attention_via_mask(
    sequence_to_decode: torch.Tensor,
    attention_sequence: torch.Tensor,
    pad_token_id: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> bool:
    """Check if the token sequence is compatible with its attention mask.

    For each position in the sequence, the following must hold:
      - If the attention mask is 1, the token must not be the pad token.
      - If the attention mask is 0, the token must be the pad token.

    Args:
        sequence_to_decode (Tensor): A tensor of token IDs.
        attention_sequence (Tensor): A tensor representing the attention mask (0 or 1).
        pad_token_id (int): The token ID used for padding.
        verbosity:
            Verbosity level for logging.
        logger:
            Logger for warnings.

    Returns:
        bool: True if the sequence is compatible with the attention mask; False otherwise.

    """
    # Create boolean masks for invalid positions.
    # For attention == 1, a token is invalid if it equals the pad token.
    invalid_when_attention_on: torch.Tensor = (attention_sequence == 1) & (sequence_to_decode == pad_token_id)
    # For attention == 0, a token is invalid if it is not the pad token.
    invalid_when_attention_off: torch.Tensor = (attention_sequence == 0) & (sequence_to_decode != pad_token_id)

    # Log each individual mismatch for further inspection.
    # torch.nonzero returns the indices where the condition is True.
    if verbosity >= Verbosity.NORMAL:
        for idx in torch.nonzero(
            input=invalid_when_attention_on,
            as_tuple=False,
        ).tolist():
            logger.warning(
                msg=f"Mismatch at index {idx = }: attention is 1 but token is pad_token_id",  # noqa: G004 - low overhead
            )
        for idx in torch.nonzero(
            input=invalid_when_attention_off,
            as_tuple=False,
        ).tolist():
            logger.warning(
                msg=f"Mismatch at index {idx = }: attention is 0 but token is not pad_token_id",  # noqa: G004 - low overhead
            )

    # Determine overall compatibility.
    has_invalid = invalid_when_attention_on.any().item() or invalid_when_attention_off.any().item()

    return not has_invalid


def check_if_sequence_is_compatible_with_attention_via_loop(
    sequence_to_decode: torch.Tensor,
    attention_sequence: torch.Tensor,
    pad_token_id: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> bool:
    """Check if the sequence is compatible with the attention mask."""
    check_passed = True
    for sequence_entry, attention_entry in zip(
        sequence_to_decode,
        attention_sequence,
        strict=True,
    ):
        if attention_entry == 1 and sequence_entry == pad_token_id:
            check_passed = False
            if verbosity >= Verbosity.NORMAL:
                logger.warning(
                    msg="Attention entry is 1 but sequence entry is pad_token_id",
                )
        if attention_entry == 0 and sequence_entry != pad_token_id:
            check_passed = False
            if verbosity >= Verbosity.NORMAL:
                logger.warning(
                    msg="Attention entry is 0 but sequence entry is not pad_token_id",
                )

    # If no check failed, we went through this loop without setting check_passed to False

    return check_passed


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


if __name__ == "__main__":
    main()
