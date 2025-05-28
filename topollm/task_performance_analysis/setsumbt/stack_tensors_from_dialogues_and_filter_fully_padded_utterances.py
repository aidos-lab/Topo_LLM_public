# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Stack tensors from SetSUMBT dataloader dialogues and filter fully padded utterances."""

import logging

import torch

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def stack_tensors_from_dialogues_and_filter_fully_padded_utterances(
    dataloader_processed: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict[str, torch.Tensor | list]:
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
    for _index, (
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
        for _turn_index, (
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

    # Stack the tensors of the individual turns into a single tensor.
    input_ids_stacked: torch.Tensor = torch.stack(
        tensors=input_ids_to_stack,
        dim=0,
    )
    attention_masks_stacked: torch.Tensor = torch.stack(
        tensors=attention_masks_to_stack,
        dim=0,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{input_ids_stacked.shape = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{attention_masks_stacked.shape = }",  # noqa: G004 - low overhead
        )

    result: dict[str, torch.Tensor | list] = {
        "input_ids": input_ids_stacked,
        "attention_mask": attention_masks_stacked,
        "dialogue_ids": dialogue_ids_to_concatenate,
    }

    return result
