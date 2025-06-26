"""Collate a batch of data for embedding computation and move to device."""

from typing import Any

import torch

from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.loaded_model_container import LoadedModelContainer

INPUT_IDS_COLUMN_NAME: str = "input_ids"
ATTENTION_MASK_COLUMN_NAME: str = "attention_mask"

default_model_input_names: list[str] = [
    INPUT_IDS_COLUMN_NAME,
    ATTENTION_MASK_COLUMN_NAME,
]


def _pad_sequence(
    seq: list[int],
    target_length: int,
    pad_val: int,
) -> list[int]:
    """Pad or truncate ``seq`` on the right up to ``target_length``."""
    if len(seq) >= target_length:
        return seq[:target_length]
    return seq + [pad_val] * (target_length - len(seq))


def collate_batch(
    batch: list,
    loaded_model_container: LoadedModelContainer,
    model_input_names: list[str] | None = None,
    token_level_metadata_keys: list[str] | None = None,
    metadata_pad_val: int = -100,
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Collate the batch into input dictionary and preserve metadata.

    Args:
    ----
        batch:
            The batch to collate.
        loaded_model_container:
            Loaded model container containing the tokenizer and model configuration.
        model_input_names:
            List of input names for the model.
        token_level_metadata_keys:
            Keys in each sample whose values are token-level lists that should be
            padded/truncated to the same length as the model inputs.
            If ``None`` the function infers token-level keys by
            inspecting the first sample for list-valued fields.
            Padded sequences remain regular Python lists (they are *not* converted to tensors).

    Returns:
    -------
        result:
            Dictionary containing model inputs and metadata.
            - 'model_inputs':
                A dictionary containing collated tensors for model input.
            - 'metadata':
                A dictionary containing metadata for each instance.

    """
    if model_input_names is None:
        model_input_names = default_model_input_names

    # ------------------------------------------------------------------ #
    # Collate model input fields                                         #
    # ------------------------------------------------------------------ #
    pad_token_id: int = loaded_model_container.tokenizer.pad_token_id  # type: ignore[attr-defined]

    # Determine target length: Take an explicit max length from the config,
    # and check against the longest sequence in the current batch.
    explicit_max_len: int = loaded_model_container.tokenizer_config.max_length
    longest_in_batch: int = max(
        len(sample[field]) for sample in batch for field in model_input_names if field != ATTENTION_MASK_COLUMN_NAME
    )
    if longest_in_batch > explicit_max_len:
        msg: str = (
            f"Longest sequence in batch ({longest_in_batch=}) exceeds explicit max length "
            f"({explicit_max_len=}) from tokenizer config."
        )
        raise ValueError(
            msg,
        )
    # Use the explicit max length from the tokenizer config as the target length.
    target_len: int = explicit_max_len

    # Ensure we always return an ``attention_mask``.
    if ATTENTION_MASK_COLUMN_NAME not in model_input_names:
        model_input_names.append(ATTENTION_MASK_COLUMN_NAME)

    # Collect padded sequences for every requested field.
    padded: dict[str, list[list[int]]] = {k: [] for k in model_input_names}

    for sample in batch:
        for field in model_input_names:
            if field == ATTENTION_MASK_COLUMN_NAME:
                if field in sample:
                    seq = sample[field]
                    padded_seq: list[int] = _pad_sequence(
                        seq=seq,
                        target_length=target_len,
                        pad_val=0,
                    )
                else:
                    # Create a mask from the first token field
                    ref_field = INPUT_IDS_COLUMN_NAME if INPUT_IDS_COLUMN_NAME in sample else model_input_names[0]
                    real_len = min(
                        len(sample[ref_field]),
                        target_len,
                    )
                    padded_seq = [1] * real_len + [0] * (target_len - real_len)
                padded[field].append(padded_seq)
            else:
                seq = sample[field]
                padded[field].append(
                    _pad_sequence(
                        seq=seq,
                        target_length=target_len,
                        pad_val=pad_token_id,
                    ),
                )

    # Convert lists to tensors
    collated_batch: dict[str, torch.Tensor] = {
        name: torch.tensor(
            data=values,
            dtype=torch.long,
        )
        for name, values in padded.items()
    }

    # Check that the input_ids and attention_mask are present and have the same shape
    if INPUT_IDS_COLUMN_NAME not in collated_batch:
        msg: str = f"Collated batch is missing required model input names: {INPUT_IDS_COLUMN_NAME=}."
        raise ValueError(
            msg,
        )
    if ATTENTION_MASK_COLUMN_NAME not in collated_batch:
        msg: str = f"Collated batch is missing required model input names: {ATTENTION_MASK_COLUMN_NAME=}."
        raise ValueError(
            msg,
        )
    if collated_batch[INPUT_IDS_COLUMN_NAME].shape != collated_batch[ATTENTION_MASK_COLUMN_NAME].shape:
        msg: str = (
            f"Collated batch has mismatched shapes for {INPUT_IDS_COLUMN_NAME=} and {ATTENTION_MASK_COLUMN_NAME=}: "
            f"{collated_batch[INPUT_IDS_COLUMN_NAME].shape=} vs {collated_batch[ATTENTION_MASK_COLUMN_NAME].shape=}"
        )
        raise ValueError(
            msg,
        )

    # ------------------------------------------------------------------ #
    # Collate metadata                                                   #
    # ------------------------------------------------------------------ #
    if token_level_metadata_keys is None:
        token_level_metadata_keys = [
            k for k, v in batch[0].items() if k not in model_input_names and isinstance(v, list)
        ]

    sequence_level_metadata_keys: list = [
        k for k in batch[0] if k not in model_input_names and k not in token_level_metadata_keys
    ]

    metadata_batch: dict[
        str,
        Any,
    ] = {}

    # Sequence-level metadata (no padding)
    for k in sequence_level_metadata_keys:
        metadata_batch[k] = [item[k] for item in batch]

    # Token-level metadata (pad / truncate, keep as Python lists)
    for k in token_level_metadata_keys:
        metadata_batch[k] = [
            _pad_sequence(
                seq=item[k],
                target_length=target_len,
                pad_val=metadata_pad_val,
            )
            for item in batch
        ]

    return {
        "model_inputs": collated_batch,
        "metadata": metadata_batch,
    }


def move_collated_batch_to_device(
    collated_batch: dict,
    device: torch.device,
    model_input_names: list[str] | None = None,
) -> dict:
    """Move collated batch tensors to the specified device."""
    if model_input_names is None:
        model_input_names = default_model_input_names

    model_inputs: dict = collated_batch["model_inputs"]
    model_inputs_on_device: dict = {
        key: value.to(device=device)
        for key, value in model_inputs.items()
        if (
            key in model_input_names
            and isinstance(
                value,
                torch.Tensor,
            )
        )
    }

    return {
        "model_inputs": model_inputs_on_device,
        "metadata": collated_batch["metadata"],
    }


def collate_batch_and_move_to_device(
    batch: list,
    device: torch.device,
    loaded_model_container: LoadedModelContainer,
    model_input_names: list[str] | None = None,
) -> dict:
    """Collate the batch, move model input tensors to device, and keep metadata.

    Args:
    ----
        batch:
            The batch to collate.
        device:
            The device to move the tensors to.
        tokenizer_config:
            Configuration for the tokenizer.
        model_input_names:
            List of input names for the model.

    Returns:
    -------
        result:
            Dictionary containing model inputs moved to device and metadata.
            - 'model_inputs': A dictionary containing tensors for model input moved to the specified device.
            - 'metadata': A dictionary containing metadata for each instance.

    """
    collated_batch: dict = collate_batch(
        batch=batch,
        loaded_model_container=loaded_model_container,
        model_input_names=model_input_names,
    )
    collated_batch: dict = move_collated_batch_to_device(
        collated_batch=collated_batch,
        device=device,
        model_input_names=model_input_names,
    )

    return collated_batch
