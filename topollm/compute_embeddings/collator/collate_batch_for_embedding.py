"""Collate a batch of data for embedding computation and move to device."""

from typing import Any

import torch

from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_model_input_names: list[str] = [
    "input_ids",
    "attention_mask",
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
        model_input_names:
            List of input names for the model.

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

    # Collate model input fields
    pad_token_id: int = loaded_model_container.tokenizer.pad_token_id  # type: ignore[attr-defined]
    # TODO: This assumes that for a given model_input_name, the elements item[model_input_name] all have the same shape.
    # TODO: If not, we need to pad using the tokenizer.pad_token_id

    collated_batch: dict[
        str,
        torch.Tensor,
    ] = {
        model_input_name: torch.tensor(data=[item[model_input_name] for item in batch])
        for model_input_name in model_input_names
    }

    # TODO: We need to check whether the attention masks already exist, and if not need to create the attention mask accordingly.

    # Collate metadata fields into lists
    metadata_keys: list = [k for k in batch[0] if k not in model_input_names]
    metadata_batch: dict[
        str,
        list[Any],
    ] = {metadata_key: [item[metadata_key] for item in batch] for metadata_key in metadata_keys}

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
