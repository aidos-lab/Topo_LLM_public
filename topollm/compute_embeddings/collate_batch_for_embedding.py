# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

from typing import Any

import torch

default_model_input_names: list[str] = [
    "input_ids",
    "attention_mask",
]


def collate_batch(
    batch: list,
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
                A list of dictionaries containing metadata for each instance.

    """
    if model_input_names is None:
        model_input_names = default_model_input_names

    # Collate model input fields
    collated_batch: dict[
        str,
        torch.Tensor,
    ] = {
        model_input_name: torch.tensor(data=[item[model_input_name] for item in batch])
        for model_input_name in model_input_names
    }

    # Collate metadata fields into lists
    metadata_keys = [k for k in batch[0] if k not in model_input_names]
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

    model_inputs = collated_batch["model_inputs"]
    model_inputs = {
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
        "model_inputs": model_inputs,
        "metadata": collated_batch["metadata"],
    }


def collate_batch_and_move_to_device(
    batch: list,
    device: torch.device,
    model_input_names: list[str],
) -> dict:
    """Collate the batch, move model input tensors to device, and keep metadata.

    Args:
    ----
        batch:
            The batch to collate.
        device:
            The device to move the tensors to.
        model_input_names:
            List of input names for the model.

    Returns:
    -------
        result:
            Dictionary containing model inputs moved to device and metadata.
            - 'model_inputs': A dictionary containing tensors for model input moved to the specified device.
            - 'metadata': A dictionary containing metadata for each instance.

    """
    collated_batch = collate_batch(
        batch=batch,
        model_input_names=model_input_names,
    )
    collated_batch = move_collated_batch_to_device(
        collated_batch=collated_batch,
        device=device,
        model_input_names=model_input_names,
    )

    return collated_batch
