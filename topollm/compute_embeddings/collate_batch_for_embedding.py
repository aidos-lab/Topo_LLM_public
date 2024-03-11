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
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Third party imports
import torch

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def collate_batch(
    batch: list,
    model_input_names: list[str] = [
        "input_ids",
        "attention_mask",
    ],
) -> dict:
    """
    Function to collate the batch.

    Args:
        batch:
            The batch to collate.
    Returns:
        collated_batch:
            The collated batch.
    """
    collated_batch: dict[str, torch.Tensor] = {
        model_input_name: torch.tensor([item[model_input_name] for item in batch])
        for model_input_name in model_input_names
    }

    return collated_batch


def move_collated_batch_to_device(
    collated_batch: dict,
    device: torch.device,
    model_input_names: list[str] = [
        "input_ids",
        "attention_mask",
    ],
):
    collated_batch = {
        key: value.to(device)
        for key, value in collated_batch.items()
        if key in model_input_names
    }

    return collated_batch


def collate_batch_and_move_to_device(
    batch: list,
    device: torch.device,
    model_input_names: list[str],
) -> dict:
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
