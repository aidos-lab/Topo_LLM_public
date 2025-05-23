# Copyright 2024
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


def compute_last_save_step(
    total_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    save_steps: int,
) -> int:
    """Compute the last save step for the Huggingface Trainer.

    Args:
    ----
        total_samples:
            Total number of samples in the training dataset.
        batch_size:
            Batch size for training.
        gradient_accumulation_steps:
            Number of gradient accumulation steps.
        num_epochs:
            Number of training epochs.
        save_steps:
            Frequency of saving checkpoints.

    Returns:
    -------
        int: The last save step.

    """
    # Calculate the effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Calculate the number of steps per epoch
    steps_per_epoch = total_samples // effective_batch_size

    # Calculate the total number of training steps
    total_training_steps = steps_per_epoch * num_epochs

    # Calculate the last save step
    last_save_step = (total_training_steps // save_steps) * save_steps

    return last_save_step
