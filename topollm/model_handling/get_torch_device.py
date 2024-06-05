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

"""Get the preferred torch device."""

import logging

import torch

from topollm.typing.enums import PreferredTorchBackend, Verbosity

default_logger = logging.getLogger(__name__)


def get_torch_device(
    preferred_torch_backend: PreferredTorchBackend,
    verbosity: int = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> torch.device:
    """Get the preferred torch device."""
    # Directly select 'cpu' if preferred,
    # since it is always available
    if preferred_torch_backend == PreferredTorchBackend.CPU:
        device = torch.device("cpu")
    # For 'cuda', check if it is the preference
    # and if it is available
    elif preferred_torch_backend == PreferredTorchBackend.CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
    # For 'mps', check if it is the preference
    # and if it is available
    elif (
        preferred_torch_backend == PreferredTorchBackend.MPS and torch.backends.mps.is_available()
    ) or torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbosity >= 1:
        logger.info(
            f"{device = }",  # noqa: G004 - low overhead
        )

    return device
