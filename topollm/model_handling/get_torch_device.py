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
