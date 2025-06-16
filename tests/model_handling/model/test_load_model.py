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


import logging
import os

import pytest
import torch
from transformers import PreTrainedModel

from tests.model_handling.parameter_lists import (
    example_pretrained_model_name_or_path_list,
)
from topollm.model_handling.model.load_model import load_model


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    example_pretrained_model_name_or_path_list,
)
@pytest.mark.uses_transformers_models
def test_load_model(
    pretrained_model_name_or_path: str | os.PathLike,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    """Test the load_model function."""
    model = load_model(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device_fixture,
        verbosity=1,
        logger=logger_fixture,
    )

    assert model is not None  # noqa: S101 - pytest assert
    assert isinstance(  # noqa: S101 - pytest assert
        model,
        PreTrainedModel,
    )
