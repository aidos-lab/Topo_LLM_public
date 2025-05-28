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


"""Test the do_finetuning_process function."""

import logging

import pytest
import torch

from topollm.config_classes.main_config import MainConfig
from topollm.model_finetuning.do_finetuning_process import do_finetuning_process


@pytest.mark.uses_transformers_models
@pytest.mark.high_memory_usage
@pytest.mark.slow
@pytest.mark.very_slow
def test_do_finetuning_process(
    main_config: MainConfig,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    """Test the do_finetuning_process function."""
    do_finetuning_process(
        main_config=main_config,
        device=device_fixture,
        logger=logger_fixture,
    )
