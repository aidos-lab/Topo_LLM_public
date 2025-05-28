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


"""Test the do_inference function."""

import logging

import pytest

from topollm.config_classes.main_config import MainConfig
from topollm.model_inference.do_inference import do_inference


@pytest.mark.uses_transformers_models()
@pytest.mark.slow()
def test_do_inference(
    main_config: MainConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test the do_inference function."""
    results = do_inference(
        main_config=main_config,
        prompts=None,
        logger=logger_fixture,
    )

    logger_fixture.info(
        "results:\n%s",
        results,
    )
