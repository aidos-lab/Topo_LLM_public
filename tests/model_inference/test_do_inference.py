"""Test the do_inference function."""

import logging

import pytest

from topollm.config_classes.main_config import MainConfig
from topollm.model_inference.do_inference import do_inference
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


@pytest.mark.uses_transformers_models
@pytest.mark.slow
def test_do_inference(
    main_config: MainConfig,
    embeddings_path_manager: EmbeddingsPathManager,
    logger_fixture: logging.Logger,
) -> None:
    """Test the do_inference function."""
    results: list[list] = do_inference(
        main_config=main_config,
        embeddings_path_manager=embeddings_path_manager,
        logger=logger_fixture,
    )

    logger_fixture.info(
        msg=f"results:\n{results}",  # noqa: G004 - low overhead
    )
