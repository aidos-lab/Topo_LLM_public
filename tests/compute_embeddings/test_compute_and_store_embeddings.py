"""Test the compute_and_store_embeddings function."""

import logging

import pytest
import torch

from topollm.compute_embeddings.compute_and_store_embeddings import (
    compute_and_store_embeddings,
)
from topollm.config_classes.main_config import MainConfig


@pytest.mark.uses_transformers_models
@pytest.mark.slow
def test_compute_and_store_embeddings(
    main_config: MainConfig,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    """Test the compute_and_store_embeddings function."""
    # Set a smaller number of samples for testing purposes.
    # Otherwise, the embedding computation will take too long.
    main_config.data.data_subsampling.number_of_samples = 200

    compute_and_store_embeddings(
        main_config=main_config,
        logger=logger_fixture,
    )
