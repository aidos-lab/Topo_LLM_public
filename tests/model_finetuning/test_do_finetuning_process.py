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
