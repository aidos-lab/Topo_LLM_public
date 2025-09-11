import logging

import pytest
import torch
from transformers import PreTrainedModel

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.model_handling.model.load_model_from_language_model_config import load_model_from_language_model_config
from topollm.typing.enums import Verbosity


@pytest.mark.uses_transformers_models
def test_load_model(
    language_model_config: LanguageModelConfig,
    device_fixture: torch.device,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
    from_pretrained_kwargs_instance: dict | None = None,
) -> None:
    """Test the load_model function."""
    model: PreTrainedModel = load_model_from_language_model_config(
        language_model_config=language_model_config,
        from_pretrained_kwargs_instance=from_pretrained_kwargs_instance,
        device=device_fixture,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    logger_fixture.info(
        msg=f"Loaded model:\n{model}",  # noqa: G004 - low overhead
    )

    assert model is not None  # noqa: S101 - pytest assert
    assert isinstance(  # noqa: S101 - pytest assert
        model,
        PreTrainedModel,
    )
