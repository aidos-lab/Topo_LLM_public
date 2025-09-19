import logging
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedModel

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.model_finetuning.generate_from_pretrained_kwargs_instance import generate_from_pretrained_kwargs_instance
from topollm.model_handling.model.load_model_from_language_model_config import load_model_from_language_model_config
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
    from topollm.model_handling.model.token_classification_from_pretrained_kwargs import (
        TokenClassificationFromPretrainedKwargs,
    )

default_device = torch.device(
    device="cpu",
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def load_base_model_from_finetuning_config(
    finetuning_config: FinetuningConfig,
    label_list: list[str] | None = None,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreTrainedModel:
    """Interface function to load a model from a FinetuningConfig object."""
    language_model_config: LanguageModelConfig = finetuning_config.base_model

    from_pretrained_kwargs_instance: TokenClassificationFromPretrainedKwargs | None = (
        generate_from_pretrained_kwargs_instance(
            finetuning_config=finetuning_config,
            label_list=label_list,
        )
    )

    # Note that the subsequent model loading functions might add additional entries to the `from_pretrained_kwargs`.
    # In particular, we handle the dropout rate for the model later.
    model: PreTrainedModel = load_model_from_language_model_config(
        language_model_config=language_model_config,
        from_pretrained_kwargs_instance=from_pretrained_kwargs_instance,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    return model
