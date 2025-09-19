import logging

import peft.mapping_func
import peft.peft_model
import torch
from peft.tuners.lora.config import LoraConfig
from transformers import PreTrainedModel

from topollm.logging.log_model_info import log_model_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_lora_model(
    base_model: PreTrainedModel,
    lora_config: LoraConfig,
    device: torch.device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> peft.peft_model.PeftModel:
    """Prepare a model for LoRA training by injecting the LoRA adapter into the base model."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Preparing LoRA adapter ...",
        )
        logger.info(
            msg="base_model before modification:",
        )
        log_model_info(
            model=base_model,
            model_name="base_model",
            logger=logger,
        )

    # Get the model prepared with PEFT
    # (here: LoRA)
    lora_model = peft.mapping_func.get_peft_model(
        model=base_model,
        peft_config=lora_config,
        adapter_name="default",
    )
    lora_model.print_trainable_parameters()

    if not isinstance(
        lora_model,
        peft.peft_model.PeftModel,
    ):
        msg = f"Expected peft.peft_model.PeftModel, but got {type(lora_model) = }"
        raise TypeError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="lora_model after modification:",
        )
        log_model_info(
            model=lora_model,
            model_name="lora_model",
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Moving model to {device = } ...",  # noqa: G004 - low overhead
        )
    lora_model.to(
        device=device,  # type: ignore - problem with torch.device type
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Moving model to {device = } DONE",  # noqa: G004 - low overhead
        )

    return lora_model
