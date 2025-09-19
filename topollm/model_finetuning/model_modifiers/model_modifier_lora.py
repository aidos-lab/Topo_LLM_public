import logging

import peft.peft_model
import torch
from peft.tuners.lora.config import LoraConfig
from transformers import PreTrainedModel

from topollm.model_finetuning.model_modifiers.prepare_lora_model import (
    prepare_lora_model,
)
from topollm.typing.enums import Verbosity

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


class ModelModifierLora:
    def __init__(
        self,
        lora_config: LoraConfig,
        device: torch.device = default_device,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        self.lora_config = lora_config
        self.device = device

        self.verbosity = verbosity
        self.logger = logger

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> peft.peft_model.PeftModel | PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Preparing LoRA adapter ...")
            self.logger.info(
                "self.lora_config:\n%s",
                self.lora_config,
            )

        modified_model = prepare_lora_model(
            base_model=model,
            lora_config=self.lora_config,
            device=self.device,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Preparing LoRA adapter DONE.")

        return modified_model
