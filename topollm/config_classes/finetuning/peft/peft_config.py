"""Configuration class for the PEFT model."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import FinetuningMode


class PEFTConfig(ConfigBaseModel):
    """Configurations for the PEFT model."""

    finetuning_mode: FinetuningMode = Field(
        default=FinetuningMode.STANDARD,
        description="The finetuning mode of the PEFT model.",
    )

    r: int = Field(
        default=8,
        description="The r (rank) parameter of the PEFT model for LoRA.",
    )

    lora_alpha: int = Field(
        default=32,
        description="The alpha parameter of the PEFT model for LoRA.",
    )

    target_modules: list[str] | str | None = Field(
        default=[
            "query",
            "key",
            "value",
        ],
        description="The target modules of the PEFT model for LoRA.",
    )

    lora_dropout: float = Field(
        default=0.01,
        description="The dropout rate of the PEFT model for LoRA.",
    )

    use_rslora: bool = Field(
        default=False,
        description="Whether to use the RSLora variant of the PEFT model.",
    )
