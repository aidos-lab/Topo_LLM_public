from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.constants import NAME_PREFIXES


from pydantic import Field


import pathlib


class LanguageModelConfig(ConfigBaseModel):
    pretrained_model_name_or_path: str | pathlib.Path = Field(
        ...,
        title="Model identifier for huggingface transformers model.",
        description=f"The model identifier for the huggingface transformers model "
        f"to use for computing embeddings.",
    )

    short_model_name: str = Field(
        ...,
        title="Short model name.",
        description="The short model name.",
    )

    masking_mode: str = Field(
        ...,
        title="Masking mode.",
        description="The masking mode.",
    )

    @property
    def lanugage_model_config_description(
        self,
    ) -> str:
        # Construct and return the model parameters description

        return (
            f"{NAME_PREFIXES['model']}{self.short_model_name}"
            f"_"
            f"{NAME_PREFIXES['masking_mode']}{self.masking_mode}"
        )