"""Configuration class for inference."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class InferenceConfig(ConfigBaseModel):
    """Configurations for running model inference."""

    max_length: int = Field(
        default=100,
        title="Maximum length of generated text.",
        description="The maximum length of generated text.",
    )

    num_return_sequences: int = Field(
        default=3,
        title="Number of returned sequences.",
        description="The number of returned sequences.",
    )

    prompts: list[str] | None = Field(
        default=None,
        title="Prompts for inference.",
        description="A list of prompts to use for inference. If None, default prompts will be used.",
    )

    include_timestamp_in_filename: bool = Field(
        default=True,
        title="Include timestamp in filename.",
        description="If True, the timestamp will be included in the filename of the inference results.",
    )
