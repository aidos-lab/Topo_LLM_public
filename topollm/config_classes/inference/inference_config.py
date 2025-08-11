"""Configuration class for inference."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class InferenceConfig(ConfigBaseModel):
    """Configurations for running model inference."""

    max_length: int | None = Field(
        default=None,
        title="Maximum length of generated text.",
        description=(
            "The maximum length of generated text. "
            "If the input is longer than max_length, it can lead to unexpected behavior. "
            "Thus you need to set a suitable max_length or, better yet, set max_new_tokens."
        ),
    )

    max_new_tokens: int | None = Field(
        default=100,
        title="Maximum number of new tokens to generate.",
        description="The maximum number of new tokens to generate.",
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
        default=False,
        title="Include timestamp in filename.",
        description="If True, the timestamp will be included in the filename of the inference results.",
    )
