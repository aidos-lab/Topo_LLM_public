"""Configurations for specifying sampling in the embeddings data prep step."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode


class SamplingConfig(ConfigBaseModel):
    """Configurations for specifying sampling in the embeddings data prep step."""

    num_samples: int = Field(
        default=30_000,
        title="Number of samples.",
        description="The number of samples to be extracted."
        "Choose size of a sample which is used to take subsets for a point-wise computation of local estimators.",
    )

    sampling_mode: EmbeddingsDataPrepSamplingMode = Field(
        default=EmbeddingsDataPrepSamplingMode.RANDOM,
        title="Sampling mode.",
        description="The sampling mode to be used.",
    )

    seed: int = Field(
        default=42,
        title="Seed.",
        description="The seed for the random number generator.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        desc: str = (
            f"{NAME_PREFIXES['sampling_mode']}{KV_SEP}{self.sampling_mode}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['seed']}{KV_SEP}{self.seed!s}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['num_samples']}{KV_SEP}{self.num_samples!s}"
        )

        return desc
