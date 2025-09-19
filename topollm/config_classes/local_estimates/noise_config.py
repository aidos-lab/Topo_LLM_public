"""Configurations for adding artificial noise into the local estimates computation."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import ArtificialNoiseMode


class LocalEstimatesNoiseConfig(ConfigBaseModel):
    """Configurations for adding artificial noise into the local estimates computation."""

    artificial_noise_mode: ArtificialNoiseMode = Field(
        default=ArtificialNoiseMode.DO_NOTHING,
        title="Artificial noise mode.",
        description="Which kind of noise will be added before the local estimates computation.",
    )

    distortion_parameter: float = Field(
        default=0.0,
        title="Distortion parameter.",
        description="For example, this will be the standard deviation of the Gaussian noise.",
    )

    seed: int = Field(
        default=0,
        title="Noise seed.",
        description="The random seed for the noise generation.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        match self.artificial_noise_mode:
            case ArtificialNoiseMode.DO_NOTHING:
                description = (
                    f"{NAME_PREFIXES['local_estimates_noise_artificial_noise_mode']}"
                    + KV_SEP
                    + f"{str(object=self.artificial_noise_mode)}"
                )
            case ArtificialNoiseMode.GAUSSIAN:
                description = (
                    f"{NAME_PREFIXES['local_estimates_noise_artificial_noise_mode']}"
                    + KV_SEP
                    + f"{str(object=self.artificial_noise_mode)}"
                    + ITEM_SEP
                    + f"{NAME_PREFIXES['local_estimates_noise_distortion_parameter']}"
                    + KV_SEP
                    + f"{str(object=self.distortion_parameter)}"
                    + ITEM_SEP
                    + f"{NAME_PREFIXES['local_estimates_noise_seed']}"
                    + KV_SEP
                    + f"{str(object=self.seed)}"
                )
            case _:
                msg: str = f"Unknown {self.artificial_noise_mode = }"
                raise ValueError(
                    msg,
                )

        return description
