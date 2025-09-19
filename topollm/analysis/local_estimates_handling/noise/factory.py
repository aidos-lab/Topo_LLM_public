"""Factory function to get the noise adding module."""

import logging
import pprint

from topollm.analysis.local_estimates_handling.noise.gaussian_noiser import GaussianNoiser
from topollm.analysis.local_estimates_handling.noise.identity_noiser import IdentityNoiser
from topollm.analysis.local_estimates_handling.noise.protocol import PreparedDataNoiser
from topollm.config_classes.local_estimates.noise_config import LocalEstimatesNoiseConfig
from topollm.typing.enums import ArtificialNoiseMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_prepared_data_noiser(
    local_estimates_noise_config: LocalEstimatesNoiseConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreparedDataNoiser:
    """Get the noiser for the data array."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data noiser ...",
        )
        logger.info(
            msg=f"local_estimates_noise_config:\n{pprint.pformat(object=local_estimates_noise_config)}",  # noqa: G004 - low overhead
        )

    match local_estimates_noise_config.artificial_noise_mode:
        case ArtificialNoiseMode.DO_NOTHING:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Creating IdentityNoiser ...",
                )
            result = IdentityNoiser()
        case ArtificialNoiseMode.GAUSSIAN:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Creating GaussianNoiser ...",
                )
            result = GaussianNoiser(
                distortion_param=local_estimates_noise_config.distortion_parameter,
                seed=local_estimates_noise_config.seed,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Unknown {local_estimates_noise_config.artificial_noise_mode = }"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data noiser DONE",
        )

    return result
