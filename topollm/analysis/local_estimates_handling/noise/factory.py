# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory function to get the noise adding module."""

import logging

from topollm.analysis.local_estimates_handling.noise.gaussian_noiser import GaussianNoiser
from topollm.analysis.local_estimates_handling.noise.identity_noiser import IdentityNoiser
from topollm.analysis.local_estimates_handling.noise.protocol import PreparedDataNoiser
from topollm.config_classes.local_estimates.local_estimates_config import LocalEstimatesNoiseConfig
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
