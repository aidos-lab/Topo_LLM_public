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

import logging
import pprint

import skdim

from topollm.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from topollm.typing.enums import EstimatorMethodType, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_estimator_from_local_estimates_config(
    local_estimates_config: LocalEstimatesConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> skdim._commonfuncs.GlobalEstimator:
    """Get the estimator from the local estimates configuration."""
    match local_estimates_config.estimator.method_type:
        case EstimatorMethodType.TWONN:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using the TwoNN estimator.",
                )
            estimator = skdim.id.TwoNN(
                discard_fraction=local_estimates_config.estimator.twonn_discard_fraction,
            )
        case EstimatorMethodType.LPCA:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using lPCA estimator.",
                )
            estimator = skdim.id.lPCA(
                ver=local_estimates_config.estimator.lpca_ver,
                alphaRatio=local_estimates_config.estimator.lpca_alphaRatio,
            )
            # TODO: Add additional parameters to the lPCA estimator construction
        # Note: You can add additional estimators here.
        case _:
            msg: str = f"Unsupported estimator method type: {local_estimates_config.estimator.method_type =}"
            logger.error(
                msg=msg,
            )
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"estimator:\n{estimator}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"estimator.__dict__:\n{pprint.pformat(object=estimator.__dict__)}",  # noqa: G004 - low overhead
        )

    return estimator
