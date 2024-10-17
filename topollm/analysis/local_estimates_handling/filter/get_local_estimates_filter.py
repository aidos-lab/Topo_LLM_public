# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Factory function to get the local estimates filter."""

import logging

from topollm.analysis.local_estimates_handling.filter.identity_filter import IdentityFilter
from topollm.analysis.local_estimates_handling.filter.protocol import LocalEstimatesFilter
from topollm.analysis.local_estimates_handling.filter.remove_zero_vectors_filter import RemoveZeroVectorsFilter
from topollm.config_classes.local_estimates.filtering_config import LocalEstimatesFilteringConfig
from topollm.typing.enums import Verbosity, ZeroVectorHandlingMode

default_logger = logging.getLogger(__name__)


def get_local_estimates_filter(
    local_estimates_filtering_config: LocalEstimatesFilteringConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LocalEstimatesFilter:
    """Get the filter for the local estimates computation."""
    if verbosity >= Verbosity.NORMAL:
        logger.info("Getting local estimates filter ...")

    if local_estimates_filtering_config.zero_vector_handling_mode == ZeroVectorHandlingMode.KEEP:
        if verbosity >= Verbosity.NORMAL:
            logger.info("Returning IdentityFilter ...")
        local_estimates_filter = IdentityFilter()
    elif local_estimates_filtering_config.zero_vector_handling_mode == ZeroVectorHandlingMode.REMOVE:
        if verbosity >= Verbosity.NORMAL:
            logger.info("Returning RemoveZeroVectorsFilter ...")
        local_estimates_filter = RemoveZeroVectorsFilter()
    else:
        msg = f"Unknown zero vector handling mode:\n{local_estimates_filtering_config.zero_vector_handling_mode = }"
        raise ValueError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Getting local estimates filter DONE")

    return local_estimates_filter
