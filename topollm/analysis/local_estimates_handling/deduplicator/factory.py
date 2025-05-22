# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

from topollm.analysis.local_estimates_handling.deduplicator.array_deduplicator import ArrayDeduplicator
from topollm.analysis.local_estimates_handling.deduplicator.identity_deduplicator import IdentityDeduplicator
from topollm.analysis.local_estimates_handling.deduplicator.protocol import PreparedDataDeduplicator
from topollm.config_classes.local_estimates.filtering_config import LocalEstimatesFilteringConfig
from topollm.typing.enums import DeduplicationMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_prepared_data_deduplicator(
    local_estimates_filtering_config: LocalEstimatesFilteringConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreparedDataDeduplicator:
    """Get the filter for the local estimates computation."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data deduplicator ...",
        )

    match local_estimates_filtering_config.deduplication_mode:
        case DeduplicationMode.IDENTITY:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Returning IdentityDeduplicator ...",
                )
            deduplicator = IdentityDeduplicator()
        case DeduplicationMode.ARRAY_DEDUPLICATOR:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Returning ArrayDeduplicator ...",
                )
            deduplicator = ArrayDeduplicator(
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Unknown {local_estimates_filtering_config.deduplication_mode = }"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data deduplicator DONE",
        )

    return deduplicator
