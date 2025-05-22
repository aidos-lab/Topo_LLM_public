# Copyright 2024-2025
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

"""Worker function which goes through the pipeline."""

import logging

from topollm.analysis.local_estimates_computation.global_and_pointwise_local_estimates_worker import (
    global_and_pointwise_local_estimates_worker,
)
from topollm.compute_embeddings.compute_and_store_embeddings import compute_and_store_embeddings
from topollm.config_classes.constants import logger_section_separation_line
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.embeddings_data_prep_worker import embeddings_data_prep_worker
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def worker_for_pipeline(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the worker which goes through the pipeline."""
    verbosity: Verbosity = main_config.verbosity

    # # # # # # # # # # # # # # # #
    # Compute embeddings worker
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=logger_section_separation_line,
        )
    if not main_config.feature_flags.compute_and_store_embeddings.skip_compute_and_store_embeddings_in_pipeline:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling compute embeddings worker ...",
            )

        compute_and_store_embeddings(
            main_config=main_config,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling compute embeddings worker DONE",
            )
    elif verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Skipping compute embeddings worker because of feature flag.",
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=logger_section_separation_line,
        )

    # # # # # # # # # # # # # # # #
    # Data prep worker
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=logger_section_separation_line,
        )
    if not main_config.feature_flags.embeddings_data_prep.skip_embeddings_data_prep_in_pipeline:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling data prep worker ...",
            )

        embeddings_data_prep_worker(
            main_config=main_config,
            verbosity=main_config.verbosity,
            logger=logger,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling data prep worker DONE",
            )
    elif verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Skipping data prep worker because of feature flag.",
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=logger_section_separation_line,
        )

    # # # # # # # # # # # # # # # #
    # Local estimates worker
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=logger_section_separation_line,
        )
        logger.info(
            msg="Calling local estimates worker ...",
        )

    global_and_pointwise_local_estimates_worker(
        main_config=main_config,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling local estimates worker DONE",
        )
        logger.info(
            msg=logger_section_separation_line,
        )
