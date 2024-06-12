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

import logging

import torch

from topollm.analysis.twonn.twonn_worker import twonn_worker
from topollm.compute_embeddings.compute_and_store_embeddings import compute_and_store_embeddings
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.embeddings_data_prep_worker import embeddings_data_prep_worker
from topollm.pipeline_scripts.run_pipeline_embeddings_data_prep_local_estimate import (
    global_logger,
    logger_section_separation_line,
)
from topollm.typing.enums import Verbosity

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def worker_for_pipeline(
    main_config: MainConfig,
    device: torch.device = default_device,
    logger: logging.Logger = default_logger,
):
    """Run the worker which goes through the pipeline."""
    verbosity: Verbosity = main_config.verbosity

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            logger_section_separation_line,
        )
        logger.info("Calling compute embeddings worker ...")

    # # # # # # # # # # # # # # # #
    # Compute embeddings worker
    compute_and_store_embeddings(
        main_config=main_config,
        device=device,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling compute embeddings worker DONE")
        logger.info(
            logger_section_separation_line,
        )

    # # # # # # # # # # # # # # # #
    # Data prep worker
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            logger_section_separation_line,
        )
        logger.info("Calling data prep worker ...")

    embeddings_data_prep_worker(
        main_config=main_config,
        device=device,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling data prep worker DONE")
        logger.info(
            logger_section_separation_line,
        )

    # # # # # # # # # # # # # # # #
    # Local estimates worker
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            logger_section_separation_line,
        )
        logger.info("Calling local estimates worker ...")

    # ! TODO: There appears to be an error in the paths of the twonn_worker
    twonn_worker(
        main_config=main_config,
        device=device,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling local estimates worker DONE")
        logger.info(
            logger_section_separation_line,
        )
