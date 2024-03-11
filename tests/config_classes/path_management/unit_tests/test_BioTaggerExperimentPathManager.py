# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import logging
import pathlib

# Third-party imports
import pytest

# Local imports
from convlab.tda.config_classes.config_factory import experiment_config_list
from convlab.tda.config_classes.ExperimentConfig import ExperimentConfig
from convlab.tda.config_classes.path_management.BioTaggerExperimentPathManager import (
    BioTaggerExperimentPathManager,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of the logging module

logger = logging.getLogger(__name__)

# END Configuration of the logging module
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(scope="session")
def config_fixture(
    experiment_config_from_factory: ExperimentConfig,
) -> ExperimentConfig:
    return experiment_config_from_factory


@pytest.fixture(scope="session")
def logger_fixture() -> logging.Logger:
    return logger


@pytest.fixture(scope="session")
def path_manager(
    config_fixture: ExperimentConfig,
    logger_fixture: logging.Logger,
) -> BioTaggerExperimentPathManager:
    return BioTaggerExperimentPathManager(
        config=config_fixture,
        logger=logger_fixture,
    )


class TestBioTaggerExperimentPathManager:
    def test_get_output_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_output_dir_absolute_path()
        logger.info(f"output_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_model_files_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_model_files_dir_absolute_path()
        logger.info(f"model_files_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_dataloaders_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_dataloaders_dir_absolute_path()
        logger.info(f"dataloaders_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_tensorboard_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_tensorboard_dir_absolute_path()
        logger.info(f"tensorboard_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_logging_file_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_logging_file_absolute_path()
        logger.info(f"logging_file_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_metrics_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_metrics_dir_absolute_path()
        logger.info(f"metrics_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_best_model_scores_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_best_model_scores_dir_absolute_path()
        logger.info(f"best_model_scores_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_model_predictions_dir_absolute_path(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.get_model_predictions_dir_absolute_path()
        logger.info(f"model_predictions_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_construct_model_description_string(
        self,
        path_manager: BioTaggerExperimentPathManager,
    ):
        result = path_manager.construct_model_description_string()
        logger.info(f"model_description_string: {result = }")

        assert isinstance(
            result,
            str,
        )
        assert result != ""
