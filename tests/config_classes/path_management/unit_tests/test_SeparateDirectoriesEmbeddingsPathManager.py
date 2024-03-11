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
from topollm.config_classes.enums import DatasetType, Split

# Local imports
from topollm.config_classes.path_management.SeparateDirectoriesEmbeddingsPathManager import (
    SeparateDirectoriesEmbeddingsPathManager,
)
from topollm.config_classes.Configs import (
    DataConfig,
    EmbeddingsConfig,
    PathsConfig,
    TransformationsConfig,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of the logging module

logger = logging.getLogger(__name__)

# END Configuration of the logging module
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    return DataConfig(
        column_name="summary",
        context="dataset_entry",
        dataset_description_string="xsum",
        dataset_identifier="xsum",
        dataset_type=DatasetType.HUGGINGFACE_DATASET,
        number_of_samples=5000,
        split=Split.TRAIN,
    )


# TODO: Fixtures for embeddings_config, paths_config, transformations_config


@pytest.fixture(scope="session")
def logger_fixture() -> logging.Logger:
    return logger


@pytest.fixture(scope="session")
def embeddings_path_manager(
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    paths_config: PathsConfig,
    transformations_config: TransformationsConfig,
    logger_fixture: logging.Logger,
) -> SeparateDirectoriesEmbeddingsPathManager:
    return SeparateDirectoriesEmbeddingsPathManager(
        data_config=data_config,
        embeddings_config=embeddings_config,
        paths_config=paths_config,
        transformations_config=transformations_config,
        logger=logger_fixture,
    )


class TestSeparateDirectoriesEmbeddingsPathManager:
    # TODO: Update the tests

    def test_get_output_dir_absolute_path(
        self,
        embeddings_path_manager: SeparateDirectoriesEmbeddingsPathManager,
    ):
        result = embeddings_path_manager.get_output_dir_absolute_path()
        logger.info(f"output_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_model_files_dir_absolute_path(
        self,
        embeddings_path_manager: SeparateDirectoriesEmbeddingsPathManager,
    ):
        result = embeddings_path_manager.get_model_files_dir_absolute_path()
        logger.info(f"model_files_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_dataloaders_dir_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_dataloaders_dir_absolute_path()
        logger.info(f"dataloaders_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_tensorboard_dir_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_tensorboard_dir_absolute_path()
        logger.info(f"tensorboard_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_logging_file_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_logging_file_absolute_path()
        logger.info(f"logging_file_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_metrics_dir_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_metrics_dir_absolute_path()
        logger.info(f"metrics_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_best_model_scores_dir_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_best_model_scores_dir_absolute_path()
        logger.info(f"best_model_scores_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_get_model_predictions_dir_absolute_path(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.get_model_predictions_dir_absolute_path()
        logger.info(f"model_predictions_dir_absolute_path: {result = }")

        assert isinstance(
            result,
            pathlib.Path,
        )

    def test_construct_model_description_string(
        self,
        embeddings_path_manager: BioTaggerExperimentPathManager,
    ):
        result = embeddings_path_manager.construct_model_description_string()
        logger.info(f"model_description_string: {result = }")

        assert isinstance(
            result,
            str,
        )
        assert result != ""
