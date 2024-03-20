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
from datetime import datetime

# Third-party imports
import pytest

# Local imports
from topollm.config_classes.Configs import (
    DataConfig,
    DatasetMapConfig,
    EmbeddingExtractionConfig,
    LanguageModelConfig,
    PathsConfig,
    TokenizerConfig,
    TransformationsConfig,
)
from topollm.config_classes.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.enums import DatasetType, Level, Split, AggregationType
from topollm.config_classes.path_management.SeparateDirectoriesEmbeddingsPathManager import (
    SeparateDirectoriesEmbeddingsPathManager,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of the logging module

logger = logging.getLogger(__name__)

# END Configuration of the logging module
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of pytest


def pytest_addoption(
    parser: pytest.Parser,
) -> None:
    parser.addoption(
        "--keep-test-data",
        action="store_true",
        help="Keep test data after tests are done",
    )

    return None


def pytest_configure(
    config: pytest.Config,
) -> None:
    """
    Create a custom path to the log file
    if log_file is not mentioned in pytest.ini file
    """
    if not config.option.log_file:
        timestamp = datetime.strftime(
            datetime.now(),
            "%Y-%m-%d_%H-%M-%S",
        )
        config.option.log_file = "temp_files/logs/pytest-logs_" + timestamp + ".log"

    return None


# END Configuration of pytest
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(scope="session")
def test_data_dir(
    tmp_path_factory: pytest.TempPathFactory,
    pytestconfig: pytest.Config,
) -> pathlib.Path:
    if pytestconfig.getoption(
        name="--keep-test-data",
    ):
        # Create a more permanent directory
        base_dir = pathlib.Path(
            pathlib.Path.cwd(),
            "temp_files",
            "test_data",
        )
        base_dir.mkdir(
            exist_ok=True,
        )
        return base_dir
    else:
        # Use pytest's tmp_path_factory for a truly temporary directory
        return tmp_path_factory.mktemp(
            basename="data-",
            numbered=True,
        )


@pytest.fixture(scope="session")
def repository_base_path() -> pathlib.Path:
    return pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
    )


@pytest.fixture(scope="session")
def logger_fixture() -> logging.Logger:
    return logger


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


@pytest.fixture(scope="session")
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        add_prefix_space=False,
        max_length=512,
    )


@pytest.fixture(scope="session")
def dataset_map_config() -> DatasetMapConfig:
    return DatasetMapConfig(
        batch_size=1000,
        num_proc=2,
    )


@pytest.fixture(scope="session")
def language_model_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        pretrained_model_name_or_path="roberta-base",
        masking_mode="no_masking",
    )


@pytest.fixture(scope="session")
def embedding_extraction_config() -> EmbeddingExtractionConfig:
    return EmbeddingExtractionConfig(
        layer_indices=[-1],
        aggregation=AggregationType.MEAN,
    )


@pytest.fixture(scope="session")
def embeddings_config(
    tokenizer_config: TokenizerConfig,
    dataset_map_config: DatasetMapConfig,
    language_model_config: LanguageModelConfig,
    embedding_extraction_config: EmbeddingExtractionConfig,
) -> EmbeddingsConfig:
    return EmbeddingsConfig(
        tokenizer=tokenizer_config,
        dataset_map=dataset_map_config,
        batch_size=32,
        language_model=language_model_config,
        embedding_extraction=embedding_extraction_config,
        level=Level.TOKEN,
        num_workers=1,
    )


@pytest.fixture(scope="session")
def paths_config(
    test_data_dir: pathlib.Path,
    repository_base_path: pathlib.Path,
) -> PathsConfig:
    return PathsConfig(
        data_dir=test_data_dir,
        repository_base_path=repository_base_path,
    )


@pytest.fixture(scope="session")
def transformations_config() -> TransformationsConfig:
    return TransformationsConfig(
        normalization="None",
    )


@pytest.fixture(scope="session")
def separate_directories_embeddings_path_manager(
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
        verbosity=1,
        logger=logger_fixture,
    )


# import tempfile
# import shutil
# from typing import Generator
#
# @pytest.fixture(scope="session")
# def session_tmp_path() -> Generator[
#     pathlib.Path,
#     None,
#     None,
# ]:
#     # Create a temporary directory for the session
#     temp_dir = tempfile.mkdtemp()
#     yield pathlib.Path(
#         temp_dir,
#     )
#     # Cleanup the temporary directory at the end of the session
#     shutil.rmtree(temp_dir)
