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

import logging
import os
import pathlib
from datetime import datetime

import pytest
import torch
import transformers
from dotenv import find_dotenv, load_dotenv

from topollm.config_classes.data.DataConfig import DataConfig
from topollm.config_classes.data.DatasetMapConfig import DatasetMapConfig
from topollm.config_classes.embeddings.EmbeddingExtractionConfig import (
    EmbeddingExtractionConfig,
)
from topollm.config_classes.embeddings.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.enums import (
    AggregationType,
    ArrayStorageType,
    DatasetType,
    FinetuningMode,
    Level,
    LMmode,
    MetadataStorageType,
    PreferredTorchBackend,
    Split,
)
from topollm.config_classes.finetuning.BatchSizesConfig import BatchSizesConfig
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.finetuning.FinetuningDatasetsConfig import (
    FinetuningDatasetsConfig,
)
from topollm.config_classes.finetuning.peft.PEFTConfig import PEFTConfig
from topollm.config_classes.inference.InferenceConfig import InferenceConfig
from topollm.config_classes.language_model.LanguageModelConfig import (
    LanguageModelConfig,
)
from topollm.config_classes.MainConfig import MainConfig
from topollm.config_classes.PathsConfig import PathsConfig
from topollm.config_classes.StorageConfig import StorageConfig
from topollm.config_classes.tokenizer.TokenizerConfig import TokenizerConfig
from topollm.config_classes.TransformationsConfig import TransformationsConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_tokenizer
from topollm.path_management.embeddings.EmbeddingsPathManagerSeparateDirectories import (
    EmbeddingsPathManagerSeparateDirectories,
)
from topollm.path_management.finetuning.FinetuningPathManagerBasic import (
    FinetuningPathManagerBasic,
)
from topollm.path_management.finetuning.FinetuningPathManagerProtocol import (
    FinetuningPathManager,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of the logging module

logger = logging.getLogger(__name__)

# END Configuration of the logging module
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of pytest


@pytest.fixture(
    scope="session",
    autouse=True,
)
def load_env() -> None:
    """
    https://stackoverflow.com/questions/48211784/best-way-to-use-python-dotenv-with-pytest-or-best-way-to-have-a-pytest-test-dev
    """
    env_file = find_dotenv(".test.env")
    result = load_dotenv(
        dotenv_path=env_file,
        verbose=True,
    )

    if result:
        logger.info(f"Loaded environment variables " f"from {env_file = }")
    else:
        logger.warning(f"No environment variables loaded " f"from {env_file = }")

    return None


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
        # Note: the doubling {{ and }} is necessary to escape the curly braces
        config.option.log_file = f"logs/pytest-logs_{timestamp}.log"

    return None


# END Configuration of pytest
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(
    scope="session",
)
def repository_base_path() -> pathlib.Path:
    # Get the values from the
    # 'TOPO_LLM_REPOSITORY_BASE_PATH' environment variable
    topo_llm_repository_base_path = os.getenv(
        key="TOPO_LLM_REPOSITORY_BASE_PATH",
    )

    if topo_llm_repository_base_path is None:
        raise ValueError(
            f"The 'TOPO_LLM_REPOSITORY_BASE_PATH' " f"environment variable is not set."
        )

    path = pathlib.Path(
        topo_llm_repository_base_path,
    )

    return path


@pytest.fixture(
    scope="session",
)
def temp_files_dir() -> pathlib.Path:
    # Get the values from the 'TEMP_FILES_DIR' environment variable
    temp_files_dir = os.getenv("TEMP_FILES_DIR")

    if temp_files_dir is None:
        raise ValueError("The 'TEMP_FILES_DIR' environment variable is not set.")

    path = pathlib.Path(
        temp_files_dir,
    )

    return path


@pytest.fixture(
    scope="session",
)
def test_data_dir(
    repository_base_path: pathlib.Path,
    temp_files_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    pytestconfig: pytest.Config,
) -> pathlib.Path:
    if pytestconfig.getoption(
        name="--keep-test-data",
    ):
        # Create a more permanent directory
        base_dir = pathlib.Path(
            repository_base_path,
            temp_files_dir,
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


@pytest.fixture(
    scope="session",
)
def logger_fixture() -> logging.Logger:
    return logger


@pytest.fixture(
    scope="session",
)
def data_config() -> DataConfig:
    config = DataConfig(
        column_name="summary",
        context="dataset_entry",
        dataset_description_string="xsum",
        dataset_type=DatasetType.HUGGINGFACE_DATASET,
        data_dir=None,
        dataset_path="xsum",
        dataset_name=None,
        number_of_samples=10,
        split=Split.TRAIN,
    )

    return config


@pytest.fixture(
    scope="session",
)
def tokenizer_config() -> TokenizerConfig:
    config = TokenizerConfig(
        add_prefix_space=False,
        max_length=512,
    )

    return config


@pytest.fixture(
    scope="session",
)
def dataset_map_config() -> DatasetMapConfig:
    config = DatasetMapConfig(
        batch_size=1000,
        num_proc=2,
    )

    return config


@pytest.fixture(
    scope="session",
    params=[
        (
            "roberta-base",
            "roberta-base",
            LMmode.MLM,
        ),
        (
            "bert-base-uncased",
            "bert-base-uncased",
            LMmode.MLM,
        ),
        # ("gpt2-large", "gpt2-large", LMmode.CLM), # ! TODO The code does not work for this yet
        # TODO For the gpt2-large model, a padding token is missing:
        # "ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token..."
    ],
)
def language_model_config(
    request: pytest.FixtureRequest,
) -> LanguageModelConfig:
    pretrained_model_name_or_path, short_model_name, lm_mode = request.param
    config = LanguageModelConfig(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        short_model_name=short_model_name,
        masking_mode="no_masking",
        lm_mode=lm_mode,
    )

    return config


@pytest.fixture(
    scope="session",
)
def embedding_extraction_config() -> EmbeddingExtractionConfig:
    config = EmbeddingExtractionConfig(
        layer_indices=[-1],
        aggregation=AggregationType.MEAN,
    )

    return config


@pytest.fixture(
    scope="session",
)
def embeddings_config(
    dataset_map_config: DatasetMapConfig,
    language_model_config: LanguageModelConfig,
    embedding_extraction_config: EmbeddingExtractionConfig,
) -> EmbeddingsConfig:
    return EmbeddingsConfig(
        dataset_map=dataset_map_config,
        batch_size=32,
        language_model=language_model_config,
        embedding_extraction=embedding_extraction_config,
        level=Level.TOKEN,
        num_workers=1,
    )


@pytest.fixture(
    scope="session",
)
def paths_config(
    test_data_dir: pathlib.Path,
    repository_base_path: pathlib.Path,
) -> PathsConfig:
    return PathsConfig(
        data_dir=test_data_dir,
        repository_base_path=repository_base_path,
    )


@pytest.fixture(
    scope="session",
)
def transformations_config() -> TransformationsConfig:
    return TransformationsConfig(
        normalization="None",
    )


@pytest.fixture(
    scope="session",
)
def peft_config() -> PEFTConfig:
    config = PEFTConfig(
        finetuning_mode=FinetuningMode.LORA,
    )

    return config


@pytest.fixture(
    scope="session",
)
def finetuning_datasets_config(
    data_config: DataConfig,
) -> FinetuningDatasetsConfig:
    config = FinetuningDatasetsConfig(
        train_dataset=data_config,
        eval_dataset=data_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def batch_sizes_config() -> BatchSizesConfig:
    config = BatchSizesConfig(
        train=8,
        eval=16,
    )

    return config


@pytest.fixture(
    scope="session",
)
def finetuning_config(
    batch_sizes_config: BatchSizesConfig,
    finetuning_datasets_config: FinetuningDatasetsConfig,
    peft_config: PEFTConfig,
    tokenizer_config: TokenizerConfig,
) -> FinetuningConfig:
    config = FinetuningConfig(
        peft=peft_config,
        batch_sizes=batch_sizes_config,
        finetuning_datasets=finetuning_datasets_config,
        pretrained_model_name_or_path="roberta-base",
        short_model_name="roberta-base",
        tokenizer=tokenizer_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def device_fixture() -> torch.device:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return device


@pytest.fixture(
    scope="session",
)
def inference_config() -> InferenceConfig:
    config = InferenceConfig(
        max_length=50,
        num_return_sequences=3,
    )

    return config


@pytest.fixture(
    scope="session",
)
def storage_config() -> StorageConfig:
    config = StorageConfig(
        array_storage_type=ArrayStorageType.ZARR,
        metadata_storage_type=MetadataStorageType.PICKLE,
        chunk_size=1000,
    )

    return config


@pytest.fixture(
    scope="session",
)
def transformers_config() -> TransformationsConfig:
    config = TransformationsConfig(
        normalization="None",
    )

    return config


@pytest.fixture(
    scope="session",
)
def main_config(
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    finetuning_config: FinetuningConfig,
    inference_config: InferenceConfig,
    paths_config: PathsConfig,
    storage_config: StorageConfig,
    tokenizer_config: TokenizerConfig,
    transformations_config: TransformationsConfig,
) -> MainConfig:
    config = MainConfig(
        data=data_config,
        embeddings=embeddings_config,
        finetuning=finetuning_config,
        inference=inference_config,
        paths=paths_config,
        preferred_torch_backend=PreferredTorchBackend.CPU,
        storage=storage_config,
        tokenizer=tokenizer_config,
        transformations=transformations_config,
        verbosity=1,
    )

    return config


@pytest.fixture(
    scope="session",
)
def tokenizer(
    main_config: MainConfig,
    logger_fixture: logging.Logger,
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.embeddings.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.tokenizer,
        verbosity=1,
        logger=logger_fixture,
    )

    return tokenizer


@pytest.fixture(
    scope="session",
)
def embeddings_path_manager(
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    paths_config: PathsConfig,
    tokenizer_config: TokenizerConfig,
    transformations_config: TransformationsConfig,
    logger_fixture: logging.Logger,
) -> EmbeddingsPathManagerSeparateDirectories:
    path_manager = EmbeddingsPathManagerSeparateDirectories(
        data_config=data_config,
        embeddings_config=embeddings_config,
        paths_config=paths_config,
        tokenizer_config=tokenizer_config,
        transformations_config=transformations_config,
        verbosity=1,
        logger=logger_fixture,
    )

    return path_manager


@pytest.fixture(
    scope="session",
)
def finetuning_path_manager_basic(
    data_config: DataConfig,
    paths_config: PathsConfig,
    finetuning_config: FinetuningConfig,
    logger_fixture: logging.Logger,
) -> FinetuningPathManager:
    path_manager = FinetuningPathManagerBasic(
        data_config=data_config,
        paths_config=paths_config,
        finetuning_config=finetuning_config,
        verbosity=1,
        logger=logger_fixture,
    )

    return path_manager
