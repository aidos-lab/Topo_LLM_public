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

"""Test the do_inference function."""

import logging

import pytest
import torch

from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.data.data_split_config import DataSplitConfig, Proportions
from topollm.config_classes.data.dataset_map_config import DatasetMapConfig
from topollm.config_classes.embeddings.embedding_extraction_config import EmbeddingExtractionConfig
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import EmbeddingsDataPrepConfig
from topollm.config_classes.finetuning.batch_sizes_config import BatchSizesConfig
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.finetuning.finetuning_datasets_config import FinetuningDatasetsConfig
from topollm.config_classes.finetuning.gradient_modifier.gradient_modifier_config import GradientModifierConfig
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.finetuning.tokenizer_modifier_config import TokenizerModifierConfig
from topollm.config_classes.inference.inference_config import InferenceConfig
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.config_classes.storage.storage_config import StorageConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.config_classes.transformations.transformations_config import TransformationsConfig
from topollm.config_classes.wandb.wandb_config import WandBConfig
from topollm.pipeline_scripts.worker_for_pipeline import worker_for_pipeline
from topollm.typing.enums import (
    ArrayStorageType,
    DatasetType,
    DataSplitMode,
    LMmode,
    MetadataStorageType,
    PreferredTorchBackend,
    Split,
    TokenizerModifierMode,
    Verbosity,
)


@pytest.fixture(
    scope="session",
)
def small_data_config() -> DataConfig:
    """Return a DataConfig object."""
    config = DataConfig(
        column_name="body",
        context="dataset_entry",
        dataset_description_string="one-year-of-tsla-on-reddit",
        dataset_type=DatasetType.HUGGINGFACE_DATASET,
        data_dir=None,
        dataset_path="SocialGrep/one-year-of-tsla-on-reddit",
        dataset_name="comments",
        number_of_samples=40,
        split=Split.TRAIN,
        data_split=DataSplitConfig(
            data_split_mode=DataSplitMode.PROPORTIONS,
            proportions=Proportions(),
        ),
    )

    return config


@pytest.fixture(
    scope="session",
)
def main_config_with_small_dataset_and_model(
    small_data_config: DataConfig,
    paths_config: PathsConfig,
) -> MainConfig:
    """Return a MainConfig object."""
    pretrained_model_name_or_path = "hf-internal-testing/tiny-random-RobertaModel"
    short_model_name = "tiny-random-RobertaModel"

    tokenizer_config = TokenizerConfig()
    tokenizer_modifier_config = TokenizerModifierConfig(
        mode=TokenizerModifierMode.DO_NOTHING,
        padding_token="<pad>",  # noqa: S106 - This is the hardcoded padding token
    )
    language_model_config = LanguageModelConfig(
        lm_mode=LMmode.MLM,
        masking_mode="no_masking",
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        short_model_name=short_model_name,
        tokenizer_modifier=tokenizer_modifier_config,
    )

    inference_config = InferenceConfig()
    storage_config = StorageConfig(
        array_storage_type=ArrayStorageType.ZARR,
        metadata_storage_type=MetadataStorageType.PICKLE,
        chunk_size=512,
    )
    dataset_map_config = DatasetMapConfig()
    batch_sizes_config = BatchSizesConfig()
    gradient_modifier_config = GradientModifierConfig()
    peft_config = PEFTConfig()

    finetuning_datasets_config = FinetuningDatasetsConfig(
        train_dataset=small_data_config,
        eval_dataset=small_data_config,
    )

    finetuning_config = FinetuningConfig(
        finetuning_datasets=finetuning_datasets_config,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        gradient_modifier=gradient_modifier_config,
        peft=peft_config,
        tokenizer=tokenizer_config,
        tokenizer_modifier=tokenizer_modifier_config,
        batch_sizes=batch_sizes_config,
    )

    embeddings_extraction_config = EmbeddingExtractionConfig()
    embeddings_config = EmbeddingsConfig(
        dataset_map=dataset_map_config,
        embedding_extraction=embeddings_extraction_config,
    )

    embeddings_data_prep_config = EmbeddingsDataPrepConfig(
        num_samples=1_000,
    )

    config = MainConfig(
        data=small_data_config,
        embeddings=embeddings_config,
        embeddings_data_prep=embeddings_data_prep_config,
        finetuning=finetuning_config,
        inference=inference_config,
        language_model=language_model_config,
        paths=paths_config,
        preferred_torch_backend=PreferredTorchBackend.CPU,
        storage=storage_config,
        tokenizer=tokenizer_config,
        transformations=TransformationsConfig(),
        wandb=WandBConfig(),
        verbosity=Verbosity.VERBOSE,
    )

    return config


@pytest.mark.uses_transformers_models()
@pytest.mark.slow()
def test_worker_for_pipeline(
    main_config_with_small_dataset_and_model: MainConfig,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    """Test the pipeline function."""
    worker_for_pipeline(
        main_config=main_config_with_small_dataset_and_model,
        device=device_fixture,
        logger=logger_fixture,
    )
