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
from functools import partial
from typing import TYPE_CHECKING

import datasets
import hydra
import hydra.core.hydra_config
import omegaconf
import transformers

from topollm.compute_embeddings.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingface,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.factory import get_embedding_dataloader_preparer
from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import EmbeddingDataLoaderPreparer
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_finetuning.evaluate_trainer import evaluate_trainer
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.pipeline_scripts.worker_for_pipeline import worker_for_pipeline
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # # # #
    # Load and prepare model
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    # Put the model in evaluation mode.
    # For example, dropout layers behave differently during evaluation.
    loaded_model_container.model.eval()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # Note: This collation function will not be used in this evalutation script
    partial_collate_fn = partial(
        collate_batch_and_move_to_device,
        device=loaded_model_container.device,
        model_input_names=loaded_model_container.tokenizer.model_input_names,
    )

    preparer_context = EmbeddingDataLoaderPreparerContext(
        data_config=main_config.data,
        embeddings_config=main_config.embeddings,
        tokenizer_config=main_config.tokenizer,
        tokenizer=loaded_model_container.tokenizer,
        collate_fn=partial_collate_fn,
        verbosity=verbosity,
        logger=logger,
    )
    embedding_dataloader_preparer: EmbeddingDataLoaderPreparerHuggingface = get_embedding_dataloader_preparer(
        preparer_context=preparer_context,
    )  # type: ignore - This script only works with EmbeddingDataLoaderPreparerHuggingface

    # # # #
    # Prepare the dataset
    dataset: datasets.Dataset = embedding_dataloader_preparer.dataset_preparer.prepare_dataset()
    dataset_mapped: datasets.Dataset = embedding_dataloader_preparer.create_dataset_tokenized(
        dataset=dataset,
    )

    training_args = None

    trainer: transformers.Trainer = transformers.Trainer(
        model=loaded_model_container.model,
        args=training_args,
        data_collator=None,  # TODO: Find out if we can set this to None
        train_dataset=None,  # type: ignore - typing issue with Dataset
        eval_dataset=dataset_mapped,  # type: ignore - typing issue with Dataset
        tokenizer=loaded_model_container.tokenizer,  # type: ignore - typing issue with Tokenizer
        compute_metrics=None,
    )

    # TODO: Currently, this does not return the eval loss
    result: dict = evaluate_trainer(
        trainer=trainer,
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
