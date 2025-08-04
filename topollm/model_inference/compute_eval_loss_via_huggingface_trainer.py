"""Compute the evaluation loss of a model on a dataset using the Huggingface Trainer."""

import json
import logging
import os
import pathlib
from functools import partial
from typing import TYPE_CHECKING

import datasets
import hydra
import hydra.core.hydra_config
import omegaconf
import transformers
import wandb

from topollm.compute_embeddings.collator.collate_batch_for_embedding import (
    collate_batch_and_move_to_device,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_context import (
    EmbeddingDataLoaderPreparerContext,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.embedding_dataloader_preparer_huggingface import (
    EmbeddingDataLoaderPreparerHuggingfaceWithTokenization,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.factory import get_embedding_dataloader_preparer
from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import EmbeddingDataLoaderPreparer
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_omega_conf import setup_omega_conf
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.data_handling.dataset_preparer.select_random_elements import log_selected_dataset_elements_info
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_finetuning.evaluate_trainer import evaluate_trainer
from topollm.model_finetuning.initialize_wandb import initialize_wandb
from topollm.model_finetuning.prepare_data_collator import prepare_data_collator
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.pipeline_scripts.worker_for_pipeline import worker_for_pipeline
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


# Increase the wandb service wait time to prevent errors on HPC cluster.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"

# The "core" argument is only available from wandb 0.17 onwards
#
# > wandb.require(
# >     "core",
# > )

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initialize wandb
    if main_config.feature_flags.wandb.use_wandb:
        initialize_wandb(
            main_config=main_config,
            config=config,
            logger=logger,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
        # Note: Do not set `os.environ["WANDB_DISABLED"] = "true"` because this will raise the error
        # `RuntimeError: WandbCallback requires wandb to be installed. Run `pip install wandb`.`
        main_config.finetuning.report_to = [
            "tensorboard",
        ]

    # # # #
    # Path management
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

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
    embedding_dataloader_preparer: EmbeddingDataLoaderPreparerHuggingfaceWithTokenization = (
        get_embedding_dataloader_preparer(
            preparer_context=preparer_context,
        )
    )  # type: ignore - This script only works with EmbeddingDataLoaderPreparerHuggingface

    # # # #
    # Prepare the dataset
    dataset: datasets.Dataset = embedding_dataloader_preparer.dataset_preparer.prepare_dataset()
    dataset_mapped: datasets.Dataset = embedding_dataloader_preparer.create_dataset_tokenized(
        dataset=dataset,
    )

    if verbosity >= Verbosity.NORMAL:
        log_huggingface_dataset_info(
            dataset=dataset_mapped,
            dataset_name="dataset_mapped",
            logger=logger,
        )
        log_selected_dataset_elements_info(
            dataset=dataset_mapped,
            dataset_name="dataset_mapped",
            seed=main_config.global_seed,
            logger=logger,
        )

    # # # #
    # Huggingface Trainer

    data_collator: transformers.DataCollatorForLanguageModeling | transformers.DataCollatorForTokenClassification = (
        prepare_data_collator(
            task_type=main_config.language_model.task_type,
            mlm_probability=0.15,
            tokenizer=loaded_model_container.tokenizer,
            verbosity=verbosity,
            logger=logger,
        )
    )

    output_parent_dir = pathlib.Path(
        embeddings_path_manager.get_training_and_evaluation_losses_dir_absolute_path(),
    )
    output_parent_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = transformers.TrainingArguments(
        output_dir=str(object=output_parent_dir),
        logging_dir=None,
    )

    trainer: transformers.Trainer = transformers.Trainer(
        model=loaded_model_container.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=None,  # type: ignore - typing issue with Dataset
        eval_dataset=dataset_mapped,  # type: ignore - typing issue with Dataset
        tokenizer=loaded_model_container.tokenizer,  # type: ignore - typing issue with Tokenizer
        compute_metrics=None,
    )

    result: dict = evaluate_trainer(
        trainer=trainer,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Save the results

    result_save_path: pathlib.Path = pathlib.Path(
        output_parent_dir,
        "eval_results.json",
    )

    # Save the results as json file
    result_save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with result_save_path.open(
        mode="w",
    ) as file:
        json.dump(
            obj=result,
            fp=file,
            indent=4,
        )

    if main_config.feature_flags.wandb.use_wandb:
        # We need to manually finish the wandb run
        # so that the hydra multi-run submissions are not summarized in the same run
        wandb.finish()

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
