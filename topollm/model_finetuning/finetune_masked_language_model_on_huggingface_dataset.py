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

"""
Script for fine-tuning language model on huggingface datasets.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import logging
import math
import os

# Third party imports
import datasets
import hydra
import hydra.core.hydra_config
import omegaconf
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel


# Local imports
from topollm.config_classes.MainConfig import MainConfig
from topollm.config_classes.FinetuningConfig import FinetuningConfig
from topollm.config_classes.path_management.FinetuningPathManagerFactory import (
    get_finetuning_path_manager,
)
from topollm.config_classes.path_management.FinetuningPathManagerProtocol import (
    FinetuningPathManager,
)
from topollm.data_handling.DatasetPreparerFactory import get_dataset_preparer
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.get_torch_device import get_torch_device

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# Set the transformers logging level
transformers.logging.set_verbosity_info()

# torch.set_num_threads(1)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""

    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    run_finetuning(
        main_config=main_config,
        logger=global_logger,
    )

    return None


def run_finetuning(
    main_config: MainConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    finetuning_config = main_config.finetuning

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Use accelerator if available
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=global_logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load data
    train_dataset_preparer = get_dataset_preparer(
        dataset_type=finetuning_config.finetuning_datasets.train_dataset.dataset_type,
        data_config=finetuning_config.finetuning_datasets.train_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    train_dataset = train_dataset_preparer.prepare_dataset()

    eval_dataset_preparer = get_dataset_preparer(
        dataset_type=finetuning_config.finetuning_datasets.eval_dataset.dataset_type,
        data_config=finetuning_config.finetuning_datasets.eval_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    eval_dataset = eval_dataset_preparer.prepare_dataset()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    tokenizer = load_tokenizer(
        finetuning_config=finetuning_config,
        logger=logger,
    )

    base_model = load_base_model(
        finetuning_config=finetuning_config,
        device=device,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare model input

    train_dataset_mapped, eval_dataset_mapped = prepare_model_input(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        finetuning_config=finetuning_config,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=finetuning_config.mlm_probability,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Output paths

    finetuning_path_manager = get_finetuning_path_manager(
        config=main_config,
        logger=logger,
    )

    finetuned_model_dir = prepare_finetuned_model_dir(
        finetuning_path_manager=finetuning_path_manager,
        logger=logger,
    )

    logging_dir = prepare_logging_dir(
        finetuning_path_manager=finetuning_path_manager,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Finetuning setup

    training_args = prepare_training_args(
        finetuning_config=finetuning_config,
        finetuned_model_dir=finetuned_model_dir,
        logging_dir=logging_dir,
    )

    trainer = transformers.Trainer(
        model=base_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_mapped,  # type: ignore
        eval_dataset=eval_dataset_mapped,  # type: ignore
        tokenizer=tokenizer,
    )

    finetune_model(
        trainer=trainer,
        logger=logger,
    )

    save_tuned_model(
        trainer=trainer,
        logger=logger,
    )

    evaluate_tuned_model(
        trainer=trainer,
        logger=logger,
    )

    return None


def load_base_model(
    finetuning_config: FinetuningConfig,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
):
    logger.info(
        f"Loading model " f"{finetuning_config.pretrained_model_name_or_path = } ..."
    )
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logger.info(
        f"Loading model " f"{finetuning_config.pretrained_model_name_or_path = } DONE"
    )

    logger.info(f"model:\n{model}")
    if hasattr(
        model,
        "config",
    ):
        logger.info(f"model.config:\n{model.config}")

    # Check type of model
    assert isinstance(
        model,
        PreTrainedModel,
    )

    # Move the model to GPU if available
    logger.info(f"Moving model to {device = } ...")
    model.to(
        device,  # type: ignore
    )
    logger.info(f"Moving model to {device = } DONE")

    return model


def prepare_finetuned_model_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = logging.getLogger(__name__),
) -> os.PathLike:
    finetuned_model_dir = finetuning_path_manager.finetuned_model_dir
    logger.info(f"{finetuned_model_dir = }")

    # Create the output directory if it does not exist
    finetuned_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return finetuned_model_dir


def prepare_logging_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = logging.getLogger(__name__),
) -> os.PathLike | None:
    logging_dir = finetuning_path_manager.logging_dir
    logger.info(f"{logging_dir = }")

    # Create the logging directory if it does not exist
    if logging_dir is not None:
        logging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        logger.info(
            f"No logging directory specified. "
            f"Using default logging from transformers.Trainer."
        )

    return logging_dir


def finetune_model(
    trainer: transformers.Trainer,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    logger.info(f"Calling trainer.train() ...")

    training_call_output = trainer.train(
        resume_from_checkpoint=False,
    )

    logger.info(f"Calling trainer.train() DONE")

    logger.info(f"training_call_output:\n" f"{training_call_output}")

    return None


def prepare_model_input(
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    finetuning_config: FinetuningConfig,
) -> tuple[
    datasets.Dataset,
    datasets.Dataset,
]:
    def tokenize_function(
        dataset_entries: dict[
            str,
            list,
        ],
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        # NOTE: This assumes that train and eval datasets use the same column name
        column_name = finetuning_config.finetuning_datasets.train_dataset.column_name

        result = tokenizer(
            dataset_entries[column_name],
            padding="max_length",
            truncation=True,
            max_length=finetuning_config.max_length,
        )

        return result

    train_dataset_mapped = train_dataset.map(
        tokenize_function,
        batched=True,
    )
    eval_dataset_mapped = eval_dataset.map(
        tokenize_function,
        batched=True,
    )

    return train_dataset_mapped, eval_dataset_mapped


def load_tokenizer(
    finetuning_config: FinetuningConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    # ? Do we need other config arguments for the tokenizer for finetuning here?

    logger.info(
        f"Loading tokenizer "
        f"{finetuning_config.pretrained_model_name_or_path = } ..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logger.info(
        f"Loading tokenizer "
        f"{finetuning_config.pretrained_model_name_or_path = } DONE"
    )
    logger.info(f"tokenizer:\n{tokenizer}")

    # Make sure not to accidentally modify the tokenizer pad token (tokenizer.pad_token) here.
    # In particular, it is not custom to set the pad token to the eos token for masked language model training.

    return tokenizer


def prepare_training_args(
    finetuning_config: FinetuningConfig,
    finetuned_model_dir: os.PathLike,
    logging_dir: os.PathLike | None = None,
):
    training_args = transformers.TrainingArguments(
        output_dir=str(finetuned_model_dir),
        overwrite_output_dir=True,
        num_train_epochs=finetuning_config.num_train_epochs,
        learning_rate=finetuning_config.learning_rate,
        weight_decay=finetuning_config.weight_decay,
        per_device_train_batch_size=finetuning_config.batch_sizes.train,
        per_device_eval_batch_size=finetuning_config.batch_sizes.eval,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        gradient_checkpointing=finetuning_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        fp16=finetuning_config.fp16,
        warmup_steps=finetuning_config.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=finetuning_config.eval_steps,
        save_steps=finetuning_config.save_steps,
        logging_dir=logging_dir,  # type: ignore
        log_level=finetuning_config.log_level,
        logging_steps=finetuning_config.logging_steps,
        use_cpu=finetuning_config.use_cpu,
        seed=finetuning_config.seed,
    )

    return training_args


def save_tuned_model(
    trainer: transformers.Trainer,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    logger.info(f"Calling trainer.save_model() ...")
    trainer.save_model()
    logger.info(f"Calling trainer.save_model() DONE")

    return None


def evaluate_tuned_model(
    trainer: transformers.Trainer,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    logger.info(f"Evaluating the model ...")

    eval_results = trainer.evaluate()

    perplexity = math.exp(eval_results["eval_loss"])
    logger.info(f"perplexity:\n{perplexity:.2f}")
    logger.info(f"eval_results:\n{eval_results}")

    logger.info(f"Evaluating the model DONE")

    return None


if __name__ == "__main__":
    main()
