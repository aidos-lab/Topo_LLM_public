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

"""Gradient modifier that does not modify the model."""

import logging
from collections.abc import Callable

import datasets
import numpy as np
import pandas as pd
import transformers
from transformers.integrations import WandbCallback

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.finetuning.trainer_modifier.trainer_modifier_config import TrainerModifierConfig
from topollm.typing.enums import TaskType, Verbosity

default_logger = logging.getLogger(__name__)


class TrainerModifierWandbPredictionProgressCallback:
    """Gradient modifier that does not modify the model."""

    def __init__(
        self,
        finetuning_config: FinetuningConfig,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        dataset: datasets.Dataset,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.finetuning_config = finetuning_config
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.verbosity = verbosity
        self.logger = logger

    def modify_trainer(
        self,
        trainer: transformers.Trainer,
    ) -> transformers.Trainer:
        trainer_modifier_config: TrainerModifierConfig = self.finetuning_config.trainer_modifier

        decode_predictions_function = get_decode_predictions_function(
            task_type=self.finetuning_config.base_model.task_type,
        )

        # Instantiate the WandbPredictionProgressCallback
        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            val_dataset=self.dataset,
            num_samples=trainer_modifier_config.num_samples,
            frequency=trainer_modifier_config.frequency,
            decode_predictions_function=decode_predictions_function,
        )

        # Add the callback to the trainer
        trainer.add_callback(
            callback=progress_callback,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Returning Trainer with added callback.")

        return trainer


def get_decode_predictions_function(
    task_type: TaskType,
) -> Callable:
    """Get the decode_predictions function for the task."""
    match task_type:
        case TaskType.MASKED_LM | TaskType.CAUSAL_LM:
            decode_predictions_function = decode_predictions_language_modeling
        case TaskType.TOKEN_CLASSIFICATION:
            # TODO: Implement decode_predictions for the token classification task

            raise NotImplementedError("Token classification task not implemented.")
        case _:
            msg = f"{task_type = } not supported."
            raise ValueError(msg)

    return decode_predictions_function


def decode_predictions_language_modeling(
    tokenizer: transformers.PreTrainedTokenizer,
    prediction_output: transformers.trainer_utils.PredictionOutput,
) -> dict:
    """Decode predictions and labels for language modeling tasks."""
    # `prediction_output.label_ids.shape`: (10, 512)
    #
    # Replace -100 with the pad token id,
    # because otherwise the tokeinzer decode function results in an error:
    # `OverflowError: out of range integral type conversion attempted`
    label_ids: np.ndarray = prediction_output.label_ids.copy()  # type: ignore - this is a numpy array and not a tuple
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # `len(labels_decoded)`: 10
    labels_decoded: list[str] = tokenizer.batch_decode(
        label_ids,
    )
    # `prediction_output.predictions.shape`: (10, 512, 13)
    # `logits_argmax.shape`: (10, 512)
    logits_argmax = prediction_output.predictions.argmax(  # type: ignore - problem with tuple type
        axis=-1,
    )
    # `len(prediction_text)`: 10
    prediction_text: list[str] = tokenizer.batch_decode(
        logits_argmax,
    )
    return {
        "label_ids": prediction_output.label_ids.tolist(),  # type: ignore - this is a numpy array and not a tuple
        "labels_decoded": labels_decoded,
        "logits_argmax": logits_argmax.tolist(),  # type: ignore - this is a numpy array and not a tuple
        "prediction_text": prediction_text,
    }


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This code is inspired by the example provided here:
    https://docs.wandb.ai/guides/integrations/huggingface

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes
    ----------
        trainer: The Hugging Face Trainer instance.
        tokenizer: The tokenizer associated with the model.
        sample_dataset: A subset of the validation dataset
          for generating predictions.
        num_samples: Number of samples to select from
          the validation dataset for generating predictions.
        freq: Frequency of logging. Defaults to 2.

    """

    def __init__(
        self,
        trainer: transformers.Trainer,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        val_dataset: datasets.Dataset,
        num_samples: int = 100,
        frequency: int = 400,
        decode_predictions_function: Callable = decode_predictions_language_modeling,
    ) -> None:
        """Initialize the WandbPredictionProgressCallback instance.

        Args:
        ----
            trainer: The Hugging Face Trainer instance.
            tokenizer: The tokenizer associated
              with the model.
            val_dataset: The validation dataset.
            num_samples: Number of samples to select from
              the validation dataset for generating predictions.
            frequency: Frequency of logging.

        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.frequency = frequency
        self.decode_predictions_function = decode_predictions_function

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,  # noqa: ANN003 - no type annotations for kwargs
    ) -> None:
        super().on_evaluate(
            args,
            state,
            control,
            **kwargs,
        )
        # Control the frequency of logging by logging the predictions
        # every `frequency` global steps.
        if state.epoch is None:
            return

        if state.global_step % self.frequency == 0:
            # generate predictions
            predictions_output: transformers.trainer_utils.PredictionOutput = self.trainer.predict(
                self.sample_dataset,  # type: ignore - problem with datasets.Dataset type
            )
            # decode predictions and labels
            predictions_output_decoded = self.decode_predictions_function(
                self.tokenizer,
                predictions_output,
            )
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(
                predictions_output_decoded,
            )
            predictions_df["epoch"] = state.epoch
            predictions_df["step"] = state.global_step
            predictions_df["sample_dataset"] = self.sample_dataset.to_list()

            records_table = self._wandb.Table(
                dataframe=predictions_df,
                allow_mixed_types=True,  # This is important to avoid problems with `None` values in the dataset fields
            )
            # log the table to wandb
            self._wandb.log(
                {"sample_predictions": records_table},
            )
