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

import transformers

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class TrainerModifierWandbPredictionProgressCallback:
    """Gradient modifier that does not modify the model."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_trainer(
        self,
        trainer: transformers.Trainer,
    ) -> transformers.Trainer:
        # TODO: Implement this

        raise NotImplementedError("Not implemented yet")

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Returning unmodified trainer.")

        return trainer


def decode_predictions(
    tokenizer,
    predictions,
):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


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
        tokenizer: transformers.AutoTokenizer,
        val_dataset: datasets.Dataset,
        num_samples: int = 100,
        freq: int = 2,
    ):
        """Initialize the WandbPredictionProgressCallback instance.

        Args:
        ----
            trainer: The Hugging Face Trainer instance.
            tokenizer: The tokenizer associated
              with the model.
            val_dataset: The validation dataset.
            num_samples: Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq: Frequency of logging. Defaults to 2.

        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if state.epoch % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})
