# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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
from collections.abc import Callable

import evaluate
import numpy as np
import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.typing.enums import ComputeMetricsMode, TaskType, Verbosity

default_logger = logging.getLogger(__name__)


def prepare_compute_seqeval_metrics(
    label_list: list[str],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> Callable:
    """Prepare the compute_seqeval_metrics function for the Trainer."""
    # Load the seqeval library.
    # We move this here into the function to avoid loading it
    # if it is not needed.
    seqeval_from_evaluate_load = evaluate.load("seqeval")

    def compute_seqeval_metrics(
        p: transformers.EvalPrediction,
    ) -> dict:
        predictions, labels = p
        predictions = np.argmax(
            predictions,
            axis=2,
        )

        ignore_label = -100
        true_predictions = [
            [
                label_list[p]
                for (p, l) in zip(  # noqa: E741 - variable name is not a problem here
                    prediction,
                    label,
                    strict=True,
                )
                if l != ignore_label
            ]
            for prediction, label in zip(
                predictions,
                labels,
                strict=True,
            )
        ]
        true_labels = [
            [
                label_list[l]
                for (p, l) in zip(  # noqa: E741 - variable name is not a problem here
                    prediction,
                    label,
                    strict=True,
                )
                if l != ignore_label
            ]
            for prediction, label in zip(
                predictions,
                labels,
                strict=True,
            )
        ]

        results = seqeval_from_evaluate_load.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        if results is None:
            if verbosity >= Verbosity.NORMAL:
                logger.warning("seqeval results are None.")
            return {
                "precision": -1.0,
                "recall": -1.0,
                "f1": -1.0,
                "accuracy": -1.0,
            }

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_seqeval_metrics


def get_compute_metrics(
    finetuning_config: FinetuningConfig,
    label_list: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> Callable | None:
    """Get the compute_metrics function for the Trainer."""
    match finetuning_config.compute_metrics_mode:
        case ComputeMetricsMode.NONE:
            if verbosity >= Verbosity.NORMAL:
                logger.info("Returning None for compute_metrics.")
            compute_metrics = None
        case ComputeMetricsMode.FROM_TASK_TYPE:
            match finetuning_config.base_model.task_type:
                case TaskType.CAUSAL_LM | TaskType.MASKED_LM:
                    if verbosity >= Verbosity.NORMAL:
                        logger.info("Returning None for compute_metrics.")
                    compute_metrics = None
                case TaskType.TOKEN_CLASSIFICATION:
                    if label_list is None:
                        msg = "`label_list` is None, but we need it to compute the seqeval metrics."
                        raise ValueError(msg)
                    compute_metrics = prepare_compute_seqeval_metrics(
                        label_list=label_list,
                        verbosity=verbosity,
                        logger=logger,
                    )
                case _:
                    msg = f"Unknown {finetuning_config.base_model.task_type = }"
                    raise ValueError(msg)
        case _:
            msg = f"Unknown {finetuning_config.compute_metrics_mode = }"
            raise ValueError(msg)

    return compute_metrics
