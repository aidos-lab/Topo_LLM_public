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

from collections.abc import Callable

import evaluate
import numpy as np

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.typing.enums import ComputeMetricsMode, TaskType


def prepare_compute_seqeval_metrics() -> Callable:
    # TODO: pass the label_list as an argument

    # Load the seqeval library
    seqeval = evaluate.load("seqeval")

    def compute_seqeval_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_seqeval_metrics


def get_compute_metrics(
    finetuning_config: FinetuningConfig,
) -> Callable | None:
    """Get the compute_metrics function for the Trainer."""
    match finetuning_config.compute_metrics_mode:
        case ComputeMetricsMode.NONE:
            compute_metrics = None
        case ComputeMetricsMode.FROM_TASK_TYPE:
            match finetuning_config.base_model.task_type:
                case TaskType.CAUSAL_LM | TaskType.MASKED_LM:
                    compute_metrics = None
                case TaskType.TOKEN_CLASSIFICATION:
                    compute_metrics = prepare_compute_seqeval_metrics()
                case _:
                    msg = f"Unknown {finetuning_config.base_model.task_type = }"
                    raise ValueError(msg)
        case _:
            msg = f"Unknown {finetuning_config.compute_metrics_mode = }"
            raise ValueError(msg)

    return compute_metrics
