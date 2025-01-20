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

"""Evaluate the tuned model."""

import logging
import math

import transformers

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def evaluate_tuned_model(
    trainer: transformers.Trainer,
    logger: logging.Logger = default_logger,
) -> None:
    """Evaluate the model."""
    logger.info(
        msg="Evaluating the model ...",
    )

    eval_results = trainer.evaluate()
    logger.info(
        msg=f"eval_results:\n{eval_results}",  # noqa: G004 - low overhead
    )

    # Since the model evaluation might not return the 'eval_loss' key, we need to check for it
    if "eval_loss" in eval_results:
        perplexity = math.exp(eval_results["eval_loss"])
        logger.info(
            msg=f"perplexity:\n{perplexity:.2f}",  # noqa: G004 - low overhead
        )
    else:
        logger.warning(
            msg="Could not calculate perplexity, because 'eval_loss' was not in eval_results",
        )

    logger.info(
        msg="Evaluating the model DONE",
    )
