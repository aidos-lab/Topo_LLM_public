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
Create embedding vectors from dataset.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Local imports
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.load_tokenizer import load_tokenizer
from topollm.model_handling.load_model import load_model
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.config_classes.Configs import MainConfig
from topollm.config_classes.path_management.EmbeddingsPathManagerFactory import (
    get_embeddings_path_manager,
)

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

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

    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=global_logger,
    )

    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.embeddings.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.embeddings.tokenizer,
        logger=global_logger,
        verbosity=main_config.verbosity,
    )
    model = load_model(
        pretrained_model_name_or_path=main_config.embeddings.language_model.pretrained_model_name_or_path,
        device=device,
        logger=global_logger,
        verbosity=main_config.verbosity,
    )

    # TODO Continue here

    global_logger.info("DONE")

    return


if __name__ == "__main__":
    main()
