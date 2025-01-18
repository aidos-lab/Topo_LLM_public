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

# TODO This script is under development and not yet finished.

"""Load and compare model checkpoints."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.get_data_dir import get_data_dir
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)
setup_omega_conf()


@hydra.main(
    config_path="../../../configs",
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
        logger=global_logger,
    )

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # #
    # Load the models

    model_files_root_dir = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models",
        "finetuned_models",
        "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000",
        "model-roberta-base",
        "ftm-standard",
        "lora-None",
        "gradmod-freeze_layers_target-freeze-['encoder.layer.0.', 'encoder.layer.1.', 'encoder.layer.2.', 'encoder.layer.3.', 'encoder.layer.4.', 'encoder.layer.5.']"
        "lr-5e-05_lr_scheduler_type-constant_wd-0.01",
        "ep-50",
        "model_files",
    )

    # This holds the list of model files to compare
    models_identifier_or_paths_list: list[str] = []

    # This holds the list of parameters to compare
    layer_names_to_compare: list[str] = [
        "roberta.encoder.layer.1.attention.self.query.weight",
        "roberta.encoder.layer.11.attention.self.query.weight",
    ]

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
