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

"""Load and compare model checkpoints."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)
setup_omega_conf()


@hydra.main(
    config_path="../../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    data_dir: pathlib.Path = main_config.paths.data_dir
    logger.info(
        f"{data_dir = }",  # noqa: G004 - low overhead
    )

    # # # #
    # Load the models

    model_files_root_dir = pathlib.Path(
        data_dir,
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

    model_paths_list: list[pathlib.Path] = []

    layer_names_to_compare: list[str] = [
        "roberta.encoder.layer.1.attention.self.query.weight",
        "roberta.encoder.layer.11.attention.self.query.weight",
    ]

    # TODO This script is not finished

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
