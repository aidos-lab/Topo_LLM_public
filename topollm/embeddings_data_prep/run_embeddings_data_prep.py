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

"""Prepare the embedding data of a model and its metadata for further analysis.

The script outputs two numpy arrays of subsamples
of the respective arrays that correspond to the
embeddings of the base model and the fine-tuned model,
respectively.
The arrays are stored in the directory where this
script is executed.
Since paddings are removed from the embeddings,
the resulting size of the arrays will usually be
significantly lower than the specified sample size
(often ~5% of the specified size).
"""

import logging
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.embeddings_data_prep.embeddings_data_prep_worker import embeddings_data_prep_worker
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.get_torch_device import get_torch_device

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

# logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
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

    embeddings_data_prep_worker(
        main_config=main_config,
        device=device,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    global_logger.info("Script finished.")


if __name__ == "__main__":
    main()
