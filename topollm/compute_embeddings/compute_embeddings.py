# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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
Create embedding vectors.

# TODO This script is under development
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import argparse
import logging
import pathlib
import pprint

# Third party imports
from hydra import main
import torch
import torch.utils.data

# Local imports
from topollm.utils.Configs import EmbeddingsConfig

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@main(
    config_path="configs",
    config_name="config",
    version_base="1.2",
)
def main(
    config,
):
    """Run the script."""

    pprint.pprint(
        config,
    )

    config = EmbeddingsConfig.model_validate(
        config,
    )

    pprint.pprint(
        config,
    )


if __name__ == "__main__":
    main()
