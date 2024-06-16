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


from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES


class TokenizerConfig(ConfigBaseModel):
    """Configurations for the tokenizer."""

    add_prefix_space: bool = Field(
        default=False,
        title="Add prefix space.",
        description="Whether to add prefix space.",
    )

    max_length: int = Field(
        default=512,
        title="Maximum length of the input sequence.",
        description="The maximum length of the input sequence.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the tokenizer config.

        Returns
        -------
            str: The description of the tokenizer.

        """
        desc = (
            f"{NAME_PREFIXES['add_prefix_space']}"
            f"{KV_SEP}"
            f"{str(self.add_prefix_space)}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['max_length']}"
            f"{KV_SEP}"
            f"{str(self.max_length)}"
        )

        return desc
