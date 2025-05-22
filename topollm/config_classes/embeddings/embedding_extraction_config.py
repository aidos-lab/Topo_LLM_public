# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

from pydantic import BaseModel, Field

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part
from topollm.typing.enums import AggregationType


class EmbeddingExtractionConfig(BaseModel):
    """Configuration for specifying embedding extraction."""

    layer_indices: list[int] = Field(
        default_factory=lambda: [-1],  # [-1] denotes the last layer
    )
    aggregation: AggregationType = Field(
        default=AggregationType.MEAN,
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the embedding extraction.

        Returns
        -------
            str: The description of the embedding extraction.

        """
        desc: str = (
            f"{NAME_PREFIXES['layer']}"
            f"{KV_SEP}"
            f"{convert_layer_indices_to_path_part(self.layer_indices)}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['aggregation']}"
            f"{KV_SEP}"
            f"{str(object=self.aggregation)}"
        )

        return desc


def convert_layer_indices_to_path_part(
    layer_indices: list[int],
) -> str:
    """Convert a list of layer indices to a string suitable for file paths."""
    return convert_list_to_path_part(
        input_list=layer_indices,
    )
