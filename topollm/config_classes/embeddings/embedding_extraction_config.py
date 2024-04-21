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

from pydantic import BaseModel

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.enums import AggregationType


class EmbeddingExtractionConfig(BaseModel):
    layer_indices: list[int]
    aggregation: AggregationType = AggregationType.MEAN  # type: ignore

    @property
    def embedding_extraction_config_description(
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
            f"{self.layer_indices!r}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['aggregation']}"
            f"{KV_SEP}"
            f"{self.aggregation!r}"
        )

        return desc
