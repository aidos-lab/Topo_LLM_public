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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports

# Third party imports
import torch

# Local imports
from typing import Protocol

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class LayerAggregator(Protocol):
    dimension_multiplier: int

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        This method aggregates the layers to be extracted into a single tensor.
        """
        ...


class MeanLayerAggregator:
    """
    Implementation of the LayerAggregator protocol
    which computes the mean of the layers to be extracted.
    """

    def __init__(self):
        self.dimension_multiplier = 1

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        # Mean across the layers
        aggregated_layers = torch.mean(
            torch.stack(
                layers_to_extract,
                dim=0,
            ),
            dim=0,
        )
        return aggregated_layers


class ConcatenateLayerAggregator:
    """
    Implementation of the LayerAggregator protocol
    which concatenates the layers to be extracted.
    """

    def __init__(self):
        self.dimension_multiplier = (
            1  # ! TODO: This needs to be set flexibly depending on the method
        )

    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        # Concatenate across the last dimension
        aggregated_layers = torch.cat(
            layers_to_extract,
            dim=-1,
        )
        return aggregated_layers
