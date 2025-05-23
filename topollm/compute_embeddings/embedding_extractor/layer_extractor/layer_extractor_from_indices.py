# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_2 (author2@example.com)
# AUTHOR_1 (author1@example.com)
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

import torch


class LayerExtractorFromIndices:
    """Implementation of the LayerExtractor protocol which is configured from a list of layer indices."""

    def __init__(
        self,
        layer_indices: list[int],
    ):
        """Initialize the LayerExtractorFromIndices with the layer indices to extract."""
        self.layer_indices = layer_indices

    def extract_layers_from_model_outputs(
        self,
        hidden_states: tuple,
    ) -> list[torch.Tensor]:
        layers_to_extract = [hidden_states[i] for i in self.layer_indices]
        return layers_to_extract
