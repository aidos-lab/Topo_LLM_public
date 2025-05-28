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
