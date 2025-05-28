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


from typing import Protocol

import torch


class LayerExtractor(Protocol):
    """Protocol for the layer extractor."""

    def extract_layers_from_model_outputs(
        self,
        hidden_states,
    ) -> list[torch.Tensor]:
        """Extract layers from the model outputs."""
        ...  # pragma: no cover
