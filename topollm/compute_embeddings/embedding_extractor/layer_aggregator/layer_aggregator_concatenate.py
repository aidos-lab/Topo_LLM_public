# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import torch


class LayerAggregatorConcatenate:
    """Implementation of the LayerAggregator protocol which concatenates the layers to be extracted."""

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
