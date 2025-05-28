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


class LayerAggregatorMean:
    """Implementation of the LayerAggregator protocol which computes the mean of the layers to be extracted."""

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
