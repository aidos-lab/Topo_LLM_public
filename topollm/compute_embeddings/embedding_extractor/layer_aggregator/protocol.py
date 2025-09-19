from typing import Protocol

import torch


class LayerAggregator(Protocol):
    def aggregate_layers(
        self,
        layers_to_extract: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate the layers to be extracted into a single tensor."""
        ...
