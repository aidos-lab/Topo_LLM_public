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
