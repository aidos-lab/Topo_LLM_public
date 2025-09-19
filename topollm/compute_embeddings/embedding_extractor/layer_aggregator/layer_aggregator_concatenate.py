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
