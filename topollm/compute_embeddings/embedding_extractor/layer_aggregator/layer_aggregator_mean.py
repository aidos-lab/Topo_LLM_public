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
