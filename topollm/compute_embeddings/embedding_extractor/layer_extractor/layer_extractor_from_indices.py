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
