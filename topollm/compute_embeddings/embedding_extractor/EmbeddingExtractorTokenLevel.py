from topollm.compute_embeddings.embedding_extractor.LayerAggregatorProtocol import (
    LayerAggregator,
)
from topollm.compute_embeddings.embedding_extractor.LayerExtractorProtocol import (
    LayerExtractor,
)


import numpy as np
import transformers.modeling_outputs


class EmbeddingExtractorTokenLevel:
    """
    Implementation of the EmbeddingExtractor protocol
    which extracts token level embeddings.
    """

    def __init__(
        self,
        layer_extractor: LayerExtractor,
        layer_aggregator: LayerAggregator,
        embedding_dimension: int,
    ):
        self.layer_extractor = layer_extractor
        self.layer_aggregator = layer_aggregator
        self.embedding_dimension = embedding_dimension

    def extract_embeddings_from_model_outputs(
        self,
        model_outputs: transformers.modeling_outputs.BaseModelOutput,
    ) -> np.ndarray:
        # Ensure the model outputs hidden states
        if not hasattr(
            model_outputs,
            "hidden_states",
        ):
            raise ValueError("Model outputs do not contain 'hidden_states'")

        hidden_states = (
            model_outputs.hidden_states
        )  # Assuming this is a tuple of tensors

        if hidden_states is None:
            raise ValueError(
                f"'hidden_states' is None. "
                f"Did you call the model with 'output_hidden_states=True'?"
            )

        # Extract specified layers
        layers_to_extract = self.layer_extractor.extract_layers_from_model_outputs(
            hidden_states=hidden_states,
        )

        # Aggregate the extracted layers
        embeddings = self.layer_aggregator.aggregate_layers(
            layers_to_extract=layers_to_extract,
        )

        return embeddings.cpu().numpy()
