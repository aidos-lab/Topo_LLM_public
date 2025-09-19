from typing import Protocol

import numpy as np
import transformers.modeling_outputs


class EmbeddingExtractor(Protocol):
    """Extract embeddings from model outputs."""

    embedding_dimension: int

    def extract_embeddings_from_model_outputs(
        self,
        model_outputs: transformers.modeling_outputs.BaseModelOutput,
    ) -> np.ndarray:
        """Extract embeddings from the model outputs."""
        ...  # pragma: no cover
