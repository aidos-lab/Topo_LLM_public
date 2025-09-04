"""Protocol class for token masking in the embeddings data preparation."""

from typing import Protocol

import numpy as np

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class TokenMasker(Protocol):
    """Protocol class for token masking in the embeddings data preparation."""

    def mask_tokens(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Mask tokens in the arrays and metadata in the input_data.

        Returns
        -------
            A tuple of the prepared data and the indices of the non-masked tokens.

        """
        ...  # pragma: no cover
