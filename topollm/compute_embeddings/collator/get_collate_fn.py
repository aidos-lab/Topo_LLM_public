"""Factory function to get the collate function for the data loader."""

from collections.abc import Callable
from functools import partial

from topollm.compute_embeddings.collator.collate_batch_for_embedding import collate_batch_and_move_to_device
from topollm.model_handling.loaded_model_container import LoadedModelContainer


def get_collate_fn(
    loaded_model_container: LoadedModelContainer,
) -> Callable:
    """Get the collate function for the data loader."""
    partial_collate_fn = partial(
        collate_batch_and_move_to_device,
        device=loaded_model_container.device,
        model_input_names=loaded_model_container.tokenizer.model_input_names,
    )

    return partial_collate_fn
