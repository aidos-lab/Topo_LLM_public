from typing import Any

import torch


def move_batch_to_cpu(
    batch: dict,
) -> dict:
    """Move all tensors in the batch to CPU, including nested dictionaries."""

    def recursive_to_cpu(
        data: Any,  # noqa: ANN401 - Any type is necessary here
    ) -> Any:  # noqa: ANN401 - Any type is necessary here
        if isinstance(
            data,
            torch.Tensor,
        ):
            return data.cpu()
        elif isinstance(
            data,
            dict,
        ):
            return {key: recursive_to_cpu(value) for key, value in data.items()}
        elif isinstance(
            data,
            list,
        ):
            return [recursive_to_cpu(item) for item in data]
        elif isinstance(
            data,
            tuple,
        ):
            return tuple(recursive_to_cpu(item) for item in data)
        # Add other structures if necessary
        return data

    return recursive_to_cpu(
        data=batch,
    )
