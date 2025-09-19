"""Count the number of trainable parameters in a PyTorch model."""

from torch import nn


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Count the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
