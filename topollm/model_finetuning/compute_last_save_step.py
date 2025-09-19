def compute_last_save_step(
    total_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    save_steps: int,
) -> int:
    """Compute the last save step for the Huggingface Trainer.

    Args:
    ----
        total_samples:
            Total number of samples in the training dataset.
        batch_size:
            Batch size for training.
        gradient_accumulation_steps:
            Number of gradient accumulation steps.
        num_epochs:
            Number of training epochs.
        save_steps:
            Frequency of saving checkpoints.

    Returns:
    -------
        int: The last save step.

    """
    # Calculate the effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Calculate the number of steps per epoch
    steps_per_epoch = total_samples // effective_batch_size

    # Calculate the total number of training steps
    total_training_steps = steps_per_epoch * num_epochs

    # Calculate the last save step
    last_save_step = (total_training_steps // save_steps) * save_steps

    return last_save_step
