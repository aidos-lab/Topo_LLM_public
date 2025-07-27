"""Prepare the training arguments for the finetuning process."""

import logging
import os

import torch
import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_training_args(
    finetuning_config: FinetuningConfig,
    finetuned_model_dir: os.PathLike,
    device: torch.device,
    logging_dir: os.PathLike | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> transformers.TrainingArguments:
    """Prepare the training arguments for the finetuning process."""
    match device.type:
        case "cpu":
            fp16: bool = finetuning_config.fp16
            if finetuning_config.use_cpu is False:
                logger.warning(
                    msg="Selected device is CPU, but use_cpu is set to False. Setting use_cpu to True.",
                )

            use_cpu: bool = True
        case "mps":
            # MPS backend does not support fp16
            fp16: bool = False
            if finetuning_config.fp16:
                logger.warning(
                    msg="MPS backend does not support fp16. Setting fp16 to False.",
                )

            use_cpu: bool = finetuning_config.use_cpu
        case _:
            fp16: bool = finetuning_config.fp16
            use_cpu: bool = finetuning_config.use_cpu

    # Note: the `label_names` argument appears to be necessary for the PEFT evaluation to work.
    # https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
    training_args = transformers.TrainingArguments(
        output_dir=str(finetuned_model_dir),
        overwrite_output_dir=True,
        num_train_epochs=finetuning_config.num_train_epochs,
        max_steps=finetuning_config.max_steps,
        learning_rate=finetuning_config.learning_rate,
        lr_scheduler_type=finetuning_config.lr_scheduler_type,
        weight_decay=finetuning_config.weight_decay,
        per_device_train_batch_size=finetuning_config.batch_sizes.train,
        per_device_eval_batch_size=finetuning_config.batch_sizes.eval,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        gradient_checkpointing=finetuning_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        fp16=fp16,
        warmup_steps=finetuning_config.warmup_steps,
        eval_strategy="steps",
        eval_steps=finetuning_config.eval_steps,
        save_steps=finetuning_config.save_steps,
        label_names=[
            "labels",
        ],
        logging_dir=logging_dir,  # type: ignore - typing problem with None and str
        report_to=finetuning_config.report_to,
        log_level=finetuning_config.log_level,
        logging_steps=finetuning_config.logging_steps,
        use_cpu=use_cpu,
        seed=finetuning_config.seed,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Prepared training arguments:\n{training_args}",  # noqa: G004 - low overhead
        )

    return training_args
