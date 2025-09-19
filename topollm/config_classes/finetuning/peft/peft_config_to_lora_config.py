from peft.tuners.lora.config import LoraConfig

from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig


def peft_config_to_lora_config(
    peft_config: PEFTConfig,
) -> LoraConfig:
    """Convert a PEFTConfig to a LoraConfig.

    https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig
    """
    # Note: The 'task_type' argument is not necessary, i.e.,
    # `task_type=peft.utils.peft_types.TaskType.CAUSAL_LM` is not necessary.
    lora_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        target_modules=peft_config.target_modules,
        lora_dropout=peft_config.lora_dropout,
        use_rslora=peft_config.use_rslora,
    )

    return lora_config
