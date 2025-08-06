"""Basic path manager for the PEFT finetuning mode."""

import logging
import pathlib

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.path_management.target_modules_to_path_part import target_modules_to_path_part
from topollm.typing.enums import DescriptionType, FinetuningMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class PEFTPathManagerBasic:
    """Path manager for the PEFT finetuning mode."""

    def __init__(
        self,
        peft_config: PEFTConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the PEFTPathManagerBasic."""
        self.peft_config: PEFTConfig = peft_config

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    @property
    def peft_description_subdir(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_finetuning_mode_description(
                description_type=DescriptionType.LONG,
            ),
            self.get_lora_description(
                description_type=DescriptionType.LONG,
            ),
        )

        return path

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    self.get_finetuning_mode_description(
                        description_type=DescriptionType.LONG,
                    )
                    + ITEM_SEP
                    + self.get_lora_description(
                        description_type=DescriptionType.LONG,
                    )
                )
            case DescriptionType.SHORT:
                description: str = (
                    self.get_finetuning_mode_description(
                        description_type=DescriptionType.SHORT,
                        short_description_separator=short_description_separator,
                    )
                    + short_description_separator
                    + self.get_lora_description(
                        description_type=DescriptionType.SHORT,
                        short_description_separator=short_description_separator,
                    )
                )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description

    def get_finetuning_mode_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",  # noqa: ARG002 - not used but kept for consistent interface
    ) -> str:
        match description_type:
            case DescriptionType.LONG:
                match self.peft_config.finetuning_mode:
                    case FinetuningMode.STANDARD:
                        description: str = f"{NAME_PREFIXES['FinetuningMode']}{KV_SEP}standard"
                    case FinetuningMode.LORA:
                        description = f"{NAME_PREFIXES['FinetuningMode']}{KV_SEP}lora"
                    case _:
                        msg: str = f"Unknown {self.peft_config.finetuning_mode = }"
                        raise ValueError(
                            msg,
                        )
            case DescriptionType.SHORT:
                match self.peft_config.finetuning_mode:
                    case FinetuningMode.STANDARD:
                        description = "standard"
                    case FinetuningMode.LORA:
                        description = "lora"
                    case _:
                        msg: str = f"Unknown {self.peft_config.finetuning_mode = }"
                        raise ValueError(
                            msg,
                        )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description

    def get_lora_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        match description_type:
            case DescriptionType.LONG:
                match self.peft_config.finetuning_mode:
                    case FinetuningMode.STANDARD:
                        description: str = "lora-None"
                    case FinetuningMode.LORA:
                        description: str = (
                            f"{NAME_PREFIXES['lora_r']}"
                            f"{KV_SEP}"
                            f"{str(object=self.peft_config.r)}"
                            f"{ITEM_SEP}"
                            f"{NAME_PREFIXES['lora_alpha']}"
                            f"{KV_SEP}"
                            f"{str(object=self.peft_config.lora_alpha)}"
                            f"{ITEM_SEP}"
                            f"{NAME_PREFIXES['lora_target_modules']}"
                            f"{KV_SEP}"
                            f"{target_modules_to_path_part(target_modules=self.peft_config.target_modules)}"
                            f"{ITEM_SEP}"
                            f"{NAME_PREFIXES['lora_dropout']}"
                            f"{KV_SEP}"
                            f"{self.peft_config.lora_dropout}"
                            f"{ITEM_SEP}"
                            f"{NAME_PREFIXES['use_rslora']}"
                            f"{KV_SEP}"
                            f"{self.peft_config.use_rslora}"
                        )
                    case _:
                        msg: str = f"Unknown finetuning_mode: {self.peft_config.finetuning_mode = }"
                        raise ValueError(
                            msg,
                        )
            case DescriptionType.SHORT:
                match self.peft_config.finetuning_mode:
                    case FinetuningMode.STANDARD:
                        description: str = "None"
                    case FinetuningMode.LORA:
                        description: str = (
                            f"{self.peft_config.r}"
                            f"{short_description_separator}"
                            f"{self.peft_config.lora_alpha}"
                            f"{short_description_separator}"
                            f"{target_modules_to_path_part(target_modules=self.peft_config.target_modules)}"
                            f"{short_description_separator}"
                            f"{self.peft_config.lora_dropout}"
                            f"{short_description_separator}"
                            f"{self.peft_config.use_rslora}"
                        )
                    case _:
                        msg: str = f"Unknown finetuning_mode: {self.peft_config.finetuning_mode = }"
                        raise ValueError(
                            msg,
                        )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description
