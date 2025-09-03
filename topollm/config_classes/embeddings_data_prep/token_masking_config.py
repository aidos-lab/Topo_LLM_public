"""Configurations for specifying token masking in the embeddings data prep step."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.typing.enums import EmbeddingsDataPrepTokenMaskingMode


class TokenMaskingConfig(ConfigBaseModel):
    """Configurations for specifying token masking in the embeddings data prep step."""

    token_masking_mode: EmbeddingsDataPrepTokenMaskingMode = Field(
        default=EmbeddingsDataPrepTokenMaskingMode.NO_MASKING,
        title="Masking mode.",
        description="The masking mode to be used.",
    )

    token_mask_meta_column_name: str = Field(
        default="token_mask",
        title="Token mask meta column name.",
        description="The name of the meta column to be used for token masking.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        desc: str = (
            f"{NAME_PREFIXES['embeddings_data_prep_token_masking_mode']}{KV_SEP}{self.token_masking_mode}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['embeddings_data_prep_token_mask_meta_column_name']}{KV_SEP}{self.token_mask_meta_column_name}"
        )

        return desc
