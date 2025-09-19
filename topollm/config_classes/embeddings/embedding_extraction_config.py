from pydantic import BaseModel, Field

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part
from topollm.typing.enums import AggregationType


class EmbeddingExtractionConfig(BaseModel):
    """Configuration for specifying embedding extraction."""

    layer_indices: list[int] = Field(
        default_factory=lambda: [-1],  # [-1] denotes the last layer
    )
    aggregation: AggregationType = Field(
        default=AggregationType.MEAN,
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the embedding extraction.

        Returns
        -------
            str: The description of the embedding extraction.

        """
        desc: str = (
            f"{NAME_PREFIXES['layer']}"
            f"{KV_SEP}"
            f"{convert_layer_indices_to_path_part(self.layer_indices)}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['aggregation']}"
            f"{KV_SEP}"
            f"{str(object=self.aggregation)}"
        )

        return desc


def convert_layer_indices_to_path_part(
    layer_indices: list[int],
) -> str:
    """Convert a list of layer indices to a string suitable for file paths."""
    return convert_list_to_path_part(
        input_list=layer_indices,
    )
