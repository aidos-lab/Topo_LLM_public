"""Test the FinetuningConfig class with Hydra."""

import logging
import pprint
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_module

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig

if TYPE_CHECKING:
    import omegaconf

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def test_hydra_with_finetuning_config() -> None:
    """Test Hydra with FinetuningConfig."""
    with initialize_config_module(
        version_base=None,
        config_module="configs.finetuning",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="finetuning_for_masked_lm",
            overrides=[
                "batch_sizes.eval=42",
            ],
        )

        logger.info(
            "cfg:\n%s",
            pprint.pformat(cfg),
        )

        # This tests whether the configuration is valid
        config = FinetuningConfig.model_validate(
            obj=cfg,
        )

        logger.info(
            f"{type(config) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "config:\n%s",
            pprint.pformat(config),
        )
