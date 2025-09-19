"""Test the MainConfig class.

Inspired by the examples from the hydra repository:
https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
"""

import logging
import pprint
from typing import TYPE_CHECKING

from hydra import compose, initialize

from topollm.config_classes.main_config import MainConfig

if TYPE_CHECKING:
    import omegaconf

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def test_hydra_with_main_config() -> None:
    """Test the MainConfig class with Hydra."""
    with initialize(
        config_path="../../configs",
        version_base=None,
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="main_config",
            overrides=[
                "data.data_subsampling.number_of_samples=6000",
            ],
        )

        logger.info(
            msg=f"cfg:\n{pprint.pformat(object=cfg)}",  # noqa: G004 - low overhead
        )

        # This tests whether the configuration is valid
        config: MainConfig = MainConfig.model_validate(
            obj=cfg,
        )

        logger.info(
            msg=f"{type(config) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
        )
