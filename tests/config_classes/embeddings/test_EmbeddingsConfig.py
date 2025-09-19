import logging
import pprint

import omegaconf
from hydra import compose, initialize_config_module

from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig

logger = logging.getLogger(__name__)


def test_hydra_with_EmbeddingsConfig() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="configs.embeddings",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="basic_embeddings",
            overrides=[],
        )

        logger.info(f"cfg:\n{pprint.pformat(cfg)}")

        # This tests whether the configuration is valid
        config = EmbeddingsConfig.model_validate(
            obj=cfg,
        )

        logger.info(f"{type(config) = }")
        logger.info(f"config:\n{pprint.pformat(config)}")
