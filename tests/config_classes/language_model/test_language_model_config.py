import logging
import pprint
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_module

from topollm.config_classes.language_model.language_model_config import (
    LanguageModelConfig,
)

if TYPE_CHECKING:
    import omegaconf

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@pytest.mark.parametrize(
    argnames="config_name",
    argvalues=[
        "gpt2-medium",
        "roberta-base",
        "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
    ],
)
def test_hydra_with_language_model_config(
    config_name: str,
) -> None:
    """Test the LanguageModelConfig class with Hydra."""
    with initialize_config_module(
        version_base=None,
        config_module="configs.language_model",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name=config_name,
            overrides=[
                "pretrained_model_name_or_path=overridden_pretrained_model_name_or_path",
                "short_model_name=overridden_short_model_name",
            ],
        )

        logger.info(
            msg=f"cfg:\n{pprint.pformat(object=cfg)}",  # noqa: G004 - low overhead
        )

        # This tests whether the configuration is valid
        config: LanguageModelConfig = LanguageModelConfig.model_validate(
            obj=cfg,
        )

        logger.info(
            msg=f"{type(config) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
        )
