# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Setup OmegaConf with custom resolvers."""

import omegaconf

from topollm.config_classes.sanitize_dirname import sanitize_dirname


def setup_omega_conf() -> None:
    """Set up OmegaConf with custom resolvers."""
    omegaconf.OmegaConf.register_new_resolver(
        name="sanitize_override_dirname",
        resolver=sanitize_dirname,
    )
