"""Setup OmegaConf with custom resolvers."""

import omegaconf

from topollm.config_classes.sanitize_dirname import sanitize_dirname


def setup_omega_conf() -> None:
    """Set up OmegaConf with custom resolvers."""
    omegaconf.OmegaConf.register_new_resolver(
        name="sanitize_override_dirname",
        resolver=sanitize_dirname,
    )
