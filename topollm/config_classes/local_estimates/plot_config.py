# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class PlotSavingConfig(ConfigBaseModel):
    """Configurations for specifying saving options of the plot."""

    save_html: bool = True
    save_pdf: bool = True
    save_csv: bool = True


class LocalEstminatesPlotConfig(ConfigBaseModel):
    """Configurations for specifying parameters of the local estimates plot."""

    pca_n_components: int | None = Field(
        default=50,
        title="Number of PCA components before t-SNE.",
        description="The number of PCA components before t-SNE to use for embedding data preparation.",
    )
    saving: PlotSavingConfig = Field(
        default_factory=PlotSavingConfig,
        title="Configuration for saving the plots.",
        description="Configurations for specifying saving options of the plot.",
    )

    tsne_n_components: int = Field(
        default=2,
        title="Number of t-SNE components",
        description="The number of t-SNE components to use for embedding data preparation.",
    )
    tsne_random_state: int = Field(
        default=42,
        title="Random state for t-SNE",
        description="The random state to use for t-SNE algorithm.",
    )
