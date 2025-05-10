import itertools
import logging
import pathlib

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import seaborn as sns
import wandb
from tqdm.auto import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH, TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.config_classes.main_config import MainConfig
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.typing.enums import Verbosity

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run main function."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    # Load configuration
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # ================================================== #
    # WandB API
    # ================================================== #

    use_scan_history = False

    # Project is specified by <entity/project-name>
    wandb_id: str = main_config.analysis.wandb_export.wandb_id
    project_name: str = main_config.analysis.wandb_export.project_name

    (
        wandb_filters,
        wandb_filters_desc,
    ) = (
        {
            "$or": [
                {"config.dataset.frac_train": 0.3},
                {"config.dataset.frac_train": 0.15},  # TODO: There appears to be a problem with the values 0.1 and 0.15
            ],
        },
        "selected_frac_train",
    )
    # wandb_filters, wandb_filters_desc = None, "None"

    samples: int = main_config.analysis.wandb_export.samples

    save_root_dir: pathlib.Path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "saved_plots",
        "wandb_export",
        f"{project_name=}",
        f"{wandb_filters_desc=}",
        f"{use_scan_history=}",
        f"{samples=}",
    )

    concatenated_df_save_path: pathlib.Path = pathlib.Path(
        save_root_dir,
        "concatenated_df.csv",
    )

    match main_config.analysis.wandb_export.use_saved_concatenated_df:
        case False:
            api = wandb.Api()
            runs = api.runs(
                path=f"{wandb_id}/{project_name}",
                filters=wandb_filters,
            )

            concatenated_df: pd.DataFrame = pd.DataFrame()
            match use_scan_history:
                case False:
                    history_list: list[pd.DataFrame] = []

                    for run in tqdm(
                        runs,
                        desc="Iterating through runs",
                    ):
                        logger.info(
                            msg=f"{run.name=}",  # noqa: G004 - low overhead
                        )
                        # Default value: `samples=500`
                        history: pd.DataFrame = run.history(
                            samples=samples,
                        )

                        history["name"] = run.name
                        history["dataset.frac_train"] = run.config["dataset"]["frac_train"]

                        history_list.append(history)

                    concatenated_df: pd.DataFrame = pd.concat(
                        objs=history_list,
                        ignore_index=True,
                    )

                    pass  # TODO: This is here for setting breakpoints
                case True:
                    for run in runs:
                        run_name = run.name
                        logger.info(
                            msg=f"{run_name=}",  # noqa: G004 - low overhead
                        )

                        # When you pull data from history, by default it's sampled to 500 points.
                        # Get all the logged data points using run.scan_history().
                        # Here's an example downloading all the loss data points logged in history.
                        #
                        history = run.scan_history()

                        # metrics_list = [
                        #     (
                        #         row[x_axis_name],
                        #         row[metric_name],
                        #     )
                        #     for row in tqdm(history)
                        # ]

                        concatenated_df = pd.DataFrame()  # TODO: This is a placeholder

                        pass  # TODO: This is here for setting breakpoints
                case _:
                    msg: str = f"Unknown value for use_scan_history: {use_scan_history=}"
                    raise ValueError(
                        msg,
                    )

            if concatenated_df.empty:
                msg: str = f"Concatenated DataFrame is empty. {concatenated_df=}"
                raise ValueError(
                    msg,
                )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving concatenated_df to {concatenated_df_save_path=} ...",  # noqa: G004 - low overhead
                )
            concatenated_df_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            concatenated_df.to_csv(
                path_or_buf=concatenated_df_save_path,
                index=False,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving concatenated_df to {concatenated_df_save_path=} DONE",  # noqa: G004 - low overhead
                )
        case True:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Loading concatenated_df from {concatenated_df_save_path=} ...",  # noqa: G004 - low overhead
                )
            if not concatenated_df_save_path.exists():
                msg: str = f"Concatenated DataFrame does not exist. {concatenated_df_save_path=}"
                raise FileNotFoundError(
                    msg,
                )

            concatenated_df = pd.read_csv(
                filepath_or_buffer=concatenated_df_save_path,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Loading concatenated_df from {concatenated_df_save_path=} DONE",  # noqa: G004 - low overhead
                )

    # # # #
    # Select a specific subset of the data and create corresponding plots
    x_axis_name_options: list[str] = [
        "step",
    ]
    x_axis_name_to_short_description: dict[str, str] = {
        "step": "Step",
    }

    x_range_step_max_options: list[int] = [
        15_000,
        60_000,
    ]

    metric_name_options: list[str] = [
        "train.accuracy",
        "val.accuracy",
        "train.take_all.desc=twonn_samples=3000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing.n-neighbors-mode=absolute_size_n-neighbors=64.mean",
    ]

    metric_to_short_description: dict[str, str] = {
        "train.accuracy": "Training Accuracy",
        "val.accuracy": "Validation Accuracy",
        "train.take_all.desc=twonn_samples=3000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing.n-neighbors-mode=absolute_size_n-neighbors=64.mean": "Mean local dimension (N=3000; L=64)",
    }

    group_by_column_name_options: list[str | None] = [
        None,
        "dataset.frac_train",
    ]

    add_legend_options: list[bool] = [
        False,
        True,
    ]

    for (
        x_axis_name,
        x_range_step_max,
        metric_name,
        add_legend,
        group_by_column_name,
    ) in itertools.product(
        x_axis_name_options,
        x_range_step_max_options,
        metric_name_options,
        add_legend_options,
        group_by_column_name_options,
    ):
        # ---------- Prepare data ----------
        plot_df: pd.DataFrame = concatenated_df[concatenated_df[x_axis_name] < x_range_step_max]

        # ---------- OO figure/axes ----------
        (
            fig,
            ax,
        ) = plt.subplots(figsize=(12, 8))  # creates both Figure and Axes objects

        # ---------- 3. Draw the lines ----------

        match group_by_column_name:
            case None:
                # - In this case, draw the curves for different runs separately.
                # - Still, color the curves with the same training data fraction with the same color.
                sns.lineplot(
                    data=plot_df,
                    x=x_axis_name,
                    y=metric_name,
                    hue="dataset.frac_train",  # colour chosen by frac_train
                    units="name",  # keep individual runs separate
                    estimator=None,  # no mean/CI aggregation
                    alpha=0.9,
                    legend="auto",
                    ax=ax,  # <-- draw on *this* Axes, not the implicit one
                )
            case _:
                # - Group the runs with the same value in the group_by_column_name
                # - Draw standard deviation bands.
                sns.lineplot(
                    data=plot_df,
                    x=x_axis_name,
                    y=metric_name,
                    hue="dataset.frac_train",  # colour chosen by frac_train
                    units=None,
                    estimator="mean",
                    errorbar=("ci", 95),
                    alpha=0.9,
                    legend="auto",
                    ax=ax,  # <-- draw on *this* Axes, not the implicit one
                )

        # ---------- 4. Axis cosmetics ----------
        ax.set(
            title=None,  # No title, since we add this in the TeX
            xlabel=x_axis_name_to_short_description[x_axis_name],
            ylabel=metric_to_short_description[metric_name],
        )

        # ---------- 5. Custom legend ----------
        match add_legend:
            case False:
                # Remove legend if this is set.
                pass  # TODO: Add or remove the legend depending on this argument

        # ---------- 6. Layout ----------
        fig.tight_layout()

        # # # #
        # Save plots to disk

        plot_output_directory: pathlib.Path = pathlib.Path(
            save_root_dir,
            f"{x_axis_name=}",
            f"{x_range_step_max=}",
            f"{metric_name=}",
            f"{group_by_column_name=}",
        )

        plot_output_path = pathlib.Path(
            plot_output_directory,
            f"plot_{add_legend=}.pdf",
        )
        plot_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_output_path=} ...",  # noqa: G004 - low overhead
            )
        plt.savefig(
            plot_output_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_output_path=} DONE",  # noqa: G004 - low overhead
            )

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
