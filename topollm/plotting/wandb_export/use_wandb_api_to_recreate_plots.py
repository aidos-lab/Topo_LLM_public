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
    verbosity: Verbosity = Verbosity.NORMAL
    logger.info(
        msg="Running script ...",
    )

    api = wandb.Api()

    use_scan_history = False

    # Project is specified by <entity/project-name>
    # dialgroup-hhu/grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_large_p_new_topological_analysis
    wandb_id: str = "dialgroup-hhu"
    project_name: str = (
        "grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_large_p_new_topological_analysis"
    )

    wandb_filters: dict | None = {
        "$or": [
            {"config.dataset.frac_train": 0.3},
            {"config.dataset.frac_train": 0.15},  # TODO: There appears to be a problem with the values 0.1 and 0.15
        ],
    }
    # wandb_filters = None

    runs = api.runs(
        path=f"{wandb_id}/{project_name}",
        filters=wandb_filters,
    )

    samples = 5_000

    save_root_dir: pathlib.Path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "saved_plots",
        "wandb",
        f"{project_name=}",
        f"{use_scan_history=}",
        f"{samples=}",
    )

    x_axis_name = "step"
    x_range_step_max = 15_000

    metric_name = "val.accuracy"

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
                history_list,
                ignore_index=True,
            )

            concatenated_df_save_path: pathlib.Path = pathlib.Path(
                save_root_dir,
                "concatenated_df.csv",
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

            # # # #
            # Select a specific subset of the data and create corresponding plots

            # Restrict the x-axis to a certain range
            plot_df = concatenated_df[concatenated_df[x_axis_name] < x_range_step_max]

            sns.lineplot(
                x=x_axis_name,
                y=metric_name,
                hue="name",
                data=plot_df,
            )
            plt.show()

            pass  # TODO: This is here for setting breakpoints
        case True:
            for run in runs:
                run_name = run.name
                print(f"{run_name=}")

                # When you pull data from history, by default it's sampled to 500 points.
                # Get all the logged data points using run.scan_history().
                # Here's an example downloading all the loss data points logged in history.
                #
                history = run.scan_history()

                metrics_list = [
                    (
                        row[x_axis_name],
                        row[metric_name],
                    )
                    for row in tqdm(history)
                ]

                pass  # TODO: This is here for setting breakpoints

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
