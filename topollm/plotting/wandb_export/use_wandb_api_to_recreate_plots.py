import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm.auto import tqdm


def main():
    api = wandb.Api()

    use_scan_history = False

    # Project is specified by <entity/project-name>
    # dialgroup-hhu/grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_large_p_new_topological_analysis
    wandb_id: str = "dialgroup-hhu"
    project_name: str = (
        "grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_large_p_new_topological_analysis"
    )

    runs = api.runs(
        path=f"{wandb_id}/{project_name}",
        filters={
            "$or": [
                {"config.dataset.frac_train": 0.3},
                {"config.dataset.frac_train": 0.15},  # TODO: There appears to be a problem with the values 0.1 and 0.15
            ],
        },
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
                print(
                    f"{run.name=}",
                )
                # Default value: `samples=500`
                history: pd.DataFrame = run.history(
                    samples=5_000,
                )

                history["name"] = run.name
                history["dataset.frac_train"] = run.config["dataset"]["frac_train"]

                history_list.append(history)

            df: pd.DataFrame = pd.concat(
                history_list,
                ignore_index=True,
            )

            # Restrict the x-axis to a certain range
            df = df[df[x_axis_name] < x_range_step_max]

            sns.lineplot(
                x=x_axis_name,
                y=metric_name,
                hue="name",
                data=df,
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


if __name__ == "__main__":
    main()
