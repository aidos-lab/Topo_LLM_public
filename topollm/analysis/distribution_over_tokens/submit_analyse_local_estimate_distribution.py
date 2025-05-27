"""Launch analyse_local_estimate_distribution_over_tokens.py with various argument combinations."""

import logging
import os
import subprocess

import click

from topollm.logging.create_and_configure_global_logger import create_and_configure_global_logger

global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def run_command(
    script,
    args,
    dry_run,
):
    cmd = [
        "uv",
        "run",
        "python",
        script,
        *args,
    ]

    print(f"Launching: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


@click.command()
@click.option(
    "--launcher",
    default="basic",
    show_default=True,
    help="Hydra launcher to use (basic or hpc_submission)",
)
@click.option(
    "--run-mode",
    type=click.Choice(["regular", "dry_run"]),
    default="regular",
    show_default=True,
    help="Run mode: 'regular' executes commands, 'dry_run' only prints them",
)
def main(
    launcher,
    run_mode,
):
    """Run analyse_local_estimate_distribution_over_tokens.py for various argument combinations."""
    script = "topollm/analysis/distribution_over_tokens/analyse_local_estimate_distribution_over_tokens.py"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # determine dry_run from run_mode
    match run_mode:
        case "regular":
            dry_run = False
        case "dry_run":
            dry_run = True
        case _:
            msg = f"Unknown run_mode: {run_mode}"
            raise ValueError(msg)

    # --- Launcher options ---
    match launcher:
        case "basic":
            launcher_args = ["hydra/launcher=basic"]
        case "hpc_submission":
            launcher_args = [
                "hydra/launcher=hpc_submission",
                "hydra.launcher.template=CPU",
                "hydra.launcher.memory=17",
                "hydra.launcher.ncpus=3",
                "hydra.launcher.ngpus=0",
                "hydra.launcher.walltime=00:15:00",
            ]
        case _:
            raise ValueError(f"Unknown launcher: {launcher}")

    base_args = ["--multirun", "hydra/sweeper=basic", "tokenizer.add_prefix_space=False"]

    embeddings_args = [
        "embeddings.embedding_data_handler.mode=regular",
        "embeddings.embedding_extraction.layer_indices=[-1]",
        "embeddings_data_prep.sampling.num_samples=150000",
        "embeddings_data_prep.sampling.sampling_mode=random",
        "embeddings_data_prep.sampling.seed=42",
    ]

    local_estimates_args = [
        "local_estimates=twonn",
        "local_estimates.pointwise.n_neighbors_mode=absolute_size",
        "local_estimates.filtering.deduplication_mode=array_deduplicator",
        "local_estimates.filtering.num_samples=60000",
        "local_estimates.pointwise.absolute_n_neighbors=128",
    ]

    # --- Trippy-R case ---
    trippy_r_data_args = [
        "data=trippy_dataloaders_processed",
        "data.data_subsampling.split=train,dev,test",
        "data.data_subsampling.sampling_mode=random",
        "data.data_subsampling.number_of_samples=7000",
        "data.data_subsampling.sampling_seed=778",
    ]

    trippy_r_models = [
        "roberta-base-trippy_r_multiwoz21_short_runs",
        "roberta-base-trippy_r_multiwoz21_long_runs",
    ]

    trippy_r_checkpoints_short = [
        "1775,3550,5325,7100,8875,10650,12425,14200,15975,17750,19525,21300,23075,24850,26625,28400,30175,31950,33725,35500"
    ]
    trippy_r_checkpoints_long = [
        "1775,3550,5325,7100,8875,10650,12425,14200,15975,17750,19525,21300,23075,24850,26625,28400,30175,31950,33725,35500,37275,39050,40825,42600,44375,46150,47925,49700,51475,53250,55025,56800,58575,60350,62125,63900,65675,67450,69225,71000,72775,74550,76325,78100,79875,81650,83425,85200,86975,88750"
    ]

    # --- Regular case ---
    regular_data_args_list = [
        "data=multiwoz21",
        "data=sgd",
        # Add more datasets here if needed
    ]

    regular_model_args_list = [
        "language_model=roberta-base",
        "language_model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "language_model=gpt2-medium",
        # Add more model configurations as needed
    ]

    regular_checkpoints = [
        "-1",  # Use -1 for all checkpoints, or list specific ones
        "1200",
        "2800",
    ]

    modes_to_process = ["regular", "trippy_r"]

    for mode in modes_to_process:
        match mode:
            case "regular":
                print("Running in REGULAR mode")
                for data in regular_data_args_list:
                    for model in regular_model_args_list:
                        for checkpoint in regular_checkpoints:
                            args = (
                                base_args
                                + launcher_args
                                + [data, model, f"language_model.checkpoint_no={checkpoint}"]
                                + embeddings_args
                                + local_estimates_args
                            )
                            print(f"Launching: {data}, {model}, checkpoint: {checkpoint}")
                            run_command(script, args, dry_run)
            case "trippy_r":
                print("Running in TRIPPY-R mode")
                for model in trippy_r_models:
                    if model == "roberta-base-trippy_r_multiwoz21_short_runs":
                        for checkpoints in trippy_r_checkpoints_short:
                            args = (
                                base_args
                                + launcher_args
                                + trippy_r_data_args
                                + [f"language_model={model}", f"language_model.checkpoint_no={checkpoints}"]
                                + embeddings_args
                                + local_estimates_args
                            )
                            print(f"Launching: {model} [Short Runs], checkpoints: {checkpoints}")
                            run_command(script, args, dry_run)
                    elif model == "roberta-base-trippy_r_multiwoz21_long_runs":
                        for checkpoints in trippy_r_checkpoints_long:
                            args = (
                                base_args
                                + launcher_args
                                + trippy_r_data_args
                                + [f"language_model={model}", f"language_model.checkpoint_no={checkpoints}"]
                                + embeddings_args
                                + local_estimates_args
                            )
                            print(f"Launching: {model} [Long Runs], checkpoints: {checkpoints}")
                            run_command(script, args, dry_run)
                    else:
                        print(f"Unknown TRIPPY-R model: {model}")
            case _:
                raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
