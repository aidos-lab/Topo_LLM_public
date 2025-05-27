"""Launch analyse_local_estimate_distribution_over_tokens.py with various argument combinations."""

import logging
import subprocess
from enum import StrEnum

import click
from tqdm import tqdm

from topollm.logging.create_and_configure_global_logger import create_and_configure_global_logger
from topollm.typing.enums import Verbosity

global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class Launcher(StrEnum):
    """Enum for launcher types."""

    BASIC = "basic"
    HPC_SUBMISSION = "hpc_submission"


class RunMode(StrEnum):
    """Enum for run modes."""

    REGULAR = "regular"
    DRY_RUN = "dry_run"


def run_command(
    script: str,
    args: list[str],
    run_mode: RunMode = RunMode.REGULAR,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run a command with the given script and arguments."""
    cmd: list[str] = [
        "uv",
        "run",
        "python",
        script,
        *args,
    ]

    logger.info(
        msg=f"Launching: {' '.join(cmd)}",  # noqa: G004 - low overhead
    )
    match run_mode:
        case RunMode.DRY_RUN:
            logger.info("Running in DRY_RUN mode, command will not be executed.")

        case RunMode.REGULAR:
            subprocess.run(
                args=cmd,
                check=True,
            )
        case _:
            logger.warning(msg="Unknown run mode, command will not be executed.")


@click.command()
@click.option(
    "--launcher",
    type=click.Choice(choices=list(Launcher)),
    default=Launcher.BASIC,
    show_default=True,
    help="Hydra launcher to use.",
)
@click.option(
    "--run-mode",
    type=click.Choice(choices=list(RunMode)),
    default=RunMode.REGULAR,
    show_default=True,
    help="Run mode: 'regular' executes commands, 'dry_run' only prints them.",
)
def main(
    launcher: Launcher,
    run_mode: RunMode,
) -> None:
    """Run analyse_local_estimate_distribution_over_tokens.py for various argument combinations."""
    # --- Global options ---
    logger: logging.Logger = global_logger
    verbosity: Verbosity = Verbosity.NORMAL
    script = "topollm/analysis/distribution_over_tokens/analyse_local_estimate_distribution_over_tokens.py"

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
            msg = f"Unknown launcher: {launcher}"
            raise ValueError(msg)

    base_args: list[str] = [
        "--multirun",
        "hydra/sweeper=basic",
        "tokenizer.add_prefix_space=False",
    ]

    embeddings_args: list[str] = [
        "embeddings.embedding_data_handler.mode=regular",
        "embeddings.embedding_extraction.layer_indices=[-1]",
        "embeddings_data_prep.sampling.num_samples=150000",
        "embeddings_data_prep.sampling.sampling_mode=random",
        "embeddings_data_prep.sampling.seed=42",
    ]

    local_estimates_args: list[str] = [
        "local_estimates=twonn",
        "local_estimates.pointwise.n_neighbors_mode=absolute_size",
        "local_estimates.filtering.deduplication_mode=array_deduplicator",
        "local_estimates.filtering.num_samples=60000",
        "local_estimates.pointwise.absolute_n_neighbors=128",
    ]

    # --- Base models case ---
    regular_data_args: list[str] = [
        "data=multiwoz21,sgd,one-year-of-tsla-on-reddit,wikitext-103-v1",
        "data.data_subsampling.split=train,validation,test",
        "data.data_subsampling.sampling_mode=random",
        "data.data_subsampling.number_of_samples=10000",
        "data.data_subsampling.sampling_seed=778",
        # Add more datasets here if needed
    ]

    base_models: list[str] = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2-medium",
    ]

    # --- Fine-tuning case ---
    roberta_base_fine_tuned_models: list[str] = [
        "roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
    ]

    gpt2_medium_fine_tuned_models: list[str] = [
        "gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "gpt2-medium-causal_lm-defaults_sgd-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
        "gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5",
    ]

    # --- Trippy-R case ---
    trippy_r_data_args: list[str] = [
        "data=trippy_r_dataloaders_processed",
        "data.data_subsampling.split=train,dev,test",
        "data.data_subsampling.sampling_mode=random",
        "data.data_subsampling.number_of_samples=7000",
        "data.data_subsampling.sampling_seed=778",
    ]

    trippy_r_models: list[str] = [
        "roberta-base-trippy_r_multiwoz21_short_runs",
        "roberta-base-trippy_r_multiwoz21_long_runs",
    ]

    trippy_r_checkpoints_short: str = (
        "1775,3550,5325,7100,8875,10650,12425,14200,15975,17750,19525,"
        "21300,23075,24850,26625,28400,30175,31950,33725,35500"
    )
    trippy_r_checkpoints_long: str = (
        "1775,3550,5325,7100,8875,10650,12425,14200,15975,17750,19525,"
        "21300,23075,24850,26625,28400,30175,31950,33725,35500,"
        "37275,39050,40825,42600,44375,46150,47925,49700,51475,"
        "53250,55025,56800,58575,60350,62125,63900,65675,67450,"
        "69225,71000,72775,74550,76325,78100,79875,81650,83425,85200,86975,88750"
    )

    # --- ERC models case ---
    erc_data_args: list[str] = [
        "data=ertod_emowoz",
        "data.dataset_seed=50",
        "data.data_subsampling.split=train,validation,test",
        "data.data_subsampling.sampling_mode=random",
        "data.data_subsampling.number_of_samples=7000",
        "data.data_subsampling.sampling_seed=778",
    ]

    # Notes:
    # - We should have computed the local estimates for the ERC models for all checkpoints
    #   from 0 to 50, but here, we only run the analysis for checkpoints 0 to 10.
    erc_model_args: list[str] = [
        "language_model=bert-base-uncased-ContextBERT-ERToD_emowoz_basic_setup",
        "language_model.seed=50,51,52,53,54",
        "language_model.num_train_epochs=50",
        "language_model.checkpoint_no=0,1,2,3,4,5,6,7,8,9,10",
    ]

    # --- Modes to process ---
    modes_to_process: list[str] = [
        "base_models",
        "regular_fine_tuned_models",
        "trippy_r",
        "erc_models",
    ]

    for mode in tqdm(
        iterable=modes_to_process,
        desc="Iterating over modes",
    ):
        match mode:
            case "base_models":
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for base models.",
                    )
                for model in base_models:
                    args: list[str] = (
                        base_args
                        + launcher_args
                        + regular_data_args
                        + [
                            f"language_model={model}",
                            "++language_model.checkpoint_no=-1",
                        ]
                        + embeddings_args
                        + local_estimates_args
                    )
                    run_command(
                        script=script,
                        args=args,
                        run_mode=run_mode,
                    )

            case "regular_fine_tuned_models":
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for regular fine-tuned models.",
                    )

                # Notes:
                # - We keep the submissions for different underlying base models separate,
                #   so that we can set the model checkpoint more flexibly for different settings

                # >> roberta-base fine-tuned models
                args: list[str] = (
                    base_args
                    + launcher_args
                    + regular_data_args
                    + [
                        f"language_model={','.join(roberta_base_fine_tuned_models)}",
                        "language_model.checkpoint_no=1200,2800",
                    ]
                    + embeddings_args
                    + local_estimates_args
                )
                run_command(
                    script=script,
                    args=args,
                    run_mode=run_mode,
                )

                # >> gpt2-medium fine-tuned models
                args: list[str] = (
                    base_args
                    + launcher_args
                    + regular_data_args
                    + [
                        f"language_model={','.join(gpt2_medium_fine_tuned_models)}",
                        "language_model.checkpoint_no=1200,2800",
                    ]
                    + embeddings_args
                    + local_estimates_args
                )
                run_command(
                    script=script,
                    args=args,
                    run_mode=run_mode,
                )
            case "trippy_r":
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for TripPy-R models.",
                    )

                # >> Base model for the TripPy-R analysis
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for TripPy-R base model.",
                    )

                args: list[str] = (
                    base_args
                    + launcher_args
                    + trippy_r_data_args
                    + [
                        "language_model=robert-base",
                        "++language_model.checkpoint_no=-1",
                    ]
                    + embeddings_args
                    + local_estimates_args
                )
                run_command(
                    script=script,
                    args=args,
                    run_mode=run_mode,
                )

                # >> TripPy-R checkpoints
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for TripPy-R checkpoints.",
                    )
                for model in trippy_r_models:
                    match model:
                        case "roberta-base-trippy_r_multiwoz21_short_runs":
                            checkpoints = trippy_r_checkpoints_short
                        case "roberta-base-trippy_r_multiwoz21_long_runs":
                            checkpoints = trippy_r_checkpoints_long
                        case _:
                            msg: str = f"Unknown TRIPPY-R model: {model}"
                            raise ValueError(msg)

                    args: list[str] = (
                        base_args
                        + launcher_args
                        + trippy_r_data_args
                        + [
                            f"language_model={model}",
                            "language_model.seed=40,41,42,43,44",
                            f"language_model.checkpoint_no={checkpoints}",
                        ]
                        + embeddings_args
                        + local_estimates_args
                    )

                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg=f"Launching for: {model=}; {checkpoints=}",  # noqa: G004 - low overhead
                        )

                    run_command(
                        script=script,
                        args=args,
                        run_mode=run_mode,
                    )

            case "erc_models":
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for ERC models.",
                    )

                # >> Base model for the ERC analysis
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for ERC base model.",
                    )

                args: list[str] = (
                    base_args
                    + launcher_args
                    + erc_data_args
                    + [
                        "language_model=bert-base-uncased",
                        "++language_model.checkpoint_no=-1",
                    ]
                    + embeddings_args
                    + local_estimates_args
                )
                run_command(
                    script=script,
                    args=args,
                    run_mode=run_mode,
                )

                # >> ERC checkpoints
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Running for ERC checkpoints.",
                    )

                args: list[str] = (
                    base_args + launcher_args + erc_data_args + erc_model_args + embeddings_args + local_estimates_args
                )
                run_command(
                    script=script,
                    args=args,
                    run_mode=run_mode,
                )
            case _:
                msg: str = f"Unknown {mode=}"
                raise ValueError(msg)


if __name__ == "__main__":
    main()
