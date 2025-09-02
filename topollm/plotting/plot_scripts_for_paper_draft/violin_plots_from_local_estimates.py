"""Generate violin plots for local estimates from different models."""

import itertools
import logging
import pathlib
from collections import defaultdict
from enum import StrEnum, auto, unique

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@unique
class BaseModelMode(StrEnum):
    """The different modes for base models."""

    ROBERTA_BASE = auto()

    GPT2_MEDIUM = auto()

    LLAMA_32_1B = auto()
    LLAMA_32_3B = auto()
    LLAMA_31_8B = auto()

    PHI_35_MINI_INSTRUCT = auto()
    PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS = auto()


@unique
class SaveFormat(StrEnum):
    """The different formats for saving plots."""

    PDF = "pdf"
    PNG = "png"


@click.command()
@click.option(
    "--model-group",
    type=click.Choice(
        choices=[
            "all",
            "phi",
            "phi_luster",
            "llama",
        ],
    ),
    default="all",
    help="Select base model group.",
)
def main(
    model_group: str,
) -> None:
    """Generate violin plots for local estimates from different models."""
    logger: logging.Logger = default_logger
    verbosity: Verbosity = Verbosity.NORMAL

    ddof = 1  # Delta degrees of freedom for standard deviation calculation
    save_arrays_in_output_dir: bool = True

    # Define the dataset names and masked levels
    dataset_name_choices: list[str] = [
        "data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        "data=wikitext-103-v1_strip-True_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
        # LUSTER data
        "data=luster_column=source_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
    ]
    split_choices: list[str] = [
        "train",
        "validation",
        "test",
    ]
    samples_choices: list[int] = [
        7000,
        10000,
    ]
    sampling_mode_choices: list[str] = [
        "random",
        "take_first",
    ]
    edh_mode_choices: list[str] = [
        "masked_token",
        "regular",
    ]

    # List to store all relative paths
    relative_paths: list[pathlib.Path] = []

    # Prepare base_model_modes based on command-line option
    match model_group:
        case "all":
            base_model_modes: list[BaseModelMode] = list(BaseModelMode)
        case "phi":
            base_model_modes: list[BaseModelMode] = [
                BaseModelMode.PHI_35_MINI_INSTRUCT,
            ]
        case "phi_luster":
            base_model_modes: list[BaseModelMode] = [
                BaseModelMode.PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS,
            ]
        case "llama":
            base_model_modes: list[BaseModelMode] = [
                BaseModelMode.LLAMA_32_1B,
                BaseModelMode.LLAMA_32_3B,
                BaseModelMode.LLAMA_31_8B,
            ]
        case _:
            msg: str = f"Unsupported model group: {model_group=}"
            raise ValueError(
                msg,
            )

    for base_model_mode in base_model_modes:
        # List of second models and corresponding labels

        match base_model_mode:
            case BaseModelMode.ROBERTA_BASE:
                checkpoint_no: int = 2800
                first_label: str = "RoBERTa"
                first_model_path: str = "model=roberta-base_task=masked_lm_dr=defaults"
                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        f"model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        f"RoBERTa fine-tuned on MultiWOZ gs={checkpoint_no}",
                    ),
                    (
                        f"model=roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        f"RoBERTa fine-tuned on Reddit gs={checkpoint_no}",
                    ),
                    (
                        f"model=roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        f"RoBERTa fine-tuned on Wikitext gs={checkpoint_no}",
                    ),
                ]
            case BaseModelMode.GPT2_MEDIUM:
                checkpoint_no: int = 1200
                first_label: str = "GPT-2"
                first_model_path: str = "model=gpt2-medium_task=masked_lm_dr=defaults"
                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        f"model=gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        f"GPT-2 fine-tuned on MultiWOZ gs={checkpoint_no}",
                    ),
                    (
                        f"model=gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        f"GPT-2 fine-tuned on Reddit gs={checkpoint_no}",
                    ),
                    (
                        f"model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        f"GPT-2 fine-tuned on Wikitext gs={checkpoint_no}",
                    ),
                ]
            case BaseModelMode.PHI_35_MINI_INSTRUCT:
                first_label: str = "Phi-3.5-mini-instruct"
                first_model_path: str = "model=Phi-3.5-mini-instruct_task=masked_lm_dr=defaults"
                checkpoint_no_options: list[int] = [
                    800,
                    1200,
                    2800,
                ]
                second_models_and_labels: list[tuple[str, str]] = []

                for checkpoint_no in checkpoint_no_options:
                    second_models_and_labels.extend(
                        [
                            (
                                f"model=Phi-3.5-mini-instruct_multiwoz21_train-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
                                f"_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                                f"Phi-3.5-mini-instruct fine-tuned on MultiWOZ gs={checkpoint_no}",
                            ),
                            (
                                f"model=Phi-3.5-mini-instruct_one-year-of-tsla-on-reddit_train-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
                                f"_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                                f"Phi-3.5-mini-instruct fine-tuned on Reddit gs={checkpoint_no}",
                            ),
                        ],
                    )
            case BaseModelMode.PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS:
                first_label: str = "Phi-3.5-mini-instruct"
                first_model_path: str = "model=Phi-3.5-mini-instruct_task=causal_lm_dr=defaults"

                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        "model=luster-base_task=causal_lm_dr=defaults",
                        "LUSTER base model",
                    ),
                    (
                        "model=luster-base-emotion_task=causal_lm_dr=defaults",
                        "LUSTER base emotion model",
                    ),
                    (
                        "model=luster-chitchat_task=causal_lm_dr=defaults",
                        "LUSTER chitchat model",
                    ),
                    (
                        "model=luster-full_task=causal_lm_dr=defaults",
                        "LUSTER full model",
                    ),
                    (
                        "model=luster-rl-sent_task=causal_lm_dr=defaults",
                        "LUSTER RL sentiment model",
                    ),
                    (
                        "model=luster-rl-succ_task=causal_lm_dr=defaults",
                        "LUSTER RL success model",
                    ),
                ]
            case BaseModelMode.LLAMA_32_1B | BaseModelMode.LLAMA_32_3B | BaseModelMode.LLAMA_31_8B:
                match base_model_mode:
                    case BaseModelMode.LLAMA_32_1B:
                        base_model_str_part = "Llama-3.2-1B"
                    case BaseModelMode.LLAMA_32_3B:
                        base_model_str_part = "Llama-3.2-3B"
                    case BaseModelMode.LLAMA_31_8B:
                        base_model_str_part = "Llama-3.1-8B"
                    case _:
                        msg: str = f"Unsupported base model mode for Llama models: {base_model_mode=}"
                        raise ValueError(
                            msg,
                        )

                first_label: str = base_model_str_part
                first_model_path: str = f"model={base_model_str_part}_task=causal_lm_dr=defaults"

                checkpoint_no_options: list[int] = [
                    800,
                ]
                second_models_and_labels: list[tuple[str, str]] = []

                for checkpoint_no in checkpoint_no_options:
                    second_models_and_labels.extend(
                        [
                            (
                                f"model={base_model_str_part}-causal_lm-defaults_multiwoz21-r-T-dn-ner_tags_tr-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_q_proj_k_proj_v_proj-0.01-T_5e-05-linear-0.01-f-None-5"
                                f"_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                                f"{base_model_str_part} fine-tuned on MultiWOZ gs={checkpoint_no}",
                            ),
                            (
                                f"model={base_model_str_part}-causal_lm-defaults_one-year-of-tsla-on-reddit-r-T-pr-T-0-0.8-0.1-0.1-ner_tags_tr-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_q_proj_k_proj_v_proj-0.01-T_5e-05-linear-0.01-f-None-5"
                                f"_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                                f"{base_model_str_part} fine-tuned on Reddit gs={checkpoint_no}",
                            ),
                        ],
                    )
            case _:
                msg: str = f"Unsupported base model mode: {base_model_mode=}"
                raise ValueError(
                    msg,
                )

        data_subsampling_sampling_seed: int = 778
        add_prefix_space: bool = False

        local_estimates_directory: pathlib.Path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
            "analysis",
            "local_estimates",
        )

        # Iterate over different choices

        # Prepare the iterable for combinations
        product_iter: itertools.product = itertools.product(
            dataset_name_choices,
            split_choices,
            samples_choices,
            sampling_mode_choices,
            edh_mode_choices,
            second_models_and_labels,
        )

        total_iterations: int = (
            len(dataset_name_choices)
            * len(split_choices)
            * len(samples_choices)
            * len(sampling_mode_choices)
            * len(edh_mode_choices)
            * len(second_models_and_labels)
        )

        for (
            dataset_name,
            split,
            samples,
            sampling_mode,
            edh_mode,
            (
                second_model,
                second_label,
            ),
        ) in tqdm(
            iterable=product_iter,
            total=total_iterations,
            desc="Generating violin plots",
        ):
            match sampling_mode:
                case "random":
                    split_and_subsample_path: str = (
                        f"split={split}"
                        f"_samples={samples}"
                        f"_sampling={sampling_mode}"
                        f"_sampling-seed={data_subsampling_sampling_seed}"
                    )
                case "take_first":
                    split_and_subsample_path: str = f"split={split}_samples={samples}_sampling={sampling_mode}"
                case _:
                    msg: str = f"Unsupported sampling mode: {sampling_mode=}"
                    raise ValueError(
                        msg,
                    )

            # Define the base directories dynamically using the dataset name, masked level, and second model
            base_directory_1 = pathlib.Path(
                local_estimates_directory,
                dataset_name,
                split_and_subsample_path,
                f"edh-mode={edh_mode}_lvl=token",
                f"add-prefix-space={add_prefix_space}_max-len=512",
                first_model_path,
                "layer=-1_agg=mean",
                "norm=None",
                "sampling=random_seed=42_samples=150000",
                "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing",
                "n-neighbors-mode=absolute_size_n-neighbors=128",
            )

            base_directory_2 = pathlib.Path(
                local_estimates_directory,
                dataset_name,
                split_and_subsample_path,
                f"edh-mode={edh_mode}_lvl=token",
                f"add-prefix-space={add_prefix_space}_max-len=512",
                f"{second_model}",
                "layer=-1_agg=mean",
                "norm=None",
                "sampling=random_seed=42_samples=150000",
                "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing",
                "n-neighbors-mode=absolute_size_n-neighbors=128",
            )

            # Specify the file paths
            file_path_1: pathlib.Path = pathlib.Path(
                base_directory_1,
                "local_estimates_pointwise_array.npy",
            )
            file_path_2: pathlib.Path = pathlib.Path(
                base_directory_2,
                "local_estimates_pointwise_array.npy",
            )
            print(  # noqa: T201 - we want this script to print
                f"file_path_1:\n{file_path_1}",
            )
            print(  # noqa: T201 - we want this script to print
                f"file_path_2:\n{file_path_2}",
            )

            try:
                # Load the .npy files
                data_array_1: np.ndarray = np.load(
                    file=file_path_1,
                )
                data_array_2: np.ndarray = np.load(
                    file=file_path_2,
                )

                # Combine the data into a single dataset with labels
                data_combined: list[np.ndarray] = [
                    data_array_1,
                    data_array_2,
                ]

                labels: list[str] = [
                    first_label,
                    second_label,
                ]

                # Create a violin plot with adjusted font size
                plt.figure(
                    figsize=(7.0, 2.5),
                )  # Adjust figure size for paper format

                sns.violinplot(
                    data=data_combined,
                    density_norm="width",
                    inner="quartile",
                    split=True,
                )

                fontsize = 18

                # Add title and labels with smaller font size
                plt.ylabel(
                    ylabel="TwoNN",
                    fontsize=fontsize,
                )
                plt.xticks(
                    ticks=range(len(labels)),
                    labels=labels,
                    fontsize=fontsize,
                )
                plt.yticks(fontsize=fontsize)

                # Retrieve colors for each violin from the plot
                colors = [
                    violin.get_facecolor().mean(axis=0)  # type: ignore - problem with the color type
                    for violin in plt.gca().collections[: len(data_combined)]
                ]

                # Create custom legend handles with correct colors
                legend_handles = [
                    mpatches.Patch(
                        color=colors[i],
                        label=f"Mean={np.mean(data_combined[i]):.2f}; "
                        f"Median={np.median(data_combined[i]):.2f}; "
                        f"Std={np.std(a=data_combined[i], ddof=ddof):.2f}",
                    )
                    for i in range(len(labels))
                ]

                # Add the legend with custom handles
                plt.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=12,
                )

                # Reduce whitespace around the plot for compactness
                plt.tight_layout(pad=0.1)

                # Create directories for saving the plots
                save_dir = pathlib.Path(
                    TOPO_LLM_REPOSITORY_BASE_PATH,
                    "data",
                    "saved_plots",
                    "violin_plots",
                    f"{ddof=}",
                    f"add-prefix-space={add_prefix_space}",
                    f"edh-mode={edh_mode}",
                    split_and_subsample_path,
                    second_model.replace("=", "-").replace("/", "_"),
                    dataset_name.replace("=", "-").replace("/", "_"),
                )
                save_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for save_format in SaveFormat:
                    save_path = pathlib.Path(
                        save_dir,
                        f"violin_plot.{save_format.value}",
                    )
                    relative_paths.append(save_path)  # Add the relative path to the list

                    if verbosity >= Verbosity.NORMAL:
                        print(  # noqa: T201 - we want this script to print
                            f"Saving plot to: {save_path=}",
                        )

                    plt.savefig(
                        save_path,
                        format=save_format.value,
                        bbox_inches="tight",
                    )

                # Note: Only close the plot after saving in all formats.
                plt.close()

                if save_arrays_in_output_dir:
                    data_array_1_output_path = pathlib.Path(
                        save_dir,
                        "local_estimates_pointwise_array_1.npy",
                    )
                    data_array_2_output_path = pathlib.Path(
                        save_dir,
                        "local_estimates_pointwise_array_2.npy",
                    )
                    print(  # noqa: T201 - we want this script to print
                        f"Saving data arrays to:\n{data_array_1_output_path=}\n{data_array_2_output_path=}",
                    )

                    np.save(
                        file=data_array_1_output_path,
                        arr=data_array_1,
                    )
                    np.save(
                        file=data_array_2_output_path,
                        arr=data_array_2,
                    )

            except FileNotFoundError as e:
                print(  # noqa: T201 - we want this script to print
                    f"File not found for Dataset: {dataset_name}; "
                    f"Masked Level: {edh_mode}; "
                    f"Second Model: {second_model}: {e}",
                )
            except Exception as e:  # noqa: BLE001 - we want the script to continue in case of other exceptions
                print(  # noqa: T201 - we want this script to print
                    f"An error occurred for Dataset: {dataset_name}; "
                    f"Masked Level: {edh_mode}; "
                    f"Second Model: {second_model}: {e}",
                )

    # Group the relative paths by the second_model component
    grouped_paths = defaultdict(list)

    for path in relative_paths:
        # Extract the second_model name from the path
        # Assumes the structure where second_model is part of the path
        components = path.parts
        for comp in components:
            if comp.startswith("model-"):
                grouped_paths[comp].append(path)
                break

    # Print the grouped relative paths
    print(  # noqa: T201 - we want this script to print
        "\nRelative paths of saved plots (grouped by second_model):",
    )
    for (
        second_model,
        paths,
    ) in grouped_paths.items():
        print(  # noqa: T201 - we want this script to print
            f"\nPaths for {second_model}:",
        )
        for path in paths:
            print(  # noqa: T201 - we want this script to print
                "data/twonn_violin_plots/" + str(path),
            )


if __name__ == "__main__":
    main()
