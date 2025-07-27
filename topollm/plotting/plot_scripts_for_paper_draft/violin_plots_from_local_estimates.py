"""Generate violin plots for local estimates from different models."""

import itertools
import pathlib
from collections import defaultdict
from enum import StrEnum, auto, unique

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


@unique
class BaseModelMode(StrEnum):
    """The different modes for base models."""

    ROBERTA_BASE = auto()

    GPT2_MEDIUM = auto()
    PHI_35_MINI_INSTRUCT = auto()


def main() -> None:
    """Generate violin plots for local estimates from different models."""
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
    ]
    split_choices: list[str] = [
        "train",
        "validation",
        "test",
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

    for base_model_mode in BaseModelMode:
        # List of second models and corresponding labels

        match base_model_mode:
            case BaseModelMode.ROBERTA_BASE:
                checkpoint_no: int = 2800
                first_label: str = "RoBERTa"
                first_model_path: str = "model=roberta-base_task=masked_lm_dr=defaults"
                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        f"model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        "RoBERTa fine-tuned on MultiWOZ",
                    ),
                    (
                        f"model=roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        "RoBERTa fine-tuned on Reddit",
                    ),
                    (
                        f"model=roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
                        "RoBERTa fine-tuned on Wikitext",
                    ),
                ]
            case BaseModelMode.GPT2_MEDIUM:
                checkpoint_no: int = 1200
                first_label: str = "GPT-2"
                first_model_path: str = "model=gpt2-medium_task=masked_lm_dr=defaults"
                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        f"model=gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        "GPT-2 fine-tuned on MultiWOZ",
                    ),
                    (
                        f"model=gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        "GPT-2 fine-tuned on Reddit",
                    ),
                    (
                        f"model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=causal_lm_dr=defaults",
                        "GPT-2 fine-tuned on Wikitext",
                    ),
                ]
            case BaseModelMode.PHI_35_MINI_INSTRUCT:
                checkpoint_no: int = (
                    -1
                )  # Placeholder, which we will replace once we have fine-tuned version of the Phi-models
                first_label: str = "Phi-3.5-mini-instruct"
                first_model_path: str = "model=Phi-3.5-mini-instruct_task=masked_lm_dr=defaults"
                # TODO: To create first versions of these plots, we will use the Phi-3.5-mini-instruct model again as the second model.
                # TODO: Replace this with the fine-tuned versions once they are available.
                second_models_and_labels: list[tuple[str, str]] = [
                    (
                        "model=Phi-3.5-mini-instruct_task=masked_lm_dr=defaults",
                        "Phi-3.5-mini-instruct",
                    ),
                ]
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
        product_iter = itertools.product(
            dataset_name_choices,
            split_choices,
            sampling_mode_choices,
            edh_mode_choices,
            second_models_and_labels,
        )

        total_iterations: int = (
            len(dataset_name_choices)
            * len(split_choices)
            * len(sampling_mode_choices)
            * len(edh_mode_choices)
            * len(second_models_and_labels)
        )

        for (
            dataset_name,
            split,
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
                        f"_samples=10000"
                        f"_sampling={sampling_mode}"
                        f"_sampling-seed={data_subsampling_sampling_seed}"
                    )
                case "take_first":
                    split_and_subsample_path: str = f"split={split}_samples=10000_sampling={sampling_mode}"
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
                save_path = pathlib.Path(
                    save_dir,
                    "violin_plot.pdf",
                )

                plt.savefig(
                    save_path,
                    format="pdf",
                    bbox_inches="tight",
                )  # Save for submission
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

                relative_paths.append(save_path)  # Add the relative path to the list

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
