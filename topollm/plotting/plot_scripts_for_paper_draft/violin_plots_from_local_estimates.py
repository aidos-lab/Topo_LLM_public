import itertools
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def main() -> None:
    """Generate violin plots for local estimates from different models."""
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
    edh_mode_choices: list[str] = [
        "masked_token",
        "regular",
    ]

    # List of second models and corresponding labels
    checkpoint_no: int = 2800
    second_models_and_labels = [
        (
            f"model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
            "RoBERTa fine-tuned on MultiWOZ",
        ),
        (
            f"model=roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
            "RoBERTa fine-tuned on Wikitext",
        ),
        (
            f"model=roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{checkpoint_no}_task=masked_lm_dr=defaults",
            "RoBERTa fine-tuned on Reddit",
        ),
    ]

    # second_models_and_labels = [
    #     ("model=gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Multiwoz"),
    #     ("model=gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Reddit"),
    #     ("model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Wikitext")
    # ]

    data_subsampling_sampling_seed: int = 778
    add_prefix_space: bool = False

    # List to store all relative paths
    relative_paths: list[pathlib.Path] = []

    local_estimates_directory: pathlib.Path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "analysis",
        "local_estimates",
    )

    # Iterate over different choices
    for (
        dataset_name,
        edh_mode,
        (
            second_model,
            second_label,
        ),
    ) in itertools.product(
        dataset_name_choices,
        edh_mode_choices,
        second_models_and_labels,
    ):
        # Define the base directories dynamically using the dataset name, masked level, and second model
        base_directory_1 = pathlib.Path(
            local_estimates_directory,
            dataset_name,
            f"split=validation_samples=10000_sampling=random_sampling-seed={data_subsampling_sampling_seed}",
            f"edh-mode={edh_mode}_lvl=token",
            f"add-prefix-space={add_prefix_space}_max-len=512",
            "model=roberta-base_task=masked_lm_dr=defaults",
            "layer=-1_agg=mean/norm=None/",
            "sampling=random_seed=42_samples=150000",
            "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
        )

        base_directory_2 = pathlib.Path(
            local_estimates_directory,
            dataset_name,
            f"split=validation_samples=10000_sampling=random_sampling-seed={data_subsampling_sampling_seed}",
            f"edh-mode={edh_mode}_lvl=token",
            f"add-prefix-space={add_prefix_space}_max-len=512",
            f"{second_model}",
            "layer=-1_agg=mean/norm=None/",
            "sampling=random_seed=42_samples=150000",
            "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
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
            data_array_1 = np.load(
                file=file_path_1,
            )
            data_array_2 = np.load(
                file=file_path_2,
            )

            # Combine the data into a single dataset with labels
            data_combined: list[np.ndarray] = [
                data_array_1,
                data_array_2,
            ]

            # labels = [
            #     "GPT-2",
            #     second_label,
            # ]
            labels: list[str] = [
                "RoBERTa",
                second_label,
            ]

            # Create a violin plot with adjusted font size
            plt.figure(figsize=(8.5, 2.5))  # Adjust figure size for ICML one-column format
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

            import matplotlib.patches as mpatches

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
                    f"Std={np.std(data_combined[i]):.2f}",
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
                f"add-prefix-space={add_prefix_space}",
                f"edh-mode={edh_mode}",
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
        except Exception as e:
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
        print(f"\nPaths for {second_model}:")
        for path in paths:
            print("data/twonn_violin_plots/" + str(path))


if __name__ == "__main__":
    main()
