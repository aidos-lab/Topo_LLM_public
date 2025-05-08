import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the dataset names and masked levels
dataset_names = [
    "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
    "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
    "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
]
masked_levels = ["regular_lvl", "masked_token_lvl"]

# List of second models and corresponding labels
second_models_and_labels = [
    (
        "model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-2800_task=masked_lm_dr=defaults",
        "RoBERTa fine-tuned on Multiwoz",
    ),
    (
        "model=roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-2800_task=masked_lm_dr=defaults",
        "RoBERTa fine-tuned on Reddit",
    ),
    (
        "model=roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-2800_task=masked_lm_dr=defaults",
        "RoBERTa fine-tuned on Wikitext",
    ),
]

# second_models_and_labels = [
#     ("model=gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Multiwoz"),
#     ("model=gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Reddit"),
#     ("model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults", "GPT-2 fine-tuned on Wikitext")
# ]

# List to store all relative paths
relative_paths = []

# Iterate over dataset_names, masked_levels, and second_models_and_labels
for dataset_name in dataset_names:
    for masked_level in masked_levels:
        for second_model, second_label in second_models_and_labels:
            # Define the base directories dynamically using the dataset name, masked level, and second model
            base_directory_1 = os.path.join(
                dataset_name,
                f"split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode={masked_level}=token/add-prefix-space=True_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
            )

            base_directory_2 = os.path.join(
                dataset_name,
                f"split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode={masked_level}=token/add-prefix-space=True_max-len=512/{second_model}/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
            )

            # Specify the file paths
            file_path_1 = os.path.join(base_directory_1, "local_estimates_pointwise_array.npy")
            file_path_2 = os.path.join(base_directory_2, "local_estimates_pointwise_array.npy")

            try:
                # Load the .npy files
                data_array_1 = np.load(file_path_1)
                data_array_2 = np.load(file_path_2)

                # Combine the data into a single dataset with labels
                data_combined = [data_array_1, data_array_2]
                # labels = ["GPT-2", second_label]
                labels = ["RoBERTa", second_label]

                # Create a violin plot with adjusted font size
                plt.figure(figsize=(8.5, 2.5))  # Adjust figure size for ICML one-column format
                sns.violinplot(data=data_combined, scale="width", inner="quartile", split=True)

                fontsize = 18

                # Add title and labels with smaller font size
                plt.ylabel("TwoNN", fontsize=fontsize)
                plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                import matplotlib.patches as mpatches

                # Retrieve colors for each violin from the plot
                colors = [violin.get_facecolor().mean(axis=0) for violin in plt.gca().collections[: len(data_combined)]]

                # Create custom legend handles with correct colors
                legend_handles = [
                    mpatches.Patch(
                        color=colors[i],
                        label=f"Mean={np.mean(data_combined[i]):.2f}, Median={np.median(data_combined[i]):.2f}, Std={np.std(data_combined[i]):.2f}",
                    )
                    for i in range(len(labels))
                ]

                # Add the legend with custom handles
                plt.legend(handles=legend_handles, loc="upper right", fontsize=12)

                # Reduce whitespace around the plot for compactness
                plt.tight_layout(pad=0.1)

                # Create directories for saving the plots
                save_dir = os.path.join(
                    masked_level,
                    second_model.replace("=", "-").replace("/", "_"),
                    dataset_name.replace("=", "-").replace("/", "_"),
                )
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "violin_plot.pdf")

                plt.savefig(save_path, format="pdf", bbox_inches="tight")  # Save for submission
                relative_paths.append(save_path)  # Add the relative path to the list

                plt.close()

            except FileNotFoundError as e:
                print(
                    f"File not found for Dataset: {dataset_name}, Masked Level: {masked_level}, Second Model: {second_model}: {e}"
                )
            except Exception as e:
                print(
                    f"An error occurred for Dataset: {dataset_name}, Masked Level: {masked_level}, Second Model: {second_model}: {e}"
                )


# Group the relative paths by the second_model component
grouped_paths = defaultdict(list)
for path in relative_paths:
    # Extract the second_model name from the path
    # Assumes the structure where second_model is part of the path
    components = path.split(os.sep)
    for comp in components:
        if comp.startswith("model-"):
            grouped_paths[comp].append(path)
            break

# Print the grouped relative paths
print("\nRelative paths of saved plots (grouped by second_model):")
for second_model, paths in grouped_paths.items():
    print(f"\nPaths for {second_model}:")
    for path in paths:
        print("data/twonn_violin_plots/" + path)
