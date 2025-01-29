import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# List of sampling seeds to iterate over
sampling_seeds = [778, 779, 780, 781, 782]
sample_sizes = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

split_names = ["train", "test", "validation"]
dataset_name = "multiwoz"

# Base directory template with placeholders for split, sample size, and seed
base_dir_template = (
    "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/"
    "split={split}_samples={sample_size}_sampling=random_sampling-seed={seed}/edh-mode=regular_lvl=token/add-prefix-space=True_max-len=512/"
    "model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/"
    "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128"
)

# Loop over each split name
for split_name in split_names:
    # Initialize a dictionary to store data for each sample size and seed
    data_per_sample_size = {}

    # Loop over each sample size and seed
    for sample_size in sample_sizes:
        for seed in sampling_seeds:
            dir_with_sample_size_and_seed = base_dir_template.format(
                split=split_name, sample_size=sample_size, seed=seed
            )
            array_path = os.path.join(dir_with_sample_size_and_seed, "local_estimates_pointwise_array.npy")

            try:
                # Load the array
                data = np.load(array_path)
                if sample_size not in data_per_sample_size:
                    data_per_sample_size[sample_size] = {}
                data_per_sample_size[sample_size][seed] = data
            except FileNotFoundError:
                print(f"File not found for split {split_name}, sample size {sample_size}, seed {seed}: {array_path}")

    # Filter out sample sizes with missing data
    data_per_sample_size = {size: seeds for size, seeds in data_per_sample_size.items() if seeds}

    # Ensure there's data to plot
    if not data_per_sample_size:
        print(f"No data found for split {split_name} for any of the specified sample sizes.")
        continue

    # Prepare data for the boxplot
    means_per_sample_size = []

    for sample_size, seeds_data in data_per_sample_size.items():
        # Calculate the means for each seed
        means = [np.mean(data) for data in seeds_data.values()]
        means_per_sample_size.append(means)

    # Create the boxplot with the means
    plt.figure(figsize=(6, 4))  # Adjust size for a single-column layout
    plt.boxplot(means_per_sample_size, positions=range(len(sample_sizes)), showmeans=True)

    # Customize font sizes
    plt.xticks(range(len(sample_sizes)), [str(size) for size in sample_sizes], fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Sequence Sample Size", fontsize=12)
    plt.ylabel("Mean TwoNN Estimate", fontsize=12)
    plt.ylim(8, 10)

    # Adjust line widths
    plt.gca().spines["top"].set_linewidth(0.8)
    plt.gca().spines["right"].set_linewidth(0.8)
    plt.gca().spines["left"].set_linewidth(0.8)
    plt.gca().spines["bottom"].set_linewidth(0.8)

    # Tweak layout for ICML standards
    plt.tight_layout()

    # Save the plot as a PDF
    save_name = f"{dataset_name}_{split_name}_twonn_sample_sizes.pdf"

    with PdfPages(save_name) as pdf:
        pdf.savefig()  # Save the current figure

    plt.show()

    print(f"Plot saved as {save_name}")
