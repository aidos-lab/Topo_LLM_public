import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Base directory with placeholders for the sampling seed
base_dir = "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode=regular_lvl=token/add-prefix-space=True_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed={seed}_samples=150000/desc=twonn_samples=100000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128"

# List of sampling seeds to iterate over
sampling_seeds = [42, 43, 44, 45, 46]

# List of truncation sizes
truncation_sizes = range(10000, 100001, 10000)

# Initialize a dictionary to store data for each seed
data_per_seed = {}

# Loop over each seed and load the respective array
for seed in sampling_seeds:
    dir_with_seed = base_dir.format(seed=seed)
    array_path = os.path.join(dir_with_seed, "local_estimates_pointwise_array.npy")

    try:
        # Load the array
        data = np.load(array_path)
        data_per_seed[seed] = data
    except FileNotFoundError:
        print(f"File not found for seed {seed}: {array_path}")

# Filter out seeds with missing data
data_per_seed = {seed: data for seed, data in data_per_seed.items() if data is not None}

# Ensure there's data to plot
if not data_per_seed:
    print("No data found for any of the specified seeds.")
else:
    # Prepare data for the boxplot
    means_per_trunc_size = []

    for trunc_size in truncation_sizes:
        # Calculate the means for each seed (after truncation)
        means = [np.mean(data[:trunc_size]) for data in data_per_seed.values() if len(data) >= trunc_size]
        means_per_trunc_size.append(means)

    # Create the boxplot with the means
    plt.figure(figsize=(6, 4))  # Adjust size for a single-column layout
    plt.boxplot(means_per_trunc_size, positions=range(len(truncation_sizes)), showmeans=True)

    # Customize font sizes
    plt.xticks(range(len(truncation_sizes)), [str(size) for size in truncation_sizes], fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Sample Size", fontsize=12)
    plt.ylabel("Mean TwoNN Estimate", fontsize=12)

    # Set the y-axis range
    plt.ylim(13.7, 14.3)

    # Adjust line widths
    plt.gca().spines['top'].set_linewidth(0.8)
    plt.gca().spines['right'].set_linewidth(0.8)
    plt.gca().spines['left'].set_linewidth(0.8)
    plt.gca().spines['bottom'].set_linewidth(0.8)

    # Tweak layout for ICML standards
    plt.tight_layout()

    # Save the plot as a PDF
    split_name = "validation"
    dataset_name = "reddit"
    save_name = f"{dataset_name}_{split_name}_twonn_sample_sizes.pdf"

    with PdfPages(save_name) as pdf:
        pdf.savefig()  # Save the current figure

    plt.show()

    print(f"Plot saved as {save_name}")
