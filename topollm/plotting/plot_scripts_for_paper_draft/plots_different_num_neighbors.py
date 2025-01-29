import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dataset_name = "wiki"
# Base directory template with placeholders for the sampling seed, n-neighbors, and split
base_dir_template = (
    "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/"
    "split={split}_samples=10000_sampling=random_sampling-seed=777/"
    "edh-mode=regular_lvl=token/add-prefix-space=True_max-len=512/"
    "model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/"
    "sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_"
    "dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors={n_neighbors}"
)

# List of n-neighbors values
n_neighbors_values = [16, 32, 64, 128, 256, 512, 1024]

# Adjustable split parameter
split = "validation"  # You can change this to "train", "test", etc.

# Initialize a list to store data for plotting
all_data = []

# Iterate over n-neighbors values
for n_neighbors in n_neighbors_values:
    # Update the base directory with the current n-neighbors value and split
    base_dir = base_dir_template.format(n_neighbors=n_neighbors, split=split)
    file_path = os.path.join(base_dir, "local_estimates_pointwise_array.npy")

    try:
        # Load the data
        data = np.load(file_path)
        all_data.append(data)
    except FileNotFoundError:
        print(f"File not found for n-neighbors={n_neighbors}: {file_path}")
        all_data.append([])  # Append empty data if file not found

# Plotting the boxplots
plt.figure(figsize=(3.5, 2.5))  # Adjust figure size for one-column layout
plt.boxplot(all_data, labels=n_neighbors_values, showfliers=True)
plt.xlabel("Number of Neighbors", fontsize=9)
plt.ylabel("TwoNN Estimates", fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PDF, including the split in the file name
save_name = f"{dataset_name}_{split}_twonn_boxplot.pdf"

with PdfPages(save_name) as pdf:
    pdf.savefig()  # Save the current figure

plt.show()

print(f"Plot saved as {save_name}")
