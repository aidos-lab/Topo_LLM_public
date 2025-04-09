import os
import re
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # Added for nearest neighbor computation

# Create directories to save plots
save_dir = "persistence_plots"

# Determine subdirectory based on model_name
#model_name = "model=roberta-base_task=masked_lm_dr=defaults"
#model_name = "model=roberta-base-trippy_r_multiwoz21_seed-42_ckpt-1775_task=masked_lm_dr=defaults"
model_name = "model=roberta-base-trippy_r_multiwoz21_seed-42_ckpt-35500_task=masked_lm_dr=defaults"

if "ckpt" in model_name:
    # Look for the pattern "ckpt-" followed by one or more digits.
    match = re.search(r'ckpt-(\d+)', model_name)
    if match:
        checkpoint = "ckpt-" + match.group(1)
        base_dir = os.path.join(save_dir, checkpoint)
    else:
        base_dir = os.path.join(save_dir, "base")
else:
    base_dir = os.path.join(save_dir, "base")

# Create the directories if they don't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# === Load your data ===
path = "./data=trippy_r_dataloaders_processed_multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=7000_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/"+model_name+"/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/array_for_estimator.npy"
#path = "./data=setsumbt_dataloaders_processed_0_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=512_sampling=take_first/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/" + model_name + "/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=3000/desc=twonn_samples=500_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/array_for_estimator.npy"

X = np.load(path)
sample_size = 1000
X = X[np.random.choice(len(X), size=sample_size, replace=False)]

# === Compute persistent homology up to dimension 3 ===
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)
diagrams = VR.fit_transform([X])  # Input must be 3D: (n_samples, n_features) → shape (1, n_samples, n_features)


def plot_persistence_diagrams(diagrams):
    """Plot persistence diagrams for each present homology dimension (starting axes at 0) and save the figure."""
    if diagrams.shape[0] == 0:
        print("No persistence features found.")
        return

    unique_dims = sorted(np.unique(diagrams[:, 2]).astype(int))
    num_dims = len(unique_dims)
    plt.figure(figsize=(12, 4 * num_dims))

    for i, dim in enumerate(unique_dims):
        dgm = diagrams[diagrams[:, 2] == dim]
        if dgm.shape[0] == 0:
            continue
        birth = np.clip(dgm[:, 0], 0, None)
        death = np.clip(dgm[:, 1], 0, None)

        ax = plt.subplot(num_dims, 1, i + 1)
        ax.scatter(birth, death, alpha=0.6, edgecolor="k")

        max_val = max(birth.max(), death.max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"Persistence Diagram (H{dim})")
        ax.grid(True)
        ax.set_aspect("equal")

    plt.tight_layout()
    # Save the figure with sample_size included in the name, inside the designated subdirectory.
    filename = os.path.join(base_dir, f"persistence_diagrams_sample_size{sample_size}.png")
    plt.savefig(filename, dpi=300)
    # plt.show()


def plot_persistence_distributions(diagram):
    """Plot and save histograms of birth times, death times, and lifetimes for each homology dimension."""
    unique_dims = sorted(np.unique(diagram[:, 2]).astype(int))

    for dim in unique_dims:
        dgm = diagram[diagram[:, 2] == dim]
        births = np.clip(dgm[:, 0], 0, None)
        deaths = np.clip(dgm[:, 1], 0, None)
        lifetimes = deaths - births

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].hist(births, bins=30, alpha=0.7, edgecolor='black')
        axs[0].set_title(f'Birth Times Distribution (H{dim})')
        axs[0].set_xlabel('Birth Time')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(deaths, bins=30, alpha=0.7, edgecolor='black')
        axs[1].set_title(f'Death Times Distribution (H{dim})')
        axs[1].set_xlabel('Death Time')
        axs[1].set_ylabel('Frequency')

        axs[2].hist(lifetimes, bins=30, alpha=0.7, edgecolor='black')
        axs[2].set_title(f'Lifetime Distribution (H{dim})')
        axs[2].set_xlabel('Lifetime (Death - Birth)')
        axs[2].set_ylabel('Frequency')

        plt.tight_layout()
        # Save each distribution plot with sample_size and the dimension in the filename.
        filename = os.path.join(base_dir, f"persistence_distribution_H{dim}_sample_size{sample_size}.png")
        plt.savefig(filename, dpi=300)
        # plt.show()


def plot_nearest_neighbor_distances(X):
    """Compute and plot the distribution of the nearest neighbor distances in the dataset."""
    # Fit a nearest neighbor model; here we use 2 neighbors since the closest neighbor is the point itself.
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_model.fit(X)
    distances, _ = nn_model.kneighbors(X)
    # Take the second column, as the first column is the zero distance to itself.
    nn_distances = distances[:, 1]

    plt.figure(figsize=(8, 6))
    plt.hist(nn_distances, bins=30, alpha=0.75, edgecolor='black')
    plt.xlabel("Nearest Neighbor Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Nearest Neighbor Distances")
    plt.grid(True)

    # Save the figure with sample_size included in the name.
    filename = os.path.join(base_dir, f"nearest_neighbor_distance_sample_size{sample_size}.png")
    plt.savefig(filename, dpi=300)
    # Optionally, display the plot interactively:
    # plt.show()


print(diagrams[0])

# --- Plot and save persistence diagrams ---
plot_persistence_diagrams(diagrams[0])

# --- Plot and save distributions of birth times, death times, and lifetimes ---
plot_persistence_distributions(diagrams[0])

# --- Plot and save the nearest neighbor distance distribution ---
plot_nearest_neighbor_distances(X)
