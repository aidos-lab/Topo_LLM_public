########################################################

# This is a script to calculate and store lPCA dimension
# estimates for two given arrays for a comparison
# of embeddings of a base model and a corresponding
# fine-tuned variant. To obtain these arrays, the
# `data_prep.py` may be used.

# third party imports
import os
import pathlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skdim
from scipy.spatial import KDTree


def k_nearest_neighbor_distances(data, k):
    """
    Calculate the k nearest neighbor distances for a given dataset using KDTree.

    Parameters:
        data (numpy.ndarray): The dataset, where each row represents a data point.
        k (int): The number of nearest neighbors to consider.

    Returns:
        numpy.ndarray: Array containing the distances to the k nearest neighbors for each data point.
    """
    # Create KDTree from the data
    kdtree = KDTree(data)

    # Query the k nearest neighbors and distances
    distances, _ = kdtree.query(data, k=k + 1)  # +1 to exclude the point itself
    return distances[:, 1:]  # Exclude the first column which corresponds to the point itself


@hydra.main(
    config_path="../../configs/analysis",
    config_name="comparison",
    version_base="1.2",
)
def main(cfg):
    array_name_1 = (
        "embeddings_" + str(cfg.embedding_level_1) + "_" + str(cfg.samples_1) + "_samples_paddings_removed.npy"
    )
    array_name_2 = (
        "embeddings_" + str(cfg.embedding_level_2) + "_" + str(cfg.samples_2) + "_samples_paddings_removed.npy"
    )

    layer_1 = "layer-" + str(cfg.layer) + "_agg-mean"
    layer_2 = layer_1
    path_1 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name,
        cfg.level_1,
        cfg.prefix_1,
        cfg.model_1,
        layer_1,
        cfg.norm_1,
        cfg.array_dir_1,
        array_name_1,
    )

    path_2 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name,
        cfg.level_2,
        cfg.prefix_2,
        cfg.model_2,
        layer_2,
        cfg.norm_2,
        cfg.array_dir_2,
        array_name_2,
    )

    arr_no_pad = np.load(path_1)
    arr_no_pad_finetuned = np.load(path_2)

    np.random.seed(2)
    sample_size = 1500
    sample_size = min(sample_size, arr_no_pad.shape[0], arr_no_pad_finetuned.shape[0])
    # if sample_size>len(arr_no_pad):
    #     idx = np.random.choice(
    #         range(len(arr_no_pad)),
    #         replace=False,
    #         size=len(arr_no_pad),
    #     )
    # else:
    #     idx = np.random.choice(
    #         range(len(arr_no_pad)),
    #         replace=False,
    #         size=sample_size,
    #     )
    # if sample_size>len(arr_no_pad_finetuned):
    #     idx_finetuned = np.random.choice(
    #         range(len(arr_no_pad_finetuned)),
    #         replace=False,
    #         size=len(arr_no_pad_finetuned),
    #     )
    # else:
    #     idx_finetuned = np.random.choice(
    #         range(len(arr_no_pad_finetuned)),
    #         replace=False,
    #         size=sample_size,
    #     )
    #
    # arr_no_pad = arr_no_pad[idx]
    # arr_no_pad_finetuned = arr_no_pad_finetuned[idx_finetuned]

    arr_no_pad = arr_no_pad[:sample_size]
    arr_no_pad_finetuned = arr_no_pad_finetuned[:sample_size]

    # Number of nearest neighbors to consider
    k = 3

    # Calculate k nearest neighbor distances
    knn_distances = k_nearest_neighbor_distances(arr_no_pad, k)
    knn_distances = knn_distances[:, -1]

    knn_distances_finetuned = k_nearest_neighbor_distances(arr_no_pad_finetuned, k)
    knn_distances_finetuned = knn_distances_finetuned[:, -1]

    print(knn_distances)
    print(knn_distances_finetuned)

    neigh_frame = pd.DataFrame({"knn_dist_finetuned": list(knn_distances_finetuned), "knn_dist": list(knn_distances)})

    print(neigh_frame.corr())

    plt.ioff()
    scatter_plot = sns.scatterplot(x=list(knn_distances), y=list(knn_distances_finetuned))
    scatter_fig = scatter_plot.get_figure()

    # use savefig function to save the plot and give
    # a desired name to the plot.

    file_name = ""
    file_name += str(cfg.data_name) + "_"
    file_name += str(cfg.model_1) + "_"
    file_name += str(cfg.model_2) + "_"

    save_path = "../../data/analysis/codensity/" + str(cfg.embedding_level_1) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + file_name + "_" + str(cfg.layer) + "_" + str(k) + ".pkl"
    scatter_fig.savefig(save_name + ".png")
    neigh_frame.to_pickle(save_name)

    # plt.show()
    plt.close()
    return None


if __name__ == "__main__":
    main()  # type: ignore
