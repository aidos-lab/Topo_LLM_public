"""Compare kernel density estimates of two arrays."""

import os
import pathlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import kl_div
from sklearn.neighbors import KernelDensity


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

    path_1 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name_1,
        cfg.level_1,
        cfg.prefix_1,
        cfg.model_1,
        cfg.layer_1,
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
        cfg.data_name_2,
        cfg.level_2,
        cfg.prefix_2,
        cfg.model_2,
        cfg.layer_2,
        cfg.norm_2,
        cfg.array_dir_2,
        array_name_2,
    )

    arr_no_pad = np.load(path_1)
    arr_no_pad_finetuned = np.load(path_2)

    # provide bandwidth for the computation
    bandwidth = 0.2

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(arr_no_pad)
    kde_finetuned = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(arr_no_pad_finetuned)

    x = kde.score_samples(arr_no_pad)
    y = kde_finetuned.score_samples(arr_no_pad_finetuned)

    print(f"KL divergence: {sum(kl_div(x, y))}")
    print(sum(kl_div(x, y)))

    density_frame = pd.DataFrame({"kernel_density_finetuned": list(y), "kernel_density": list(x)})

    print(density_frame.corr())

    plt.ioff()
    scatter_plot = sns.scatterplot(x=list(x), y=list(y))
    scatter_fig = scatter_plot.get_figure()

    # use savefig function to save the plot and give
    # a desired name to the plot.

    file_name = ""
    if cfg.data_name_1 == cfg.data_name_2:
        file_name += str(cfg.data_name_1) + "_"
    if cfg.model_1 == cfg.model_2:
        file_name += str(cfg.model_1) + "_"
    names = set([name[:-1] for name in cfg.keys()])
    for name in names:
        if cfg[name + str(1)] != cfg[name + str(2)]:
            additional_string = str(cfg[name + str(1)]) + "_vs_" + str(cfg[name + str(2)] + "_")
            file_name += additional_string

    save_path = "../../data/analysis/density/" + str(cfg.embedding_level_1) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + file_name + str(len(arr_no_pad)) + "_samples"
    scatter_fig.savefig(  # type: ignore - savefig exists
        save_name + ".png",
    )
    density_frame.to_pickle(save_name)

    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
