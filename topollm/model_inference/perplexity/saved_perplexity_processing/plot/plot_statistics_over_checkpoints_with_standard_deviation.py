import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# TODO: Test this script in our current setup
# TODO: Write logic for applying this script on all the data that we have created


def create_color_map(unique_models):
    """
    Creates a color map for the given models.

    Args:
        unique_models (list): List of unique model identifiers.

    Returns:
        dict: A dictionary mapping each model to a specific color.
    """
    print("Creating color map for models...")
    colors = cm.get_cmap("tab10", len(unique_models))
    return {model: colors(i) for i, model in enumerate(unique_models)}


def filter_base_model_data(model_data):
    """
    Filters the base model data from the given model data.

    Args:
        model_data (DataFrame): Data for the specific model.

    Returns:
        DataFrame: Filtered base model data.
    """
    return model_data[
        model_data["checkpoint"].isna() & (model_data["model_without_checkpoint"] == "roberta-base_task-masked_lm")
    ]


def align_dataframes(mean_df, std_df):
    """
    Aligns mean and standard deviation DataFrames by merging them on 'model_without_checkpoint' and 'checkpoint'.

    Args:
        mean_df (DataFrame): DataFrame containing mean values.
        std_df (DataFrame): DataFrame containing standard deviation values.

    Returns:
        DataFrame: Merged DataFrame with aligned mean and standard deviation values.
    """
    print("Aligning mean and standard deviation DataFrames...")
    merged_df = pd.merge(mean_df, std_df, on=["model_without_checkpoint", "checkpoint"], suffixes=("_mean", "_std"))
    merged_df_clean = merged_df.dropna(subset=["token_perplexity_mean", "token_perplexity_std"])
    print("Alignment complete. Returning cleaned DataFrame.")
    return merged_df_clean


def plot_metrics(ax, model_data, color, model, base_model_data):
    """
    Plots perplexity and log perplexity metrics on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        model_data (DataFrame): Data for the specific model.
        color (str): Color to use for plotting.
        model (str): Model identifier.
        base_model_data (DataFrame): Data for the base model.
    """
    print(f"Plotting metrics for model: {model}")
    checkpoints = model_data["checkpoint"].values
    mean_perplexity = model_data["token_perplexity_mean"].values
    std_perplexity = model_data["token_perplexity_std"].values
    mean_perplexity = np.where(mean_perplexity > 0, mean_perplexity, np.nan)  # Ensure positive values for log
    log_perplexity = np.log(mean_perplexity)

    if len(checkpoints) > 0:
        print(f"Checkpoints available for model {model}: {len(checkpoints)}")
        ax.plot(checkpoints, mean_perplexity, linestyle="-", label=f"Perplexity ({model})", color=color)
        ax.fill_between(
            checkpoints, mean_perplexity - std_perplexity, mean_perplexity + std_perplexity, alpha=0.1, color=color
        )
        ax.plot(checkpoints, log_perplexity, linestyle="--", label=f"Log Perplexity ({model})", color=color)
        # Plotting the base model point if exists
        if not base_model_data.empty:
            print(f"Base model data found for model: {model}")
            base_perplexity = base_model_data["token_perplexity_mean"].values[0]
            ax.scatter(0, base_perplexity, color=color, marker="o", label=f"Base Model Perplexity ({model})")


def plot_local_estimates(ax, model_local_data, color, model, base_model_data):
    """
    Plots local estimates on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        model_local_data (DataFrame): Data for the specific model's local estimates.
        color (str): Color to use for plotting.
        model (str): Model identifier.
        base_model_data (DataFrame): Data for the base model.
    """
    print(f"Plotting local estimates for model: {model}")
    checkpoints = model_local_data["checkpoint"].values
    local_estimates = model_local_data["local_estimate_mean"].values
    local_std = model_local_data["local_estimate_std"].values

    if len(checkpoints) > 0:
        print(f"Checkpoints available for local estimates of model {model}: {len(checkpoints)}")
        ax.plot(checkpoints, local_estimates, linestyle=":", label=f"Local Estimate ({model})", color=color)
        ax.fill_between(checkpoints, local_estimates - local_std, local_estimates + local_std, alpha=0.2, color=color)
        # Plotting the base model point if exists
        if not base_model_data.empty:
            print(f"Base model data found for local estimates of model: {model}")
            base_local_estimate = base_model_data["local_estimate_mean"].values[0]
            ax.scatter(0, base_local_estimate, color=color, marker="o", label=f"Base Model Local Estimate ({model})")


def create_side_by_side_plots(merged_df_clean, merged_local_df_clean, unique_models, filename="side_by_side_plots.pdf"):
    """
    Creates side by side plots for perplexity/log perplexity and local estimates for each model.

    Args:
        merged_df_clean (DataFrame): Cleaned DataFrame containing mean perplexity and std.
        merged_local_df_clean (DataFrame): Cleaned DataFrame containing local estimates.
        unique_models (list): List of unique model identifiers.
        filename (str): Name of the file to save the plot as a PDF.
    """
    print("Creating side by side plots...")
    model_colors = create_color_map(unique_models)

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    for model in unique_models:
        print(f"Processing model: {model}")
        model_data = merged_df_clean[merged_df_clean["model_without_checkpoint"] == model]
        model_local_data = merged_local_df_clean[merged_local_df_clean["model_without_checkpoint"] == model]
        base_model_data = filter_base_model_data(model_data)

        color = model_colors[model]

        # Plot metrics and local estimates
        plot_metrics(axs[0], model_data, color, model, base_model_data)
        plot_local_estimates(axs[1], model_local_data, color, model, base_model_data)

    # Setting labels, titles, and legends for both subplots
    print("Setting labels, titles, and legends...")
    axs[0].set_xlabel("Model Checkpoint")
    axs[0].set_ylabel("Value")
    axs[0].set_title("Perplexity and Log Perplexity Over Model Checkpoints for Each Model")
    axs[0].grid()

    axs[1].set_xlabel("Model Checkpoint")
    axs[1].set_ylabel("Local Estimate")
    axs[1].set_title("Local Estimate Over Model Checkpoints for Each Model")
    axs[1].grid()

    # Setting a common legend below both plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=3, fontsize="small")

    plt.tight_layout()
    print("Displaying and saving the plot...")
    plt.show()
    # Save the figure as a PDF
    fig.savefig(filename, format="pdf")


def main():
    """
    Main function to align dataframes and create plots based on the aligned data.
    """
    # Update the paths to point to the correct locations of the mean and std DataFrames
    mean_df_path = "path/to/mean_data.csv"
    std_df_path = "path/to/std_data.csv"

    print("Loading data...")
    mean_df = pd.read_csv(mean_df_path)
    std_df = pd.read_csv(std_df_path)

    # Align the mean and std DataFrames
    merged_df_clean = align_dataframes(mean_df, std_df)

    # Get the list of unique models
    unique_models = merged_df_clean["model_without_checkpoint"].unique()

    # Create side by side plots
    create_side_by_side_plots(merged_df_clean, merged_df_clean, unique_models)


if __name__ == "__main__":
    main()
