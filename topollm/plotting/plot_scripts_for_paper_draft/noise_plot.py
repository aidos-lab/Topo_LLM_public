import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("raw_data.csv")

# Renaming the columns for simplicity
df = df.rename(
    columns={
        "additional_distance_approximate_hausdorff_via_kdtree": "Hausdorff Distance",
        "global_estimate": "Global Estimate",
        "pointwise_results_np_mean": "Mean Pointwise",
        "pointwise_results_np_std": "Std Pointwise",
        "local_estimates_noise_distortion": "Noise Distortion",
    }
)

# Updated column names
x_col = "Hausdorff Distance"
y_cols = ["Global Estimate", "Mean Pointwise", "Std Pointwise"]
heat_col = "Noise Distortion"

# Adjust font style and size
plt.rcParams.update(
    {
        "font.size": 8,  # Smaller font size for ICML column
        "font.family": "serif",  # Change to serif font
    }
)

# Generate scatterplots and save them
for y_col in y_cols:
    plt.figure(figsize=(3.3, 2.5))  # Adjust size for single-column width
    scatter = plt.scatter(df[x_col], df[y_col], c=df[heat_col], cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label=heat_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()  # Ensure proper layout within the column
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save the plot as a PDF with a sensible name
    file_name = f"scatter_{y_col.replace(' ', '_').lower()}.pdf"
    plt.savefig(file_name, format="pdf", dpi=300)
    print(f"Saved plot as {file_name}")
    plt.close()
