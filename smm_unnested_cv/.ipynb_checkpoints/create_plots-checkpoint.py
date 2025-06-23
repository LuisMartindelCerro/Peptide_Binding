import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from scipy.stats import pearsonr

def pearson_from_pairs(pairs):
    n = len(pairs)
    if n == 0:
        return 0.0, float("inf")
    
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    
    x0 = sum(x) / n
    y0 = sum(y) / n
    
    t = nx = ny = err = 0.0
    for i in range(n):
        dx = x[i] - x0
        dy = y[i] - y0
        t += dx * dy
        nx += dx * dx
        ny += dy * dy
        err += (x[i] - y[i]) ** 2
    
    if nx * ny == 0:
        pcc = 0.0
    else:
        pcc = t / math.sqrt(nx * ny)
    
    mse = err / n
    return pcc, mse

def concat_predictions(folders, output_file):
    """
    Concatenate *_prediction files (TSV) from the given folders into a single TSV file.
    Each file must be tab-separated and have the same structure.
    """
    if os.path.exists(output_file):
        print(f"'{output_file}' already exists. Skipping concatenation.")
        return

    all_dfs = []

    for folder in folders:
        folder_path = os.path.join(os.getcwd(), f"{folder}.res")
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found - {folder}.res")
            continue

        prediction_files = [f for f in os.listdir(folder_path) if f.endswith('_prediction')]
        if not prediction_files:
            print(f"Warning: No prediction file found in {folder}")
            continue

        pred_file = os.path.join(folder_path, prediction_files[0])
        try:
            df = pd.read_csv(pred_file, sep='\t', header=None)  # NO header assumed
            df['allele'] = folder  # Optional: tag with source allele
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read {pred_file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Saved concatenated data to '{output_file}'")
    else:
        print("No valid data found to concatenate.")

all_alleles = [
    "A0101", "A0202", "A2402", "A3001", "A3101", "A6901",
    "B0801", "B3501", "B4403", "B5801", "A0201", "A2301",
    "A2403", "A3002", "A6801", "B0702", "B2705", "B4402", "B5701"
]

concat_predictions(all_alleles, "all_predictions.txt")

less500_alleles = ['A2402', 'A2301', 'A2403', 'A3002', 'B4403', 'B4402', 'B5701']

concat_predictions(less500_alleles, "less500_alleles.txt")

_500_1000_alleles = ['A3001', 'A6901', 'B0801', 'B3501', 'B5801', 'B2705']

concat_predictions(_500_1000_alleles, "_500_1000_alleles.txt")

_1000_1500_alleles = ['A0202', 'B0702']

concat_predictions(_1000_1500_alleles, "_1000_1500_alleles.txt")

more1500_alleles = ['A0101', 'A3101', 'A0201', 'A6801']

concat_predictions(more1500_alleles, "more1500_alleles.txt")

# create the plotting

def plot_multiple_auc(prediction_sets, binder_threshold=0.426, save_path=None):
    """
    Plot ROC curves from multiple prediction datasets.
    Each entry in prediction_sets should be (y_true, y_pred, label).
    Also computes and displays Pearson correlation (PCC).
    """
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Random: AUC=0.500')

    for y_true, y_pred, label in prediction_sets:
        y_binary = (y_true >= binder_threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred)
        auc = roc_auc_score(y_binary, y_pred)

        # Calculate Pearson correlation
        try:
            pcc, _ = pearsonr(y_true, y_pred)
        except:
            pcc = float("nan")

        plt.plot(fpr, tpr, label=f"{label}: AUC={auc:.3f}, PCC={pcc:.3f}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves with AUC and PCC")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{save_path}'")

def load_predictions_from_file(filename):
    """
    Load true values and predicted values from a file.
    Assumes columns: allele, peptide_name, y_true, y_pred
    """
    data = np.loadtxt(filename, usecols=(1, 2))  # load only true and predicted
    y_true = data[:, 0]
    y_pred = data[:, 1]
    return y_true, y_pred


# Load all datasets
prediction_sets = [
    (*load_predictions_from_file("all_predictions.txt"), "All Alleles"),
    (*load_predictions_from_file("less500_alleles.txt"), "<500 Peptides"),
    (*load_predictions_from_file("_500_1000_alleles.txt"), "500–999 Peptides"),
    (*load_predictions_from_file("_1000_1500_alleles.txt"), "1000–1499 Peptides"),
    (*load_predictions_from_file("more1500_alleles.txt"), "≥1500 Peptides"),
]

# Plot ROC curves
# Plot ROC curves and save as PNG
plot_multiple_auc(
    prediction_sets,
    binder_threshold=0.426,
    save_path="auc_comparison_plot.png"  # Add this line
)

