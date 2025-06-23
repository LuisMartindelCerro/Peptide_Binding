import subprocess
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve




BASE_DIR = "/mnt/c/Users/Xavi/Documents/Master_in_Bioinformatics_and_Systems_Biology/June_2025/algorithms/code/testing/data"
ALLELES = ["A0101", "A0201", "A0202", "A2301", "A2402", "A2403", "A3001", "A3002", "A3101", "A6801", "A6901", "B0702", "B0801", "B2705", "B3501", "B4402", "B4403", "B5701", "B5801"]  # Add all your allele folders here
FOLDS = ["0", "1", "2", "3", "4"]

PSSM_SCRIPT = "pssm.py"  # Assumes it's in the same directory or in PATH

global_predictions = []
global_targets = []

pcc_allele_list = []
auc_allele_list = []

for allele in ALLELES:
    print(f"Processing allele: {allele}")
    allele_dir = f"{BASE_DIR}/{allele}"
    results_dir = f"{allele_dir}/results"
    subprocess.run(["mkdir", "-p", results_dir])

    all_predictions = []
    all_targets = []

    for fold in FOLDS:
        f_file = f"{allele_dir}/f00{fold}"
        c_file = f"{allele_dir}/c00{fold}"
        train_tmp = f"{allele_dir}/train.tmp.{fold}"

        # Filter training data for binders only
        binders = [line for line in open(f_file) if float(line.strip().split()[1]) >= 0.426]
        print(f"[INFO] Fold {fold} - #Binders used for training: {len(binders)}")

        with open(train_tmp, 'w') as fout:
            for line in binders:
                fout.write(line.strip().split()[0] + '\n')


        # Run pssm.py
        print(f"  Fold {fold}: training on {f_file}, testing on {c_file}")
        print(f"  Running: python3 {PSSM_SCRIPT} -f {train_tmp} -z {c_file}")
        pred_file = f"{results_dir}/c00{fold}.pred"
        with open(pred_file, 'w') as pred_out:
            subprocess.run([
                "python3", PSSM_SCRIPT,
                "-b", "100.0",
                "-f", train_tmp,
                "-z", c_file,
                "-w"
            ], stdout=pred_out)
            

        # Collect predictions and targets
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        pred = float(parts[1])
                        target = float(parts[2])
                        all_predictions.append(pred)
                        all_targets.append(target)
                    except ValueError:
                        continue
                    
    global_predictions.extend(all_predictions)
    global_targets.extend(all_targets)
    
    # Compute global PCC using scipy's pearsonr
    if all_predictions and all_targets:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        pcc = pearsonr(all_targets, all_predictions)
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        print(f"\n>>> {allele} Combined PCC (pearsonr): {pcc[0]:.5f}, RMSE: {rmse:.5f}\n")
        pcc_allele_list.append((allele, float(pcc[0])))

    # Convert true targets to binary labels (binders = 1 if score ≥ 0.426)    
    labels = (all_targets >= 0.426).astype(int)
    scores = all_predictions

    # Calculate AUC
    auc = roc_auc_score(labels, scores)
    print(f">>> {allele} AUC: {auc:.5f}")
    auc_allele_list.append((allele, float(auc)))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{allele} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{allele}_roc.png")
    plt.close()

    # After computing PCC
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.title(f"{allele} PSSM Predictions\nPCC = {pcc[0]:.3f}")
    plt.xlabel("True Binding Score")
    plt.ylabel("Predicted Score")
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{results_dir}/{allele}_pcc_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Finished allele {allele}. Results saved in: {results_dir}\n")

for allele, pcc_value in pcc_allele_list:
    print(f"{allele}: PCC: {pcc_value:.5f}")  
       
for allele, auc_score in auc_allele_list:
    print(f"{allele}: AUC: {auc_score:.5f}") 


global_predictions = np.array(global_predictions)
global_targets = np.array(global_targets)

global_pcc = pearsonr(global_targets, global_predictions)[0]
global_auc = roc_auc_score((global_targets >= 0.426).astype(int), global_predictions)

fpr, tpr, _ = roc_curve((global_targets >= 0.426).astype(int), global_predictions)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {global_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve (All Alleles)")
plt.legend()

plt.text(0.6, 0.2, f"PCC = {global_pcc:.3f}", bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.grid(True)
concat_dir = f"{BASE_DIR}/concatenated.res"
subprocess.run(["mkdir", "-p", concat_dir])
plt.savefig(f"{concat_dir}/global_roc_pcc.png")
plt.close()