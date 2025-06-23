import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Step 1: Parse full PCC data ---
pattern = r"^(\w+)\s+lambda\s+([0-9.]+)\s+epsilon\s+([0-9.]+)\s+PCC\s+([0-9.]+)"
data = []

with open("training_output.log") as f:
    for line in f:
        print(line)
        match = re.match(pattern, line)
        print(match)
        if match:
            allele = match.group(1)
            lam = float(match.group(2))
            eps = float(match.group(3))
            pcc = float(match.group(4))
            data.append({'allele': allele, 'lambda': lam, 'epsilon': eps, 'PCC': pcc})

df = pd.DataFrame(data)

# --- Step 2: Create heatmaps per allele ---
alleles = df['allele'].unique()

ncols = 4
nrows = (len(alleles) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

for i, allele in enumerate(alleles):
    ax = axes.flat[i]
    subset = df[df['allele'] == allele]
    pivot = subset.pivot(index="lambda", columns="epsilon", values="PCC")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False)
    ax.set_title(allele)

# Hide unused subplots
for j in range(i + 1, nrows * ncols):
    axes.flat[j].axis("off")

plt.tight_layout()
plt.savefig("allele_parameter_heatmaps.png", dpi=300)

