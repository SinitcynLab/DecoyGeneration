import numpy as np
import matplotlib.pyplot as plt

# for professional look:
plt.style.use("bmh")

# Data
categories = ['ESM650', 'Shuffle', 'Reverse']

group1 = [0.546, 0.633]
error1 = [0.006, 0.012]

group2 = [0.511, 0.610]
error2 = [0.005, 0.005]

group3 = [0.518, 0.650]
error3 = [0.002, 0.013]

data = np.array([group1, group2, group3])
err = np.array([error1, error2, error3])

# Positions
x = np.arange(3)
width = 0.35

# Plot
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, data[:, 0], width, label='Peptide level', yerr=err[:,0], capsize=2)
plt.bar(x + width/2, data[:, 1], width, label='Sequence level', yerr=err[:,1], capsize=2)

# Labels and title
plt.xlabel('Decoy type and AUC')
plt.ylabel('PLM-free RNN ROC AUC')
plt.ylim(0.4, 0.75)
plt.title('Grouped Bar Chart of PLM-free RNN classifier performance')
plt.xticks(x, categories)
plt.legend()

# Layout and display
plt.tight_layout()
plt.savefig("src/visualization/images/plm_free.png")