import numpy as np
import matplotlib.pyplot as plt

# for professional look:
plt.style.use("bmh")

# Data
categories = ['8M', '35M', '150M', '650M']

group1 = [0.660, 0.656]
error1 = [0.063, 0.065]

group2 = [0.596, 0.629]
error2 = [0.032, 0.079]

group3 = [0.588, 0.578]
error3 = [0.064, 0.047]

group4 = [0.583, 0.563]
error4 = [0.042, 0.042]

data = np.array([group1, group2, group3, group4])
err = np.array([error1, error2, error3, error4])

# Positions
x = np.arange(4)
width = 0.35

# Plot
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, data[:, 0], width, label='16 bit', yerr=err[:,0], capsize=2)
plt.bar(x + width/2, data[:, 1], width, label='32 bit', yerr=err[:,1], capsize=2)

# Labels and title
plt.xlabel('Parameter count and parameter precision')
plt.ylabel('RNN ROC AUC')
plt.ylim(0.4, 0.75)
plt.title('Grouped Bar Chart of ESM-generator performance')
plt.xticks(x, categories)
plt.legend()

# Layout and display
plt.tight_layout()
plt.savefig("src/visualization/images/param_chart.png")
