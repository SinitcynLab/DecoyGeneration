import numpy as np
import matplotlib.pyplot as plt

# for professional look:
plt.style.use("bmh")

# Data
categories = ['Target', 'Shuffle', 'Reverse', 'DIA-NN', 'ESM 8M', 'ESM 650M']

group1 = [0, 0.519]
error1 = [0, 0.020]

group2 = [0, 0.957]
error2 = [0, 0.012]

group3 = [0, 0.940]
error3 = [0, 0.012]

group4 = [0, 0.702]
error4 = [0, 0.070]

group5 = [0, 0.656]
error5 = [0, 0.042]

group6 = [0, 0.563]
error6 = [0, 0.065]

data = np.array([group1, group2, group3, group4, group5, group6])
err = np.array([error1, error2, error3, error4, error5, error6])

# Positions
x = np.arange(6)
width = 0.2

# Plot
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, data[:, 0], width, label='MLP Classifier', yerr=err[:,0], capsize=2)
plt.bar(x + width/2, data[:, 1], width, label='RNN Classifier', yerr=err[:,1], capsize=2)

# Labels and title
plt.xlabel('Decoy generator')
plt.ylabel('ROC AUC of Classifier')
plt.ylim(0.4, 1.05)
plt.title('Grouped Bar Chart of ROC AUC by classifier and decoy type')
plt.xticks(x, categories)
plt.legend()

# Layout and display
plt.tight_layout()
plt.show()
plt.savefig("src/visualization/images/quality_bar_chart.png")