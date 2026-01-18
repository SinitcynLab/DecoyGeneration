import numpy as np
import matplotlib.pyplot as plt

# for professional look:
plt.style.use("seaborn-v0_8-colorblind")

# Data
categories = ['Shuffle', 'Reverse', 'DIA-NN', 'ESM 8M', 'ESM 650M']
group1 = [0, 0, 0, 0]
group2 = [0, 0, 0, 0]
group3 = [0, 0, 0, 0]
group4 = [0, 0, 0, 0]

data = np.array([group1, group2, group3, group4])

# Positions
x = np.arange(4)
width = 0.2

# Plot
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, data[:, 0], width, label='MLP Classifier')
plt.bar(x + width/2, data[:, 1], width, label='RNN Classifier')

# Labels and title
plt.xlabel('Decoy generator')
plt.ylabel('ROC AUC of Classifier')
plt.ylim(0.5, 0.7)
plt.title('Grouped Bar Chart of ROC AUC by classifier and decoy type')
plt.xticks(x, categories)
plt.legend()

# Layout and display
plt.tight_layout()
plt.show()
plt.savefig("src/visualization/images/quality_bar_chart.png")