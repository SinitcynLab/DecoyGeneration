import numpy as np
import matplotlib.pyplot as plt

# Data
categories = ['8M', '35M', '150M', '650M']
group1 = [0.660, 0.656]
group2 = [0.553, 0.629]
group3 = [0.588, 0.578]
group4 = [0.583, 0.563]

data = np.array([group1, group2, group3, group4])

# Positions
x = np.arange(4)
width = 0.35

# Plot
plt.figure(figsize=(7, 5))
plt.bar(x - width/2, data[:, 0], width, label='16 bit')
plt.bar(x + width/2, data[:, 1], width, label='32 bit')

# Labels and title
plt.xlabel('Parameter count and parameter precision')
plt.ylabel('AUC')
plt.ylim(0.5, 0.7)
plt.title('Grouped Bar Chart of ESM-generator performance, mask_count = 1')
plt.xticks(x, categories)
plt.legend()

# Layout and display
plt.tight_layout()
plt.show()
plt.savefig("src/visualization/images/AUC_chart.png")
