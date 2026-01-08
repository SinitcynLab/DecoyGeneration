import matplotlib.pyplot as plt

# Data
labels = ['shuffle, \nskip prob 0', 'diann', 'random_replace', 'ESM8M, \ncount=1', 'esm650M, \ncount=1', 'rel_diff_smart_\nmasking_8M,\n count=1']
times = [0.797, 0.234, 0.323, 19.905, 701.3, 24457.525]  # times in minutes (or any unit)

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(labels, times)

# Labels and title
plt.xlabel('Generator')
plt.ylabel('Time (seconds)')
plt.title('Time spent by generator to convert 100 sequences')

# Optional: show values on top of bars
for i, value in enumerate(times):
    plt.text(i, value, f'{value}', ha='center', va='bottom')

# Display
plt.tight_layout()
plt.show()
