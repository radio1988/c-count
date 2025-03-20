import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example data: Replace with your actual labels
data = {
    "Ashley": [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    "Logan":  [0, 1, 0, 0, 1, 1, 1, 0, 1, 1],
    "John":   [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    "Model":  [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

# Define pairs for confusion matrices
comparisons = [
    ("Model", "Ashley"),
    ("Model", "Logan"),
    ("Model", "John"),
    ("Ashley", "Logan"),
    ("Ashley", "John"),
    ("Logan", "John"),
]

# Set up the figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Generate confusion matrices for each pair
for i, (a, b) in enumerate(comparisons):
    cm = confusion_matrix(df[a], df[b])
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=axes[i])
    axes[i].set_title(f"{a} vs. {b}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()
