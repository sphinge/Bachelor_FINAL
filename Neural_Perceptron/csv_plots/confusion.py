from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data in the format: [(predicted_label, real_label), ...]
samples = [
    (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(0,1),(1,1),(1,1),(1,1),(1,1),(1,1),(0,1),(1,1),(1,1),(0,0),(0,0),(0,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(0,0),(0,1),(1,1),(0,1),(0,1),(1,1),(1,1),(0,1),(1,1),(0,0),(1,1),(0,0),(0,0),(0,0),(0,0),(0,0),(1,1),(0,1),(0,0),(1,1),(1,1),(0,0),(0,0),(0,0),(0,0),(1,1),(0,0),(1,1),(1,1),(0,0),(0,0),(1,0),(0,0),(1,1),(0,0),(1,1),(0,1),(0,0),(1,1),(0,0),(1,1),(1,1),(0,0),(0,0),(1,0)
]

# Separate the predicted and real labels
predicted_labels = [predicted for predicted, _ in samples]
real_labels = [real for _, real in samples]

# Generate the confusion matrix
conf_matrix = confusion_matrix(real_labels, predicted_labels)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1], annot_kws={"size": 20})  # Changed here
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()