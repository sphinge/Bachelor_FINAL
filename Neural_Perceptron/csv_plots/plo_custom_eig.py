import matplotlib.pyplot as plt
import re

# Read data from file
with open("acc2eigen.txt", "r") as f:
    lines = f.readlines()

# Initialize lists to store epochs and accuracies
epochs = []
accuracies = []

# Extract epochs and accuracies using regular expressions
for line in lines:
    match = re.search("Epoch (\d+): Accuracy = ([\d\.]+)", line)
    if match:
        epoch = int(match.group(1))
        accuracy = float(match.group(2))
        epochs.append(epoch)
        accuracies.append(accuracy)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, marker='o')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
