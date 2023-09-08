import matplotlib.pyplot as plt

# Initialize lists to store epochs and accuracies
epochs = []
accuracies = []

# Read the file
with open('acc.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        epoch = int(parts[1])
        accuracy = float(parts[-1][:-1])  # Remove the '%' and convert to float
        epochs.append(epoch)
        accuracies.append(accuracy)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy per Epoch')
plt.grid(True)
plt.show()
