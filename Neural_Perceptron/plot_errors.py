import matplotlib.pyplot as plt

# Load the data from the CSV file
epochs = []
errors = []
with open('plot_errors.csv', 'r') as f:
    next(f)  # skip the header
    for line in f:
        epoch, error = line.strip().split(', ')
        epochs.append(int(epoch))
        errors.append(float(error))

# Plot the data
plt.plot(epochs, errors, '-o', markersize=4, label='MSE')
plt.xlabel('Epoch')
plt.ylabel('Error (MSE)')
plt.title('Error vs. Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
