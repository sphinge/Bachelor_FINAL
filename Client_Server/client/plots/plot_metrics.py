import matplotlib.pyplot as plt
import numpy as np

def plot_client_data(filename):
    # Read data from the given filename
    with open(filename, 'r') as f:
        lines = f.readlines()

    header = lines.pop(0)

    epochs = [int(line.split(",")[0]) for line in lines]
    losses = [float(line.split(",")[1]) for line in lines]
    accuracies = [float(line.split(",")[2]) for line in lines]

    # Calculate the number of exchanges based on epoch repetitions
    exchanges = np.cumsum(np.array(epochs) == 1)

    fig, ax1 = plt.subplots()

    # Plotting Loss on left y-axis
    ax1.set_xlabel('Exchanges')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(exchanges, losses, 'o-', color='tab:blue', label="Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Creating a second y-axis for plotting Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.plot(exchanges, accuracies, 's-', color='tab:red', label="Accuracy")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Setting grid, title and showing the plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"Loss and Accuracy for {filename}")
    plt.show()

# CHANGE FOR MORE CLIENTS!!!!!!!!!!!!
# Generate plots for each client's metrics file
for i in range(1, 3):  # For clients 1, 2, and 3
    plot_client_data(f"client{i}_metrics.csv")
