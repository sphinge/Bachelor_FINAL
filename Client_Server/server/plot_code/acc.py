import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # <-- Add this line

def read_data(file_path):
    accuracy_values = []

    with open(file_path, 'r') as f:
        for line in f:
            accuracy_values.append(float(line.strip()))
    
    epochs = list(range(1, len(accuracy_values) + 1))  # epoch values will be [1, 2, ... len(accuracy_values)]
    return epochs, accuracy_values

def plot_data(epochs, accuracy_values):
    fig, ax = plt.subplots(figsize=(10, 6))  # <-- Modify this line

    ax.plot(epochs, accuracy_values, label='Accuracy', color='b')
    ax.set_xlabel('Exchanges')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy per Exchange')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # <-- Add this line
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "/home/wiktoria/Desktop/Thesis/Client_Server/server/plots/accuracy.csv"  # Replace with the path to your data file
    epochs, accuracy_values = read_data(file_path)
    plot_data(epochs, accuracy_values)

