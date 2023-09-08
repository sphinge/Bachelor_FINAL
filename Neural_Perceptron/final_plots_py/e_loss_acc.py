import matplotlib.pyplot as plt
import sys

def read_data(file_path):
    epochs = []
    mse_values = []
    accuracy_values = []

    with open(file_path, 'r') as f:
        for line in f:
            epoch, mse, accuracy = map(float, line.strip().split(","))
            epochs.append(epoch)
            mse_values.append(mse)
            accuracy_values.append(accuracy)
            
    return epochs, mse_values, accuracy_values

def plot_data(epochs, mse_values, accuracy_values):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, mse_values, label='Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_values, label='Accuracy', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch vs Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file path.")
        sys.exit(1)
    
    file_path = sys.argv[1]  # Get the file path from command line arguments
    epochs, mse_values, accuracy_values = read_data(file_path)
    plot_data(epochs, mse_values, accuracy_values)
