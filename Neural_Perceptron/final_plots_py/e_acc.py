import matplotlib.pyplot as plt
import sys

def read_data(file_path):
    epochs = []
    accuracy_values = []

    with open(file_path, 'r') as f:
        for line in f:
            epoch, accuracy = map(float, line.strip().split(","))
            epochs.append(epoch)
            accuracy_values.append(accuracy)
            
    return epochs, accuracy_values

def plot_data(epochs, accuracy_values):
    plt.figure(figsize=(10, 6))

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
    
    file_path = sys.argv[1]
    epochs, accuracy_values = read_data(file_path)
    plot_data(epochs, accuracy_values)
