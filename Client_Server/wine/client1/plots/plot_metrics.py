import matplotlib.pyplot as plt

def read_data(file_path):
    epochs = []
    loss_values = []
    accuracy_values = []

    with open(file_path, 'r') as f:
        for index, line in enumerate(f, start=1):  # enumerate with start=1 to count indexes from 1
            loss, accuracy = map(float, line.strip().split(","))
            epochs.append(index)
            loss_values.append(loss)
            accuracy_values.append(accuracy)
            
    return epochs, loss_values, accuracy_values

def plot_data(epochs, loss_values, accuracy_values):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, label='Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
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
    file_path = "/home/wiktoria/Desktop/Thesis/Client_Server/wine/client1/plots/training_metrics.csv"  # Replace with the path to your data file
    epochs, loss_values, accuracy_values = read_data(file_path)
    plot_data(epochs, loss_values, accuracy_values)
