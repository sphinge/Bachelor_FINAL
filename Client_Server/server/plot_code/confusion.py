import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def read_data(file_path):
    y_true = []
    y_pred = []

    with open(file_path, 'r') as f:
        data = f.read().strip()
        # Remove trailing comma if it exists
        if data[-1] == ',':
            data = data[:-1]
        
        # Properly split the pairs
        pairs = data.split("),(")
        
        for pair in pairs:
            pair = pair.replace("(", "").replace(")", "")
            try:
                pred, true = map(int, pair.split(","))
                y_pred.append(pred)
                y_true.append(true)
            except ValueError as e:
                print(f"Skipping invalid data: {pair}. Error: {e}")

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Use text annotations to show the numbers inside the heatmap cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                    size=16)  # Change size to control the font size
            
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.show()

if __name__ == "__main__":
    file_path = "/home/wiktoria/Desktop/Thesis/Client_Server/server/plots/confusion.csv"  # Replace with the path to your data file
    y_true, y_pred = read_data(file_path)
    plot_confusion_matrix(y_true, y_pred)
