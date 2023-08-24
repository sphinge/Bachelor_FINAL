import pandas as pd
import matplotlib.pyplot as plt

# Read the loss data from the CSV file
data = pd.read_csv('loss.csv')

# Plot the loss values
plt.figure(figsize=(10,6))
plt.plot(data['Epoch'], data['Loss'], label='Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
