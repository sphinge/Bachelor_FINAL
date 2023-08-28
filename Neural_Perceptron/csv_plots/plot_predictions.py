import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("predictions.csv")

# Extract columns
sample = data["Sample"].tolist()
predicted = data["Predicted"].tolist()
real = data["Real"].tolist()

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(sample, predicted, color='blue', label='Predicted', s=100, marker='x')
plt.scatter(sample, real, color='red', label='Real', s=100, marker='o')
# plt.plot(sample, real, color='grey', linestyle='dashed', linewidth=1)  # Optional: Real values as a dashed line
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Real vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
