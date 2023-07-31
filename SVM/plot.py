import matplotlib.pyplot as plt
import re

# Define the list to store your data
data = []

# Open your file
with open('output.csv', 'r') as file:
    for line in file:
        # Extract the numerical values from the line
        nums = re.findall(r'\d+\.?\d*', line)
        # Convert strings to appropriate type (int or float)
        iteration = int(nums[0])
        accuracy = float(nums[1])
        error = float(nums[2])
        # Append as a dictionary
        data.append({'Iteration': iteration, 'Training Accuracy': accuracy, 'Average Error': error})

# Parse the iteration numbers, accuracies, and errors into lists
iterations = [d['Iteration'] for d in data]
accuracies = [d['Training Accuracy'] for d in data]
errors = [d['Average Error'] for d in data]

# Now plot the data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(iterations, accuracies, 'r')
plt.title('Training Accuracy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(iterations, errors, 'b')
plt.title('Average Error over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Average Error')

plt.tight_layout()
plt.show()
