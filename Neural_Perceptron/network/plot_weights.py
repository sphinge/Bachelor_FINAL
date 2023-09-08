import matplotlib.pyplot as plt

input_weights_all = []
input_biases_all = []
output_weights_all = []
output_biases_all = []
iterations = []

with open('weights.csv', 'r') as file:
    next(file)
    for line in file:
        values = line.strip().split(',')

        iteration = int(values[0])
        iterations.append(iteration)

        # Input layer weights (4x8)
        input_weights = [float(x) for x in values[1:1 + 4*8]]
        input_weights_all.append(input_weights)

        # Input layer biases (8)
        input_biases = [float(x) for x in values[1 + 4*8:1 + 4*8 + 8]]
        input_biases_all.append(input_biases)

        # Output layer weights (8x3)
        output_weights_start = 1 + 4*8 + 8
        output_weights = [float(x) for x in values[output_weights_start:output_weights_start + 8*3]]
        output_weights_all.append(output_weights)

        # Output layer biases (3)
        output_biases = [float(x) for x in values[output_weights_start + 8*3:] if x]
        output_biases_all.append(output_biases)


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(15,10))

# Input Weights
for i in range(len(input_weights_all[0])):
    axs[0, 0].plot(iterations, [item[i] for item in input_weights_all], label=f'Weight {i+1}')
axs[0, 0].set_title('Hidden Layer Weights')
axs[0, 0].legend()

# Input Biases
for i in range(len(input_biases_all[0])):
    axs[0, 1].plot(iterations, [item[i] for item in input_biases_all], label=f'Bias {i+1}')
axs[0, 1].set_title('Hidden Layer Biases')
axs[0, 1].legend()

# Output Weights
for i in range(len(output_weights_all[0])):
    axs[1, 0].plot(iterations, [item[i] for item in output_weights_all], label=f'Weight {i+1}')
axs[1, 0].set_title('Output Layer Weights')
axs[1, 0].legend()

# Output Biases
for i in range(len(output_biases_all[0])):
    axs[1, 1].plot(iterations, [item[i] for item in output_biases_all], label=f'Bias {i+1}')
axs[1, 1].set_title('Output Layer Biases')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

