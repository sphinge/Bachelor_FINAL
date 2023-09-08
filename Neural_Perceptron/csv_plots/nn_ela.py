import matplotlib.pyplot as plt

# Your data as a string
data_str = """1,0.492908,87.8049
2,0.311027,85.3659
3,0.217862,89.0244
4,0.158662,84.1463
5,0.12264,81.7073
6,0.1,84.1463
7,0.0865134,84.1463
8,0.0755538,84.1463
9,0.0665401,84.1463
10,0.059075,84.1463
11,0.052877,82.9268
12,0.0476616,82.9268
13,0.0432367,82.9268
14,0.0394561,82.9268
15,0.0362042,81.7073
16,0.033389,81.7073
17,0.0309367,81.7073
18,0.0287883,81.7073
19,0.0268958,84.1463
20,0.02522,84.1463
21,0.0237292,84.1463
22,0.0224235,84.1463
23,0.0213237,84.1463
24,0.0203309,84.1463
25,0.0194273,84.1463
26,0.0186022,85.3659
27,0.0178465,85.3659
28,0.0171524,85.3659
29,0.0165132,85.3659
30,0.0159232,87.8049
31,0.0153773,89.0244
32,0.014871,90.2439
33,0.0144005,90.2439
34,0.0139625,91.4634
35,0.0135539,91.4634
36,0.013172,91.4634
37,0.0128146,91.4634
38,0.0124795,91.4634
39,0.0121649,91.4634
40,0.011869,91.4634
41,0.0115905,91.4634
42,0.0113278,91.4634
43,0.0110798,91.4634
44,0.0108454,90.2439
45,0.0106237,91.4634
46,0.0104136,91.4634
47,0.0102143,90.2439
48,0.0100252,90.2439
49,0.00984544,90.2439
50,0.00967451,90.2439"""

# Split the data into lines and then split each line into its columns
data = [line.split(",") for line in data_str.split("\n")]

# Extract the columns into separate lists
indexes = [int(row[0]) for row in data]
losses = [float(row[1]) for row in data]
accuracies = [float(row[2]) for row in data]

# Create the plot
fig, ax1 = plt.subplots()

# Plot loss
color = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', color=color, fontsize=14)
ax1.plot(indexes, losses, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Create another y-axis for the accuracies
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color, fontsize=14)
ax2.plot(indexes, accuracies, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

# Add title and show plot
plt.title('Loss and Accuracy Plot', fontsize=16)
plt.show()