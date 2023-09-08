import csv

# Read data from the original CSV file
with open('cancer.csv', 'r') as infile:
    csvreader = csv.reader(infile)
    header = next(csvreader)  # Skip header
    
    # Initialize an empty list to store the transformed rows
    transformed_data = []
    
    # Initialize variables to store min and max values for each feature
    min_values = None
    max_values = None
    
    for row in csvreader:
        # Skip the 'id' column (row[0])
        # Transform 'diagnosis' from 'M' and 'B' to 1 and 0
        label = 1 if row[1] == 'M' else 0
        # Convert remaining columns to float
        features = list(map(float, row[2:]))
        
        # Initialize min and max values if they are None
        if min_values is None:
            min_values = features.copy()
            max_values = features.copy()
        
        # Update min and max values
        min_values = [min(min_val, feat) for min_val, feat in zip(min_values, features)]
        max_values = [max(max_val, feat) for max_val, feat in zip(max_values, features)]
        
        # Combine label and features
        transformed_row = [label] + features
        transformed_data.append(transformed_row)

# Normalize the features
for i in range(len(transformed_data)):
    label = transformed_data[i][0]
    features = transformed_data[i][1:]
    
    normalized_features = [(feat - min_val) / (max_val - min_val) for feat, min_val, max_val in zip(features, min_values, max_values)]
    transformed_data[i] = [label] + normalized_features

# Write transformed data to a new CSV file
with open('transformed_normalized_cancer.csv', 'w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    
    # Write a new header (you can modify this to match your needs)
    new_header = ['label'] + header[2:]
    csvwriter.writerow(new_header)
    
    # Write transformed and normalized rows
    for row in transformed_data:
        csvwriter.writerow(row)
