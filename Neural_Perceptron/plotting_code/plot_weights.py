def remove_lines(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            if "1: " not in line and "2: " not in line:
                outfile.write(line)

# Example usage:
input_file = "plot_weights.csv"  # Replace with your input filename
output_file = "plot_weights_n1.txt"  # Replace with your desired output filename
remove_lines(input_file, output_file)
