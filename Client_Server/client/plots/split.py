def split_metrics_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Remove header
    header = lines.pop(0)
    
    client1 = []
    client2 = []
    
    switch_to_client1 = True

    block = []
    for line in lines:
        epoch = int(line.split(',')[0])
        if epoch == 1 and block:
            if switch_to_client1:
                client1.extend(block)
            else:
                client2.extend(block)
            switch_to_client1 = not switch_to_client1
            block = []
        block.append(line)
    
    # Add the last block to the appropriate client
    if block:
        if switch_to_client1:
            client1.extend(block)
        else:
            client2.extend(block)
    
    # Write to client1_metrics.csv and client2_metrics.csv
    with open('client1_metrics.csv', 'w') as f:
        f.write(header)
        f.writelines(client1)

    with open('client2_metrics.csv', 'w') as f:
        f.write(header)
        f.writelines(client2)

split_metrics_file('training_metrics.csv')
