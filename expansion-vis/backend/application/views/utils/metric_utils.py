# Read the metric.txt file and convert data to JSON format
def process_metric_data(file_path):
    with open(file_path, 'r') as file:
        metric_data = file.readlines()
        metric_list = []

        for line in metric_data:
            data = line.strip().split(', ')
            if len(data) >= 6:  # Check if line contains all necessary data
                use_time = data[0].split(': ')[1]
                epoch = int(data[1].split(': ')[1])
                class_name = data[2].split(': ')[1]
                delta_entropy = float(data[3].split(': ')[1])
                divergence = float(data[4].split(': ')[1])
                augmented_output_prob_top1 = float(data[5].split(': ')[1])

                metric_entry = {
                    'use_time': use_time,
                    'epoch': epoch,
                    'class_name': class_name,
                    'delta_entropy': delta_entropy,
                    'divergence': divergence,
                    'augmented_output_prob_top1': augmented_output_prob_top1
                }

                metric_list.append(metric_entry)
            else:
                # Handle incomplete line
                print(f"Incomplete line: {line}")

    return metric_list

# Group data by hour, ensuring that the hour is sequentially increasing
def process_metric_data_by_hour(file_path, class_names):
    with open(file_path, 'r') as file:
        metric_data = file.readlines()
        metric_dict = {}

        for line in metric_data:
            data = line.strip().split(', ')
            if len(data) >= 6: 
                use_time = int(data[0].split(': ')[1])
                epoch = int(data[1].split(': ')[1])
                class_name = data[2].split(': ')[1]
                delta_entropy = float(data[3].split(': ')[1])
                divergence = float(data[4].split(': ')[1])
                augmented_output_prob_top1 = float(data[5].split(': ')[1])

                if use_time not in metric_dict:
                    metric_dict[use_time] = {}

                if class_name not in metric_dict[use_time]:
                    metric_dict[use_time][class_name] = {
                        'delta_entropy': [],
                        'divergence': [],
                        'augmented_output_prob_top1': []
                    }

                metric_dict[use_time][class_name]['delta_entropy'].append(delta_entropy)
                metric_dict[use_time][class_name]['divergence'].append(divergence)
                metric_dict[use_time][class_name]['augmented_output_prob_top1'].append(augmented_output_prob_top1)
            else:
                # Handle incomplete line
                print(f"Incomplete line: {line}")

    # Calculate average values
    for use_time in metric_dict:
        for class_name in class_names:
            # For cases with missing data for class_name at hour 0, set to 0
            if use_time == 0 and class_name not in metric_dict[use_time]:
                metric_dict[use_time][class_name] = {
                        'delta_entropy': [0],
                        'divergence': [0],
                        'augmented_output_prob_top1': [0]
                    }
            # For cases with missing data for class_name at other hours, use data from the previous hour
            if class_name not in metric_dict[use_time] and use_time != 0:
                metric_dict[use_time][class_name] = metric_dict[use_time-1][class_name]
            # If data is complete for class_name at this hour, calculate average values
            elif class_name in metric_dict[use_time]:
                metric_dict[use_time][class_name]['delta_entropy'] = sum(metric_dict[use_time][class_name]['delta_entropy']) / len(metric_dict[use_time][class_name]['delta_entropy'])
                metric_dict[use_time][class_name]['divergence'] = sum(metric_dict[use_time][class_name]['divergence']) / len(metric_dict[use_time][class_name]['divergence'])
                metric_dict[use_time][class_name]['augmented_output_prob_top1'] = sum(metric_dict[use_time][class_name]['augmented_output_prob_top1']) / len(metric_dict[use_time][class_name]['augmented_output_prob_top1'])

    return metric_dict

# Group data by epoch
def process_metric_data_by_epoch(file_path):
    with open(file_path, 'r') as file:
        metric_data = file.readlines()
        metric_dict = {}

        for line in metric_data:
            data = line.strip().split(', ')
            if len(data) >= 6:  # Check if line contains all necessary data
                use_time = data[0].split(': ')[1]
                epoch = int(data[1].split(': ')[1])
                class_name = data[2].split(': ')[1]
                delta_entropy = float(data[3].split(': ')[1])
                divergence = float(data[4].split(': ')[1])
                augmented_output_prob_top1 = float(data[5].split(': ')[1])

                if epoch not in metric_dict:
                    metric_dict[epoch] = {}

                if class_name not in metric_dict[epoch]:
                    metric_dict[epoch][class_name] = {
                        'delta_entropy': [],
                        'divergence': [],
                        'augmented_output_prob_top1': []
                    }

                metric_dict[epoch][class_name]['delta_entropy'].append(delta_entropy)
                metric_dict[epoch][class_name]['divergence'].append(divergence)
                metric_dict[epoch][class_name]['augmented_output_prob_top1'].append(augmented_output_prob_top1)
            else:
                # Handle incomplete line
                print(f"Incomplete line: {line}")

    # Calculate average values
    for epoch in metric_dict:
        for class_name in metric_dict[epoch]:
            metric_dict[epoch][class_name]['delta_entropy'] = sum(metric_dict[epoch][class_name]['delta_entropy']) / len(metric_dict[epoch][class_name]['delta_entropy'])
            metric_dict[epoch][class_name]['divergence'] = sum(metric_dict[epoch][class_name]['divergence']) / len(metric_dict[epoch][class_name]['divergence'])
            metric_dict[epoch][class_name]['augmented_output_prob_top1'] = sum(metric_dict[epoch][class_name]['augmented_output_prob_top1']) / len(metric_dict[epoch][class_name]['augmented_output_prob_top1'])
    
    return metric_dict
