import json

def add_mode_to_log(input_file, output_file):
    """
    Adds a 'mode' field to each JSON log entry in the input file and writes the updated entries to the output file.

    Parameters:
    input_file (str): Path to the input log file where each line is a JSON object.
    output_file (str): Path to the output file where the updated log entries will be saved.
    """

    # Read all lines from the input file
    with open(input_file, 'r') as file:
        log_lines = file.readlines()

    # Initialize a list to store the updated log entries
    updated_logs = []

    # Process each log entry
    for line in log_lines:
        # Load the JSON object from the line
        log_entry = json.loads(line.strip())

        # Check if 'epoch' key is present to determine the mode
        if 'epoch' in log_entry:
            log_entry['mode'] = 'train'  # Set mode to 'train' if 'epoch' is present
        else:
            log_entry['mode'] = 'val'  # Otherwise, set mode to 'val'
        
        # Append the updated log entry to the list
        updated_logs.append(log_entry)

    # Write the updated log entries to the output file
    with open(output_file, 'w') as file:
        for log_entry in updated_logs:
            # Convert the log entry back to a JSON string and write it to the file
            file.write(json.dumps(log_entry) + '\n')


def add_epoch_to_val_log(input_file, output_file):
    """
    Adds an 'epoch' field to each JSON log entry with 'mode' set to 'val' by copying the value from the 'step' field.
    The updated log entries are saved to a new output file.

    Parameters:
    input_file (str): Path to the input log file where each line is a JSON object.
    output_file (str): Path to the output file where the updated log entries will be saved.
    """

    # Read all lines from the input file
    with open(input_file, 'r') as file:
        log_lines = file.readlines()

    # Initialize a list to store the updated log entries
    updated_logs = []

    # Process each log entry
    for line in log_lines:
        # Load the JSON object from the line
        log_entry = json.loads(line.strip())

        # Check if the 'mode' is 'val'
        if log_entry['mode'] == 'val':
            # Add 'epoch' field with value copied from 'step'
            log_entry['epoch'] = log_entry['step']

        # Append the updated log entry to the list
        updated_logs.append(log_entry)

    # Write the updated log entries to the output file
    with open(output_file, 'w') as file:
        for log_entry in updated_logs:
            # Convert the log entry back to a JSON string and write it to the file
            file.write(json.dumps(log_entry) + '\n')


def update_log_file(input_file, output_file):
    """
    Update log file to add 'mode' and 'epoch' fields.
    
    Parameters:
    - input_file: Path to the input log file
    - output_file: Path to the output log file
    """

    # Read all lines from the input file
    with open(input_file, 'r') as file:
        log_lines = file.readlines()

    # Initialize a list to store the updated log entries
    updated_logs = []

    # Process each log entry
    for line in log_lines:
        # Load the JSON object from the line
        log_entry = json.loads(line.strip())

        # Add 'mode' field based on the presence of 'epoch'
        if 'epoch' in log_entry:
            log_entry['mode'] = 'train'
        else:
            log_entry['mode'] = 'val'
        
        # Add 'epoch' field to validation logs
        if log_entry['mode'] == 'val':
            log_entry['epoch'] = log_entry['step']

        # Append the updated log entry to the list
        updated_logs.append(log_entry)

    # Write the updated log entries to the output file
    with open(output_file, 'w') as file:
        for log_entry in updated_logs:
            # Convert the log entry back to a JSON string and write it to the file
            file.write(json.dumps(log_entry) + '\n')
    

