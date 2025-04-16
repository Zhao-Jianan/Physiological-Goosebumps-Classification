import json

def calculate_total_time_for_train(json_file_path):
    """
    Calculate the total training time from a JSON log file.

    Parameters:
    - json_file_path: Path to the JSON log file.

    Returns:
    - total_time: The total time accumulated for training entries.
    """
    
    total_time = 0.0 # Initialize the total time to 0

    # Open and read the JSON log file
    with open(file_path, 'r') as file:
        # Process each line in the log file
        for line in file:
            # Parse the JSON log entry
            log_entry = json.loads(line.strip())
            
            # Only process entries with 'train' mode
            if log_entry['mode'] == 'train':
                # Add the time for this log entry to the total time
                # Default to 0 if 'time' key is missing
                total_time += log_entry.get('time', 0)
    
    return total_time


