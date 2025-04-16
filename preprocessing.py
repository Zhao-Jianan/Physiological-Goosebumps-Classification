import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import random
from collections import defaultdict
from moviepy.editor import VideoFileClip, vfx


# Extract Frames
def extract_and_split_frames(path):
    """
    Extracts and splits frames from a video file into four parts: 
    left thigh, right thigh, dominant arm, and dominant calf.
    
    Parameters:
    path (str): The file path to the video.
    
    Returns:
    tuple: A tuple containing:
        - frames_count (int): The total number of frames in the video.
        - timestamps (list): A list of timestamps for each frame in seconds.
        - frames (list): A list containing four lists, each with frames for the 
                         respective body part (left thigh, right thigh, dominant arm, dominant calf).
    """
    # Open the video file
    cap = cv2.VideoCapture(path)

    # Get the total number of frames, frame width, and frame height
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the height and width for cropping frames
    split_height = height // 2
    # 735 is a suitable value after many attempts
    split_width = 735

    # List to store timestamps of each frame in seconds
    timestamps = []

    # Lists to store frames for each part
    left_thigh_frames = []
    right_thigh_frames = []
    dom_arm_frames = []
    dom_calf_frames = []

    # Iterate over each frame in the video
    for frame in range(frames_count):
        ret, frames = cap.read()
        if not ret:
           break

        # Get the timestamp of the current frame in seconds
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamps.append(timestamp)

        # Top left part of the frame (right thigh)
        top_left = frames[:split_height, :split_width]
        right_thigh_frames.append(top_left)

        # Top right part of the frame (dominant arm)
        top_right = frames[:split_height, split_width:2*split_width]
        dom_arm_frames.append(top_right)

        # Bottom left part of the frame (left thigh)
        bottom_left = frames[split_height:, :split_width]
        left_thigh_frames.append(bottom_left)

        # Bottom right part of the frame (dominant calf)
        bottom_right = frames[split_height:, split_width:2*split_width]
        dom_calf_frames.append(bottom_right)

    # Release the video capture object
    cap.release()

    # Combine all frames lists into a single list
    frames = [left_thigh_frames, right_thigh_frames, dom_arm_frames, dom_calf_frames]

    return frames_count,timestamps,frames


def load_and_parse_labels(label_file):
    """
    Loads and parses a label file to extract 'class' and 'state' columns 
    based on the 'Behavior' column.

    Parameters:
    label_file (str): The file path to the label file, which can be a CSV or XLSX file.

    Returns:
    pd.DataFrame: A DataFrame with the parsed label data, or an empty DataFrame if the 
                  file does not exist, is empty, or does not contain 'beginning' labels.
    """
    
    # Check if the label file exists
    if label_file == 'Not exist' or not os.path.exists(label_file):
        print(f"Label file {label_file} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist

    # Read the label file into a DataFrame based on its extension
    if label_file.endswith('.csv'):
        labels_df = pd.read_csv(label_file)
    elif label_file.endswith('.xlsx'):
        labels_df = pd.read_excel(label_file)
    else:
        raise ValueError(f"Unsupported file type: {label_file}")

    # Check if the DataFrame is empty
    if labels_df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty
    
    # Define the mapping from Behaviour to class and state
    behavior_to_class_state = {
        1: (1, 'beginning'),
        2: (1, 'peak'),
        3: (1, 'end'),
        4: (2, 'beginning'),
        5: (2, 'peak'),
        6: (2, 'end'),
        7: (3, 'beginning'),
        8: (3, 'peak'),
        9: (3, 'end')
    }

    # Parse the Behaviour column to map to 'class' and 'state' columns
    labels_df['class'] = labels_df['Behavior'].apply(lambda x: behavior_to_class_state[x][0])
    labels_df['state'] = labels_df['Behavior'].apply(lambda x: behavior_to_class_state[x][1])

    # Check if there are any 'beginning' labels
    if not any(labels_df['state'] == 'beginning'):
        return pd.DataFrame()  # Return an empty DataFrame if no 'beginning' labels are found

    return labels_df


def match_frames_with_labels(frames, timestamps, labels_df):
    """
    Matches frames with labels based on their timestamps and the given labels DataFrame.
    
    Parameters:
    frames (list): List of frames extracted from the video.
    timestamps (list): List of timestamps corresponding to each frame.
    labels_df (pd.DataFrame): DataFrame containing the labels with 'Time', 'class', and 'state' columns.
    
    Returns:
    tuple: Two lists - frame_classes and frame_states, indicating the class and state of each frame.
    """

    # Initialise lists to store the class and state
    frame_classes = [0] * len(frames)
    frame_states = [0] * len(frames)

    # Check if the labels DataFrame is empty
    if labels_df.empty:
        return frame_classes, frame_states  # Return all zeros if no valid labels are found


    # Iterate over each row in the labels DataFrame
    for i, row in labels_df.iterrows():
        behavior_time = row['Time']
        behavior_class = row['class']
        behavior_state = row['state']

        # Find the start and end timestamps
        if behavior_state == 'beginning':
            start_time = behavior_time
            end_time = labels_df.loc[(labels_df['class'] == behavior_class) & (labels_df['state'] == 'end') & (labels_df['Time'] > behavior_time), 'Time'].min()

            # If no corresponding end time is found, use the last time point of that class
            if pd.isna(end_time):
                end_time = labels_df.loc[(labels_df['class'] == behavior_class) & (labels_df['Time'] > behavior_time), 'Time'].max()

            # Assign class and state to frames based on the time interval
            for j, timestamp in enumerate(timestamps):
                if start_time <= timestamp <= end_time:
                    frame_classes[j] = behavior_class
                    frame_states[j] = 'intermediate' if timestamp != start_time and timestamp != end_time else behavior_state

    return frame_classes, frame_states


def save_video_segments(frames, frame_classes, file_index, camera_name):
    """
    Saves video segments based on frame classes into separate directories for each class.
    
    Parameters:
    frames (list): List of frames extracted from the video.
    frame_classes (list): List of classes corresponding to each frame.
    file_index (int): Index to be used in the output video file names.
    camera_name (str): Name of the camera, used to create a directory for the output videos.
    """

    # Base directory to save the video segments
    base_output_dir = 'Goosebumps_videos_multi_task'
    camera_dir = os.path.join(base_output_dir, camera_name)
    os.makedirs(camera_dir, exist_ok=True)

    # Drop the first two frames
    frames = frames[2:]
    frame_classes = frame_classes[2:]

    current_class = None
    start_frame_idx = 0

    # Iterate over each frame class
    for i, frame_class in enumerate(frame_classes):
        # Check if the current frame class is different from the previous one
        if frame_class != current_class:
            if current_class is not None:
                # Save the previous segment, split into chunks of 10 frames
                segment_frames = frames[start_frame_idx:i]
                class_dir = os.path.join(camera_dir, f'class_{current_class}')
                os.makedirs(class_dir, exist_ok=True)

                for j in range(0, len(segment_frames), 10):
                    segment_chunk = segment_frames[j:j + 10]
                    segment_video_path = os.path.join(class_dir, f'{file_index}_class_{current_class}_segment_{start_frame_idx + j + 2}_{start_frame_idx + j + len(segment_chunk) + 2}_part_{j // 10}.mp4')
                    # Create a VideoWriter object to write the segment video
                    out = cv2.VideoWriter(segment_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (735, 540))
                    for frame in segment_chunk:
                        out.write(frame)
                    out.release()

            # Update to the new class segment
            current_class = frame_class
            start_frame_idx = i

    # Save the last segment, split into chunks of 10 frames
    if current_class is not None:
        class_dir = os.path.join(camera_dir, f'class_{current_class}')
        os.makedirs(class_dir, exist_ok=True)
        segment_frames = frames[start_frame_idx:]

        for j in range(0, len(segment_frames), 10):
            segment_chunk = segment_frames[j:j + 10]
            segment_video_path = os.path.join(class_dir, f'{file_index}_class_{current_class}_segment_{start_frame_idx + j + 3}_{start_frame_idx + j + len(segment_chunk) + 3}.mp4')
            # Create a VideoWriter object to write the segment video
            out = cv2.VideoWriter(segment_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (735, 540))
            for frame in segment_chunk:
                out.write(frame)
            out.release()


def process_videos(frames, timestamps, label_csv, file_name_prefix, camera_name):
    """
    Processes videos by loading labels, matching frames with labels, and saving segmented videos.
    
    Parameters:
    frames (list): List of frames extracted from the video.
    timestamps (list): List of timestamps corresponding to each frame.
    label_csv (str): The file path to the label CSV file.
    file_name_prefix (str): Prefix to be used in the output video file names.
    camera_name (str): Name of the camera, used to create a directory for the output videos.
    """
    # Load and parse the labels from the CSV file
    labels = load_and_parse_labels(label_csv)
    # Match frames with the parsed labels to get frame classes and states
    frame_classes, frame_states = match_frames_with_labels(frames, timestamps, labels)
    # Save video segments based on frame classes
    save_video_segments(frames, frame_classes, file_name_prefix, camera_name)


def process_videos_in_directory(directory):
    """
    Processes all video files in the specified directory by extracting frames,
    loading labels, matching frames with labels, and saving segmented videos.
    
    Parameters:
    directory (str): The path to the directory containing video files.
    """
    
    # List all files in the directory
    all_files = os.listdir(directory)

    for file_name in all_files:
        # Construct the full path of the file
        file_path = os.path.join(directory, file_name)

        # Check if the file is a video file
        if file_name.endswith('.mp4'):
            print(f'Processing video: {file_path}')

            # Call the extract_and_split_frames function
            frames_count, timestamps, frames = extract_and_split_frames(file_path)
            print(f'Frames Count: {frames_count}')
            
            # Prepare parameters for further processing
            video_index = file_name[:3]
            video_filename = f'{video_index}-Both'
            base_path = './Data/Manual video codes'
            csv_paths = ['Left thigh', 'Right thigh', 'Dom arm', 'Dorm calf']
            file_name_prefix = ['Left_thigh', 'Right_thigh', 'Dom_arm', 'Dom_calf']
            
            for i in range(4):
                # Try to find the corresponding .csv or .xlsx file
                csv_file_path = os.path.join(base_path, csv_paths[i], f'{video_filename}.csv')
                xlsx_file_path = os.path.join(base_path, csv_paths[i], f'{video_filename}.xlsx')

                if os.path.exists(csv_file_path):
                    label_csv = csv_file_path
                elif os.path.exists(xlsx_file_path):
                    label_csv = xlsx_file_path
                else:
                    print(f'No label file found for {csv_paths[i]} in {video_filename}, Set label for no goosebumps!')
                    label_csv = 'Not exist'

                # Save video segments
                file_name = f'{video_index}_{file_name_prefix[i]}'
                print(f'label_csv: {label_csv}')
                process_videos(frames[i], timestamps, label_csv, file_name, file_name_prefix[i])
    print('Process End')


# Data Augment
def spatial_augmentation(video_path, augment_dir, count):
    """
    Performs spatial augmentations on a video by randomly mirroring and rotating it, then saves the augmented video.
    
    Parameters:
    video_path (str): The path to the original video file.
    augment_dir (str): The directory where the augmented video will be saved.
    count (int): A unique count or index to differentiate the augmented video file name.
    
    Returns:
    str: The path to the augmented video file.
    
    Raises:
    FileNotFoundError: If the video file does not exist.
    """

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")
        
    # Load the video file
    clip = VideoFileClip(video_path)  

    # Randomly apply horizontal and vertical mirroring with 50% probability
    if random.random() < 0.5:
        clip = clip.fx(vfx.mirror_x)
    if random.random() < 0.5:
        clip = clip.fx(vfx.mirror_y)

    # Randomly choose a rotation angle from [0, 90, 180, 270] degrees
    rotation_angle = random.choice([0, 90, 180, 270])
    if rotation_angle != 0:
        clip = clip.rotate(rotation_angle)

    # Construct and save the augmented video file
    video_name = os.path.basename(video_path).replace('.mp4', f'_augmented_{count}.mp4')
    augmented_video_path = os.path.join(augment_dir, video_name)
    clip.write_videofile(augmented_video_path, codec='mpeg4')
    
    return augmented_video_path


def augment_videos_to_target_count(video_list, target_count, augment_dir, video_dir):
    """
    Augments videos until the target count is reached by applying spatial augmentations to random videos from the list.
    
    Parameters:
    video_list (list): List of original video file names.
    target_count (int): The desired total number of videos after augmentation.
    augment_dir (str): Directory where the augmented videos will be saved.
    video_dir (str): Directory containing the original video files.
    
    Returns:
    list: List of video file paths (including augmented videos) up to the target count.
    """

    # Copy the original video list to the augmented list
    augmented_list = video_list.copy()

    # Create the augmentation directory if it doesn't exist
    if not os.path.exists(augment_dir):
        os.makedirs(augment_dir)
    count = 1
    # Loop until the augmented list reaches the target count
    while len(augmented_list) < target_count:
        # Randomly select a video from the original list
        video = random.choice(video_list)
        video_path = os.path.join(video_dir, video)

        # Apply spatial augmentation to the selected video
        augmented_video = spatial_augmentation(video_path, augment_dir, count)

        # Add the path of the augmented video to the augmented list
        augmented_list.append(augmented_video)
        
        count += 1
        
    return augmented_list


def augment_classes_to_max_samples(video_dir, classes):
    """
    Augments video samples in each class directory to match the maximum sample count among all classes.
    
    Parameters:
    video_dir (str): The directory containing class subdirectories with video files.
    classes (list): List of class subdirectories to process.
    """

    # Dictionary to map each class to its list of video files
    class_dirs = classes
    class_video_map = defaultdict(list)

    # Collect all video files and their respective classes
    for class_dir in class_dirs:
        class_path = os.path.join(video_dir, class_dir)
        if os.path.exists(class_path):
            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    class_video_map[class_dir].append(os.path.join(class_dir, video_file))
    
    # Find the maximum number of samples among all classes
    max_samples = max(len(videos) for videos in class_video_map.values())

    # Augment data for all classes to have the same number of samples as the class with the maximum samples
    for class_dir in class_dirs:
        videos = class_video_map[class_dir]
        if len(videos) < max_samples:
            augment_dir = os.path.join(video_dir, f'{class_dir}_augment')
            augment_videos_to_target_count(videos, max_samples, augment_dir,video_dir)


# Split Dataset
def split_data(video_dir, task, train_ratio=0.8, val_ratio=0.1):
    """
    Splits video files into training, validation, and test sets based on the provided ratios.
    
    Parameters:
    video_dir (str): Directory containing class subdirectories with video files.
    task (str): Task type for splitting ('normal' or 'subclass').
    train_ratio (float): Ratio of data to be used for training (default is 0.8).
    val_ratio (float): Ratio of data to be used for validation (default is 0.1).
    
    Returns:
    tuple: Three lists containing (video_file, class_dir) tuples for training, validation, and test sets.
    """

    # List of class directories
    class_dirs = ['class_0', 'class_1', 'class_2', 'class_3']
    # Create corresponding augment directories
    class_augment_dirs = [f'{d}_augment' for d in class_dirs]
    # Dictionary to map class directories to lists of video file names
    class_video_map = defaultdict(list)

    # Collect all video file names and their classes
    for class_dir in class_dirs:
        class_path = os.path.join(video_dir, class_dir)
        if os.path.exists(class_path):
            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    class_video_map[class_dir].append(video_file)

    # Collect video file names from augmentation directories
    for augment_dir in class_augment_dirs:
        augment_path = os.path.join(video_dir, augment_dir)
        if os.path.exists(augment_path):
             class_idx = class_dirs.index(augment_dir.replace('_augment', ''))
             class_dir = class_dirs[class_idx]
             for video_file in os.listdir(augment_path):
                  if video_file.endswith('.mp4'):
                     class_video_map[class_dir].append(video_file)
                    
    # Shuffle the video lists for each class
    for class_dir in class_dirs:
        random.shuffle(class_video_map[class_dir])

    # Initialize lists to hold training, validation, and test data
    train_data = []
    val_data = []
    test_data = []

    # Determine the maximum number of videos across all classes
    max_samples = max(len(class_video_map['class_1']), len(class_video_map['class_2']), len(class_video_map['class_3']))

    # Calculate the total number of samples (excluding 'class_0')
    normal_samples = sum(len(class_video_map[class_dir]) for class_dir in class_dirs if class_dir != 'class_0')

    # Process each class directory
    for class_dir in class_dirs:
        videos = class_video_map[class_dir]
        total_videos = len(videos)
        target_samples = 0
        
        if task == 'normal':
            if class_dir == 'class_0':
                target_samples = normal_samples
            else:
                target_samples = total_videos
        elif task == 'subclass':
            target_samples = max_samples
        else:
            raise ValueError(f"Unsupported task: {task}")  

        # Calculate indices for splitting the data into training, validation, and test sets
        train_end_idx = int(target_samples * train_ratio)
        val_end_idx = train_end_idx + int(target_samples * val_ratio)
        test_end_idx = target_samples

        # Split the data and add to respective lists
        train_data.extend([(video, class_dir) for video in videos[:train_end_idx]])
        val_data.extend([(video, class_dir) for video in videos[train_end_idx:val_end_idx]])
        test_data.extend([(video, class_dir) for video in videos[val_end_idx:test_end_idx]])

    # Shuffle the resulting datasets to ensure randomness
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def save_to_file(data, file_path, task, camera_dir, mode='w'):
    """
    Saves video file paths and their corresponding class indices to a text file.
    
    Parameters:
    data (list of tuples): Each tuple contains a video file name and its class directory.
    file_path (str): Path to the file where the data will be saved.
    task (str): The type of task for class index determination ('normal' or 'subclass').
    camera_dir (str): The base directory for the camera-related video files.
    mode (str): File opening mode ('w' for write, 'a' for append). Default is 'w'.
    """
    
    with open(file_path, mode) as f:
        for video_file, class_dir in data:
            class_idx = None

            # Skip augmentation folders for non-augmentation tasks
            if task in ['normal', 'subclass']:
                if 'augment' in class_dir:
                    continue
            
            if task == 'normal':
                if class_dir == 'class_0':
                    class_idx = 0  # Assign index 0 for 'class_0'
                else:
                    class_idx = 1  # Assign index 1 for other classes
            elif task == 'subclass':
                if class_dir != 'class_0':
                    class_idx = int(class_dir[-1]) -1   # Determine index based on class suffix           
            else:
                raise ValueError(f"Unsupported task: {task}") # Raise an error for unsupported tasks

            # Process video file path and class index
            if class_idx is not None:
                if 'augmented' in video_file:
                    class_dir += '_augment'

                # Construct the relative path for the video file
                camera_class_path = os.path.join(camera_dir, class_dir)
                relative_path = os.path.join(camera_class_path, video_file)

                # Write the relative path and class index to the file
                f.write(f'{relative_path} {class_idx}\n')


def shuffle_labels(file_path):
    """
    Shuffles the lines of a text file in place.
    
    Parameters:
    file_path (str): Path to the text file whose lines will be shuffled.
    """
    
    # Read all lines from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Shuffle the list of lines randomly
    random.shuffle(lines)
    
    # Write the shuffled lines back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)