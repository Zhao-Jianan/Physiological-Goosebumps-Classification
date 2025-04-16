import os
import re
import cv2
import numpy as np
from itertools import groupby
from mmaction.apis import init_recognizer, inference_recognizer
from operator import itemgetter


def parse_filename(filename):
    """
    Parses the filename to extract details.

    Parameters:
    - filename (str): The video file name to parse.

    Returns:
    - dict: Parsed components including sequence, view, class, start frame,
      end frame, part (if available), and the original filename.
    - None: If the filename doesn't match the expected patterns.
    """
    
    # Pattern for filenames with part number
    pattern_with_part = r"(\d+)_([A-Za-z_]+)_class_(\d+)_segment_(\d+)_(\d+)_part_(\d+).mp4"
    # Pattern for filenames without part number
    pattern_without_part = r"(\d+)_([A-Za-z_]+)_class_(\d+)_segment_(\d+)_(\d+).mp4"
    
    # Try pattern with part number
    match = re.match(pattern_with_part, filename)
    if match:
        return {
            "sequence": int(match.group(1)),
            "view": match.group(2),
            "class": int(match.group(3)),
            "start_frame": int(match.group(4)),
            "end_frame": int(match.group(5)),
            "part": int(match.group(6)),
            "filename": filename
        }
    
    # Try pattern without part number
    match = re.match(pattern_without_part, filename)
    if match:
        return {
            "sequence": int(match.group(1)),
            "view": match.group(2),
            "class": int(match.group(3)),
            "start_frame": int(match.group(4)),
            "end_frame": int(match.group(5)),
            "part": None, 
            "filename": filename
        }
    
    # Return None if no pattern matches
    return None


def collect_videos_info(input_folder, exclude_folders):
    """
    Collects and parses information from video files in the specified folder, 
    excluding certain subfolders.

    Parameters:
    - input_folder (str): The directory to search for video files.
    - exclude_folders (list of str): List of subfolder names to exclude from the search.

    Returns:
    - list of dict: A list of dictionaries containing parsed video information, 
                    sorted by sequence, view, start frame, end frame, and part.
    """
    videos_info = []

    # Traverse the directory tree
    for root, dirs, files in os.walk(input_folder):
        # Exclude specified subfolders from the search
        dirs[:] = [d for d in dirs if d not in exclude_folders]

        # Process each file in the current directory
        for filename in files:
            # Check if the file is a video file
            if filename.endswith('.mp4'):
                # Parse the filename to extract information
                info = parse_filename(filename)
                if info:
                    # Add the class (folder name) to the information
                    info['class'] = os.path.basename(root)
                    # Append the information to the list
                    videos_info.append(info)

    # Sort the collected information
    videos_info.sort(key=lambda x: (x['sequence'], x['view'], x['start_frame'], 
                                    x['end_frame'], (x['part'] if x['part'] is not None else 0)))
    
    return videos_info


def process_videos(videos_info, input_folder, output_folder, model_task_normal, model_task_subclass, 
                   label_map_task_normal, label_map_task_subclass):
    """
    Processes video files by combining them into a single video for each sequence and view,
    and annotates the video frames with inference results and real labels.

    Parameters:
    - videos_info (list of dict): List of video file information including metadata.
    - input_folder (str): Directory where input video files are stored.
    - output_folder (str): Directory where combined and annotated video files will be saved.
    - model_task_normal (object): Model for the normal task inference.
    - model_task_subclass (object): Model for the subclass task inference.
    - label_map_task_normal (dict): Mapping from label indices to class names for the normal task.
    - label_map_task_subclass (dict): Mapping from subclass label indices to class names.
    """
    
    # Group videos by sequence number and view
    grouped_videos = groupby(sorted(videos_info, key=itemgetter('sequence', 'view')), 
                             key=lambda x: (x['sequence'], x['view']))
    for key, group in grouped_videos:
        group = list(group)
        if not group:
            continue

        # Get the path of the first video to determine video properties
        first_video_path = os.path.join(input_folder, group[0]['class'], group[0]['filename'])
        cap = cv2.VideoCapture(first_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Output file path
        output_file = os.path.join(output_folder, f'{key[0]}_{key[1]}_combined.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for video_info in group:
            video_path = os.path.join(input_folder, video_info['class'], video_info['filename'])

            # Perform inference with the normal Task Model
            result_task_normal = inference_recognizer(model_task_normal, video_path)
            pred_label_task_normal = result_task_normal.pred_label.item()
            pred_score_task_normal = result_task_normal.pred_score[pred_label_task_normal].item()

            # Perform inference with the subclass model if the normal task label is 1
            if pred_label_task_normal == 1:
                result_task_subclass = inference_recognizer(model_task_subclass, video_path)
                pred_label_task_subclass = result_task_subclass.pred_label.item()
                pred_score_task_subclass = result_task_subclass.pred_score[pred_label_task_subclass].item()
                label_task_subclass = label_map_task_subclass[pred_label_task_subclass]
            else:
                pred_label_task_subclass = pred_score_task_subclass = label_task_subclass = None

            # Read and process the video frames
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Real label
                real_label = int(video_info['class'].split('_')[-1])
                real_label_text = f'Real label: {label_map_task_normal[int(real_label==1)]}' \
                if real_label == 0 \
                else f'Real label: {label_map_task_subclass[real_label-1]}'

                # Real label position and color
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_size_real_label = cv2.getTextSize(real_label_text, font, font_scale, font_thickness)[0]
                text_x_real_label = (width - text_size_real_label[0]) // 2
                text_y_real_label = height - 80 
                cv2.putText(frame, real_label_text, (text_x_real_label, text_y_real_label), 
                            font, font_scale, (255, 255, 255), font_thickness)
                
                # Display Task Normal inference results
                text_task_normal = f'Inference: {label_map_task_normal[pred_label_task_normal]} (Pred score: {pred_score_task_normal:.2f})'
                text_color_normal = (0, 255, 0) \
                if (pred_label_task_normal == 0 and real_label == 0) \
                or (pred_label_task_normal == 1 and int(real_label) > 0) else (0, 0, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_size_task_normal = cv2.getTextSize(text_task_normal, font, font_scale, font_thickness)[0]
                text_x_task_normal = (width - text_size_task_normal[0]) // 2
                text_y_task_normal = height - 40

                cv2.putText(frame, text_task_normal, (text_x_task_normal, text_y_task_normal), 
                            font, font_scale, text_color_normal, font_thickness)

                # Display Task Subclass inference results
                if pred_label_task_subclass is not None:
                    text_task_subclass = f' {label_task_subclass} (Pred score: {pred_score_task_subclass:.2f})'
                    text_color_subclass = (0, 255, 0) if pred_label_task_subclass == (real_label - 1) else (0, 0, 255)
                    text_size_task_subclass = cv2.getTextSize(text_task_subclass, font, font_scale, font_thickness)[0]
                    text_x_task_subclass = (width - text_size_task_subclass[0]) // 2
                    text_y_task_subclass = height - 10

                    cv2.putText(frame, text_task_subclass, (text_x_task_subclass, text_y_task_subclass), 
                                font, font_scale, text_color_subclass, font_thickness)

                out.write(frame)

            cap.release()

        out.release()
        print(f'Output and saved video to: {output_file}')