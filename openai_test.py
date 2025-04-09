import os
import json
import re
import cv2
import base64
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap # Not used, can be removed
import matplotlib.gridspec as gridspec # Not used, can be removed
from collections import Counter # Not used, can be removed
from IPython.display import Image, display # Not used in script execution, typically for notebooks
from pydantic import BaseModel # Not used, can be removed

# Import OpenAI client (requires 'openai' package installed: pip install openai)
from openai import OpenAI

warnings.filterwarnings("ignore")

# ------------------------ Global Configuration ------------------------
# --- User Configuration Required ---
# Please set the OPENAI_API_KEY environment variable before running the script.
# Optionally, set OPENAI_BASE_URL if using a custom endpoint.
# Adjust the paths below to match your environment.

MODEL_NAME = "gpt-4o"  # Model name (used for processing and analysis identification)
OUTPUT_DIR = "./output_results"  # Directory for intermediate and final JSON results
# Ensure this path points to your input Parquet file
INPUT_PARQUET_FILE = "./data/qa.parquet" # Example path, change as needed
# Root directory where your video files are stored
VIDEO_ROOT_DIR = "./data/video" # Example path, change as needed

# --- Script Configuration ---
MAX_WORKERS = 4  # Number of concurrent threads for processing
MAX_API_RETRIES = 3 # Number of retries for OpenAI API calls
MAX_NORMALIZE_RETRIES = 3 # Number of retries for normalizing the answer
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.json")  # Final results file path
DATA_ROOT_DIR = OUTPUT_DIR  # Directory used during the analysis phase

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Ensure the example data directories exist if using defaults, or instruct user to create them
# os.makedirs(os.path.dirname(INPUT_PARQUET_FILE), exist_ok=True) # Optional: create data dir
# os.makedirs(VIDEO_ROOT_DIR, exist_ok=True) # Optional: create video dir

# ------------------------ OpenAI Client Initialization ------------------------
# The client automatically picks up OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.
# Ensure OPENAI_API_KEY is set.
print("Initializing OpenAI client...")
try:
    client = OpenAI()
    # You can optionally test the connection here if needed, e.g., by listing models
    # client.models.list()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure the OPENAI_API_KEY environment variable is set correctly.")
    print("If using a custom endpoint, also set OPENAI_BASE_URL.")
    exit(1) # Exit if client fails to initialize

# ------------------------ Utility Functions (Core Logic & Normalization) ------------------------
def make_key(item: dict) -> str:
    """
    Generates a unique key based on filename, ID, start time, and end time
    to identify if a record has already been processed.
    """
    # Assuming 'Video' key exists in the input item dictionary from the parquet file
    return f"{item.get('Video', 'UnknownVideo')}|{item.get('ID', 'UnknownID')}|{item.get('time_start', 'NaN')}|{item.get('time_end', 'NaN')}"

def get_frame(video_path):
    """
    Extracts frames from the specified video path, sampling up to 30 frames.
    Frames are converted to JPEG format and Base64 encoded.
    Returns: Tuple (list of base64_frames, sampled_fps)
    """
    base64_frames = []
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
        return [], 0.0 # Return empty list and 0 fps if video not found

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_num / fps if fps > 0 else 0
    max_frames = 30

    # Dynamically calculate sampling rate to extract at most max_frames images
    target_fps = max_frames / duration if duration > 0 else fps
    sample_interval = int(fps / target_fps) if target_fps > 0 else 1
    sample_interval = max(1, sample_interval) # Ensure interval is at least 1
    sample_fps = fps / sample_interval if sample_interval > 0 else fps

    sampled_indices = set(range(0, total_frame_num, sample_interval))
    if len(sampled_indices) > max_frames:
        # If interval calculation still yields too many frames, take the first max_frames
        sampled_indices = set(sorted(list(sampled_indices))[:max_frames])

    current_frame_index = 0
    frames_extracted = 0
    while frames_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if current_frame_index in sampled_indices:
            # Convert BGR to RGB, encode as JPEG, then Base64
            try:
                # Ensure frame is not empty
                if frame is None or frame.size == 0:
                    print(f"Warning: Read an empty frame at index {current_frame_index} from {video_path}")
                    current_frame_index += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                is_success, buffer = cv2.imencode(".jpeg", rgb_frame)
                if is_success:
                    base64_data = base64.b64encode(buffer).decode("utf-8")
                    base64_frames.append(base64_data)
                    frames_extracted += 1
                else:
                     print(f"Warning: Failed to encode frame at index {current_frame_index} from {video_path}")

            except cv2.error as e:
                print(f"Warning: OpenCV error processing frame {current_frame_index} from {video_path}: {e}")
            except Exception as e:
                print(f"Warning: Unexpected error processing frame {current_frame_index} from {video_path}: {e}")


        current_frame_index += 1

    cap.release()
    # print(f"Extracted {len(base64_frames)} frames from {video_path} at ~{sample_fps:.2f} FPS.")
    return base64_frames, sample_fps

def normalize_answer(text: str) -> str | None:
    """
    Attempts to extract an uppercase letter answer (A-E) from the text
    and formats it as "Ans='X'". Returns None if no match is found.
    """
    if not isinstance(text, str): # Handle potential non-string input
        return None

    # Patterns to find the single letter answer (A, B, C, D, or E)
    # Prioritize simpler patterns first.
    patterns = [
        r"^\s*([A-E])\s*$",              # Line contains only the letter (potentially with whitespace)
        r"Ans='([A-E])'",             # Explicit Ans='X' format
        r"Answer:\s*([A-E])",          # Answer: X format
        r"Option\s+([A-E])",           # Option X format
        r"chosen_option\s*=\s*['\"]([A-E])['\"]", # Python-like variable assignment
        r'\b([A-E])\.',                # Letter followed by a period (e.g., "A.")
        r'The correct option is\s*([A-E])', # Specific phrasing
        # Add more patterns here if needed based on observed outputs
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE) # Make case-insensitive and check across lines
        if match:
            return f"Ans='{match.group(1).upper()}'" # Return in standardized uppercase format

    # Fallback: If no specific pattern matches, check if the entire response is just a single letter A-E
    if len(text.strip()) == 1 and text.strip().upper() in "ABCDE":
        return f"Ans='{text.strip().upper()}'"

    return None # No matching answer found

# Thread lock for safely writing to the results file
write_lock = threading.Lock()

def process_item(item, index, results_dict):
    """
    Processes a single record:
    1. Generates a key to check if already processed.
    2. Constructs the full video path.
    3. Extracts frames from the video.
    4. Builds the prompt with the question and candidate answers.
    5. Calls the OpenAI API (with retries up to MAX_API_RETRIES) to get the answer.
    6. Normalizes the API response (with retries up to MAX_NORMALIZE_RETRIES).
       If normalization fails after retries, marks as "NORMALIZE_ERROR".
       If API call fails, marks as "ERROR_CALL_API".
    7. Organizes the result into a dictionary.
    8. Saves the updated results dictionary to the JSON file (thread-safe).
    """
    key = make_key(item)
    # Construct video path using VIDEO_ROOT_DIR and the 'Video' field from the item
    video_filename = item.get('Video')
    if not video_filename:
        print(f"[{index}] Error: Missing 'Video' field in item. Skipping.")
        return None # Cannot process without video filename

    video_path = os.path.join(VIDEO_ROOT_DIR, video_filename)
    print(f"[{index}] Processing video: {video_path}")

    # Extract frames
    base64_frames, sample_fps = get_frame(video_path)
    if not base64_frames:
        print(f"[{index}] Warning: No frames extracted from {video_path}. Skipping API call.")
        # Decide how to handle this: skip, or mark as an error? Let's mark as an error.
        model_output = "ERROR_NO_FRAMES"
        normalized = model_output # Keep the error status
    else:
        # Prepare frames for the API request
        frames_payload = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                     "detail": "low" # Use low detail for potentially faster processing and lower cost
                }
            }
            for frame in base64_frames
        ]

        # Generate candidate answer list, e.g., "A xxx", "B yyy"
        # Ensure 'Candidates' is a dictionary
        candidates = item.get("Candidates", {})
        if isinstance(candidates, dict):
            candidate_list = [f"{k}. {v}" for k, v in candidates.items()] # Use ". " for clarity
        else:
            print(f"[{index}] Warning: 'Candidates' field is not a dictionary. Using empty list.")
            candidate_list = []

        # Construct the question part of the prompt
        time_start = item.get('time_start', 'N/A')
        time_end = item.get('time_end', 'N/A')
        question_text = item.get('Question', 'No question provided.')

        # Consolidate question details
        que = (
            f"Observe the video frames, focusing on the time segment from {time_start} to {time_end} seconds.\n"
            f"Question: {question_text}\n"
            "Options:\n" + "\n".join(candidate_list) + "\n"
            "Based on the visual information, which option is the most accurate answer? "
            "Please provide only the letter of the correct option (A, B, C, D, or E)."
        )

        # Define the system prompt (optional, but can help guide the model)
        system_prompt = f"""You are an AI assistant specialized in analyzing video frames to answer multiple-choice questions.
Carefully observe the provided frames, which were sampled at approximately {sample_fps:.2f} FPS.
Focus on the relevant time period and visual details to answer the question accurately.
Your response should consist ONLY of the single capital letter corresponding to the best option (e.g., 'A', 'B', 'C', 'D', 'E'). Do not include explanations or any other text."""

        # Prepare messages for the API
        prompt_messages = [
             # { # Optional System Prompt - uncomment if desired
             #    "role": "system",
             #    "content": system_prompt
             # },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": que} # 'que' now contains the full context
                ] + frames_payload # Add the image frames
            }
        ]

        params = {
            "model": MODEL_NAME,
            "messages": prompt_messages,
            "max_tokens": 10,  # Restrict response length, expecting just 'A', 'B', etc.
            "temperature": 0.0, # Set temperature to 0 for deterministic output
        }

        # Call API with retries
        api_response_content = None
        for attempt in range(MAX_API_RETRIES):
            try:
                print(f"[{index}] Calling OpenAI API (Attempt {attempt + 1}/{MAX_API_RETRIES})...")
                completion = client.chat.completions.create(**params)
                api_response_content = completion.choices[0].message.content
                print(f"[{index}] API call successful.")
                break # Exit loop on success
            except Exception as e:
                print(f"[{index}] API Error (Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    print(f"[{index}] API call failed after {MAX_API_RETRIES} attempts.")
                    api_response_content = "ERROR_CALL_API" # Mark as API error

        # Normalize the response if API call was successful
        if api_response_content != "ERROR_CALL_API":
            normalized = None
            raw_response_for_log = api_response_content # Keep raw response for logging if needed
            for norm_attempt in range(MAX_NORMALIZE_RETRIES):
                normalized = normalize_answer(api_response_content)
                if normalized is not None:
                    # print(f"[{index}] Normalization successful: '{raw_response_for_log}' -> '{normalized}'")
                    break # Exit loop on successful normalization
                else:
                    # print(f"[{index}] Normalization attempt {norm_attempt + 1}/{MAX_NORMALIZE_RETRIES} failed for response: '{raw_response_for_log}'. Retrying...")
                    if norm_attempt < MAX_NORMALIZE_RETRIES - 1:
                        time.sleep(1) # Short delay before retry
                    else:
                         print(f"[{index}] Normalization failed after {MAX_NORMALIZE_RETRIES} attempts for response: '{raw_response_for_log}'. Marking as error.")
                         normalized = "NORMALIZE_ERROR" # Mark as normalization error
            model_output = normalized # Use the normalized result or error code
        else:
            model_output = api_response_content # Keep the API error code

    # Prepare the output item dictionary
    # Map fields from input 'item' to desired output structure
    out_item = {
        "id": item.get("file", item.get("Video", "UnknownFile")), # Use 'file' if available, else 'Video'
        "id_file": item.get("ID", "UnknownID"),
        "time_s": item.get("time_start", None),
        "time_e": item.get("time_end", None),
        "model_out": model_output, # This is the normalized answer or error code
        "ans": item.get("Answer", None), # Ground truth answer
        "class": item.get("Task", "UnknownTask"), # Keep 'class' for compatibility? Or rename to 'Task'? Let's keep 'Task' consistent.
        "Task": item.get("Task", "UnknownTask"),
        "que": que, # Store the full question context sent to the model
        "Source": item.get("Source", "UnknownSource"),
        # Optionally add raw model output if needed for debugging
        # "raw_model_output": raw_response_for_log if 'raw_response_for_log' in locals() else None
    }

    # Write result to the shared dictionary and save to file (thread-safe)
    with write_lock:
        results_dict[key] = out_item
        # Save snapshot after each item - can be slow but ensures data persistence
        all_data = list(results_dict.values())
        try:
            with open(OUTPUT_FILEPATH, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"[{index}] Error writing results to {OUTPUT_FILEPATH}: {e}")
        except Exception as e:
            print(f"[{index}] Unexpected error saving results: {e}")

    print(f"[{index}] Done. ID: {out_item['id_file']}, Result: {model_output}, Ground Truth: {out_item['ans']}")
    return out_item


def processing_main():
    """
    Main function for data processing:
    - Loads data from the input Parquet file.
    - Loads existing results if available to avoid reprocessing.
    - Filters records that haven't been processed or resulted in errors previously.
    - Uses a ThreadPoolExecutor to process records concurrently.
    """
    parquet_file = INPUT_PARQUET_FILE
    try:
        print(f"Loading data from Parquet file: {parquet_file}")
        df = pd.read_parquet(parquet_file, engine='pyarrow') # Specify engine if needed
        # Convert DataFrame to a list of dictionaries
        list_of_items = df.to_dict(orient='records')
        print(f"Successfully loaded {len(list_of_items)} records.")
    except FileNotFoundError:
        print(f"Error: Input Parquet file not found at {parquet_file}")
        print("Please ensure the file exists and INPUT_PARQUET_FILE is set correctly.")
        return # Exit if data cannot be loaded
    except Exception as e:
        print(f"Error reading Parquet file {parquet_file}: {e}")
        return # Exit on other read errors

    # Load existing results if the output file exists
    results_dict = {}
    if os.path.exists(OUTPUT_FILEPATH):
        print(f"Loading existing results from: {OUTPUT_FILEPATH}")
        try:
            with open(OUTPUT_FILEPATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Rebuild the dictionary using the unique key
            for record in existing_data:
                # Reconstruct the key based on the fields in the JSON record
                # Ensure keys match those used in make_key
                k = f"{record.get('id', 'UnknownVideo')}|{record.get('id_file', 'UnknownID')}|{record.get('time_s', 'NaN')}|{record.get('time_e', 'NaN')}"
                results_dict[k] = record
            print(f"Loaded {len(results_dict)} existing results.")
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding JSON from {OUTPUT_FILEPATH}. Starting with empty results. Error: {e}")
            results_dict = {} # Reset if file is corrupt
        except Exception as e:
            print(f"Warning: Error loading existing results from {OUTPUT_FILEPATH}: {e}. Starting with empty results.")
            results_dict = {} # Reset on other errors

    # Filter items that need processing
    items_to_process = []
    processed_keys = set(results_dict.keys())
    error_statuses = {"ERROR_CALL_API", "NORMALIZE_ERROR", "ERROR_NO_FRAMES"} # Statuses indicating reprocessing might be needed

    for index, item in enumerate(list_of_items):
        key = make_key(item)
        if key not in processed_keys:
            items_to_process.append((index, item)) # Add new items
        else:
            # Check if existing result was an error that we want to retry
            existing_result = results_dict[key]
            if existing_result.get("model_out") in error_statuses:
                print(f"Retrying item {index} (key: {key}) due to previous error: {existing_result.get('model_out')}")
                items_to_process.append((index, item)) # Add items with previous errors for reprocessing

    if not items_to_process:
        print("No new items to process. All records are up-to-date based on existing results.")
        return # Exit if nothing to do

    print(f"Found {len(items_to_process)} items to process (including retries for errors).")

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Store futures to track progress and results
        future_to_index = {executor.submit(process_item, item_data, item_index, results_dict): item_index for item_index, item_data in items_to_process}

        processed_count = 0
        total_items = len(items_to_process)
        for future in as_completed(future_to_index):
            item_idx = future_to_index[future]
            try:
                # Retrieve result (optional, main work is done in process_item including saving)
                result = future.result()
                processed_count += 1
                # Optional: Log success based on result if needed
                # if result:
                #     print(f"Successfully processed item {item_idx}.")
                # else:
                #     print(f"Item {item_idx} processing returned None (potentially skipped).")

            except Exception as e:
                processed_count += 1
                print(f"Error processing item at index {item_idx}: Task generated an exception: {e}")
                # Optionally update the results_dict with an error status here if not handled in process_item
                # Example: Find the key for item_idx and update its status in results_dict

            print(f"Progress: {processed_count}/{total_items} items completed.")


    print(f"\nProcessing finished! All tasks completed.")
    print(f"The final results have been saved to: {OUTPUT_FILEPATH}")

# ------------------------ Analysis and Visualization Phase ------------------------

def get_input_file_path():
    """Returns the path to the JSON file generated by the processing phase."""
    return os.path.join(DATA_ROOT_DIR, f"{MODEL_NAME}.json")

def get_output_dir():
    """Creates and returns the directory path for saving analysis results."""
    output_dir = f"./analysis_results_{MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data():
    """Loads the processed data from the JSON file."""
    file_path = get_input_file_path()
    if not os.path.exists(file_path):
        print(f"Error: Analysis input file not found: {file_path}")
        print("Please run the processing phase first to generate the results file.")
        return None # Return None if file doesn't exist
    print(f"Loading analysis data from {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} records for analysis.")
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {file_path}. Error: {e}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def scene_type(id_str):
    """
    Determines the scene type based on the 'id' (filename) string:
      - Contains 'camera' -> 'outdoor'
      - Starts with 'scene' -> 'indoor'
      - Is a 6-digit number (before extension) -> 'desktop'
      - Otherwise -> 'unknown'
    """
    if not isinstance(id_str, str):
        return 'unknown'

    filename = os.path.splitext(id_str)[0] # Get filename without extension

    if 'camera' in id_str.lower(): # Case-insensitive check
        return 'outdoor'
    elif id_str.lower().startswith('scene'): # Case-insensitive check
        return 'indoor'
    # Check if the filename part is exactly 6 digits
    elif filename.isdigit() and len(filename) == 6:
        return 'desktop'
    else:
        # Add more rules here if needed, e.g., based on other patterns
        return 'unknown'

def plot_radar_chart(ax, data_values, task_labels, title, task_counts=None):
    """
    Draws a radar chart on the given matplotlib axes (polar projection).
      - Filters out tasks with zero samples if task_counts is provided.
      - Labels data points with accuracy percentages.
    """
    if data_values is None or task_labels is None:
         print(f"Warning: Missing data or labels for radar chart '{title}'. Skipping plot.")
         ax.text(0.5, 0.5, "Data Unavailable",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)
         ax.set_title(title, size=14, color='grey', y=1.1)
         return

    filtered_tasks = []
    filtered_values = []
    # Filter based on counts if provided
    if task_counts is not None and len(task_counts) == len(task_labels):
        for label, value, count in zip(task_labels, data_values, task_counts):
            if count > 0:
                filtered_tasks.append(label)
                filtered_values.append(value)
            # else: # Optional: print which tasks are skipped
            #     print(f"Skipping task '{label}' in '{title}' radar chart due to zero samples.")
    else:
        # If no counts provided or mismatch, use all data (assume non-zero counts)
        filtered_tasks = task_labels
        filtered_values = data_values

    if not filtered_tasks:
        ax.text(0.5, 0.5, "No data with samples > 0",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(title, size=14, color='blue', y=1.1)
        # print(f"No tasks with data to plot for radar chart '{title}'.")
        return

    num_tasks = len(filtered_tasks)
    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()

    # The plot is circular, so we need to close the loop
    values_closed = filtered_values + [filtered_values[0]]
    angles_closed = angles + [angles[0]]

    # Plotting
    ax.plot(angles_closed, values_closed, marker='o', linewidth=2, label='Accuracy')
    ax.fill(angles_closed, values_closed, alpha=0.3)

    # Set ticks and labels
    ax.set_xticks(angles)
    ax.set_xticklabels(filtered_tasks, fontsize=9)
    ax.set_yticks(np.linspace(0, 1.0, 6)) # Ticks from 0.0 to 1.0
    ax.set_yticklabels([f"{i:.0%}" for i in np.linspace(0, 1.0, 6)])
    ax.set_ylim(0, 1.05) # Set Y limit slightly above 100%

    # Add percentage labels near points
    for angle, value, label in zip(angles, filtered_values, filtered_tasks):
        # Simple positioning - adjust if labels overlap
        ha = 'center' # Default horizontal alignment
        va = 'bottom' if np.sin(angle) >= 0 else 'top' # Place below if point is in upper half, above if lower
        offset = 0.08 # Distance from point
        ax.text(angle, value + offset * np.sin(angle), f'{value:.1%}',
                ha=ha, va=va, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))

    ax.set_title(title, size=14, color='blue', y=1.15) # Adjust y for spacing

def generate_analysis_report(df, output_dir):
    """
    Generates a comprehensive text report summarizing the analysis results,
    including overall, per-task, per-scene, and scene-task breakdown accuracies.
    Saves the report to a text file.
    """
    report_lines = []
    report_lines.append(f"=== Comprehensive Analysis Report for Model: {MODEL_NAME} ===")
    report_lines.append(f"Data source: {get_input_file_path()}")
    report_lines.append(f"Analysis generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("-" * 60)

    total_records = len(df)
    if total_records == 0:
        report_lines.append("No data available for analysis.")
        final_report = "\n".join(report_lines)
        print(final_report)
        report_filepath = os.path.join(output_dir, "analysis_report.txt")
        try:
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(final_report)
            print(f"Empty analysis report saved to: {report_filepath}")
        except IOError as e:
            print(f"Error saving empty analysis report: {e}")
        return

    report_lines.append(f"Total records analyzed: {total_records}")

    # Calculate overall accuracy
    overall_correct = df['is_correct'].sum()
    overall_accuracy = overall_correct / total_records if total_records else 0
    report_lines.append(f"\nOverall Correct Predictions: {overall_correct}")
    report_lines.append(f"Overall Accuracy: {overall_accuracy:.2%}")
    report_lines.append("-" * 60)

    # Accuracy per Task
    report_lines.append("\nAccuracy per Task (sorted descending):")
    # Calculate counts and accuracy per task
    task_stats = df.groupby('Task').agg(
        correct_count=('is_correct', 'sum'),
        total_count=('is_correct', 'count')
    ).reset_index()
    task_stats['accuracy'] = (task_stats['correct_count'] / task_stats['total_count']).fillna(0)
    task_stats = task_stats.sort_values(by='accuracy', ascending=False)

    if not task_stats.empty:
        for _, row in task_stats.iterrows():
             report_lines.append(f" - {row['Task']}: {row['accuracy']:.2%} ({int(row['correct_count'])}/{int(row['total_count'])})")
    else:
        report_lines.append(" - No task data found.")
    report_lines.append("-" * 60)

    # Accuracy per Scene
    report_lines.append("\nAccuracy per Scene (sorted descending):")
    # Calculate counts and accuracy per scene
    scene_stats = df.groupby('scene').agg(
        correct_count=('is_correct', 'sum'),
        total_count=('is_correct', 'count')
    ).reset_index()
    scene_stats['accuracy'] = (scene_stats['correct_count'] / scene_stats['total_count']).fillna(0)
    scene_stats = scene_stats.sort_values(by='accuracy', ascending=False)

    if not scene_stats.empty:
        for _, row in scene_stats.iterrows():
            report_lines.append(f" - {row['scene'].capitalize()}: {row['accuracy']:.2%} ({int(row['correct_count'])}/{int(row['total_count'])})")
    else:
        report_lines.append(" - No scene data found.")
    report_lines.append("-" * 60)


    # Accuracy per Scene x Task
    report_lines.append("\nAccuracy Breakdown: Scene x Task:")
    # Use pivot_table for accuracy and calculate counts separately
    try:
        pivot_accuracy = df.pivot_table(index='scene', columns='Task', values='is_correct', aggfunc='mean')
        pivot_counts = df.pivot_table(index='scene', columns='Task', values='is_correct', aggfunc='count', fill_value=0)

        # Sort scenes based on overall scene accuracy for better readability
        sorted_scenes = scene_stats['scene'].tolist() if not scene_stats.empty else pivot_accuracy.index

        for scene_name in sorted_scenes:
            if scene_name in pivot_accuracy.index:
                report_lines.append(f"\n --- Scene: {scene_name.capitalize()} ---")
                # Sort tasks within the scene, e.g., alphabetically or by accuracy within that scene
                tasks_in_scene = pivot_accuracy.loc[scene_name].dropna().sort_values(ascending=False)
                if not tasks_in_scene.empty:
                     for task_name, acc in tasks_in_scene.items():
                         count = int(pivot_counts.loc[scene_name, task_name]) if task_name in pivot_counts.columns else 0
                         correct_count = int(acc * count) if count > 0 else 0 # Calculate correct count
                         report_lines.append(f"    - Task {task_name}: {acc:.2%} ({correct_count}/{count})")
                else:
                    report_lines.append("    - No task data found for this scene.")
            # else: # Optional: Report scenes with no data if needed
            #     report_lines.append(f"\n --- Scene: {scene_name.capitalize()} ---")
            #     report_lines.append("    - No records found for this scene.")

    except Exception as e:
         report_lines.append(f"\nError generating Scene x Task breakdown: {e}")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("End of Report")

    # Print report to console
    final_report = "\n".join(report_lines)
    print("\n" + final_report + "\n")

    # Save report to file
    report_filepath = os.path.join(output_dir, "analysis_report.txt")
    try:
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(final_report)
        print(f"Analysis report saved successfully to: {report_filepath}")
    except IOError as e:
        print(f"Error: Failed to save analysis report to {report_filepath}: {e}")


def analysis_main():
    """
    Main function for the analysis phase:
    - Loads the processed JSON data.
    - Prepares the data in a Pandas DataFrame.
    - Determines the scene type for each record.
    - Calculates correctness (considers API/normalization errors as incorrect).
    - Generates and saves visualizations (radar charts, bar charts).
    - Prints summary statistics to the console.
    - Generates and saves a detailed text analysis report.
    """
    print("\n--- Starting Analysis Phase ---")
    # Load data
    data = load_data()
    if data is None:
        print("Analysis cannot proceed without data.")
        return # Stop if data loading failed

    output_dir = get_output_dir()
    print(f"Analysis results will be saved to: {output_dir}")

    # Create DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        print("The loaded data is empty. No analysis to perform.")
        return

    # --- Data Preparation ---
    # 1. Determine scene type
    if 'id' not in df.columns:
        print("Error: 'id' column not found in the data, which is needed for scene type determination.")
        # Fallback or exit? Let's try to continue but scene analysis will fail.
        df['scene'] = 'unknown'
    else:
         df['scene'] = df['id'].apply(scene_type)

    # 2. Determine if the prediction was correct
    # A prediction is correct if 'model_out' matches 'ans' AND 'model_out' is a valid normalized answer (not an error code).
    # Valid answers are expected in the format "Ans='X'" where X is A, B, C, D, or E.
    def check_correctness(row):
        model_out = row.get('model_out')
        ans = row.get('ans')
        if not isinstance(model_out, str) or not isinstance(ans, str):
            return False # Cannot compare if types are wrong or missing
        if model_out.startswith("Ans='") and len(model_out) == 6: # Check format "Ans='X'"
             predicted_letter = model_out[5] # Extract the letter
             return predicted_letter == ans # Compare predicted letter with ground truth letter
        return False # Return False if model_out is an error code or has incorrect format

    df['is_correct'] = df.apply(check_correctness, axis=1)

    # Print basic info after preparation
    print(f"\nData prepared for analysis:")
    print(f" - Total records: {len(df)}")
    print(f" - Detected scenes: {df['scene'].unique().tolist()}")
    print(f" - Detected tasks: {df['Task'].unique().tolist()}")
    print(f" - Correct predictions: {df['is_correct'].sum()}")
    error_counts = df[~df['model_out'].str.startswith("Ans='", na=False)]['model_out'].value_counts()
    if not error_counts.empty:
        print(" - Counts of non-standard 'model_out' values (errors/unnormalized):")
        for status, count in error_counts.items():
            print(f"   - {status}: {count}")


    # --- Visualization ---
    print("\nGenerating visualizations...")

    # 1. Radar Charts: Overall and Per-Scene Task Accuracy
    # Get all unique tasks present in the data
    all_tasks = sorted(df['Task'].unique())
    if not all_tasks:
        print("Warning: No 'Task' information found in the data. Skipping radar charts.")
    else:
        # Calculate overall accuracy per task
        overall_task_accuracy = df.groupby('Task')['is_correct'].mean().reindex(all_tasks, fill_value=0)
        overall_task_counts = df.groupby('Task')['is_correct'].count().reindex(all_tasks, fill_value=0)

        # Calculate accuracy per task for each scene
        scenes_present = sorted(df['scene'].unique())
        scene_accuracies = {}
        scene_task_counts = {}
        for scene in scenes_present:
            if scene == 'unknown' and len(df[df['scene'] == 'unknown']) == 0: continue # Skip unknown if empty
            subset = df[df['scene'] == scene]
            scene_acc = subset.groupby('Task')['is_correct'].mean().reindex(all_tasks, fill_value=0)
            scene_counts = subset.groupby('Task')['is_correct'].count().reindex(all_tasks, fill_value=0)
            scene_accuracies[scene] = scene_acc.tolist()
            scene_task_counts[scene] = scene_counts.tolist()

        # Determine grid size for plots (Overall + one per scene)
        num_plots = 1 + len(scene_accuracies)
        ncols = 2 #min(num_plots, 2) # Max 2 columns
        nrows = (num_plots + ncols - 1) // ncols
        fig_height = 6 * nrows
        fig_width = 7 * ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), subplot_kw=dict(polar=True))
        fig.suptitle(f'{MODEL_NAME} - Task Accuracy Radar Charts', fontsize=16, fontweight='bold')
        axs = axs.flatten() # Flatten axes array for easy iteration

        # Plot Overall Accuracy
        plot_radar_chart(axs[0], overall_task_accuracy.tolist(), all_tasks, 'Overall Task Accuracy', overall_task_counts.tolist())

        # Plot Per-Scene Accuracy
        plot_idx = 1
        for scene in scenes_present:
            if scene in scene_accuracies:
                 if plot_idx < len(axs): # Check if there's an axis available
                     plot_radar_chart(axs[plot_idx], scene_accuracies[scene], all_tasks, f'{scene.capitalize()} Task Accuracy', scene_task_counts[scene])
                     plot_idx += 1
                 else:
                      print(f"Warning: Not enough subplots allocated for scene '{scene}'. Skipping its radar chart.")


        # Hide any unused subplots
        for i in range(plot_idx, len(axs)):
            axs[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        radar_chart_path = os.path.join(output_dir, 'task_accuracy_radar_charts.png')
        try:
            plt.savefig(radar_chart_path, dpi=300)
            print(f"Radar charts saved to: {radar_chart_path}")
        except Exception as e:
            print(f"Error saving radar charts: {e}")
        plt.close(fig)


    # 2. Bar Chart: Correct vs Total Counts Per Scene
    # Calculate counts per scene
    scene_counts = df.groupby('scene').agg(
        correct_count=('is_correct', 'sum'),
        total_count=('is_correct', 'count')
    ).reset_index()
    # Sort scenes for consistent plotting (e.g., alphabetical or custom order)
    scene_order = sorted(scene_counts['scene'].unique())
    scene_counts = scene_counts.set_index('scene').reindex(scene_order).reset_index()

    if not scene_counts.empty:
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(scene_order)), 6)) # Adjust width based on number of scenes
        bar_width = 0.35
        index = np.arange(len(scene_counts))

        # Bar for correct counts
        bars1 = ax.bar(index - bar_width/2, scene_counts['correct_count'], bar_width,
                       label='Correct', color=sns.color_palette("viridis", 2)[0])
        # Bar for total counts
        bars2 = ax.bar(index + bar_width/2, scene_counts['total_count'], bar_width,
                       label='Total', color=sns.color_palette("viridis", 2)[1])

        # Add count labels on top of bars
        ax.bar_label(bars1, padding=3, fmt='%d')
        ax.bar_label(bars2, padding=3, fmt='%d')

        # Add accuracy percentage above the bars
        for i, row in scene_counts.iterrows():
            accuracy = row['correct_count'] / row['total_count'] if row['total_count'] > 0 else 0
            ax.text(index[i], row['total_count'] + max(row['total_count']*0.05, 5), # Position above total bar
                    f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_xlabel('Scene Type', fontsize=12)
        ax.set_ylabel('Number of Records', fontsize=12)
        ax.set_title(f'{MODEL_NAME} - Prediction Counts by Scene', fontsize=14, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels([s.capitalize() for s in scene_counts['scene']])
        ax.legend(title="Count Type")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        scene_counts_path = os.path.join(output_dir, 'scene_prediction_counts.png')
        try:
            plt.savefig(scene_counts_path, dpi=300)
            print(f"Scene counts bar chart saved to: {scene_counts_path}")
        except Exception as e:
            print(f"Error saving scene counts chart: {e}")
        plt.close(fig)
    else:
        print("No scene data available to generate the counts bar chart.")


    # --- Generate Text Report ---
    print("\nGenerating comprehensive analysis report...")
    generate_analysis_report(df, output_dir)

    print(f"\n--- Analysis Phase Complete ---")
    print(f"Model '{MODEL_NAME}' analysis results are available in the '{output_dir}' directory.")


# ------------------------ Main Program Entry Point ------------------------
if __name__ == '__main__':
    print("Starting script execution...")

    # Stage 1: Processing (API calls, normalization, saving initial JSON)
    print("\n--- Running Processing Phase ---")
    start_time_proc = time.time()
    processing_main()
    end_time_proc = time.time()
    print(f"--- Processing Phase took {end_time_proc - start_time_proc:.2f} seconds ---")


    # Stage 2: Analysis and Visualization (loading JSON, calculating metrics, plotting)
    print("\n--- Running Analysis Phase ---")
    start_time_analysis = time.time()
    analysis_main()
    end_time_analysis = time.time()
    print(f"--- Analysis Phase took {end_time_analysis - start_time_analysis:.2f} seconds ---")

    print("\nScript execution finished.")