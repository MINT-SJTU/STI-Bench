import os
import json
import re
import cv2
import base64
import warnings
import time
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from collections import Counter
from IPython.display import Image, display
from pydantic import BaseModel
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # Ensure this file exists or is adapted

warnings.filterwarnings("ignore")

# ------------------------ Global Configuration ------------------------
MODEL_NAME = "qwen2.5-vl"  # Model name
OUTPUT_DIR = "output"  # Output directory
JSON_INPUT_DIR = "input"   # Input directory for JSON data
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.json")  # Filepath for saving final results
DATA_ROOT_DIR = OUTPUT_DIR  # Root directory for analysis stage
VIDEO_ROOT_DIR = "videos" # Root directory for video files
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ Model Loading ------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", # Or your downloaded path
    torch_dtype=torch.float16,
    device_map="auto"  # Modified: Let transformers handle device placement automatically
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
model.eval()  # Set model to evaluation mode

# Function to move inputs to the correct device (e.g., GPU)
def prepare_model_input(model_input, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}


# ------------------------ Utility Functions ------------------------
def make_key(item: dict) -> str:
    """
    Generates a unique key based on the filename, ID, and start/end times of a record.
    Used to check if a record has already been processed.
    """
    return f"{item['file']}|{item['ID']}|{item['time_start']}|{item['time_end']}"

def get_frame(video_path):
    """
    Extracts video frames from the specified video path, up to a maximum of 30 frames.
    Converts the frames to JPEG format and then Base64 encodes them.
    Returns: (base64_frames, sample_fps)
    """
    base64_frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_num / fps if fps > 0 else 0
    max_frames = 30
    # Dynamically calculate the sampling frame rate to ensure a maximum of max_frames images are extracted.
    target_fps = max_frames / duration if duration > 0 else fps
    sample_interval = int(fps / target_fps) if target_fps > 0 else 1
    sample_interval = max(1, sample_interval)
    sample_fps = fps / sample_interval
    frame_idx = set(range(0, total_frame_num, sample_interval))
    if len(frame_idx) > max_frames:
        frame_idx = set(sorted(frame_idx)[:max_frames])

    sampled_video_path = 'sampled_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sampled_video_path, fourcc, sample_fps, (int(cap.get(3)), int(cap.get(4))))

    for index in range(total_frame_num):
        ret, frame = cap.read()
        if not ret:
            break
        if index in frame_idx:
            out.write(frame)

    cap.release()
    out.release()

    return sampled_video_path, sample_fps
def normalize_answer(text: str) -> str:
    """
    Attempts to extract a capital letter answer (A-E) from the text.
    Unifies the format to "Ans='X'".  Returns None if no match is found.
    """
    patterns = [
        r"Ans='([A-E])'",
        r'\n([A-E])\s*$',
        r'([A-E])\s*$',
        r'[Oo]ption\s+([A-E])',
        r'chosen_option\s*=\s*["\']([A-E])["\']',
        r'print\(\s*["\']?([A-E])["\']?\s*\)',
        r'\b([A-E])\.\s*',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"Ans='{match.group(1)}'"
    return None


def process_item(item, index, results_dict):
    """
    Processes a single record:
    1. Extracts frames from the video path and constructs a question prompt.
    2. Calls the Qwen model to get an answer.
    3. Normalizes the answer.
    4. Organizes the results, stores them in a dictionary, and saves them to disk.
    """
    key = make_key(item)
    video_path = os.path.join(VIDEO_ROOT_DIR, item["Video"])
    print(f"[{index}] Processing video: {video_path}")

    # Extract frames and sample video
    sampled_video_path, sample_fps = get_frame(video_path)

    # Construct prompt (adapt as needed for Qwen)
    candidate_list = [f"{k} {v}" for k, v in item["Candidates"].items()]
    question = (
        f"From {item['time_start']} seconds to {item['time_end']} seconds. "
        + item["Question"]
        + "\n" + "\n".join(candidate_list)
        + "\nPlease output only the options!"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{sampled_video_path}",
                    "max_pixels": 360 * 420,
                    "fps": sample_fps,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=sample_fps,  # Pass the calculated sample_fps
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = prepare_model_input(inputs, model.device) # Moved the inputs to the model's device.

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0]  # Take the first output

    except Exception as e:
        print(f"Inference Error: {e}")
        response = "ERROR_INFERENCE"

    # Normalize the answer
    normalized = normalize_answer(response)
    if normalized is None:
        normalized = "NORMALIZE_ERROR"

    out_item = {
        "id": item["file"],
        "id_file": item["ID"],
        "time_s": item["time_start"],
        "time_e": item["time_end"],
        "model_out": normalized,
        "ans": item["Answer"],
        "class": item["Task"],
        "que": question,
        "Source": item["Source"],
        "Task": item["Task"]
    }

    # Write the results to the dictionary and save to disk
    results_dict[key] = out_item
    all_data = list(results_dict.values())
    with open(OUTPUT_FILEPATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"[{index}] Done. model_out = {normalized}")

    #Cleanup
    if os.path.exists(sampled_video_path):
        os.remove(sampled_video_path)
    return out_item

def processing_main():
    """
    Main data processing function:
    - Loads all JSON files from JSON_INPUT_DIR.
    - Filters out records that have already been processed (or previously marked as erroneous).
    - Processes each record sequentially.
    """
    json_dir = JSON_INPUT_DIR
    all_files = os.listdir(json_dir)
    tmp = []
    for filename in all_files:
        if filename.endswith(".json"):
            full_path = os.path.join(json_dir, filename)
            try:
                data_list = json.load(open(full_path, "r", encoding="utf-8"))
                for item in data_list:
                    item["file"] = filename
                tmp += data_list
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    lis = tmp
    print(f"Loaded {len(lis)} total records from {json_dir}.")

    # Load existing results
    results_dict = {}
    if os.path.exists(OUTPUT_FILEPATH):
        try:
            old_data = json.load(open(OUTPUT_FILEPATH, "r", encoding="utf-8"))
            for rec in old_data:
                k = f"{rec['id']}|{rec['id_file']}|{rec['time_s']}|{rec['time_e']}"
                results_dict[k] = rec
        except Exception as e:
            print(f"Error loading existing results: {e}")

    # Process items
    to_process = []
    for idx, item in enumerate(lis):
        key = make_key(item)
        if key not in results_dict:
            to_process.append((idx, item))
        else:
            exist_model_out = results_dict[key].get("model_out", "")
            if exist_model_out in ["ERROR_INFERENCE", "NORMALIZE_ERROR"]:
                to_process.append((idx, item))

    print(f"Total {len(to_process)} items need to be processed (including retries for errors).")
    for idx, itm in to_process:
        process_item(itm, idx, results_dict)


    print(f"All processing done! The results have been saved to: {OUTPUT_FILEPATH}")

# ------------------------ Analysis and Visualization Stage ------------------------

def get_input_file_path():
    return os.path.join(DATA_ROOT_DIR, f"{MODEL_NAME}.json")

def get_output_dir():
    output_dir = f"./analysis_results_{MODEL_NAME}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_data():
    file_path = get_input_file_path()
    print(f"Loading data from {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def scene_type(id_str):
    """
    Determines the scene type based on the filename:
      - Contains 'camera' -> outdoor
      - Starts with 'scene' -> indoor
      - Numeric and length is 6 -> desktop
      - Otherwise -> unknown
    """
    if 'camera' in id_str:
        return 'outdoor'
    elif id_str.startswith('scene'):
        return 'indoor'
    elif id_str.split('.')[0].isdigit() and len(id_str.split('.')[0]) == 6:
        return 'desktop'
    else:
        return 'unknown'

def plot_radar_chart(ax, task_labels, data_values, title, task_counts=None):
    """
    Draws a radar chart in polar coordinates:
      - Filters out tasks with 0 samples.
      - Marks the accuracy percentage in the chart.
    """
    filtered_tasks = []
    filtered_values = []
    if task_counts is not None:
        for label, value, count in zip(task_labels, data_values, task_counts):
            if count > 0:
                filtered_tasks.append(label)
                filtered_values.append(value)
    else:
        filtered_tasks = task_labels
        filtered_values = data_values

    if not filtered_tasks:
        ax.text(0.5, 0.5, "No data available",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(title, size=14, color='blue', y=1.1)
        return

    num_tasks = len(filtered_tasks)
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    values_closed = filtered_values + [filtered_values[0]]
    angles_closed = angles + [angles[0]]
    ax.plot(angles_closed, values_closed, marker='o', linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.3)
    ax.set_xticks(angles)
    ax.set_xticklabels(filtered_tasks, fontsize=9)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    for angle, value, label in zip(angles, filtered_values, filtered_tasks):
        ha = 'left' if 0 <= angle < np.pi else 'right'
        ax.text(angle, value + 0.05, f'{value:.1%}',
                ha=ha, va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    ax.set_title(title, size=14, color='blue', y=1.1)

def generate_analysis_report(df, output_dir):
    """
    Generates a comprehensive statistical report text, including overall accuracy,
    accuracy for each task, each scene, and scene x task combinations. Also saves
    the report to a text file.
    """
    report_lines = []
    report_lines.append("=== Comprehensive Analysis Report ===")
    report_lines.append(f"Total records in the dataset: {len(df)}")

    overall_correct = df['correct'].sum()
    overall_total = len(df)
    overall_accuracy = overall_correct / overall_total if overall_total else 0
    report_lines.append(f"Number of correct predictions by the model overall: {overall_correct}")
    report_lines.append(f"Overall accuracy of the model: {overall_accuracy:.2%}")

    task_accuracy = df.groupby('Task')['correct'].mean().sort_values(ascending=False)
    report_lines.append("\nAccuracy for each task (highest to lowest):")
    for task_name, acc in task_accuracy.items():
        report_lines.append(f" - {task_name}: {acc:.2%}")

    scene_accuracy = df.groupby('scene')['correct'].mean().sort_values(ascending=False)
    report_lines.append("\nAccuracy for each scene (highest to lowest):")
    for scn, acc in scene_accuracy.items():
        report_lines.append(f" - {scn}: {acc:.2%}")

    pivot_scene_task = df.pivot_table(index='scene', columns='Task', values='correct', aggfunc='mean')
    scene_task_counts = df.groupby(['scene','Task'])['correct'].count().unstack(fill_value=0)

    report_lines.append("\nAccuracy for each scene x each task:")
    for scn in pivot_scene_task.index:
        report_lines.append(f" - Scene {scn}:")
        for tsk in pivot_scene_task.columns:
            count_ = scene_task_counts.loc[scn, tsk] if scn in scene_task_counts.index else 0
            if count_ > 0:
                acc = pivot_scene_task.loc[scn, tsk]
                acc_str = f"{acc:.2%}" if not np.isnan(acc) else "N/A"
                report_lines.append(f"    Task {tsk}: {acc_str} (Sample Count: {count_})")

    final_report = "\n".join(report_lines)
    print(final_report)
    with open(os.path.join(output_dir, "analysis_report.txt"), 'w', encoding='utf-8') as f:
        f.write(final_report)

def analysis_main():
    """
    Main analysis function:
    - Loads the processed JSON data.
    - Determines the scene based on the filename.
    - Calculates the accuracy for overall, each task, and each scene using a DataFrame (where inference errors and normalization errors are considered errors).
    - Draws radar charts and bar charts and saves the results, while also printing a detailed statistical report.
    """
    data = load_data()
    output_dir = get_output_dir()
    print(f"Analysis results will be saved to: {output_dir}")
    df = pd.DataFrame(data)
    df['scene'] = df['id'].apply(scene_type)
    df['correct'] = df.apply(
        lambda row: (
            len(row['model_out']) >= 2
            and row['model_out'][-2] in ["A","B","C","D","E"]
            and row['model_out'][-2] == row['ans']
        ),
        axis=1
    )
    print(f"Data loading complete, total {len(df)} records")
    print(f"Analyzing model: {MODEL_NAME}")

    # Calculate overall task accuracy data
    task_counts = df.groupby('Task')['correct'].count()
    overall_accuracy_by_task = df.groupby('Task')['correct'].mean().reset_index()
    tasks = overall_accuracy_by_task['Task'].tolist()
    overall_acc_values = overall_accuracy_by_task['correct'].tolist()
    overall_task_counts = task_counts.reindex(tasks).values.tolist()

    # Calculate task accuracy by scene
    scenes = ['indoor', 'desktop', 'outdoor']
    scene_accuracies = {}
    scene_task_counts = {}
    for scene in scenes:
        subset = df[df['scene'] == scene]
        task_count = subset.groupby('Task')['correct'].count().reindex(tasks, fill_value=0)
        scene_task_counts[scene] = task_count.tolist()
        scene_acc = subset.groupby('Task')['correct'].mean().reindex(tasks, fill_value=0)
        scene_accuracies[scene] = scene_acc.tolist()

    # Draw radar charts: overall and for each scene
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    fig.suptitle('Task Accuracy Radar Charts', fontsize=16, fontweight='bold')
    plot_radar_chart(axs[0, 0], tasks, overall_acc_values, 'Overall Task Accuracy', overall_task_counts)
    plot_radar_chart(axs[0, 1], tasks, scene_accuracies['indoor'], 'Indoor Task Accuracy', scene_task_counts['indoor'])
    plot_radar_chart(axs[1, 0], tasks, scene_accuracies['desktop'], 'Desktop Task Accuracy', scene_task_counts['desktop'])
    plot_radar_chart(axs[1, 1], tasks, scene_accuracies['outdoor'], 'Outdoor Task Accuracy', scene_task_counts['outdoor'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_charts.png', dpi=300)
    plt.close()

    # Draw bar charts: correct count and total count for each scene
    correct_counts = df.groupby('scene')['correct'].sum()
    total_counts = df.groupby('scene')['correct'].count()
    correct_vals = correct_counts.reindex(scenes, fill_value=0)
    total_vals = total_counts.reindex(scenes, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(scenes))
    bars1 = ax.bar(index, correct_vals, bar_width, label='Correct', color=sns.color_palette("tab10")[0])
    bars2 = ax.bar(index + bar_width, total_vals, bar_width, label='Total', color=sns.color_palette("tab10")[1])
    for i in range(len(scenes)):
        accuracy = correct_vals[i] / total_vals[i] if total_vals[i] > 0 else 0
        ax.text(i + bar_width/2, max(correct_vals[i], total_vals[i]) + 5,
                f'{accuracy:.1%}', ha='center', fontweight='bold')
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_xlabel('Scene', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title('Correct Predictions and Total Counts by Scene', fontsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([s.capitalize() for s in scenes])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scene_counts.png', dpi=300)
    plt.close()

    overall_correct = df['correct'].sum()
    overall_total = len(df)
    overall_accuracy = overall_correct / overall_total if overall_total else 0
    print(f"\n=== Overall Model Statistics ===")
    print(f"Total Records: {overall_total}")
    print(f"Correct Predictions: {overall_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    task_accuracy = df.groupby('Task')['correct'].mean().sort_values(ascending=False)
    print("\n=== Task Accuracy (Highest to Lowest) ===")
    for tsk, acc in task_accuracy.items():
        print(f"Accuracy for task {tsk}: {acc:.2%}")

    scene_accuracy = df.groupby('scene')['correct'].mean().sort_values(ascending=False)
    print("\n=== Scene Accuracy (Highest to Lowest) ===")
    for scn, acc in scene_accuracy.items():
        print(f"Accuracy for scene {scn}: {acc:.2%}")

    print("\n=== Scene x Task Accuracy (Ignoring tasks with a count of 0) ===")
    pivot_scene_task = df.pivot_table(index='scene', columns='Task', values='correct', aggfunc='mean')
    scene_task_counts = df.groupby(['scene','Task'])['correct'].count().unstack(fill_value=0)
    for scn in pivot_scene_task.index:
        print(f"\nScene {scn}:")
        for tsk in pivot_scene_task.columns:
            count_ = scene_task_counts.loc[scn, tsk] if scn in scene_task_counts.index else 0
            if count_ > 0:
                acc_val = pivot_scene_task.loc[scn, tsk]
                acc_str = f"{acc_val:.2%}" if not np.isnan(acc_val) else "N/A"
                print(f"   Task {tsk}: {acc_str} (Sample Count: {count_})")

    print("\nGenerating a comprehensive analysis report...")
    generate_analysis_report(df, output_dir)
    print(f"\nAnalysis complete! Results for model {MODEL_NAME} have been saved to the {output_dir} directory")

# ------------------------ Main Program Entry Point ------------------------
if __name__ == '__main__':
    # First, execute the processing stage
    processing_main()

    # Then, execute the data analysis and visualization stage
    analysis_main()