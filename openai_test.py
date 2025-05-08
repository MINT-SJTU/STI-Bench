#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STI-Bench batch QA script (Parquet version)

Refactor Notes:
- No longer reads multiple JSON files from a directory.
- Now reads from a single Parquet file: qa.parquet
- Configuration section is centralized.
- Core logic (frame extraction, model API call, multithreading, checkpointing) remains intact.
"""

import os
import json
import cv2
import base64
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI  # Requires: pip install openai

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
MODEL_NAME: str = "your-model-name"  # e.g., "gpt-4" or custom model name
API_KEY: str = "your-api-key"
BASE_URL: str = "http://your-api-endpoint/v1"

PARQUET_PATH: str = "/path/to/qa.parquet"
VIDEO_DIR: str = "/path/to/video/files"
OUTPUT_DIR: str = "/path/to/output"
MAX_WORKERS: int = 4  # Number of concurrent threads

# Output JSON path based on model name
OUTPUT_FILEPATH: str = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.json")
# ===================================

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========== UTILITY FUNCTIONS ==========

def make_key(item: Dict) -> str:
    """Generate a unique key based on video name and question timing."""
    return f"{item['Video']}|{item['ID']}|{item['time_start']}|{item['time_end']}"

def get_frame(video_path: str) -> Tuple[List[str], float]:
    """
    Extract up to 30 evenly spaced frames from the video.
    Returns:
        - base64_frames: List of base64-encoded JPEG frames
        - sample_fps: Approximate sampling frame rate
    """
    base64_frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return base64_frames, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    max_frames = 30

    sample_interval = max(1, int(total_frames / max_frames))
    sample_fps = fps / sample_interval

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            _, buffer = cv2.imencode(".jpeg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            base64_frames.append(base64.b64encode(buffer).decode())
            if len(base64_frames) >= max_frames:
                break
    cap.release()
    return base64_frames, sample_fps

write_lock = threading.Lock()

def process_item(item: Dict, idx: int, results_dict: Dict[str, Dict]) -> Dict:
    """Core function to process one item: extract frames, call model, and save result."""
    key = make_key(item)
    video_path = os.path.join(VIDEO_DIR, item["Video"])
    frames_b64, sample_fps = get_frame(video_path)

    frames_payload = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]

    candidate_list = [f"{k} {v}" for k, v in item["Candidates"].items()]
    question = (
        f"From {item['time_start']} seconds to {item['time_end']} seconds. "
        + item["Question"] + "\n" + "\n".join(candidate_list)
    )

    prompt_text = (
        f"Answer the question below based on the frames provided, "
        f"which are sampled at about {sample_fps:.2f} FPS.\n"
        f"Question: {question}\n"
        f"Please output only the option you choose!"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + frames_payload}]
    params = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0,
        "top_p": 1
    }

    response = "ERROR_CALL_API"
    for attempt in range(3):
        try:
            result = client.chat.completions.create(**params)
            response = result.choices[0].message.content
            break
        except Exception as e:
            print(f"[{idx}] Retry {attempt+1}/3 failed: {e}")
            time.sleep(2)

    out_item = {
        "id": item["Video"],
        "id_file": item["ID"],
        "time_s": item["time_start"],
        "time_e": item["time_end"],
        "model_out": str(response),
        "ans": item["Answer"],
        "Task": item["Task"],
        "que": question,
        "Source": item["Source"]
    }

    with write_lock:
        results_dict[key] = out_item
        with open(OUTPUT_FILEPATH, "w", encoding="utf-8") as f:
            json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=4)

    return out_item

# ========== MAIN WORKFLOW ==========

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_FILEPATH):
        with open(OUTPUT_FILEPATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Load Parquet file
    df = pd.read_parquet(PARQUET_PATH)
    records = df.to_dict("records")
    print(f"[INFO] Loaded {len(records)} records from {PARQUET_PATH}")

    # Load existing results for checkpointing
    results_dict: Dict[str, Dict] = {}
    if os.path.getsize(OUTPUT_FILEPATH) > 2:
        with open(OUTPUT_FILEPATH, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                k = f"{rec['id']}|{rec['id_file']}|{rec['time_s']}|{rec['time_e']}"
                results_dict[k] = rec

    # Filter tasks to process
    to_process = [
        (idx, item) for idx, item in enumerate(records)
        if (make_key(item) not in results_dict) or
           (results_dict[make_key(item)]["model_out"] == "ERROR_CALL_API")
    ]
    print(f"[INFO] Need to process {len(to_process)} items.")

    # Multithreaded execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(process_item, itm, idx, results_dict): idx
                   for idx, itm in to_process}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                res = fut.result()
                print(f"[{idx}] Done. model_out = {res['model_out']}")
            except Exception as e:
                print(f"[{idx}] Task exception: {e}")

    print(f"[INFO] All finished. Results saved to {OUTPUT_FILEPATH}")

if __name__ == "__main__":
    main()
