"""
Add ASR/OCR to existing video databases
"""

import os
import sys
import json
import argparse
import logging
import traceback
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
from typing import List, Optional, Tuple

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration via Environment Variables
GPT4O_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GPT4O_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT4O_MODEL_NAME = "gpt-4o"
WHISPER_MODEL_ID = "./model_zoo/whisper-base" 

_whisper_pipeline = None

def load_whisper_pipeline(model_id: str = WHISPER_MODEL_ID, device: str = "cuda"):
    global _whisper_pipeline
    if _whisper_pipeline is None:
        from transformers import pipeline
        import torch
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    return _whisper_pipeline

def extract_audio_from_video(video_path: str) -> Optional[str]:
    output_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, timeout=300
        )
        return output_path
    except Exception:
        return None

def extract_asr_whisper_hf(video_path: str, model_id: str, device: str) -> str:
    audio_path = extract_audio_from_video(video_path)
    if not audio_path: return ""
    try:
        pipe = load_whisper_pipeline(model_id, device)
        result = pipe(audio_path, return_timestamps=True, generate_kwargs={"language": None})
        return result.get("text", "").strip()
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)

def extract_ocr_gpt4o_single_frame(image_path: str) -> List[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=GPT4O_API_KEY, base_url=GPT4O_BASE_URL)
        
        with open(image_path, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = client.chat.completions.create(
            model=GPT4O_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Extract ALL visible text."}
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception: return []

def update_database_with_asr_ocr(video_db_path: str, asr_text: str, ocr_text: str) -> bool:
    try:
        from nano_vectordb import NanoVectorDB
        if not os.path.exists(video_db_path): return False
        db = NanoVectorDB(1024, storage_file=video_db_path)
        data = db.get_additional_data()
        data['asr_text'] = asr_text
        data['ocr_text'] = ocr_text
        db.store_additional_data(**data)
        db.save()
        return True
    except Exception: return False

def extract_ocr_gpt4o_batch(frames_dir: str, sample_interval: int = 15, max_workers: int = 8) -> str:
    try:
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
            key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")) if "_n" in x else 0
        )
        if not frame_files: return ""
        sampled = frame_files[::sample_interval]
        all_texts = set()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(extract_ocr_gpt4o_single_frame, os.path.join(frames_dir, f)): f
                for f in sampled
            }
            for future in as_completed(future_to_frame):
                texts = future.result()
                if texts: all_texts.update(texts)
        
        return " | ".join(sorted(all_texts))
    except Exception:
        return ""

def scan_video_databases(video_db_root: str) -> List[str]:
    video_ids = []
    for item in os.listdir(video_db_root):
        if os.path.isdir(os.path.join(video_db_root, item)) and os.path.exists(os.path.join(video_db_root, item, "database.json")):
            video_ids.append(item)
    return sorted(video_ids)

def process_single_video(video_id, video_db_root, raw_dir, whisper_model, ocr_interval, ocr_workers, skip_existing, device):
    base_dir = os.path.join(video_db_root, video_id)
    video_db_path = os.path.join(base_dir, "database.json")
    frames_dir = os.path.join(base_dir, "frames")
    
    if not os.path.exists(video_db_path): return False, "DB Not Found"
    
    video_path = os.path.join(raw_dir, video_id + ".mp4") # Simplified check
    
    asr = extract_asr_whisper_hf(video_path, whisper_model, device) if os.path.exists(video_path) else ""
    ocr = extract_ocr_gpt4o_batch(frames_dir, ocr_interval, ocr_workers) if os.path.exists(frames_dir) else ""
    
    if update_database_with_asr_ocr(video_db_path, asr, ocr):
        return True, "Updated"
    return False, "Failed update"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_db_root", type=str, default="./video_database")
    parser.add_argument("--whisper_model", type=str, default=WHISPER_MODEL_ID)
    parser.add_argument("--ocr_sample_interval", type=int, default=15)
    parser.add_argument("--ocr_workers", type=int, default=8)
    args = parser.parse_args()
    
    video_ids = scan_video_databases(args.video_db_root)
    raw_dir = os.path.join(args.video_db_root, "raw")
    
    for vid in video_ids:
        process_single_video(vid, args.video_db_root, raw_dir, args.whisper_model, args.ocr_sample_interval, args.ocr_workers, True, "cuda")

if __name__ == "__main__":
    main()