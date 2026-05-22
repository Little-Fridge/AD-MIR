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
import admir.config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration via Environment Variables
GPT4O_BASE_URL = os.environ.get("OPENAI_BASE_URL", config.LOCAL_VLLM_BASE_URL)
GPT4O_API_KEY = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
GPT4O_MODEL_NAME = os.environ.get("ADMIR_TOOL_VLM_MODEL", config.AOAI_TOOL_VLM_MODEL_NAME)
WHISPER_MODEL_ID = os.environ.get("ADMIR_ASR_MODEL", "")

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
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, timeout=300
        )
        if proc.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[-1000:])
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def extract_asr_whisper_hf(video_path: str, model_id: str, device: str) -> str:
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        raise RuntimeError(f"Could not extract audio from {video_path}")
    try:
        pipe = load_whisper_pipeline(model_id, device)
        result = pipe(audio_path, return_timestamps=True, generate_kwargs={"language": None})
        chunks = result.get("chunks") or []
        if chunks:
            lines = []
            for chunk in chunks:
                ts = chunk.get("timestamp") or (None, None)
                start = "" if ts[0] is None else f"{float(ts[0]):.2f}"
                end = "" if ts[1] is None else f"{float(ts[1]):.2f}"
                text = str(chunk.get("text", "")).strip()
                if text:
                    lines.append(f"[{start}-{end}] {text}")
            if lines:
                return "\n".join(lines)
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
                    {"type": "text", "text": "Extract ALL visible text from this image. Include titles, labels, captions, signs, logos, brand names, slogans, and written content. Return ONLY extracted text, one item per line. If no text is visible, return NO_TEXT."}
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
        db = NanoVectorDB(config.AOAI_EMBEDDING_LARGE_DIM, storage_file=video_db_path)
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

def _find_video_file(raw_dir: str, video_id: str) -> str:
    for ext in (".mp4", ".webm", ".mkv", ".mov", ".avi"):
        path = os.path.join(raw_dir, video_id + ext)
        if os.path.exists(path):
            return path
    return ""

def process_single_video(video_id, video_db_root, raw_dir, asr_model, ocr_interval, ocr_workers, skip_existing, device, skip_ocr=False):
    base_dir = os.path.join(video_db_root, video_id)
    video_db_path = os.path.join(base_dir, "database.json")
    frames_dir = os.path.join(base_dir, "frames")
    
    if not os.path.exists(video_db_path): return False, "DB Not Found"
    
    video_path = _find_video_file(raw_dir, video_id)
    
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Raw video missing for {video_id} under {raw_dir}")
    if not asr_model:
        raise FileNotFoundError("Set --asr_model or ADMIR_ASR_MODEL before running ASR.")
    if not (os.path.exists(asr_model) or ("/" in asr_model and not asr_model.startswith("."))):
        raise FileNotFoundError(f"ASR model not found: {asr_model}")
    if getattr(config, "STRICT_PAPER_MODE", False) and not skip_ocr:
        raise RuntimeError("Strict mode leaves OCR to frame_inspect; pass --skip_ocr for ASR-only augmentation.")

    if skip_existing:
        try:
            with open(video_db_path, "r", encoding="utf-8") as f:
                current_db = json.load(f)
            additional_data = current_db.get("additional_data", {}) if isinstance(current_db, dict) else {}
            has_asr = "asr_text" in current_db or "asr_text" in additional_data
            has_ocr = skip_ocr or "ocr_text" in current_db or "ocr_text" in additional_data
            if has_asr and has_ocr:
                return True, "Already exists"
        except Exception:
            pass

    asr = extract_asr_whisper_hf(video_path, asr_model, device)
    ocr = "" if skip_ocr else (extract_ocr_gpt4o_batch(frames_dir, ocr_interval, ocr_workers) if os.path.exists(frames_dir) else "")
    
    if update_database_with_asr_ocr(video_db_path, asr, ocr):
        return True, "Updated"
    return False, "Failed update"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_db_root", type=str, default="./video_database")
    parser.add_argument("--raw_video_root", type=str, default="./data/raw_videos")
    parser.add_argument("--asr_model", type=str, default=WHISPER_MODEL_ID)
    parser.add_argument("--ocr_sample_interval", type=int, default=15)
    parser.add_argument("--ocr_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_ocr", action="store_true", help="Paper-strict runs use Whisper ASR here and leave OCR to frame_inspect.")
    args = parser.parse_args()
    
    video_ids = scan_video_databases(args.video_db_root)
    raw_dir = args.raw_video_root
    
    for vid in video_ids:
        process_single_video(vid, args.video_db_root, raw_dir, args.asr_model, args.ocr_sample_interval, args.ocr_workers, True, args.device, args.skip_ocr)

if __name__ == "__main__":
    main()
