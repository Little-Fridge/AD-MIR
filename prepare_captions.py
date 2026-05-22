"""
Caption Preparation Script
Decodes videos, generates semantic captions via the configured VLM, and initializes the Admir database.
"""
import functools
import json
import multiprocessing as mp
import os
import base64
import argparse
try:
    import cv2
except ImportError:
    cv2 = None
import sys
import inspect
import requests
import traceback
import httpx
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from openai import OpenAI

# Try importing admir config
try:
    import admir.config as config
    from admir.utils import create_chat_completion, robust_json_parse
except ImportError:
    class Config:
        STRICT_PAPER_MODE = False
        pass
    config = Config()
    def create_chat_completion(client, model, messages, max_tokens=None, temperature=None, **kwargs):
        params = dict(kwargs)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        return client.chat.completions.create(model=model, messages=messages, **params)
    def robust_json_parse(text):
        return json.loads(text)

# --------------------------------------------------------------------------- #
#                             Global Configuration                            #
# --------------------------------------------------------------------------- #

GPT4O_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    getattr(config, "LOCAL_VLLM_BASE_URL", "https://api.openai.com/v1"),
)
GPT4O_API_KEY = os.environ.get("OPENAI_API_KEY", getattr(config, "OPENAI_API_KEY", "EMPTY"))
GPT4O_MODEL_NAME = os.environ.get(
    "ADMIR_CAPTION_VLM_MODEL",
    getattr(config, "AOAI_CAPTION_VLM_MODEL_NAME", ""),
)
MAX_API_RETRIES = 8


def _strict_mode() -> bool:
    return bool(getattr(config, "STRICT_PAPER_MODE", False))


def _validate_strict_ready() -> None:
    validate = getattr(config, "validate_strict_paper_config", None)
    if callable(validate):
        validate(require_api_key=True)
    if _strict_mode() and not GPT4O_MODEL_NAME:
        raise RuntimeError("Set ADMIR_CAPTION_VLM_MODEL before running caption generation.")


def _run_metadata(video_id: str = "", fps: float = 0.0, clip_secs: int = 0) -> Dict[str, Any]:
    return {
        "strict_paper_mode": _strict_mode(),
        "caption_model": GPT4O_MODEL_NAME,
        "openai_base_url": GPT4O_BASE_URL,
        "embedding_backend": getattr(config, "EMBEDDING_BACKEND", ""),
        "embedding_model": getattr(config, "AOAI_EMBEDDING_LARGE_MODEL_NAME", ""),
        "embedding_dim": getattr(config, "AOAI_EMBEDDING_LARGE_DIM", None),
        "video_id": video_id,
        "fps": float(fps) if fps else None,
        "clip_secs": int(clip_secs) if clip_secs else None,
    }


def _metadata_matches_current_run(metadata: Dict[str, Any], fps: float, clip_secs: int) -> bool:
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("strict_paper_mode") == _strict_mode()
        and metadata.get("caption_model") == GPT4O_MODEL_NAME
        and metadata.get("embedding_backend") == getattr(config, "EMBEDDING_BACKEND", "")
        and metadata.get("embedding_model") == getattr(config, "AOAI_EMBEDDING_LARGE_MODEL_NAME", "")
        and int(metadata.get("embedding_dim") or 0) == int(getattr(config, "AOAI_EMBEDDING_LARGE_DIM", 0))
        and abs(float(metadata.get("fps") or 0.0) - float(fps)) < 1e-9
        and int(metadata.get("clip_secs") or 0) == int(clip_secs)
    )


def _strip_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    clean.pop("_admir_metadata", None)
    return clean

# --------------------------------------------------------------------------- #
#                             Prompt templates                                #
# --------------------------------------------------------------------------- #

CAPTION_PROMPT = """Here are consecutive frames from a video clip. Please visually analyze the video clip and output JSON in the template below.

Output template:
{
  "clip_start_time": CLIP_START_TIME,
  "clip_end_time": CLIP_END_TIME,
  "subject_registry": {
    "<subject_i>": {
      "name": "<fill with short identity if name is unknown, e.g. 'man in red'>",
      "appearance": "<list of visual appearance descriptions>",
      "identity": "<list of inferred identity descriptions>",
      "first_seen": "<timestamp>"
    },
    ...
  },
  "clip_description": "<smooth and detailed visual narration of the video clip>"
}
"""

MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects visually refer to the same person/object, merge them
   (keep earliest `first_seen` time and union all fields).

Input (list of JSON objects):
REGISTRIES_PLACEHOLDER

Return *only* the merged `subject_registry` JSON object.
"""

SYSTEM_PROMPT = "You are a helpful assistant designed to output strict JSON only. Do not wrap JSON in markdown."

# --------------------------------------------------------------------------- #
#                      Database Builder Integration                           #
# --------------------------------------------------------------------------- #

def _infer_emb_dim(emb_endpoint: str):
    try:
        url = emb_endpoint.rstrip("/") + "/v1/embeddings"
        payload = {"model": "hf-embedding", "input": "hello"}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        emb = data["data"][0]["embedding"]
        return int(len(emb))
    except Exception as e:
        print(f"Warning: Could not infer embedding dim: {e}")
        return 1024 

def _get_emb_dim(args_emb_dim: int, emb_endpoint: str):
    if isinstance(args_emb_dim, int) and args_emb_dim > 0:
        return args_emb_dim
    for k in ["ADMIR_EMB_DIM", "EMB_DIM", "EMBED_DIM", "EMBEDDING_DIM"]:
        v = os.environ.get(k, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    for attr in ["EMB_DIM", "EMBED_DIM", "EMBEDDING_DIM", "HF_EMBEDDING_DIM"]:
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return _infer_emb_dim(emb_endpoint)

def ensure_video_db_exists(caption_file: str, video_db_path: str, emb_dim: int):
    """
    Ensures the vector database is built from the caption file.
    Uses admir.build_database.init_single_video_db
    """
    if os.path.exists(video_db_path) and os.path.getsize(video_db_path) > 0:
        return True, "Database already exists"

    try:
        from admir.build_database import init_single_video_db
        print(f"Building DB using admir.build_database.init_single_video_db...")
        init_single_video_db(caption_file, video_db_path, emb_dim)
        
        if os.path.exists(video_db_path) and os.path.getsize(video_db_path) > 0:
            return True, "Built successfully"
        else:
            return False, "Function ran but DB file is missing or empty"
            
    except ImportError:
        return False, "Could not import 'admir' package. Is it in the python path?"
    except Exception as e:
        traceback.print_exc()
        return False, f"DB build failed: {str(e)}"

# --------------------------------------------------------------------------- #
#                          Video Decoding Helper                              #
# --------------------------------------------------------------------------- #

def decode_video_to_frames(video_path: str, output_folder: str, fps: float = 1.0):
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"Frames already exist in {output_folder}, skipping decode.")
        return

    os.makedirs(output_folder, exist_ok=True)
    if cv2 is None:
        temp_pattern = os.path.join(output_folder, "frame_tmp_%06d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            temp_pattern,
        ]
        print(f"Decoding video {os.path.basename(video_path)} with ffmpeg at {fps} FPS...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        temp_files = sorted(f for f in os.listdir(output_folder) if f.startswith("frame_tmp_") and f.endswith(".jpg"))
        for idx, name in enumerate(temp_files):
            os.rename(os.path.join(output_folder, name), os.path.join(output_folder, f"frame_n{idx:06d}.jpg"))
        print(f"Decoded {len(temp_files)} frames.")
        return

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 24.0 
    
    step = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    saved_count = 0
    
    print(f"Decoding video {os.path.basename(video_path)} to {output_folder} at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % step == 0:
            # We don't necessarily use timestamp in filename here, just an index, 
            # but usually it's better to ensure strict 1FPS ordering.
            # Filename format: frame_nXXXXXX.jpg
            filename = f"frame_n{saved_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"Decoded {saved_count} frames.")

# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def gather_clip_frames(video_frame_folder: str, clip_secs: int, fps: float = 1.0) -> List[Tuple[str, Dict]]:
    """
    Groups frames into clips based on time intervals.
    Args:
        video_frame_folder: Folder containing frames
        clip_secs: Length of each clip in seconds (e.g., 30)
        fps: FPS used during decoding (must match decode_video_to_frames, default 1.0)
    """
    # 1. Filter and sort: sort by the 6-digit index after _n
    frame_files = sorted(
        [f for f in os.listdir(video_frame_folder) if f.startswith("frame_n") and f.endswith(".jpg")],
        key=lambda x: int(x.split("_n")[-1].rstrip(".jpg")),
    )
    if not frame_files: 
        return []

    # 2. Convert index to timestamp (Timestamp = Index / FPS)
    def get_ts(fname):
        idx = int(fname.split("_n")[-1].rstrip(".jpg"))
        return float(idx / fps)

    frame_data = [(f, get_ts(f)) for f in frame_files]
    
    # last_ts represents the timestamp of the last frame
    last_ts = frame_data[-1][1] if frame_data else 0

    result = []
    clip_start = 0
    while clip_start <= last_ts:
        clip_end = clip_start + clip_secs
        
        # Filter frames that fall within the current [start, end) interval
        clip_files = [
            os.path.join(video_frame_folder, f)
            for f, t in frame_data
            if clip_start <= t < clip_end 
        ]
        
        if clip_files:
            result.append((
                f"{clip_start}_{clip_end}", 
                {"files": clip_files}
            ))
        clip_start += clip_secs
        
    return result

# --------------------------------------------------------------------------- #
#                             LLM Logic                                       #
# --------------------------------------------------------------------------- #

def _get_openai_client():
    """
    Configures the OpenAI client with http_client for robust connections.
    """
    _validate_strict_ready()
    return OpenAI(
        base_url=GPT4O_BASE_URL,
        api_key=GPT4O_API_KEY,
        http_client=httpx.Client(
            base_url=GPT4O_BASE_URL,
            follow_redirects=True,
            timeout=60.0 
        ),
    )

def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    timestamp, info = task
    files = info["files"]

    ckpt_path = os.path.join(caption_ckpt_folder, f"{timestamp}.json")
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                cached = json.load(f)
            if not _strict_mode() or cached.get("_admir_metadata", {}).get("caption_model") == GPT4O_MODEL_NAME:
                return timestamp, _strip_metadata(cached)
        except json.JSONDecodeError:
            pass

    clip_start_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[1]))

    user_prompt_text = CAPTION_PROMPT.replace(
        "CLIP_START_TIME", clip_start_time).replace(
        "CLIP_END_TIME", clip_end_time)

    content_list = [{"type": "text", "text": user_prompt_text}]
    
    # Image sampling: Limit payload size by taking at most 5 frames
    if len(files) > 5:
        step = len(files) // 5
        files = files[::step][:5]

    for img_path in files:
        b64 = encode_image_to_base64(img_path)
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low" 
            }
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_list}
    ]

    # Initialize client within process
    client = _get_openai_client()
    tries = MAX_API_RETRIES
    while tries:
        tries -= 1
        try:
            kwargs = {
                "model": GPT4O_MODEL_NAME,
                "messages": messages,
                "max_tokens": 1200,
                "temperature": 0.2,
            }
            try:
                response = create_chat_completion(client, **kwargs, response_format={"type": "json_object"})
            except Exception as format_error:
                if "response_format" not in str(format_error):
                    raise
                response = create_chat_completion(client, **kwargs)
            resp_str = response.choices[0].message.content
            parsed = robust_json_parse(resp_str)
            if not parsed:
                raise ValueError(f"Invalid JSON response: {resp_str[:200]}")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                ckpt_payload = dict(parsed)
                ckpt_payload["_admir_metadata"] = _run_metadata()
                json.dump(ckpt_payload, f, indent=4, ensure_ascii=False)
            return timestamp, parsed
        except Exception as e:
            if tries == 0:
                print(f"Failed to process {timestamp} after retries: {e}")
                if _strict_mode():
                    raise
    return timestamp, {}

def merge_subject_registries(registries: List[dict]) -> dict:
    if not registries: return {}
    prompt_text = MERGE_PROMPT.replace("REGISTRIES_PLACEHOLDER", json.dumps(registries))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]
    client = _get_openai_client()
    try:
        kwargs = {
            "model": GPT4O_MODEL_NAME,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.2,
        }
        try:
            response = create_chat_completion(client, **kwargs, response_format={"type": "json_object"})
        except Exception as format_error:
            if "response_format" not in str(format_error):
                raise
            response = create_chat_completion(client, **kwargs)
        return robust_json_parse(response.choices[0].message.content) or {}
    except Exception as e:
        print(f"Error merging registries: {e}")
        if _strict_mode():
            raise
        return {}

# --------------------------------------------------------------------------- #
#                             Main Pipeline                                   #
# --------------------------------------------------------------------------- #

def process_single_video_pipeline(
    video_path: str,
    output_root: str,
    workers: int,
    emb_dim: int,
    fps: float = None,
    clip_secs: int = None,
):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    base_dir = os.path.join(output_root, video_id)
    frames_dir = os.path.join(base_dir, "frames")
    captions_dir = os.path.join(base_dir, "captions")
    caption_ckpt_folder = os.path.join(captions_dir, "ckpt") 
    caption_file_path = os.path.join(captions_dir, "captions.json")
    database_json_path = os.path.join(base_dir, "database.json")
    
    os.makedirs(caption_ckpt_folder, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    fps = float(fps if fps is not None else getattr(config, 'VIDEO_FPS', 1))
    clip_secs = int(clip_secs if clip_secs is not None else getattr(config, 'CLIP_SECS', 5))

    print(f"\n[{video_id}] Processing...")
    _validate_strict_ready()

    # Step 1: Decode
    decode_video_to_frames(video_path, frames_dir, fps=fps)

    # Step 2: Generate Captions
    if os.path.exists(caption_file_path):
        with open(caption_file_path, "r", encoding="utf-8") as f:
            existing_captions = json.load(f)
        existing_metadata = existing_captions.get("_admir_metadata", {})
        if _strict_mode() and not _metadata_matches_current_run(existing_metadata, fps, clip_secs):
            raise RuntimeError(
                f"[{video_id}] Existing captions do not match paper-strict metadata. "
                "Use a clean output_root for a strict rerun."
            )
        print(f"[{video_id}] Captions already exist, metadata matches current run.")
    else:
        clips = gather_clip_frames(frames_dir, clip_secs, fps=fps)
        if not clips:
            print(f"[{video_id}] No frames found!")
            return

        caption_func = functools.partial(_caption_clip, caption_ckpt_folder=caption_ckpt_folder)
        
        results = []
        with mp.Pool(workers) as pool:
            results = list(tqdm(pool.imap_unordered(caption_func, clips), total=len(clips), desc=f"Captioning {video_id}"))

        partial_registries = []
        frame_captions = {}
        results = sorted(results, key=lambda x: float(x[0].split("_")[0]))
        
        for ts, parsed in results:
            if parsed:
                frame_captions[ts] = {"caption": parsed.get("clip_description", "")}
                if "subject_registry" in parsed:
                    partial_registries.append(parsed["subject_registry"])

        print(f"[{video_id}] Merging entities...")
        merged_registry = merge_subject_registries(partial_registries)
        frame_captions["subject_registry"] = merged_registry
        frame_captions["_admir_metadata"] = _run_metadata(video_id, fps, clip_secs)

        with open(caption_file_path, "w", encoding="utf-8") as f:
            json.dump(frame_captions, f, indent=4, ensure_ascii=False)
        print(f"[{video_id}] Captions saved.")

    # Step 3: Build Database
    print(f"[{video_id}] Building database.json (Embedding)...")
    success, msg = ensure_video_db_exists(caption_file_path, database_json_path, emb_dim)
    
    if success:
        print(f"[{video_id}] SUCCESS: {msg}")
    else:
        print(f"[{video_id}] FAILED to build DB: {msg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./data/raw_videos", help="Video file or folder")
    parser.add_argument("--output_root", type=str, default="./data/video_database", help="Output root folder")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=0)
    parser.add_argument("--emb_endpoint", type=str, default="http://localhost:8090")
    parser.add_argument("--fps", type=float, default=getattr(config, 'VIDEO_FPS', 1))
    parser.add_argument("--clip_secs", type=int, default=getattr(config, 'CLIP_SECS', 5))
    parser.add_argument("--max_videos", type=int, default=0, help="Debug/sample mode: process at most this many videos")
    
    args = parser.parse_args()

    global GPT4O_API_KEY, GPT4O_BASE_URL
    if args.api_key:
        GPT4O_API_KEY = args.api_key
        setattr(config, "OPENAI_API_KEY", args.api_key)
    if args.api_url:
        GPT4O_BASE_URL = args.api_url
        setattr(config, "OPENAI_BASE_URL", args.api_url)
        setattr(config, "LOCAL_VLLM_BASE_URL", args.api_url)

    emb_dim = _get_emb_dim(args.emb_dim, args.emb_endpoint)
    print(f"Target Embedding Dimension: {emb_dim}")

    if os.path.isfile(args.video_path):
        process_single_video_pipeline(args.video_path, args.output_root, args.workers, emb_dim, args.fps, args.clip_secs)
    elif os.path.isdir(args.video_path):
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path) if f.lower().endswith(video_exts)]
        video_files = sorted(video_files)
        if args.max_videos and args.max_videos > 0:
            video_files = video_files[:args.max_videos]
        print(f"Found {len(video_files)} videos.")
        for vid in video_files:
            try:
                process_single_video_pipeline(vid, args.output_root, args.workers, emb_dim, args.fps, args.clip_secs)
            except Exception as e:
                print(f"Error processing {vid}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()
