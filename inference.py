"""
High-Concurrency Inference Pipeline for Admir
"""

import os
import json
import argparse
import traceback
import threading
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from admir.agent import AdmirAgent
import admir.config as config

_db_init_locks = {}
_lock_creation_lock = threading.Lock()

def get_db_lock(video_id: str) -> threading.Lock:
    with _lock_creation_lock:
        if video_id not in _db_init_locks:
            _db_init_locks[video_id] = threading.Lock()
        return _db_init_locks[video_id]

def get_video_paths(video_id: str, db_root: str) -> Dict[str, str]:
    base_dir = Path(db_root) / video_id
    return {
        "base": str(base_dir),
        "frames": str(base_dir / "frames"),
        "captions": str(base_dir / "captions" / "captions.json"),
        "database": str(base_dir / "database.json")
    }

def process_single_sample_worker(item: Dict, db_root: str, results_dir: str, max_iterations: int, max_retries: int) -> tuple:
    question_id = item.get("question_id")
    video_id = item.get("video")
    result_path = Path(results_dir) / f"{video_id}_{question_id}" / "pred.json"
    
    if result_path.exists(): return True, "Skipped"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            paths = get_video_paths(video_id, db_root)
            if not os.path.exists(paths["database"]): return False, "DB missing"
            
            with get_db_lock(video_id):
                agent = AdmirAgent(paths["database"], paths["captions"], max_iterations)
            
            result = agent.run(item.get("question"))
            
            with open(result_path, "w") as f:
                json.dump([{"prediction": result.get("answer", "")}], f)
            return True, "Success"
        except Exception as e:
            if attempt == max_retries - 1: return False, str(e)
    return False, "Failed"

def run_inference(test_file, db_root, results_dir, workers, max_retries):
    with open(test_file) as f: data = json.load(f)
    
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(process_single_sample_worker, item, db_root, results_dir, 15, max_retries): item for item in data}
        for f in tqdm(as_completed(futures), total=len(data)):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="./data/testset.json")
    parser.add_argument("--video_db_root", default="./video_database")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    run_inference(args.test_file, args.video_db_root, args.results_dir, args.workers, 2)