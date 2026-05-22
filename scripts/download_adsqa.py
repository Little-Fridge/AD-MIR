#!/usr/bin/env python3
"""Download AdsQA metadata and original videos.

The Hugging Face dataset hosts the benchmark files and a video_urls.json file.
This script snapshots the dataset repo, then downloads the full videos into
raw_videos/ using yt-dlp or direct HTTP downloads.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple


def snapshot_dataset(repo_id: str, output_root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Please install huggingface_hub to download AdsQA metadata.") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_root),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _iter_video_entries(data) -> Iterable[Tuple[str, str]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                yield str(key), value
            elif isinstance(value, dict):
                url = value.get("url") or value.get("video_url") or value.get("link")
                target_name = value.get("target_name") or value.get("filename")
                video_id = value.get("video_id") or value.get("video") or (Path(target_name).stem if target_name else key)
                if url:
                    yield str(video_id), str(url)
        return

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                yield str(i), item
            elif isinstance(item, dict):
                url = item.get("url") or item.get("video_url") or item.get("link")
                target_name = item.get("target_name") or item.get("filename")
                video_id = item.get("video_id") or item.get("video") or item.get("id") or (Path(target_name).stem if target_name else str(i))
                if url:
                    yield str(video_id), str(url)


def load_video_urls(dataset_root: Path) -> list[Tuple[str, str]]:
    url_file = dataset_root / "video_urls.json"
    if not url_file.exists():
        raise FileNotFoundError(f"Missing {url_file}")
    with url_file.open(encoding="utf-8") as f:
        data = json.load(f)
    entries = list(_iter_video_entries(data))
    if not entries:
        raise ValueError(f"No video URLs found in {url_file}")
    return entries


def _direct_download(url: str, output_path: Path) -> bool:
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(output_path)
            return True
        return False
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def download_one(video_id: str, url: str, raw_dir: Path, retries: int) -> bool:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in raw_dir.glob(f"{video_id}.*") if p.suffix != ".part"]
    if any(p.stat().st_size > 0 for p in existing):
        return True

    direct_target = raw_dir / f"{video_id}.mp4"
    for _ in range(retries):
        if _direct_download(url, direct_target):
            return True
        time.sleep(2)

    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        return False

    output_template = str(raw_dir / f"{video_id}.%(ext)s")
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--continue",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/best",
        "-o",
        output_template,
        url,
    ]
    for _ in range(retries):
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            return any(p.stat().st_size > 0 for p in raw_dir.glob(f"{video_id}.*"))
        time.sleep(2)
    return False


def download_many(entries: list[Tuple[str, str]], raw_dir: Path, retries: int, workers: int) -> list[dict]:
    failures = []
    total = len(entries)
    workers = max(1, workers)

    def _run(index: int, video_id: str, url: str) -> dict:
        print(f"[start {index}/{total}] {video_id}: {url}", flush=True)
        ok = download_one(video_id, url, raw_dir, retries)
        status = "ok" if ok else "failed"
        print(f"[{status} {index}/{total}] {video_id}", flush=True)
        return {"ok": ok, "video_id": video_id, "url": url}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run, index, video_id, url)
            for index, (video_id, url) in enumerate(entries, start=1)
        ]
        for future in as_completed(futures):
            result = future.result()
            if not result["ok"]:
                failures.append({"video_id": result["video_id"], "url": result["url"]})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="TsinghuaC3I/AdsQA")
    parser.add_argument("--output_root", default="./data/AdsQA")
    parser.add_argument("--raw_video_dir", default="")
    parser.add_argument("--skip_metadata", action="store_true")
    parser.add_argument("--skip_videos", action="store_true")
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    raw_dir = Path(args.raw_video_dir).resolve() if args.raw_video_dir else output_root / "raw_videos"

    if not args.skip_metadata:
        snapshot_dataset(args.repo_id, output_root)

    if args.skip_videos:
        return

    entries = load_video_urls(output_root)
    if args.max_videos > 0:
        entries = entries[: args.max_videos]

    failures = download_many(entries, raw_dir, args.retries, args.workers)

    if failures:
        failure_file = output_root / "video_download_failures.json"
        with failure_file.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"Failed videos: {len(failures)}. See {failure_file}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
