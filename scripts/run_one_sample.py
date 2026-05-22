#!/usr/bin/env python3
"""Build one AdsQA video database and run one AD-MIR inference sample."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import admir.config as config
from admir.agent import AdmirAgent
from add_asr_ocr import process_single_video as add_asr_ocr_for_video
from prepare_captions import _get_emb_dim, process_single_video_pipeline


VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


def _normalize_sample(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    video_id = item.get("video") or item.get("video_id") or item.get("video_name") or item.get("vid")
    question = item.get("question") or item.get("query") or item.get("Q")
    question_id = item.get("question_id") or item.get("id") or item.get("qid") or "sample"
    if not video_id or not question:
        return None
    video_id = Path(str(video_id)).stem
    return {"video_id": video_id, "question": str(question), "question_id": str(question_id), "raw": item}


def _iter_json_samples(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                sample = _normalize_sample(item)
                if sample:
                    yield sample
    elif isinstance(data, dict):
        sample = _normalize_sample(data)
        if sample:
            yield sample
        for key in ("data", "questions", "test", "samples", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                yield from _iter_json_samples(value)


def load_samples(test_file: Path) -> list[Dict[str, Any]]:
    with test_file.open(encoding="utf-8") as f:
        data = json.load(f)
    samples = list(_iter_json_samples(data))
    if not samples:
        raise ValueError(f"No AdsQA-like samples found in {test_file}")
    return samples


def auto_find_test_file(dataset_root: Path) -> Path:
    preferred = [
        "test.json",
        "testset.json",
        "qa_test.json",
        "annotations/test.json",
        "questions/test.json",
    ]
    for rel in preferred:
        path = dataset_root / rel
        if path.exists():
            return path
    for path in sorted(dataset_root.rglob("*.json")):
        if path.name == "video_urls.json":
            continue
        try:
            if load_samples(path):
                return path
        except Exception:
            continue
    raise FileNotFoundError("Could not find a QA JSON file under the dataset root.")


def find_video(raw_video_root: Path, video_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        path = raw_video_root / f"{video_id}{ext}"
        if path.exists():
            return path
    for path in raw_video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and path.stem == video_id:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./data/AdsQA")
    parser.add_argument("--raw_video_root", default="")
    parser.add_argument("--test_file", default="")
    parser.add_argument("--video_id", default="")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_db_root", default="./data/video_database")
    parser.add_argument("--results_dir", default="./results/one_sample")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip_asr_ocr", action="store_true")
    parser.add_argument("--skip_ocr", action="store_true", help="Run Whisper ASR but do not add offline OCR context.")
    parser.add_argument("--asr_model", default=os.environ.get("ADMIR_ASR_MODEL", ""))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate = getattr(config, "validate_strict_paper_config", None)
    if callable(validate):
        validate(require_api_key=True)
    if getattr(config, "STRICT_PAPER_MODE", False):
        if args.skip_asr_ocr:
            raise RuntimeError("Strict mode requires ASR; do not pass --skip_asr_ocr.")
        if not args.skip_ocr:
            raise RuntimeError("Strict mode leaves OCR to frame_inspect; pass --skip_ocr for ASR-only augmentation.")

    dataset_root = Path(args.dataset_root).resolve()
    raw_video_root = Path(args.raw_video_root).resolve() if args.raw_video_root else dataset_root / "raw_videos"
    test_file = Path(args.test_file).resolve() if args.test_file else auto_find_test_file(dataset_root)
    samples = load_samples(test_file)

    selected = None
    if args.video_id:
        for sample in samples:
            if sample["video_id"] == Path(args.video_id).stem:
                selected = sample
                break
    else:
        for sample in samples[args.sample_index:]:
            if find_video(raw_video_root, sample["video_id"]):
                selected = sample
                break
    if selected is None:
        raise FileNotFoundError("Could not find a sample with an available raw video.")

    video_path = find_video(raw_video_root, selected["video_id"])
    if video_path is None:
        raise FileNotFoundError(f"Missing raw video for {selected['video_id']} under {raw_video_root}")

    output_db_root = Path(args.output_db_root).resolve()
    emb_dim = _get_emb_dim(config.AOAI_EMBEDDING_LARGE_DIM, os.environ.get("ADMIR_EMBEDDING_ENDPOINT", ""))

    process_single_video_pipeline(
        str(video_path),
        str(output_db_root),
        workers=args.workers,
        emb_dim=emb_dim,
        fps=config.VIDEO_FPS,
        clip_secs=config.CLIP_SECS,
    )

    if not args.skip_asr_ocr:
        add_asr_ocr_for_video(
            selected["video_id"],
            str(output_db_root),
            str(raw_video_root),
        args.asr_model,
            15,
            1,
            True,
            args.device,
            args.skip_ocr,
        )

    base_dir = output_db_root / selected["video_id"]
    agent = AdmirAgent(
        str(base_dir / "database.json"),
        str(base_dir / "captions" / "captions.json"),
        max_iterations=config.MAX_ITERATIONS,
        embedding_dim=config.AOAI_EMBEDDING_LARGE_DIM,
    )
    result = agent.run(selected["question"])

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "video_id": selected["video_id"],
        "question_id": selected["question_id"],
        "question": selected["question"],
        "prediction": result.get("answer", ""),
        "raw_answer": result.get("raw_answer", ""),
        "error": result.get("error", ""),
        "traceback": result.get("traceback", ""),
        "history": result.get("history", []),
    }
    with (results_dir / "one_sample_result.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
