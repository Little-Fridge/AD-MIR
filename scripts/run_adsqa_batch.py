#!/usr/bin/env python3
"""Run AD-MIR on a batch of AdsQA questions with resumable JSONL output."""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import admir.config as config
from admir.agent import AdmirAgent
from add_asr_ocr import process_single_video as add_asr_ocr_for_video
from prepare_captions import _get_emb_dim, process_single_video_pipeline


VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


class SampleTimeoutError(TimeoutError):
    pass


def _handle_sample_timeout(signum, frame) -> None:
    raise SampleTimeoutError("Sample timed out before finish_with_answer.")


def normalize_sample(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    video_id = item.get("video") or item.get("video_id") or item.get("video_name") or item.get("vid")
    question = item.get("question") or item.get("query") or item.get("Q")
    question_id = item.get("question_id") or item.get("id") or item.get("qid") or "sample"
    if not video_id or not question:
        return None
    return {
        "video_id": Path(str(video_id)).stem,
        "question": str(question),
        "question_id": str(question_id),
        "raw": item,
    }


def iter_json_samples(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                sample = normalize_sample(item)
                if sample:
                    yield sample
    elif isinstance(data, dict):
        sample = normalize_sample(data)
        if sample:
            yield sample
        for key in ("data", "questions", "test", "samples", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                yield from iter_json_samples(value)


def load_samples(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(iter_json_samples(json.load(f)))


def load_groundtruth(path: Path) -> dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    truth = {}
    for sample in iter_json_samples(data):
        qid = sample["question_id"]
        raw = sample["raw"]
        truth[qid] = {
            "answer": raw.get("answer", raw.get("gt_answer", "")),
            "meta_info": raw.get("meta_info", ""),
            "question_type": raw.get("question_type", []),
            "raw": raw,
        }
    return truth


def find_video(raw_video_root: Path, video_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        path = raw_video_root / f"{video_id}{ext}"
        if path.exists():
            return path
    for path in raw_video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and path.stem == video_id:
            return path
    return None


def is_success_record(item: Dict[str, Any]) -> bool:
    if item.get("error") or item.get("prediction") in ("", "Error", "Max iterations"):
        return False
    for step in item.get("history", []):
        if step.get("tool") != "communication_expert_tool":
            continue
        result = str(step.get("result", ""))
        if "Error calling expert model" in result or "Expert model call failed" in result:
            return False
    return True


def read_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            qid = item.get("question_id")
            if qid and is_success_record(item):
                done.add(str(qid))
    return done


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./data/AdsQA")
    parser.add_argument("--raw_video_root", default="")
    parser.add_argument("--test_file", default="./data/AdsQA/testset_question.json")
    parser.add_argument("--groundtruth_file", default="./data/AdsQA/testset_groundtruth.json")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_db_root", default="./data/video_database")
    parser.add_argument("--results_dir", default="./results/adsqa_batch")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip_asr_ocr", action="store_true")
    parser.add_argument("--skip_ocr", action="store_true")
    parser.add_argument("--asr_model", default=os.environ.get("ADMIR_ASR_MODEL", ""))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sample_timeout_seconds",
        type=int,
        default=int(os.environ.get("ADMIR_SAMPLE_TIMEOUT_SECONDS", "0")),
    )
    args = parser.parse_args()

    validate = getattr(config, "validate_strict_paper_config", None)
    if callable(validate):
        validate(require_api_key=True)

    dataset_root = Path(args.dataset_root).resolve()
    raw_video_root = Path(args.raw_video_root).resolve() if args.raw_video_root else dataset_root / "raw_videos"
    test_file = Path(args.test_file).resolve()
    groundtruth_file = Path(args.groundtruth_file).resolve()
    output_db_root = Path(args.output_db_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "predictions.jsonl"
    manifest_path = results_dir / "manifest.json"
    summary_path = results_dir / "summary.json"
    done = read_done(predictions_path)
    truth = load_groundtruth(groundtruth_file)
    samples = load_samples(test_file)

    selected = []
    for dataset_index, sample in enumerate(samples[args.start_index:], start=args.start_index):
        video_path = find_video(raw_video_root, sample["video_id"])
        if video_path is None:
            continue
        row = dict(sample)
        row["dataset_index"] = dataset_index
        row["video_path"] = str(video_path)
        if row["question_id"] in truth:
            row.update(truth[row["question_id"]])
        selected.append(row)
        if len(selected) >= args.num_samples:
            break

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "test_file": str(test_file),
                "groundtruth_file": str(groundtruth_file),
                "raw_video_root": str(raw_video_root),
                "output_db_root": str(output_db_root),
                "num_requested": args.num_samples,
                "num_selected": len(selected),
                "samples": [
                    {
                        "dataset_index": s["dataset_index"],
                        "question_id": s["question_id"],
                        "video_id": s["video_id"],
                        "question": s["question"],
                        "answer": s.get("answer", ""),
                        "question_type": s.get("question_type", []),
                    }
                    for s in selected
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    emb_dim = _get_emb_dim(config.AOAI_EMBEDDING_LARGE_DIM, os.environ.get("ADMIR_EMBEDDING_ENDPOINT", ""))
    processed = skipped = failed = 0
    started_at = time.time()

    for local_index, sample in enumerate(selected, start=1):
        qid = str(sample["question_id"])
        if qid in done:
            skipped += 1
            print(f"[{local_index}/{len(selected)}] SKIP existing {qid}", flush=True)
            continue

        print(f"[{local_index}/{len(selected)}] RUN {qid} video={sample['video_id']}", flush=True)
        record = {
            "batch_index": local_index - 1,
            "dataset_index": sample["dataset_index"],
            "video_id": sample["video_id"],
            "question_id": qid,
            "question": sample["question"],
            "reference_answer": sample.get("answer", ""),
            "meta_info": sample.get("meta_info", ""),
            "question_type": sample.get("question_type", []),
            "prediction": "",
            "raw_answer": "",
            "error": "",
            "traceback": "",
            "history": [],
            "timing_seconds": None,
        }
        t0 = time.time()
        try:
            if args.sample_timeout_seconds > 0:
                signal.signal(signal.SIGALRM, _handle_sample_timeout)
                signal.alarm(args.sample_timeout_seconds)
            process_single_video_pipeline(
                str(sample["video_path"]),
                str(output_db_root),
                workers=args.workers,
                emb_dim=emb_dim,
                fps=config.VIDEO_FPS,
                clip_secs=config.CLIP_SECS,
            )
            if not args.skip_asr_ocr:
                add_asr_ocr_for_video(
                    sample["video_id"],
                    str(output_db_root),
                    str(raw_video_root),
                    args.asr_model,
                    15,
                    1,
                    True,
                    args.device,
                    args.skip_ocr,
                )

            base_dir = output_db_root / sample["video_id"]
            agent = AdmirAgent(
                str(base_dir / "database.json"),
                str(base_dir / "captions" / "captions.json"),
                max_iterations=config.MAX_ITERATIONS,
                embedding_dim=config.AOAI_EMBEDDING_LARGE_DIM,
            )
            result = agent.run(sample["question"])
            record["prediction"] = result.get("answer", "")
            record["raw_answer"] = result.get("raw_answer", "")
            record["error"] = result.get("error", "")
            record["traceback"] = result.get("traceback", "")
            record["history"] = result.get("history", [])
            if record["error"]:
                failed += 1
            else:
                processed += 1
        except Exception as exc:
            failed += 1
            record["prediction"] = "Error"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            print(record["traceback"], flush=True)
        finally:
            if args.sample_timeout_seconds > 0:
                signal.alarm(0)
            record["timing_seconds"] = round(time.time() - t0, 3)
            append_jsonl(predictions_path, record)
            done.add(qid)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_selected": len(selected),
                        "processed_ok": processed,
                        "skipped_existing": skipped,
                        "failed": failed,
                        "written": len(done),
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "predictions_path": str(predictions_path),
                        "manifest_path": str(manifest_path),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(
                f"[{local_index}/{len(selected)}] DONE {qid} "
                f"answer={record['prediction'][:120]!r} time={record['timing_seconds']}s",
                flush=True,
            )

    print(f"Batch complete. Results: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
