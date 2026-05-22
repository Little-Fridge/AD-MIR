#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_selected_question_ids(path):
    ids = []
    seen = set()
    for row in read_jsonl(path):
        qid = row.get("question_id")
        if qid and qid not in seen:
            seen.add(qid)
            ids.append(qid)
    return ids


def find_video(video_root, video_id):
    root = Path(video_root)
    for ext in VIDEO_EXTS:
        p = root / f"{video_id}{ext}"
        if p.exists():
            return p
    matches = list(root.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    return root / f"{video_id}.mp4"


def sample_video_frames(video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = float(total / fps) if total > 0 and fps > 0 else 0.0
    frames = []

    if total > 0:
        count = min(max_frames, total)
        indices = np.linspace(0, max(total - 1, 0), count, dtype=int)
        for idx in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
    else:
        step = 1
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, {"decoded_frames": len(frames), "total_frames": total, "fps": fps, "duration": duration}


def clean_prediction(text):
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def move_inputs_to_device(inputs, device):
    moved = {}
    for k, v in inputs.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def build_prompt(question):
    return (
        "You are answering a question about an advertising video. "
        "Use only the visual evidence in the video. "
        "Answer directly and concisely in one sentence.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_official_pred(results_dir, question_id, prediction):
    qdir = Path(results_dir) / "official_format_results" / question_id
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / "pred.json", "w", encoding="utf-8") as f:
        json.dump([{"prediction": prediction, "score": ""}], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--select-predictions", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    status_path = out_dir / "status.json"

    gt = load_json(args.groundtruth)
    by_qid = {row["question_id"]: (idx, row) for idx, row in enumerate(gt)}
    selected_qids = [qid for qid in load_selected_question_ids(args.select_predictions) if qid in by_qid]
    records = []
    for qid in selected_qids:
        dataset_index, row = by_qid[qid]
        records.append((dataset_index, row))
    records = records[args.start :]
    if args.limit:
        records = records[: args.limit]

    done = set()
    if pred_path.exists() and not args.no_resume:
        for row in read_jsonl(pred_path):
            qid = row.get("question_id")
            if qid and not row.get("error"):
                done.add(qid)

    print(f"Loading processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Loading model from {args.model_path}", flush=True)
    target_device = args.device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
    dtype = torch.bfloat16 if target_device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(target_device)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model ready on {device}; total records={len(records)}; already_done={len(done)}", flush=True)

    started = time.time()
    completed = len(done)
    with open(pred_path, "a", encoding="utf-8") as fout:
        for local_idx, (dataset_index, item) in enumerate(records):
            question_id = item["question_id"]
            if question_id in done:
                continue

            video_id = item["video"]
            video_path = find_video(args.video_root, video_id)
            row = {
                "batch_index": args.start + local_idx,
                "dataset_index": dataset_index,
                "video_id": video_id,
                "question_id": question_id,
                "question": item.get("question", ""),
                "reference_answer": item.get("answer", ""),
                "meta_info": item.get("meta_info", ""),
                "question_type": item.get("question_type", []),
                "prediction": "",
                "raw_answer": "",
                "error": "",
                "traceback": "",
                "video_path": str(video_path),
                "max_frames": args.max_frames,
            }

            t0 = time.time()
            try:
                frames, video_meta = sample_video_frames(video_path, args.max_frames)
                messages = [
                    {
                        "role": "system",
                        "content": "You are a precise advertising video question-answering model.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": str(video_path)},
                            {"type": "text", "text": build_prompt(item.get("question", ""))},
                        ],
                    },
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    videos=[frames],
                    return_tensors="pt",
                    do_sample_frames=False,
                )
                inputs = move_inputs_to_device(inputs, device)
                input_len = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                new_tokens = generated[:, input_len:]
                raw_answer = processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                answer = clean_prediction(raw_answer)
                row.update(
                    {
                        "prediction": answer,
                        "raw_answer": raw_answer,
                        "video_meta": video_meta,
                        "timing_seconds": round(time.time() - t0, 3),
                    }
                )
                make_official_pred(out_dir, question_id, answer)
                completed += 1
                done.add(question_id)
                print(
                    f"[{completed}/{len(records)}] {question_id} ok "
                    f"({row['timing_seconds']}s) {answer[:120]}",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                row["timing_seconds"] = round(time.time() - t0, 3)
                make_official_pred(out_dir, question_id, "")
                print(f"[{completed}/{len(records)}] {question_id} ERROR {row['error']}", flush=True)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "completed": completed,
                        "total": len(records),
                        "errors": None,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "output_dir": str(out_dir),
                        "predictions_jsonl": str(pred_path),
                        "official_format_results": str(out_dir / "official_format_results"),
                    },
                    sf,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    main()
