#!/usr/bin/env python3
"""Run AD-MIR on one user-provided advertising video and query."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import admir.config as config


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def _default_device() -> str:
    if os.environ.get("ADMIR_DEVICE"):
        return os.environ["ADMIR_DEVICE"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _validate_video(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"--video must point to a file, not a directory: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {path.suffix!r}. Supported: {supported}")


def _trim(text: Any, limit: int = 1600) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[:limit] + "\n[truncated]"


def _iter_trace(history: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for step_idx, item in enumerate(history, start=1):
        tool = item.get("tool", "unknown_tool")
        iteration = item.get("iter", step_idx)
        accepted = " accepted" if item.get("accepted") else ""
        yield f"### Step {step_idx}: `{tool}`{accepted}\n"
        yield f"- Iteration: `{iteration}`\n"
        if item.get("answer"):
            yield f"- Proposed answer: {_trim(item.get('answer'), 400)}\n"
        if item.get("evidence_check"):
            yield f"- Evidence check: {_trim(item.get('evidence_check'), 600)}\n"
        result = _trim(item.get("result"), 1800)
        if result:
            yield "\n```text\n" + result + "\n```\n"
        yield "\n"


def _write_trace_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# AD-MIR Run Trace\n\n",
        f"- Video: `{payload['video_path']}`\n",
        f"- Query: {payload['query']}\n",
        f"- Final answer: **{payload.get('answer', '')}**\n",
        f"- Generated at: `{payload['created_at']}`\n\n",
        "## Tool Trajectory\n\n",
    ]
    lines.extend(_iter_trace(payload.get("history", [])))
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AD-MIR multimodal memory for one ad video, then answer one user query."
    )
    parser.add_argument("--video", required=True, help="Path to your advertising video file.")
    parser.add_argument("--query", required=True, help="Natural-language question about the ad.")
    parser.add_argument("--output_db_root", default="./data/video_database")
    parser.add_argument("--results_dir", default="./results/custom_ad")
    parser.add_argument("--run_name", default="", help="Optional output subfolder name.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=config.VIDEO_FPS)
    parser.add_argument("--clip_secs", type=int, default=config.CLIP_SECS)
    parser.add_argument("--emb_dim", type=int, default=0)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--skip_asr", action="store_true", help="Skip ASR even when an ASR model is configured.")
    parser.add_argument("--asr_model", default=os.environ.get("ADMIR_ASR_MODEL", ""))
    parser.add_argument("--offline_ocr", action="store_true", help="Also pre-compute offline OCR. By default, OCR is handled by frame inspection during reasoning.")
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    _validate_video(video_path)
    if not args.query.strip():
        raise ValueError("--query cannot be empty.")

    from admir.agent import AdmirAgent
    from add_asr_ocr import process_single_video as add_asr_to_video_database
    from prepare_captions import _get_emb_dim, process_single_video_pipeline

    validate = getattr(config, "validate_strict_paper_config", None)
    if callable(validate):
        validate(require_api_key=True)
    if getattr(config, "STRICT_PAPER_MODE", False):
        if args.skip_asr:
            raise RuntimeError("Strict mode requires ASR; remove --skip_asr.")
        if args.offline_ocr:
            raise RuntimeError("Strict mode leaves OCR to frame_inspect; remove --offline_ocr.")

    video_id = video_path.stem
    output_db_root = Path(args.output_db_root).expanduser().resolve()
    base_dir = output_db_root / video_id
    if args.force_rebuild and base_dir.exists():
        shutil.rmtree(base_dir)

    emb_dim = _get_emb_dim(args.emb_dim, os.environ.get("ADMIR_EMBEDDING_ENDPOINT", ""))
    process_single_video_pipeline(
        str(video_path),
        str(output_db_root),
        workers=args.workers,
        emb_dim=emb_dim,
        fps=args.fps,
        clip_secs=args.clip_secs,
    )

    asr_status = "skipped"
    if not args.skip_asr:
        if args.asr_model:
            ok, message = add_asr_to_video_database(
                video_id,
                str(output_db_root),
                str(video_path.parent),
                args.asr_model,
                15,
                1,
                True,
                args.device,
                skip_ocr=not args.offline_ocr,
            )
            asr_status = message if ok else f"failed: {message}"
        elif getattr(config, "STRICT_PAPER_MODE", False):
            raise RuntimeError("Set --asr_model or ADMIR_ASR_MODEL before running strict mode.")
        else:
            asr_status = "skipped: no ASR model configured"

    database_path = base_dir / "database.json"
    captions_path = base_dir / "captions" / "captions.json"
    if not database_path.exists() or not captions_path.exists():
        raise FileNotFoundError(
            "AD-MIR database was not created. Check the captioning and embedding configuration."
        )

    agent = AdmirAgent(
        str(database_path),
        str(captions_path),
        max_iterations=config.MAX_ITERATIONS,
        embedding_dim=emb_dim,
    )
    result = agent.run(args.query)

    run_name = args.run_name or video_id
    run_dir = Path(args.results_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "video_id": video_id,
        "query": args.query,
        "answer": result.get("answer", ""),
        "raw_answer": result.get("raw_answer", ""),
        "error": result.get("error", ""),
        "traceback": result.get("traceback", ""),
        "history": result.get("history", []),
        "database_path": str(database_path),
        "captions_path": str(captions_path),
        "embedding_dim": emb_dim,
        "asr_status": asr_status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    answer_path = run_dir / "answer.json"
    trace_path = run_dir / "trace.md"
    answer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trace_markdown(trace_path, payload)

    print("\nAD-MIR answer:")
    print(payload["answer"])
    if payload.get("error"):
        print(f"\nError: {payload['error']}")
    print(f"\nSaved structured output: {answer_path}")
    print(f"Saved readable trace: {trace_path}")


if __name__ == "__main__":
    main()
