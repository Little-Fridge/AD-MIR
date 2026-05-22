#!/usr/bin/env python3
"""Preflight checks for AD-MIR runs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from admir import config


def _exists(path: str) -> bool:
    return Path(path).exists()


def _video_count(raw_video_root: Path) -> int:
    if not raw_video_root.exists():
        return 0
    return sum(1 for p in raw_video_root.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"})


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./data/AdsQA")
    parser.add_argument("--raw_video_root", default="./data/AdsQA/raw_videos")
    parser.add_argument("--video_id", default="")
    parser.add_argument("--asr_model", default=os.environ.get("ADMIR_ASR_MODEL", ""))
    parser.add_argument("--output_json", default="./results/preflight.json")
    parser.add_argument("--test_embedding", action="store_true")
    parser.add_argument("--test_asr", action="store_true")
    args = parser.parse_args()

    checks = {}
    try:
        import openai
        checks["official_openai_sdk"] = getattr(openai, "__file__", "")
        checks["openai_version"] = getattr(openai, "__version__", "unknown")
    except Exception as exc:
        checks["official_openai_sdk_error"] = repr(exc)

    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401
        checks["flagembedding_import"] = True
    except Exception as exc:
        checks["flagembedding_import"] = False
        checks["flagembedding_error"] = repr(exc)

    try:
        import json_repair  # noqa: F401
        checks["json_repair_import"] = True
    except Exception as exc:
        checks["json_repair_import"] = False
        checks["json_repair_error"] = repr(exc)

    dataset_root = Path(args.dataset_root)
    raw_video_root = Path(args.raw_video_root)
    sample_video = raw_video_root / f"{Path(args.video_id).stem}.mp4" if args.video_id else Path("")

    checks.update({
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "caption_model": config.AOAI_CAPTION_VLM_MODEL_NAME,
        "orchestrator_model": config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
        "frame_inspect_model": config.AOAI_FRAME_INSPECT_MODEL_NAME,
        "communication_expert_model": config.AOAI_COMMUNICATION_EXPERT_MODEL_NAME,
        "refine_model": config.AOAI_REFINE_LLM_MODEL_NAME,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "embedding_model": config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        "embedding_dim": config.AOAI_EMBEDDING_LARGE_DIM,
        "global_browse_topk": config.GLOBAL_BROWSE_TOPK,
        "clip_search_min_topk": config.CLIP_SEARCH_MIN_TOPK,
        "clip_search_max_topk": config.OVERWRITE_CLIP_SEARCH_TOPK,
        "expert_max_grid_frames": config.EXPERT_MAX_GRID_FRAMES,
        "max_iterations": config.MAX_ITERATIONS,
        "dataset_root_exists": dataset_root.exists(),
        "raw_video_count": _video_count(raw_video_root),
        "sample_video_exists": sample_video.exists(),
        "sample_video_size": sample_video.stat().st_size if sample_video.exists() else 0,
        "embedding_model_exists": Path(str(config.HF_EMBEDDING_MODEL_NAME)).exists()
        if config.EMBEDDING_BACKEND == "hf" and config.HF_EMBEDDING_MODEL_NAME
        else None,
        "asr_model_exists": Path(args.asr_model).exists() if args.asr_model else None,
        "ffmpeg_available": _ffmpeg_available(),
    })

    try:
        config.validate_strict_paper_config(require_api_key=True)
        checks["strict_config_valid"] = True
    except Exception as exc:
        checks["strict_config_valid"] = False
        checks["strict_config_error"] = f"{type(exc).__name__}: {exc}"

    if args.test_embedding:
        from admir.utils import AzureOpenAIEmbeddingService
        emb = AzureOpenAIEmbeddingService.get_embeddings(
            config.AOAI_EMBEDDING_RESOURCE_LIST,
            config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            ["Screwfix Sprint delivery speed"],
            config.OPENAI_API_KEY,
        )[0]["embedding"]
        checks["embedding_test_dim"] = len(emb)
        checks["embedding_test_first3"] = [round(float(x), 6) for x in emb[:3]]

    if args.test_asr:
        if not args.asr_model:
            raise RuntimeError("Set --asr_model or ADMIR_ASR_MODEL before running --test_asr.")
        from add_asr_ocr import extract_asr_whisper_hf
        asr_text = extract_asr_whisper_hf(str(sample_video), args.asr_model, "cuda")
        checks["asr_test_chars"] = len(asr_text)
        checks["asr_test_preview"] = asr_text[:500]

    checks["ready_to_run"] = bool(
        checks.get("strict_config_valid")
        and checks.get("flagembedding_import")
        and checks.get("json_repair_import")
        and checks.get("dataset_root_exists")
        and checks.get("raw_video_count", 0) > 0
        and (args.video_id == "" or checks.get("sample_video_exists"))
        and checks.get("ffmpeg_available")
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not checks["ready_to_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
