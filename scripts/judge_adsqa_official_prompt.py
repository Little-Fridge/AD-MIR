#!/usr/bin/env python3
"""AdsQA official-prompt evaluator against an OpenAI-compatible local endpoint.

This reproduces the public AdsQA evaluation prompt and 0/0.5/1 aggregation,
while allowing the judge model/base URL to be local vLLM instead of GPT-4o.
It evaluates a subset when predictions contain only a subset of question IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


OFFICIAL_PROMPT = """
You are an advertising expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video's Meta-Information, Question, Golden Answer, and the Response to be judged below.\n
###The meta-information includes the advertisement video's theme, creative points, and a brief content description, which can be regarded as ground-truth information, as follows::
{meta_info}

###Question: 
{question}

###Golden Answer: 
{golden_answer}

###Rule:
1. If the response to be judged contains ALL key information of the golden answer or expresses the same meaning using other sentences or synonyms, it is considered a match with the golden answer, and the output is 1.
2. If the response to be judged does NOT contain the key information from the golden answer, it is considered a mismatch, and the output is 0.
3. The response to be judged should NOT contain any content that is contradictory, conflicting, or unreasonable when inferred from the meta-information. If such content exist, it is considered a mismatch, and the output is 0.
4. If the response to be judged contains the MOST of key information of the golden answer and, do NOT contain any information that is contradictory, conflicting, or unreasonable when inferred from the meta-information, it is considered a partial match, and the output is 0.5.

###Response to be judged: 
{response}

###Instructions:
Follow the format below and do not give any extra outputs:
Answer: 0 (if the response does not match)
Answer: 0.5 (if the response partially match)
Answer: 1 (if the response matches)
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_latest_predictions(path: Path) -> dict[str, dict[str, Any]]:
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        predictions_jsonl = path / "predictions.jsonl"
        files = [predictions_jsonl] if predictions_jsonl.exists() else sorted(path.rglob("*.jsonl"))
    records: dict[str, dict[str, Any]] = {}
    for file in files:
        data = load_jsonl(file) if file.suffix == ".jsonl" else load_json(file)
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id") or item.get("qid") or item.get("id")
            if qid is None:
                continue
            prediction = item.get("prediction")
            if prediction in (None, ""):
                prediction = item.get("answer", item.get("raw_answer", ""))
            records[str(qid)] = dict(item, prediction="" if prediction is None else str(prediction))
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def official_prediction_text(prediction: Any) -> str:
    pred_answer = "" if prediction is None else str(prediction)
    if "<answer>" in pred_answer:
        pred_answer = re.sub(r"(?s).*<answer>\s*(.*?)\s*</answer>.*", r"\1", pred_answer)
    words = pred_answer.split()
    if len(words) > 30:
        pred_answer = " ".join(words[:30])
    return pred_answer


def post_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def official_score(score_text: str) -> tuple[float, float, str]:
    gptscore = score_text.replace("Answer: ", "").strip()
    if "1" in gptscore:
        return 1.0, 1.0, gptscore
    if "0.5" in gptscore:
        return 0.0, 0.5, gptscore
    return 0.0, 0.0, gptscore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("ADMIR_JUDGE_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--eval_name", default="adsqa_official_prompt_eval.json")
    args = parser.parse_args()
    if not args.model:
        raise ValueError("Set --model or ADMIR_JUDGE_MODEL before running the judge.")

    results_path = Path(args.results_path)
    output_dir = Path(args.output_dir)
    official_results_dir = output_dir / "official_format_results"
    official_results_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_latest_predictions(results_path)
    raw_groundtruth = load_json(Path(args.groundtruth))
    refs = [item for item in raw_groundtruth if str(item.get("question_id")) in predictions]
    refs_by_id = {str(item.get("question_id")): item for item in refs}
    missing = [qid for qid in predictions if qid not in refs_by_id]
    if missing:
        raise ValueError(f"Predictions missing from ground truth: {missing[:10]}")

    strict_acc_scores = {f"Type_{i}": 0.0 for i in range(1, 6)}
    strict_acc_counts = {f"Type_{i}": 0 for i in range(1, 6)}
    relax_acc_scores = {f"Type_{i}": 0.0 for i in range(1, 6)}
    relax_acc_counts = {f"Type_{i}": 0 for i in range(1, 6)}

    strict_acc = 0.0
    relaxed_acc = 0.0
    pred_nums = 0
    details: list[dict[str, Any]] = []

    for item in refs:
        qid = str(item["question_id"])
        pred_record = predictions[qid]
        pred_text = official_prediction_text(pred_record.get("prediction", ""))
        prompt = OFFICIAL_PROMPT.format(
            meta_info=item.get("meta_info", ""),
            question=item.get("question", ""),
            golden_answer=item.get("answer", item.get("gt_answer", "")),
            response=pred_text,
        )

        score_text = ""
        last_error = ""
        for attempt in range(args.retries + 1):
            try:
                score_text = post_chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                if attempt >= args.retries:
                    raise
                time.sleep(5)

        strict_score, relaxed_score, normalized_score = official_score(score_text)
        pred_nums += 1
        strict_acc += strict_score
        relaxed_acc += relaxed_score

        question_types = item.get("question_type", [])
        for typee in question_types:
            if typee not in strict_acc_scores:
                strict_acc_scores[typee] = 0.0
                strict_acc_counts[typee] = 0
                relax_acc_scores[typee] = 0.0
                relax_acc_counts[typee] = 0
            strict_acc_scores[typee] += strict_score
            relax_acc_scores[typee] += relaxed_score
            strict_acc_counts[typee] += 1
            relax_acc_counts[typee] += 1

        official_item = [
            {
                "question_id": qid,
                "prediction": pred_record.get("prediction", ""),
                "score": score_text,
            }
        ]
        qid_dir = official_results_dir / qid
        qid_dir.mkdir(parents=True, exist_ok=True)
        with (qid_dir / args.eval_name).open("w", encoding="utf-8") as f:
            json.dump(official_item, f, ensure_ascii=False, indent=4)

        details.append(
            {
                "question_id": qid,
                "video": item.get("video"),
                "question": item.get("question"),
                "answer": item.get("answer", item.get("gt_answer", "")),
                "prediction": pred_record.get("prediction", ""),
                "official_truncated_prediction": pred_text,
                "score_raw": score_text,
                "score_normalized": normalized_score,
                "strict_score": strict_score,
                "relaxed_score": relaxed_score,
                "question_type": question_types,
                "last_error": last_error,
            }
        )
        print(f"{qid}: {normalized_score}", flush=True)

    target_nums = len(refs)
    strict_by_type = {
        k: (strict_acc_scores[k] / strict_acc_counts[k] if strict_acc_counts[k] else None)
        for k in strict_acc_scores
    }
    relaxed_by_type = {
        k: (relax_acc_scores[k] / relax_acc_counts[k] if relax_acc_counts[k] else None)
        for k in relax_acc_scores
    }
    summary = {
        "results_path": str(results_path.resolve()),
        "groundtruth": str(Path(args.groundtruth).resolve()),
        "official_prompt": True,
        "judge_base_url": args.base_url,
        "judge_model": args.model,
        "eval_name": args.eval_name,
        "num_predictions": len(predictions),
        "num_scored": pred_nums,
        "num_targets": target_nums,
        "strict_accuracy": strict_acc / target_nums if target_nums else 0.0,
        "relaxed_accuracy": relaxed_acc / target_nums if target_nums else 0.0,
        "strict_by_type": strict_by_type,
        "relaxed_by_type": relaxed_by_type,
        "details_path": str((output_dir / "judge_official_prompt_details.jsonl").resolve()),
        "official_format_results_dir": str(official_results_dir.resolve()),
        "note": "Uses the public AdsQA prompt and official 0/0.5/1 aggregation with the configured judge model.",
    }

    with (output_dir / "official_subset_groundtruth.json").open("w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)
    with (output_dir / "judge_official_prompt_details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "judge_official_prompt_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
