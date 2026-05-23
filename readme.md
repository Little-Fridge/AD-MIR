# AD-MIR

**AD-MIR: Bridging the Gap from Perception to Persuasion in Advertising Video Understanding via Structured Reasoning**

<p align="center">
  <img src="assets/admir-mascot.png" alt="AD-MIR mascot" width="320">
</p>

<p align="center">
  <em>From pixels to persuasion: AD-MIR turns ad videos into auditable evidence trails.</em>
</p>

AD-MIR is a tool-grounded video reasoning framework for advertising understanding. Instead of answering from a single pass over frames, it builds structured multimodal memory, retrieves relevant evidence, asks a communication expert to reason about persuasive intent, and verifies concrete visual/OCR/ASR anchors before producing a concise answer.

This repository is the public, provider-neutral implementation. It does **not** include private API keys, local server paths, model weights, AdsQA videos, or generated evaluation outputs.

## Why AD-MIR?

Advertising videos are engineered to imply more than they literally show: product claims appear in tiny text, emotional appeals unfold across edits, and the real answer often depends on connecting a visual object to a persuasion strategy. AD-MIR makes that reasoning explicit.

- **Structured memory:** converts videos into caption, ASR, subject-registry, and retrieval-ready evidence.
- **Tool-grounded reasoning:** combines global browsing, clip search, frame inspection, and expert interpretation.
- **Visual-anchor verification:** checks concrete claims against retrieved or inspected evidence before finalizing.
- **Auditable traces:** records tool calls and observations so answers can be inspected after inference.

## Visual Tour

### Reasoning Workflow

<p align="center">
  <img src="assets/figure1-workflow.png" alt="AD-MIR reasoning workflow" width="900">
</p>

AD-MIR first builds browse-based context and a high-level advertising narrative, then zooms into precise clips and frames when the answer depends on literal visual evidence.

### System Architecture

<p align="center">
  <img src="assets/figure2-architecture.png" alt="AD-MIR architecture" width="900">
</p>

The system couples a ReAct controller with a shared multimodal database and four interaction tools: Global Browse, Communication Expert, Clip Search, and Frame Inspect.

### Real AdsQA Case Trajectories

<p align="center">
  <img src="assets/case-gallery.png" alt="AD-MIR qualitative case gallery" width="900">
</p>

The cases above are real AdsQA examples from the paper. Each card shows retrieved or inspected frames, the question, the ground-truth answer, AD-MIR's answer, and the evidence trace used to support the final response.

<p align="center">
  <img src="assets/tool-grounding-cases.png" alt="AD-MIR tool grounding cases" width="440">
  <img src="assets/intent-reasoning-cases.png" alt="AD-MIR intent reasoning cases" width="440">
</p>

Tool-grounding cases emphasize literal verification, while intent-reasoning cases show how visual evidence and OCR/ASR cues support higher-level advertising interpretation.

## What Is Inside

- `admir/agent.py`: ReAct controller, fixed communication-expert initializer, evidence verification, and answer refinement.
- `admir/build_database.py`: global browse, clip search, frame inspection, subject registry activation, and vector database construction.
- `prepare_captions.py`: video decoding, clip captioning, subject registry construction, and database initialization.
- `add_asr_ocr.py`: timestamped ASR augmentation and optional offline OCR utility.
- `scripts/`: AdsQA download helpers, batch runners, direct VLM baseline runner, preflight checks, and official-prompt judge.

## Installation

```bash
conda env create -f admir.yml
conda activate admir
pip install -r admir_requirements.txt
```

You also need `ffmpeg` available on your `PATH`.

## Configure Runtime Models

AD-MIR uses OpenAI-compatible chat/completion endpoints and either local or remote embeddings. Model names are intentionally not hard-coded in this release. Copy `.env.example` and fill in your own endpoints and deployments:

```bash
cp .env.example .env
source .env
```

Minimal variables:

```bash
export OPENAI_API_KEY="<your-api-key-or-empty-for-local>"
export OPENAI_BASE_URL="<openai-compatible-chat-endpoint>"

export ADMIR_DEFAULT_LMM_MODEL="<your-default-lmm-or-vlm>"
export ADMIR_CAPTION_VLM_MODEL="<your-caption-vlm>"
export ADMIR_ORCHESTRATOR_LLM_MODEL="<your-controller-llm>"
export ADMIR_FRAME_INSPECT_MODEL="<your-frame-inspection-vlm>"
export ADMIR_COMMUNICATION_EXPERT_MODEL="<your-expert-llm-or-vlm>"
export ADMIR_REFINE_LLM_MODEL="<your-refinement-llm>"

export ADMIR_EMBEDDING_BACKEND="hf"
export ADMIR_HF_EMBEDDING_MODEL="<local-or-hub-embedding-model>"
export ADMIR_HF_EMBEDDING_DIM="1024"
export ADMIR_ASR_MODEL="<local-or-hub-asr-model>"
```

Set `ADMIR_STRICT_PAPER_MODE=1` only when you want fail-fast checks that every component model has been explicitly configured.

## Data Layout

Download AdsQA metadata and place videos under the expected root:

```bash
python scripts/download_adsqa.py --output_root ./data/AdsQA
```

Expected layout:

```text
data/
  AdsQA/
    raw_videos/
      <video_id>.mp4
    testset_question.json
    testset_groundtruth.json
```

The full video dataset is not stored in this repository. Follow the AdsQA license and download instructions for the video files.

## Build Multimodal Memory

Generate clip captions, subject registry entries, and the vector database:

```bash
python prepare_captions.py \
  --video_path ./data/AdsQA/raw_videos \
  --output_root ./data/video_database \
  --workers 4
```

Add timestamped ASR. In strict AD-MIR runs, OCR is verified by the frame-inspection tool rather than by offline pre-aggregation, so `--skip_ocr` is recommended:

```bash
python add_asr_ocr.py \
  --video_db_root ./data/video_database \
  --raw_video_root ./data/AdsQA/raw_videos \
  --asr_model "$ADMIR_ASR_MODEL" \
  --skip_ocr
```

## Run AD-MIR

Single sample:

```bash
python scripts/run_one_sample.py \
  --dataset_root ./data/AdsQA \
  --output_db_root ./data/video_database \
  --results_dir ./results/one_sample \
  --sample_index 0 \
  --asr_model "$ADMIR_ASR_MODEL" \
  --skip_ocr
```

Batch evaluation:

```bash
python scripts/run_adsqa_batch.py \
  --dataset_root ./data/AdsQA \
  --output_db_root ./data/video_database \
  --results_dir ./results/adsqa_batch \
  --num_samples 50 \
  --workers 2 \
  --asr_model "$ADMIR_ASR_MODEL" \
  --skip_ocr
```

Generic inference on your own question JSON:

```bash
python inference.py \
  --test_file ./data/testset.json \
  --video_db_root ./data/video_database \
  --results_dir ./results/custom_run \
  --workers 4
```

## Evaluate With The AdsQA Official Prompt

```bash
python scripts/judge_adsqa_official_prompt.py \
  --results_path ./results/adsqa_batch/predictions.jsonl \
  --groundtruth ./data/AdsQA/testset_groundtruth.json \
  --output_dir ./results/adsqa_batch_judge \
  --base_url "$OPENAI_BASE_URL" \
  --api_key "$OPENAI_API_KEY" \
  --model "$ADMIR_JUDGE_MODEL"
```

The judge script implements the public AdsQA 0/0.5/1 prompt and reports both strict and relaxed accuracy.

## Preflight

```bash
python scripts/preflight_paper_strict.py \
  --dataset_root ./data/AdsQA \
  --raw_video_root ./data/AdsQA/raw_videos \
  --asr_model "$ADMIR_ASR_MODEL" \
  --output_json ./results/preflight.json
```

## Repository Hygiene

The `.gitignore` excludes model weights, datasets, generated video databases, raw videos, logs, caches, and evaluation outputs. Keep only lightweight source code, documentation, examples, and figures in commits.

## Citation

```bibtex
@inproceedings{admir2026,
  title     = {AD-MIR: Bridging the Gap from Perception to Persuasion in Advertising Video Understanding via Structured Reasoning},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

## License

This project is released under the MIT License.
