# Admir - Official Implementation

This repository contains the official implementation for the paper: **AD-MIR: Bridging the Gap from Perception to Persuasion in Advertising Video Understanding via Structured Reasoning**.

Admir is a ReAct-based video agent that leverages hybrid retrieval, context-anchored subject registries, and audio-visual fusion to solve complex video question answering tasks.

## 1. Requirements

### Environment Setup

We recommend using Conda to manage the environment:

```setup
conda env create -f conda.yml
conda activate admir
pip install -r admir_requirements.txt
```

### Key Dependencies
* **Python**: 3.11.14
* **NanoVectorDB**: For lightweight vector storage.
* **OpenAI**: For LLM and VLM API calls.
* **Transformers**: For local Whisper ASR models.
* **Torch**: Required for local embedding/ASR models.

## 2. Configuration

The system relies on environment variables for configuration to ensure security and flexibility. You can set these in a `.env` file or export them directly.

**Essential Variables:**

```bash
# API Keys
export OPENAI_API_KEY="..."

# Base URLs (if using vLLM or custom endpoints)
export ADMIR_LOCAL_VLLM_BASE_URL="xxx"
export ADMIR_EMBEDDING_ENDPOINT="xxx"

# Model Selection
export ADMIR_HF_EMBEDDING_MODEL="./model_zoo/bge-m3"
export ADMIR_CAPTION_VLM_MODEL="Qwen2.5-VL-7B-Instruct"
```

See `admir/config.py` for a full list of configurable parameters.

## 3. Data Preparation

Data preparation consists of two steps: Captioning/DB Initialization and ASR/OCR Extraction.

### Directory Structure
Prepare your raw videos in the following structure:

```
./data/
  ├── raw_videos/
  │   ├── video1.mp4
  │   ├── video2.mp4
  │   └── ...
  ├── video_database/  (Output directory)
```

### Step 1: Initialize Database & Generate Captions
This script decodes videos, generates semantic captions using VLMs, and initializes the vector database.

```bash
python prepare_captions.py \
  --video_path ./data/raw_videos \
  --output_root ./data/video_database \
  --workers 8
```

### Step 2: Add ASR and OCR
Enhance the database with Automatic Speech Recognition (Whisper) and Optical Character Recognition (GPT-4o/VLM).

```bash
python add_asr_ocr.py \
  --video_db_root ./data/video_database \
  --whisper_model ./model_zoo/whisper-base \
  --ocr_workers 8
```

## 4. Inference

To run the Admir agent on the benchmark:

```bash
python inference.py \
  --test_file ./data/testset.json \
  --video_db_root ./data/video_database \
  --results_dir ./results/exp_admir_v1 \
  --workers 8
```

**Arguments:**
* `--test_file`: JSON file containing the questions (Format: `[{"question_id": "...", "video": "...", "question": "..."}]`).
* `--video_db_root`: Path to the processed video database.
* `--workers`: Number of concurrent agents to run.

## 5. File Structure

```
.
├── admir/                  # Core package
│   ├── agent.py            # AdmirAgent (ReAct implementation)
│   ├── build_database.py   # Hybrid retrieval & tools implementation
│   ├── config.py           # Global configuration
│   ├── func_call_shema.py  # Function calling schemas
│   └── utils.py            # Helper utilities
├── add_asr_ocr.py          # Script for adding Audio/Text modalities
├── prepare_captions.py     # Script for DB init and visual captioning
├── inference.py            # Main evaluation script
└── requirements.txt        # Dependencies
```

## License

This project is licensed under the MIT License.