import os

# Backend configuration: hf (local SentenceTransformer) or openai (Remote)
EMBEDDING_BACKEND = os.environ.get("ADMIR_EMBEDDING_BACKEND", "hf")

# Local embedding model path/name
HF_EMBEDDING_MODEL_NAME = os.environ.get("ADMIR_HF_EMBEDDING_MODEL", "./model_zoo/bge-m3")
HF_EMBEDDING_DIM = int(os.environ.get("ADMIR_HF_EMBEDDING_DIM", "1024"))
HF_EMBEDDING_BATCH_SIZE = int(os.environ.get("ADMIR_HF_EMBEDDING_BATCH_SIZE", "8"))

# Video database root directory
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = "360"
VIDEO_FPS = 1
CLIP_SECS = 5

# OpenAI API Key (defaults to empty, expects env var)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Local LLM/VLLM Configuration
LOCAL_VLLM_BASE_URL = os.environ.get("ADMIR_LOCAL_VLLM_BASE_URL", "http://localhost:8000/v1")

# Service Endpoint Lists
AOAI_CAPTION_VLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_CAPTION_VLM_MODEL_NAME = os.environ.get("ADMIR_CAPTION_VLM_MODEL", "Qwen3-VL-8B-Instruct")

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = os.environ.get("ADMIR_ORCHESTRATOR_LLM_MODEL", "Qwen3-VL-8B-Instruct")

AOAI_TOOL_VLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_TOOL_VLM_MODEL_NAME = os.environ.get("ADMIR_TOOL_VLM_MODEL", "Qwen3-VL-8B-Instruct")
AOAI_TOOL_VLM_MAX_FRAME_NUM = int(os.environ.get("ADMIR_TOOL_VLM_MAX_FRAME_NUM", "20"))

# Embedding Service
# List of endpoints for embedding services
AOAI_EMBEDDING_RESOURCE_LIST = [os.environ.get("ADMIR_EMBEDDING_ENDPOINT", "http://0.0.0.0:8090")]

# Embedding model name and dimension
AOAI_EMBEDDING_LARGE_MODEL_NAME = HF_EMBEDDING_MODEL_NAME if EMBEDDING_BACKEND == "hf" else os.environ.get("ADMIR_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
AOAI_EMBEDDING_LARGE_DIM = HF_EMBEDDING_DIM if EMBEDDING_BACKEND == "hf" else 3072

# Runtime settings
LITE_MODE = False
GLOBAL_BROWSE_TOPK = 40
OVERWRITE_CLIP_SEARCH_TOPK = 8

SINGLE_CHOICE_QA = False
MAX_ITERATIONS = 15