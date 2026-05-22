"""Runtime configuration for AD-MIR.

The open-source release intentionally does not hard-code provider-specific
model names, API keys, server paths, or private endpoints. Configure all model
deployments through environment variables or a local ``.env`` file.
"""

import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


# Optional strict reproducibility mode. In the public release this only checks
# that the user explicitly configured every required component; it does not
# prescribe private model names.
STRICT_PAPER_MODE = _env("ADMIR_STRICT_PAPER_MODE", "0") == "1"

# Embedding backend: "hf" for a local HuggingFace/FlagEmbedding model, or
# "openai" for an OpenAI-compatible embeddings endpoint.
EMBEDDING_BACKEND = _env("ADMIR_EMBEDDING_BACKEND", "hf")
HF_EMBEDDING_MODEL_NAME = _env("ADMIR_HF_EMBEDDING_MODEL")
HF_EMBEDDING_DIM = _env_int("ADMIR_HF_EMBEDDING_DIM", 1024)
HF_EMBEDDING_BATCH_SIZE = _env_int("ADMIR_HF_EMBEDDING_BATCH_SIZE", 8)

# Video preprocessing.
VIDEO_DATABASE_FOLDER = _env("ADMIR_VIDEO_DATABASE_FOLDER", "./video_database/")
VIDEO_RESOLUTION = _env("ADMIR_VIDEO_RESOLUTION", "360")
VIDEO_FPS = _env_float("ADMIR_VIDEO_FPS", 1.0)
CLIP_SECS = _env_int("ADMIR_CLIP_SECS", 5)

# OpenAI-compatible chat/completion endpoint. Leave the key empty for local
# servers that do not enforce authentication.
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
LOCAL_VLLM_BASE_URL = _env("ADMIR_LOCAL_VLLM_BASE_URL", OPENAI_BASE_URL)

# Component models. Set a shared ADMIR_DEFAULT_LMM_MODEL if one model should
# serve all text/VLM roles, or set each component separately.
DEFAULT_LMM_MODEL = _env("ADMIR_DEFAULT_LMM_MODEL")
AOAI_CAPTION_VLM_MODEL_NAME = _env("ADMIR_CAPTION_VLM_MODEL", DEFAULT_LMM_MODEL)
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = _env("ADMIR_ORCHESTRATOR_LLM_MODEL", DEFAULT_LMM_MODEL)
AOAI_FRAME_INSPECT_MODEL_NAME = _env("ADMIR_FRAME_INSPECT_MODEL", DEFAULT_LMM_MODEL)
AOAI_COMMUNICATION_EXPERT_MODEL_NAME = _env("ADMIR_COMMUNICATION_EXPERT_MODEL", DEFAULT_LMM_MODEL)
AOAI_TOOL_VLM_MODEL_NAME = _env("ADMIR_TOOL_VLM_MODEL", AOAI_FRAME_INSPECT_MODEL_NAME)
AOAI_REFINE_LLM_MODEL_NAME = _env("ADMIR_REFINE_LLM_MODEL", DEFAULT_LMM_MODEL)
AOAI_TOOL_VLM_MAX_FRAME_NUM = _env_int("ADMIR_TOOL_VLM_MAX_FRAME_NUM", 20)

AOAI_CAPTION_VLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_TOOL_VLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]
AOAI_REFINE_LLM_ENDPOINT_LIST = [LOCAL_VLLM_BASE_URL]

# Embedding service. For local HF embeddings this is unused; for remote
# embeddings, point it at an OpenAI-compatible /v1 endpoint.
EMBEDDING_ENDPOINT = _env("ADMIR_EMBEDDING_ENDPOINT")
AOAI_EMBEDDING_RESOURCE_LIST = [EMBEDDING_ENDPOINT] if EMBEDDING_ENDPOINT else []
AOAI_EMBEDDING_LARGE_MODEL_NAME = (
    HF_EMBEDDING_MODEL_NAME
    if EMBEDDING_BACKEND == "hf"
    else _env("ADMIR_OPENAI_EMBEDDING_MODEL")
)
AOAI_EMBEDDING_LARGE_DIM = (
    HF_EMBEDDING_DIM
    if EMBEDDING_BACKEND == "hf"
    else _env_int("ADMIR_OPENAI_EMBEDDING_DIM", 3072)
)

# Runtime settings.
LITE_MODE = False
GLOBAL_BROWSE_TOPK = _env_int("ADMIR_GLOBAL_BROWSE_TOPK", 40)
OVERWRITE_CLIP_SEARCH_TOPK = _env_int("ADMIR_CLIP_SEARCH_TOPK", 8)
CLIP_SEARCH_MIN_TOPK = _env_int("ADMIR_CLIP_SEARCH_MIN_TOPK", 5)
LEXICAL_MATCH_BETA = _env_float("ADMIR_LEXICAL_MATCH_BETA", 2.0)
EXPERT_MAX_GRID_FRAMES = _env_int("ADMIR_EXPERT_MAX_GRID_FRAMES", 64)
TEMPORAL_STAGNATION_OVERLAP = _env_float("ADMIR_TEMPORAL_STAGNATION_OVERLAP", 0.6)
TEMPORAL_STAGNATION_REPEATS = _env_int("ADMIR_TEMPORAL_STAGNATION_REPEATS", 2)
TEMPORAL_REDIRECT_SECONDS = _env_int("ADMIR_TEMPORAL_REDIRECT_SECONDS", 15)
SINGLE_CHOICE_QA = False
MAX_ITERATIONS = _env_int("ADMIR_MAX_ITERATIONS", 8)


def configured_chat_models() -> dict[str, str]:
    return {
        "ADMIR_CAPTION_VLM_MODEL": AOAI_CAPTION_VLM_MODEL_NAME,
        "ADMIR_ORCHESTRATOR_LLM_MODEL": AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
        "ADMIR_FRAME_INSPECT_MODEL": AOAI_FRAME_INSPECT_MODEL_NAME,
        "ADMIR_COMMUNICATION_EXPERT_MODEL": AOAI_COMMUNICATION_EXPERT_MODEL_NAME,
        "ADMIR_REFINE_LLM_MODEL": AOAI_REFINE_LLM_MODEL_NAME,
    }


def validate_strict_paper_config(require_api_key: bool = False) -> None:
    """Validate that all runtime components were explicitly configured.

    The function name is kept for backward compatibility with earlier scripts.
    Public releases avoid hard-coded model/provider choices; set
    ``ADMIR_STRICT_PAPER_MODE=1`` only when you want fail-fast checks.
    """
    if not STRICT_PAPER_MODE:
        return

    missing_models = [name for name, value in configured_chat_models().items() if not value]
    if missing_models:
        raise RuntimeError(
            "ADMIR_STRICT_PAPER_MODE=1 requires explicit model names for: "
            + ", ".join(missing_models)
        )
    if AOAI_TOOL_VLM_MODEL_NAME and AOAI_FRAME_INSPECT_MODEL_NAME:
        if AOAI_TOOL_VLM_MODEL_NAME != AOAI_FRAME_INSPECT_MODEL_NAME:
            raise RuntimeError(
                "ADMIR_TOOL_VLM_MODEL must match ADMIR_FRAME_INSPECT_MODEL in strict mode."
            )
    if EMBEDDING_BACKEND == "hf" and not HF_EMBEDDING_MODEL_NAME:
        raise RuntimeError("Set ADMIR_HF_EMBEDDING_MODEL for local HF embeddings.")
    if EMBEDDING_BACKEND != "hf" and not AOAI_EMBEDDING_LARGE_MODEL_NAME:
        raise RuntimeError("Set ADMIR_OPENAI_EMBEDDING_MODEL for remote embeddings.")
    if require_api_key and not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY, or disable strict mode for unauthenticated local endpoints.")
