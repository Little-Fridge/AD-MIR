"""
Admir Utils
Common utilities for embedding generation, JSON parsing, and error handling.
"""

import base64
import copy
import json
import os
import random
import re
import uuid
import time
from mimetypes import guess_type
from typing import Optional, List, Dict, Any, Union

import cv2
import requests
import httpx
from openai import OpenAI

# Try importing admir config
try:
    import admir.config as config
except ImportError:
    class Config: pass
    config = Config()

# =====================================================================
# JSON Repair Integration
# =====================================================================
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("[WARN] json_repair not installed. Run: pip install json-repair")


def robust_json_parse(json_str: str) -> Optional[Dict]:
    """
    Multi-stage JSON parsing with fallback strategies.
    
    Stage 1: Standard json.loads
    Stage 2: json_repair library (handles common LLM JSON errors)
    Stage 3: Manual cleanup for common errors
    
    Returns:
        Parsed dict or None if all strategies fail
    """
    if not json_str or not isinstance(json_str, str):
        return None
    
    json_str = json_str.strip()
    
    # Stage 1: Try standard parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Stage 2: Try json_repair
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except Exception:
            pass
    
    # Stage 3: Manual cleanup for common LLM errors
    try:
        # Remove markdown code blocks
        cleaned = re.sub(r'^```(?:json)?\s*', '', json_str)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        # Fix trailing commas
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Fix unquoted keys (simple cases)
        cleaned = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        
        # Fix single quotes
        cleaned = cleaned.replace("'", '"')
        
        return json.loads(cleaned)
    except Exception:
        pass
    
    return None

# =====================================================================
# Retry with Exponential Backoff
# =====================================================================
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                retryable_errors = [
                    "rate limit", "timed out", "too many requests",
                    "forbidden for url", "internal", "connection",
                    "timeout", "502", "503", "504"
                ]
                
                if any(err in error_str for err in retryable_errors):
                    num_retries += 1
                    if num_retries > max_retries:
                        print(f"[ERROR] Max retries ({max_retries}) reached. Last error: {e}")
                        raise e
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"[RETRY {num_retries}/{max_retries}] Waiting {delay:.1f}s due to: {str(e)[:100]}")
                    time.sleep(delay)
                else:
                    raise e
    return wrapper


# =====================================================================
# Embedding Service
# =====================================================================
_HF_EMBED_MODEL = None
_HF_EMBED_LOCK = None

def _get_hf_embedder():
    global _HF_EMBED_MODEL, _HF_EMBED_LOCK
    if _HF_EMBED_LOCK is None:
        import threading
        _HF_EMBED_LOCK = threading.Lock()
    with _HF_EMBED_LOCK:
        if _HF_EMBED_MODEL is None:
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Use path from config or default local path
            default_local = "./model_zoo/bge-m3"
            model_path = getattr(config, "HF_EMBEDDING_MODEL_NAME", None) or default_local
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading local embedding model: {model_path} on {device}")
            _HF_EMBED_MODEL = SentenceTransformer(model_path, device=device)
    return _HF_EMBED_MODEL


class AzureOpenAIEmbeddingService:
    """
    Service wrapper for generating embeddings.
    Supports both remote OpenAI-compatible APIs and local HuggingFace models.
    """
    
    @staticmethod
    @retry_with_exponential_backoff
    def get_embeddings(
        endpoints: List[str],
        model_name: str,
        input_text: Union[str, List[str]],
        api_key: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generates embeddings for the input text.
        
        Args:
            endpoints: List of base URLs (uses the first valid one if remote).
            model_name: The model deployment name.
            input_text: Single string or list of strings to embed.
            api_key: The API key.
            
        Returns:
            A list of dicts, where each dict has an 'embedding' key containing the vector.
        """
        # Ensure input is a list
        if isinstance(input_text, str):
            input_text = [input_text]
            
        # Remove empty strings to prevent API errors
        valid_indices = [i for i, t in enumerate(input_text) if t and t.strip()]
        valid_texts = [input_text[i] for i in valid_indices]
        
        if not valid_texts:
            return []

        # 1. Local HF Embedding Backend
        backend = getattr(config, "EMBEDDING_BACKEND", "hf")
        if backend == "hf":
            model = _get_hf_embedder()
            vecs = model.encode(
                valid_texts,
                batch_size=int(getattr(config, "HF_EMBEDDING_BATCH_SIZE", 8)),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            
            # Remap to original list structure (fill empty strings with zeros or skip)
            results = []
            vec_ptr = 0
            for i in range(len(input_text)):
                if i in valid_indices:
                    results.append({"embedding": vecs[vec_ptr].tolist()})
                    vec_ptr += 1
                else:
                    # Return zero vector for empty input? Or just skip? 
                    # Usually better to maintain length or handled by caller.
                    # Here we append None to indicate invalid input at that index
                    results.append({"embedding": []}) 
            return results

        # 2. Remote OpenAI-Compatible Backend
        # Initialize Client
        # Use the first endpoint in the list (simplified logic) or env var
        base_url = endpoints[0] if endpoints else os.environ.get("ADMIR_EMBEDDING_URL", "http://0.0.0.0:8090")
        
        # Ensure URL is clean
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
             base_url += "/v1"

        client = OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-dummy", # Some local servers need a dummy key
            http_client=httpx.Client(timeout=120.0)
        )

        try:
            response = client.embeddings.create(
                input=valid_texts,
                model=model_name
            )
            
            # Map results back to original structure
            # The API returns a list of data objects sorted by index
            data_map = {item.index: item.embedding for item in response.data}
            
            results = []
            valid_ptr = 0
            for i in range(len(input_text)):
                if i in valid_indices:
                    results.append({"embedding": data_map[valid_ptr]})
                    valid_ptr += 1
                else:
                    results.append({"embedding": []})
            return results
            
        except Exception as e:
            raise RuntimeError(f"Embedding service failed: {e}")

# =====================================================================
# Utility Functions
# =====================================================================

def generate_json_error_feedback(error_type: str, original_content: str) -> str:
    """
    Generate a helpful error message for self-correction loop.
    """
    feedback = f"""[SYSTEM ERROR] JSON Decode Error detected.

Your previous response contained malformed JSON that could not be parsed.
Error type: {error_type}

Please retry with VALID JSON format. Follow this exact structure:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

Previous problematic content (first 200 chars): {original_content[:200]}...
"""
    return feedback