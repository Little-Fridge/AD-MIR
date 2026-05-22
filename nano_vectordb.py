import json
import os
from typing import Any, Dict, List

import numpy as np


class NanoVectorDB:
    """Small compatibility implementation for nano-vectordb's used surface."""

    def __init__(self, embedding_dim: int, storage_file: str):
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self._data: List[Dict[str, Any]] = []
        self._additional_data: Dict[str, Any] = {}
        if storage_file and os.path.exists(storage_file) and os.path.getsize(storage_file) > 0:
            with open(storage_file, encoding="utf-8") as f:
                payload = json.load(f)
            self._data = payload.get("data", payload if isinstance(payload, list) else [])
            self._additional_data = payload.get("additional_data", {})

    def upsert(self, data: List[Dict[str, Any]]):
        for item in data:
            clean = dict(item)
            vec = clean.get("__vector__")
            if isinstance(vec, np.ndarray):
                clean["__vector__"] = vec.tolist()
            self._data.append(clean)
        return data

    def query(self, query: List[float], top_k: int = 5):
        if isinstance(query, np.ndarray):
            q = query.astype(float)
        else:
            q = np.array(query, dtype=float)
        q_norm = np.linalg.norm(q)
        results = []
        for item in self._data:
            vec = np.array(item.get("__vector__", []), dtype=float)
            if vec.size == 0 or q_norm == 0:
                score = 0.0
            else:
                denom = float(np.linalg.norm(vec) * q_norm)
                score = float(np.dot(vec, q) / denom) if denom else 0.0
            out = dict(item)
            out["__score__"] = score
            results.append(out)
        results.sort(key=lambda x: x.get("__score__", 0.0), reverse=True)
        return results[:top_k]

    def store_additional_data(self, **kwargs):
        self._additional_data.update(kwargs)

    def get_additional_data(self):
        return dict(self._additional_data)

    def save(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(
                {"data": self._data, "additional_data": self._additional_data},
                f,
                ensure_ascii=False,
            )
