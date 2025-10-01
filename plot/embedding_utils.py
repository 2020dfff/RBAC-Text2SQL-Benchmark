"""Shared helpers for text embeddings used in visualization scripts."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # Optional dependency for OpenAI embeddings
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    OpenAI = None  # type: ignore

try:  # Optional dependency for local embeddings
    from fastembed import TextEmbedding
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    TextEmbedding = None  # type: ignore


_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()
import numpy as np

DEFAULT_BACKEND = os.environ.get("ROLE_EMBED_BACKEND", "fastembed").lower()
DEFAULT_MODEL = os.environ.get("ROLE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def _get_fastembed(model_name: str):
    if TextEmbedding is None:
        raise ModuleNotFoundError(
            "fastembed is required for the 'fastembed' embedding backend. Install via `pip install fastembed`."
        )
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _get_openai_client(api_key: str | None):
    if OpenAI is None:
        raise ModuleNotFoundError(
            "openai is required for the 'openai' embedding backend. Install via `pip install openai`."
        )
    return OpenAI(api_key=api_key)


def _resolve_backend(model_name: str, backend: str | None) -> str:
    if backend:
        return backend.lower()
    lowered = model_name.lower()
    if lowered.startswith("text-embedding-") or lowered.startswith("text-search-"):
        return "openai"
    return DEFAULT_BACKEND


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    *,
    backend: str | None = None,
    openai_api_key: str | None = None,
) -> List[np.ndarray]:
    """Embed a batch of texts using the configured backend."""
    if not texts:
        return []

    cleaned_texts = [t or "" for t in texts]
    backend_hint = backend if backend is not None else os.environ.get("ROLE_EMBED_BACKEND")
    chosen_backend = _resolve_backend(model_name, backend_hint)
    explicit_backend = backend is not None
    env_forced_backend = backend is None and backend_hint is not None

    openai_compatible = model_name.lower().startswith(("text-embedding-", "text-search-"))
    if chosen_backend == "openai" and not openai_compatible:
        if explicit_backend:
            raise ValueError(
                f"Model '{model_name}' is not compatible with the OpenAI backend."
                " Please choose an OpenAI embedding model (e.g. text-embedding-3-small)"
                " or switch backend to fastembed."
            )
        if env_forced_backend:
            warnings.warn(
                f"Environment requested OpenAI backend but model '{model_name}' does not look like an OpenAI embedding model."
                " Falling back to 'fastembed'.",
                RuntimeWarning,
            )
        chosen_backend = "fastembed"

    if chosen_backend == "openai":
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the 'openai' embedding backend.")
        client = _get_openai_client(api_key)
        print(f"[embedding_utils] Using OpenAI embeddings ({model_name}) for {len(cleaned_texts)} texts")
        response = client.embeddings.create(model=model_name, input=list(cleaned_texts))
        return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]

    if chosen_backend != "fastembed":
        raise ValueError(f"Unsupported embedding backend: {chosen_backend}")

    embedder = _get_fastembed(model_name)
    print(f"[embedding_utils] Using FastEmbed ({model_name}) for {len(cleaned_texts)} texts")
    return [np.asarray(vec, dtype=np.float32) for vec in embedder.embed(cleaned_texts)]


def cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    """Compute cosine similarity, guarding against zero vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
