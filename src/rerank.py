from typing import List, Tuple
from loguru import logger
import torch
import gc

_rbackend_model = None
_rbackend_fn = None

def set_backend(backend_type: str, model_name: str):
    global _rbackend_model, _rbackend_fn
    if backend_type == "local":
        from src.backend_transformers_rerank import load_reranker, rerank_passages

        _rbackend_model = load_reranker(model_name)
        _rbackend_fn = rerank_passages
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}. Supported types are 'local'.")


def clean_backend():
    global _rbackend_model, _rbackend_fn
    _rbackend_model = None
    _rbackend_fn = None
    gc.collect()
    torch.cuda.empty_cache()


def _call_backend(*args, **kwargs):
    global _rbackend_model, _rbackend_fn
    if _rbackend_model is None or _rbackend_fn is None:
        raise ValueError("Backend not set. Please set the backend using `set_backend`")

    return _rbackend_fn(_rbackend_model, *args, **kwargs)

def rerank_docs(query: str, doc_ids: List[str], doc_passages: List[str], model_name: str, top_k: int = None) -> Tuple[List[str], List[str]]:
    scores = _call_backend(query, doc_passages)
    logger.debug(f"Received scores: {scores}")

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if top_k:  # Limit to top_k results
        sorted_indices = sorted_indices[:top_k]

    logger.debug(f"Selected indices: {sorted_indices}")
    doc_ids = [doc_ids[i] for i in sorted_indices]
    doc_passages = [doc_passages[i] for i in sorted_indices]
    return doc_ids, doc_passages
