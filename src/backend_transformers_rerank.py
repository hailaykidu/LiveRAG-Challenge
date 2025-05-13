from typing import List, Tuple
from loguru import logger

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_reranker(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto").eval()

    device_type = next(iter(model.parameters())).device.type
    device = torch.device(device_type)
    if device_type != "cuda":
        logger.warning("No GPU available. Using CPU for reranker!")

    return tokenizer, model, device


def rerank_passages(
    initialized_model: Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device],
    query: str,
    passages: List[str],
) -> str:
    tokenizer, model, device = initialized_model
    pairs = [(query, p) for p in passages]

    with torch.inference_mode():
        inputs = tokenizer([q for q, _ in pairs], [p for _, p in pairs], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        return scores.cpu().tolist()
