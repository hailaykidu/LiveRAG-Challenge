from typing import List, Literal
from pinecone import Pinecone
import os
import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer


@cache
def has_mps():
    return torch.backends.mps.is_available()


@cache
def has_cuda():
    return torch.cuda.is_available()


@cache
def get_tokenizer(model_name: str = "intfloat/e5-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@cache
def get_model(model_name: str = "intfloat/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        model = model.to("mps")
    elif has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_query(
    query: str,
    query_prefix: str = "query: ",
    model_name: str = "intfloat/e5-base-v2",
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
) -> list[float]:
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]


def batch_embed_queries(
    queries: List[str],
    query_prefix: str = "query: ",
    model_name: str = "intfloat/e5-base-v2",
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        match pooling:
            case "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            case "avg":
                embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


@cache
def get_pinecone_index(index_name: str = None):
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
    index = pc.Index(name=index_name)
    return index


def query_pinecone(query: str, top_k: int = 10, namespace: str = None) -> dict:
    if namespace is None:
        namespace = os.getenv("PINECONE_NAMESPACE", "default")

    index = get_pinecone_index()
    results = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True,
    )

    return results
