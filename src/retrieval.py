from typing import Callable, List, Tuple
from loguru import logger

from retrieval.opensearch import query_opensearch
from retrieval.pinecone import query_pinecone


RetrievalFunction = Callable[[str, int], Tuple[List[str], List[str]]]


def dummy_retrieval(query: str, ret_top_k: int) -> Tuple[List[str], List[str]]:
    """
    This function does not perform any retrieval and returns empty lists.
    The purpose of this function is to provide a reference for the retrieval function.
    """
    return [], []


def top_k_opensearch(query: str, ret_top_k: int) -> Tuple[List[str], List[str]]:
    """
    This function performs top-k retrieval using OpenSearch.
    """
    results = query_opensearch(query, top_k=ret_top_k)
    logger.debug(f"Retrieved {len(results['hits']['hits'])} documents from OpenSearch")

    doc_ids = []
    ret_docs = []
    for matches in results["hits"]["hits"]:
        doc_ids.append(matches["_source"]["doc_id"])
        ret_docs.append(matches["_source"]["text"])
    return doc_ids, ret_docs


def top_k_pinecone(query: str, ret_top_k: int) -> Tuple[List[str], List[str]]:
    """
    This function performs top-k retrieval using Pinecone.
    """
    results = query_pinecone(query, top_k=ret_top_k)
    logger.debug(f"Retrieved {len(results['matches'])} documents from Pinecone")

    doc_ids = []
    ret_docs = []
    for matches in results["matches"]:
        doc_ids.append(matches["metadata"]["doc_id"])
        ret_docs.append(matches["metadata"]["text"])
    return doc_ids, ret_docs
