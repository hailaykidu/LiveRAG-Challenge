import argparse
from functools import cache
import time
from typing import List, Tuple, Dict, Any

import jsonlines
from tqdm import tqdm
from src.utils import save_results, setup_seeds, save_outputs, setup_logging, time_measurement, time_summarize, compound_log_name
import src.rerank as rerank
import src.retrieval as retrieval
import src.query as query
from loguru import logger
from dotenv import load_dotenv, find_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGtifier")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file, if not specified, will use the database")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to process, -1 to process all")
    parser.add_argument("--skip", type=int, default=-1, help="Number of questions to skip, -1 to skip none")

    parser.add_argument("--ret_method", type=str, default="none", choices=["none", "top_k_opensearch", "top_k_pinecone"], help="Method to retrieve documents")
    parser.add_argument("--ret_top_k", type=int, default=10, help="Number of documents to retrieve")

    parser.add_argument("--rerank_backend", type=str, default="local", choices=["local"], help="Backend to use for reranking")
    parser.add_argument("--rerank_model", type=str, help="Reramker model (Transformers sequence classification model)")
    parser.add_argument("--rerank_top_k", type=int, default=5, help="Number of documents to keep after reranking")

    parser.add_argument("--invert", action="store_true", help="Invert the order of the retrieved documents (applied after reranking, if any)")

    parser.add_argument("--query_method", type=str, default="instruct", choices=["simple", "trustrag", "astute", "instruct"], help="RAG Method to query the LLM")
    parser.add_argument("--query_backend", type=str, default="openai", choices=["local", "openai"], help="Backend to use for RAG LLM")
    parser.add_argument("--query_model", type=str, default="falcon3:10b-instruct-fp16", help="Model to use for RAG LLM")

    parser.add_argument("--retry_times", type=int, default=5, help="Retry this many times if the query fails. If reached, empty ansswer will be returned.")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Random seed, as default, current time in seconds is used to have sortable results")
    parser.add_argument("--log_name", type=str, help="Name of log and result")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (Debug level)")
    parser.add_argument("-vv", "--very-verbose", action="store_true", help="Enable very verbose logging (Trace level)")

    args = parser.parse_args()
    if not args.log_name:
        args.log_name = compound_log_name(args)
    return args


@cache
def get_query_func(method: str) -> query.QueryFunction:
    if method == "simple":
        return query.simple_query
    elif method == "trustrag":
        return query.trustrag_query
    elif method == "astute":
        return query.astute_query
    elif method == "instruct":
        return query.instructrag_query
    else:
        raise ValueError(f"Invalid query method: {method}")


@cache
def get_retrieval_func(method: str) -> retrieval.RetrievalFunction | None:
    if method == "none":
        return None
    elif method == "top_k_opensearch":
        return retrieval.top_k_opensearch
    elif method == "top_k_pinecone":
        return retrieval.top_k_pinecone
    else:
        raise ValueError(f"Invalid retrieval method: {method}")


def load_models(args: argparse.Namespace):
    logger.info(f"Using {args.query_backend} backend with model {args.query_model} for querying")
    query.set_backend(args.query_backend, args.query_model)

    if args.rerank_model:
        logger.info(f"Using {args.rerank_backend} backend with model {args.rerank_model} for reranking")
        rerank.set_backend(args.rerank_backend, args.rerank_model)


def load_dataset(args: argparse.Namespace) -> Tuple[List[dict], int]:
    logger.info("Loading dataset from " + args.dataset_path)
    dataset = []
    with jsonlines.open(args.dataset_path) as reader:
        for obj in reader:
            dataset.append(obj)

    # warnings here, because it's expected to be used with caution
    if args.skip > 0:
        logger.warning(f"Skipping first {args.skip} questions")
        dataset = dataset[args.skip :]

    if args.limit > 0 and args.limit < len(dataset):
        logger.warning(f"Limiting dataset to {args.limit} questions")
        dataset = dataset[: args.limit]

    return dataset, len(dataset)


def apply_rag(args: argparse.Namespace, question: str) -> Tuple[str, str, List[str], List[str], Dict[str, float]]:
    perf_stats = {}
    query_func = get_query_func(args.query_method)
    retrieval_func = get_retrieval_func(args.ret_method)

    context = ""
    doc_ids = []
    doc_passages = []
    if retrieval_func:
        with time_measurement(perf_stats, "retrieval"):
            doc_ids, doc_passages = retrieval_func(question, args.ret_top_k)

        with time_measurement(perf_stats, "rerank"):
            if args.rerank_model:
                doc_ids, doc_passages = rerank.rerank_docs(question, doc_ids, doc_passages, args.rerank_model, top_k=args.rerank_top_k)

        # Invert ordering of retrieved context based on
        # The Power of Noise: Redefining Retrieval for RAG Systems. SIGIR 2024
        if args.invert:
            logger.debug("Inverting document order")
            doc_ids.reverse()
            doc_passages.reverse()

        for index, document in enumerate(doc_passages):
            context += f"Externally Retrieved Document{index}:" + document + "\n"
    else:
        logger.warning("No retrieval method specified, using empty context")

    with time_measurement(perf_stats, "rag_query"):
        final_answer, final_prompt = query_func(question, context, args.query_model)

    return final_answer, final_prompt, doc_ids, doc_passages, perf_stats


def _flatten_passages(doc_ids: List[str], doc_passages: List[str]) -> List[dict]:
    structured_passages = []
    for i in range(len(doc_ids)):
        structured_passages.append(
            {
                "doc_IDs": [doc_ids[i]],  # Wrap in list as requested
                "passage": doc_passages[i],
            }
        )
    return structured_passages


def process_question(args: argparse.Namespace, qobj: dict) -> Tuple[Dict[str, Any], str | None, Dict[str, float]]:
    id = qobj.get("id")
    question = qobj.get("question")
    request_id = qobj.get("request_id", None)

    final_answer, final_prompt, doc_ids, doc_passages, perf_stats = apply_rag(args, question)

    answer = {
        "id": id,  # only numeric id is expected
        "question": question,
        "answer": final_answer,
        "final_prompt": final_prompt,
        "passages": _flatten_passages(doc_ids, doc_passages),
    }

    logger.info("Final answer: " + final_answer)
    return answer, request_id, perf_stats


def retry_question(args: argparse.Namespace, qobj: dict) -> Tuple[Dict[str, Any], str | None, Dict[str, float]]:
    logger.info(f"Question: {qobj['question']}")
    for attempt in range(args.retry_times):
        try:
            return process_question(args, qobj)
        except Exception as e:
            if attempt < args.retry_times - 1:
                logger.opt(exception=e).warning(f"Attempt {attempt + 1}/{args.retry_times} failed, retrying...")
            else:
                logger.opt(exception=e).error("Out of attempts to process the question, falling back to empty answer")

    # If all attempts fail, return "I don't know" as the answer
    return (
        {
            "id": qobj.get("id", None),
            "question": qobj.get("question", None),
            "passages": [],
            "final_prompt": "The system is not responding, answer with 'I don't know'",
            "answer": "I don't know",
        },
        qobj.get("request_id", None),
        {},
    )


def main():
    main_time = time.perf_counter()
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
    load_dotenv(dotenv_path, verbose=True, override=True)

    args = parse_args()
    setup_seeds(args.seed)
    setup_logging(args.log_name, file_level="DEBUG", console_level="TRACE" if args.very_verbose else "DEBUG" if args.verbose else "INFO")
    logger.debug("Found .env file at: " + dotenv_path)
    logger.info(args)

    load_models(args)

    answers = []
    request_ids = []
    times = []

    data, total = load_dataset(args)
    for qobj in tqdm(data, total=total, desc="Generating answers", unit="question"):
        start_time = time.perf_counter()
        with logger.contextualize(question_id=qobj["id"]):
            answer, request_id, perf_stats = retry_question(args, qobj)
        perf_stats["rag_total"] = time.perf_counter() - start_time

        answers.append(answer)
        times.append(perf_stats)
        if request_id is not None:
            request_ids.append(request_id)

    save_results(answers, args.log_name, "answers")
    if len(request_ids) > 0:
        save_outputs(request_ids, args.log_name, "request_ids")

    save_outputs(times, args.log_name, "perf_stats")
    avg_times, total_times = time_summarize(times)
    logger.info(f"Average times (seconds): {avg_times}")
    logger.info(f"Total times (seconds): {total_times}")
    logger.info(f"Total process time (seconds): {time.perf_counter() - main_time}")


if __name__ == "__main__":
    main()
