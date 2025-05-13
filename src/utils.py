import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import random
import torch
import jsonlines

from tqdm import tqdm
from loguru import logger
import time


def file_formatter(record):
    record["fn_line"] = record["name"] + ":" + record["function"] + ":" + str(record["line"])
    format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {fn_line: <32} | "

    if record["extra"]:
        format += "{extra} "

    format += "{message}\n"

    if record["exception"]:
        format += "{exception}\n"

    return format


def setup_logging(experiment_name=None, log_dir="logs", file_level="INFO", console_level="INFO"):
    """
    Configure logging for experiments with both console and file output.

    Args:
        experiment_name: Name of the experiment for the log file
        log_dir: Directory to store log files
    """
    # Remove any existing handlers
    logger.remove()

    # Add console handler with a simple format
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format="<level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> {extra} - <level>{message}</level>",
        level=console_level,
        colorize=True,
    )

    # Add file handler if experiment_name is provided
    if experiment_name and file_level is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(log_file, format=file_formatter, level=file_level)

    return logger


def compound_log_name(args: argparse.Namespace):
    log_name = f"{args.query_method}-"
    log_name += args.query_model.replace("/", "_").replace("-", "_").lower()
    log_name += f"-{args.query_backend}"
    if args.limit > 0:
        log_name += f"-{args.limit}x{args.repeat_times}"
    if args.ret_method and args.ret_method != "none":
        log_name += f"-{args.ret_method}-t{args.ret_top_k}"
    if args.rerank_model and args.rerank_model != "none":
        log_name += f"-{args.rerank_model.replace('/', '_').replace('-', '_').lower()}-t{args.rerank_top_k}"
    if args.invert:
        log_name += "-invert"
    if args.seed:
        log_name += f"-{args.seed}"
    return log_name


def time_measurement(stats_dict: Dict[str, float], operation_name: str):
    class TimerContext:
        def __enter__(self):
            logger.debug(f"Doing {operation_name}...")
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            stats_dict[operation_name] = time.perf_counter() - self.start_time
            logger.debug(f"Done {operation_name} in {stats_dict[operation_name]:.3f} seconds")

    return TimerContext()


def time_summarize(perf_stats: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    all_stats = {}

    for perf_stat in tqdm(perf_stats, desc="Calculating performance summary", unit="question", leave=False, delay=1):
        for key, value in perf_stat.items():
            if key not in all_stats:
                all_stats[key] = []
            all_stats[key].append(value)

    average_stats = {}
    total_stats = {}
    for operation, times in all_stats.items():
        sum_times = sum(times)
        avg_time = sum_times / len(times)
        average_stats[operation] = avg_time
        total_stats[operation] = sum_times

    return average_stats, total_stats


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_outputs(outputs, dir, file_name):
    json_dict = json.dumps(outputs, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f"data_cache/{dir}"):
        os.makedirs(f"data_cache/{dir}", exist_ok=True)
    with open(os.path.join(f"data_cache/{dir}", f"{file_name}.json"), "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f, indent=4)


def save_results(results, dir, file_name="answers"):
    if not os.path.exists(f"results/{dir}"):
        os.makedirs(f"results/{dir}", exist_ok=True)

    results_path = os.path.join(f"results/{dir}", f"{file_name}.jsonl")
    with jsonlines.open(results_path, "w") as writer:
        writer.write_all(results)
    logger.info(f"Results saved to {results_path}")


def load_results(dir, file_name="answers"):
    results_path = os.path.join(f"results/{dir}", f"{file_name}.jsonl")
    return load_jsonl(results_path)


def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f, indent=4)


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def load_jsonl(file_path):
    results = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            results.append(obj)
    return results


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_str(s):
    try:
        if isinstance(s, dict):
            s = " ".join([f"{k}: {v}" for k, v in s.items()])
        elif isinstance(s, list):
            s = " ".join(map(str, s))
        else:
            s = str(s)
    except Exception as e:
        logger.opt(exception=e).error("the output cannot be converted to a string")
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()
