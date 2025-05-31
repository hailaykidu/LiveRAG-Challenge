#!/bin/bash

#SBATCH --job-name=rag-eval
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
module load conda
conda activate ragtifier

python main.py \
    --dataset_path "LiveRAG_LCD_Session2_Question_file.jsonl" \
    --query_method "instruct" \
    --query_model "tiiuae/Falcon3-10B-Instruct" \
    --query_backend "local" \
    --ret_method "top_k_pinecone" \
    --ret_top_k "200" \
    --rerank_model "BAAI/bge-reranker-v2-m3" \
    --rerank_top_k "5" \
    --invert
