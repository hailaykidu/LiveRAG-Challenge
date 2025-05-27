# #SIGIR 2025 LiveRAG Challenge
The SIGIR'2025 LiveRAG Challenge is organized by TII (Technology Innovation Institute) with support from AI71, AWS, Pinecone, and Hugging Face. 

# RAGtifier  

RAGtifier is our team name. We have implemented multiple RAG query strategies, reranking techniques, and other related components.
We run all of them using different parameter combinations to determine which configuration performs best for the Datamorgana QA use case.
 
To reproduce the submitted results, use the parameters from `submit.sh`.

## Setup environment

```bash
module load conda

conda env create -n ragtifier -f env.yml
conda activate ragtifier
```

### Secrets

Make sure you have `.env` in the dirrectory from which you run the code.

This is required to get access to retrieval APIs and LLMs. Simply copy the `.env.example` file and fill in the required fields.

## Usage

All supported parameters can be found in `main.py` or retrieved by `python main.py -h`.

You can run `main.sh` script directly, e.g.:

```bash
python main.py --dataset_path questions.jsonl
```

But we were using SLURM to run the experiments, so we have a `submit.sh` script with predefined parameters.

```bash
sbatch submit.sh
```
