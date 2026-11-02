# HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation

Official code for **HARP**, published at **NeurIPS 2026**.

HARP views LLM-based recommendation as a communication process between a user agent and item agents, in four stages: **Build**, **Transmit**, **Understand**, **Sync**. It adds three operators to a frozen LLM backbone and needs no training:

- **Dual profiles (Build).** Users and items both get a structured four-channel profile (preference, quality, context, audience). User profiles come from one LLM call; item profiles come from deterministic rules with no LLM call.
- **Profile-level homophily (Understand).** The similarity between a user profile and a neighbour's profile re-weights neighbours at retrieval time, before the reranker sees any candidate.
- **Reflective profile update (Sync).** After warmup, each user profile is revised once from the user's own hits and misses, with an optional acceptance check.

Code: <https://github.com/Jove-Zhen/HARP>

## Layout

```
code/
├── configs/        one YAML per dataset (LLM endpoint, retrieval budget, paths)
├── prompts/        prompt templates (memory update, user profile, user understanding)
├── scripts/
│   ├── run_com_train.py             main entry: warmup, evaluation, metrics
│   ├── convert_instructrec.py       convert one iAgent dataset to the processed format
│   ├── convert_all_instructrec.sh   convert all four datasets
│   └── generate_fixed_eval_users.py fixed evaluation user list
├── src/
│   ├── homophily/  channel similarity and profile-level homophily
│   ├── profiles/   item profile extraction, user profile manager and reflection
│   ├── memory/     storage, graph, pruner, memory manager
│   ├── models/     agent, reranker, LLM client
│   ├── train/      warmup and evaluation loop
│   └── data/       dataset loaders
└── run_{yelp,books,goodreads,movietv}_full.sh
```

## Data

We use the four InstructRec benchmarks released with [iAgent](https://github.com/agiresearch/iAgent): **Yelp** (local businesses), **Books**, **Goodreads** (books) and **MovieTV** (movies and TV). Each user's last interaction is the test item, and the one before it is used for warmup. At evaluation time every user is ranked against 10 candidates: the ground-truth item and 9 sampled negatives.

**1. Download.** Get the iAgent files (`*.pkl` and the `combined_*_asin_mapping.csv` files) and place them in `data/iagent/`.

**2. Convert.** The conversion scripts are in `scripts/`:

```bash
bash scripts/convert_all_instructrec.sh
```

To convert a single dataset, call `scripts/convert_instructrec.py` with `--input`, `--mapping`, `--output` and `--domain` (see the header of the script for an example). Each dataset becomes one folder with four files, named after the folder:

```
data/processed/
├── instructrec-yelp/        .inter  .meta  .text  .instruction
├── instructrec-books/
├── instructrec-goodreads/
└── instructrec-movietv/
```

`.inter` holds the interactions, `.meta` the item metadata, `.text` the review text and `.instruction` the per-user request text. The converter also writes `statistics.txt` with the dataset sizes.

**3. Item profiles.** Item profiles are rule-extracted for Yelp. For the other three datasets the item side has no channel profile and the retrieval gate is neutral.

## Run

```bash
pip install -r requirements.txt
```

Put your LLM endpoint and key in the `llm:` section of the config (the committed values are placeholders; do not commit real keys). Then run one dataset:

```bash
DATA_DIR=/path/to/data/processed/instructrec-yelp bash run_yelp_full.sh
```

`DATA_DIR` is optional and defaults to `dataset.data_dir` in the config. The other scripts work the same way: `run_books_full.sh`, `run_goodreads_full.sh`, `run_movietv_full.sh`.

Each script fixes the evaluation setting for one dataset (full test split, 10 candidates, one warmup round) and logs to `logs/<run_tag>.log`. Extra arguments are passed on to the main entry, so a quick trial on a few users is:

```bash
bash run_yelp_full.sh --n_eval_users 100
```

Each run writes to `results/<run_tag>/`: the metrics (Hit@K and NDCG@K for K in 1, 3, 5, 10), per-case evaluation traces, the final user profiles and their update traces. Use a new `run_tag` for every run; the scripts do this by timestamp.

## Citation

```bibtex
@inproceedings{tao2026harp,
  title     = {{HARP}: Training-Free Dual-Profile Agentic Communication for {LLM}-Based Recommendation},
  author    = {Tao, Zhen and Zhou, Xun and Chen, Xuhui and Qiao, Ziyue and Sun, Qingqiang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

HARP builds on [MemRec](https://aclanthology.org/2026.acl-long.2061.pdf) and uses the InstructRec data from [iAgent](https://github.com/agiresearch/iAgent).
