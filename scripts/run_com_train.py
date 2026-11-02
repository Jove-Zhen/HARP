#!/usr/bin/env python3

import argparse
import yaml
import sys
import os
import csv
import logging
from datetime import datetime
from pathlib import Path


class TimestampedStream:

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, text):
        if not text:
            return
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if i > 0:
                self._stream.write('\n')
                self._at_line_start = True
            if line:
                if self._at_line_start:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self._stream.write(f'[{timestamp}] {line}')
                else:
                    self._stream.write(line)
                self._at_line_start = False
            elif i == len(lines) - 1:
                pass

    def flush(self):
        self._stream.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return hasattr(self._stream, 'isatty') and self._stream.isatty()


sys.stdout = TimestampedStream(sys.__stdout__)
sys.stderr = TimestampedStream(sys.__stderr__)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.__stdout__
)
logger = logging.getLogger('HARP')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.com_agent import ComMemRecAgent
from src.prompts.prompt_registry import PromptRegistry
from src.train.com_trainer import ComTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_llm_client(llm_config: dict):
    """Create the LLM client. Fails loudly: a run must never silently use a stand-in model."""
    from src.models.llm_client import LLMClient
    provider = llm_config.get('provider', 'openai')
    model = llm_config.get('model', 'gpt-4o-mini')
    endpoint = llm_config.get('api_base', llm_config.get('endpoint', ''))
    api_key = llm_config.get('api_key', '')
    api_version = llm_config.get('api_version', '2024-08-01-preview')

    if endpoint.startswith('${'):
        endpoint = os.environ.get(endpoint[2:-1], '')
    if api_key.startswith('${'):
        api_key = os.environ.get(api_key[2:-1], '')

    logger.info(f"LLM Provider: {provider}")
    logger.info(f"LLM Model: {model}")
    logger.info(f"LLM Endpoint: {endpoint}")

    client = LLMClient(
        api_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        model=model,
        provider_name=provider
    )
    logger.info("Real LLM client created")
    return client


def create_dataset(dataset_config: dict):
    """
    Load the dataset named in the config.

    data_format 'processed': TSV files <name>.inter / .meta / .instruction (MemRec layout)
    Raises FileNotFoundError when the data cannot be found (no silent fallback).
    """
    name = dataset_config.get('name', 'instructrec-yelp')
    data_format = dataset_config.get('data_format', 'processed')
    seed = dataset_config.get('seed', 42)
    data_dir = dataset_config.get('data_dir', None)
    project_root = Path(__file__).resolve().parent.parent

    logger.info(f"Dataset: {name} ({data_format})")

    search_dirs = []
    if data_dir:
        search_dirs += [Path(data_dir), project_root / data_dir]
    search_dirs.append(project_root / 'data' / 'processed' / name)

    if data_format == 'processed':
        from src.data.dataset_base import RecDataset
        for d in search_dirs:
            inter_path = d / f"{name}.inter"
            if inter_path.exists():
                logger.info(f"Found processed data: {d}")
                dataset = RecDataset(str(inter_path), seed=seed)
                dataset.load_instructions()
                dataset.load_item_metadata()
                return dataset
    else:
        raise ValueError(f"Unknown data_format '{data_format}' (expected 'processed')")

    raise FileNotFoundError(
        f"No {data_format} data for '{name}'. Searched: " + ", ".join(str(d) for d in search_dirs)
        + ". Set dataset.data_dir in the config to the folder that contains the data.")


def _save_summary_csv(output_dir, config, metrics, args, n_eval_users, n_eval_candidates, parallel_workers=8, warmup_parallel_workers=32):
    summary_file = Path(output_dir) / "summary.csv"
    file_exists = summary_file.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fieldnames = [
        'timestamp', 'dataset', 'model', 'seed', 'n_eval_users', 'n_eval_candidates',
        'Hit@1', 'NDCG@1', 'Hit@3', 'NDCG@3', 'Hit@5', 'NDCG@5', 'Hit@10', 'NDCG@10',
        'pruner_type', 'k', 'tau', 'llm_model', 'n_facets', 'reranker_mode',
        'eval_workers', 'warmup_workers'
    ]

    pruner_type = 'llm_generated_rules'

    memory_config = config.get('memory', {})
    llm_config = config.get('llm', {})
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    if not dataset_name.startswith('instructrec-'):
        dataset_name = f"instructrec-{dataset_name}"

    row = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'model': 'harp',
        'seed': args.seed,
        'n_eval_users': n_eval_users if n_eval_users else 'all',
        'n_eval_candidates': n_eval_candidates,
        'Hit@1':   metrics.get('Hit@1',  0.0),
        'NDCG@1':  metrics.get('NDCG@1', 0.0),
        'Hit@3':   metrics.get('Hit@3',  0.0),
        'NDCG@3':  metrics.get('NDCG@3', 0.0),
        'Hit@5':   metrics.get('Hit@5',  0.0),
        'NDCG@5':  metrics.get('NDCG@5', 0.0),
        'Hit@10':  metrics.get('Hit@10', 0.0),
        'NDCG@10': metrics.get('NDCG@10', 0.0),
        'pruner_type':   pruner_type,
        'k':             memory_config.get('k', 16),
        'tau':           memory_config.get('tau', 1800),
        'llm_model':     llm_config.get('model', 'gpt-4o-mini'),
        'n_facets':      memory_config.get('n_facets', 7),
        'reranker_mode': config.get('reranker', {}).get('mode', 'llm'),
        'eval_workers':  parallel_workers,
        'warmup_workers': warmup_parallel_workers,
    }

    with open(summary_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Summary appended to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Run HARP training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/com_books.yaml)')
    parser.add_argument('--max_interactions', type=int, default=None,
                        help='Maximum number of interactions (None = all)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--n_eval_users', type=int, default=None,
                        help='Number of users to evaluate (None = all)')
    parser.add_argument('--n_eval_candidates', type=int, default=10,
                        help='Number of candidates per user (default: 10)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Folder with <name>.inter/.meta/.instruction (overrides dataset.data_dir in the config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--parallel_workers', type=int, default=8,
                        help='Number of parallel workers for evaluation (default: 8, set 1 for serial)')
    parser.add_argument('--warmup_parallel_workers', type=int, default=None,
                        help='Number of parallel workers for warmup (default: use config or parallel_workers)')
    parser.add_argument('--debug_prompts', action='store_true',
                        help='Dump Stage-R/ReRank prompts for first N users to output_dir/debug_prompts')
    parser.add_argument('--debug_prompt_users', type=int, default=10,
                        help='Number of users to dump prompts for when --debug_prompts is enabled')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Unique run tag for isolating per-user profiles (e.g., 20260323_2200 or expA). Default: timestamp.')
    parser.add_argument('--warmup_rounds', type=int, default=None,
                        help='Override warmup rounds in config (enables warmup if set).')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.run_tag:
        args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.debug_prompts:
        debug_dir = output_dir / "debug_prompts"
        debug_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HARP_DEBUG_PROMPTS"] = "1"
        os.environ["HARP_DEBUG_PROMPTS_N"] = str(args.debug_prompt_users)
        os.environ["HARP_DEBUG_DIR"] = str(debug_dir)
        logger.info(f"Debug prompts enabled, dir: {debug_dir}, n_users: {args.debug_prompt_users}")
    logger.info("="*60)
    logger.info("HARP Training")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    logger.info("[1/6] Loading configuration...")
    config = load_config(args.config)
    from src import settings
    settings.configure(config)
    logger.info(f"Config loaded from: {args.config}")

    if args.warmup_rounds is not None:
        config.setdefault('warmup', {})['enabled'] = True
        config.setdefault('warmup', {})['rounds'] = int(args.warmup_rounds)
        logger.info(f"Warmup override: enabled=true, rounds={args.warmup_rounds}")

    if args.n_eval_users is not None:
        config.setdefault('eval', {})['n_eval_users'] = args.n_eval_users
    config.setdefault('eval', {})['candidate_size'] = args.n_eval_candidates
    config['dataset']['seed'] = args.seed
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir

    logger.info("[2/6] Creating LLM client...")
    llm_client = create_llm_client(config['llm'])

    logger.info("[3/6] Loading dataset...")
    dataset = create_dataset(config['dataset'])

    logger.info("[4/6] Loading prompt templates...")
    prompts_dir = config['prompts']['dir']

    prompt_registry = PromptRegistry(prompts_dir=prompts_dir)
    logger.info(f"Prompt templates loaded from: {prompts_dir}")
    logger.info(f"Loaded prompt templates: {list(prompt_registry.versions().keys())}")

    logger.info("[5/6] Creating ComMemRecAgent...")
    memory_config = config['memory']
    agent = ComMemRecAgent(
        dataset=dataset,
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        k=memory_config.get('k', 16),
        tau=memory_config.get('tau', 1800),
        n_facets=memory_config.get('n_facets', 7),
        mix_min_users=memory_config.get('mix_min_users', 4),
        mix_min_items=memory_config.get('mix_min_items', 6),
        fanout_cap=memory_config.get('fanout_cap', 6),
        reranker_mode=config.get('reranker', {}).get('mode', 'llm'),
        pruner_mode=config.get('pruner', {}).get('mode', 'llm_rules'),
        enable_stage_r=True,
        enable_temporal_weighting=memory_config.get('enable_temporal_weighting', False),
        debug=False
    )
    logger.info("Agent created successfully")

    logger.info("[6/6] Creating trainer and starting training...")
    trainer = ComTrainer(
        agent=agent,
        dataset=dataset,
        profile_config=config.get("profile", {})
    )
    trainer.config = config  # Ensure the trainer has access to the full config
    trainer.run_tag = args.run_tag
    trainer.output_dir = str(output_dir)

    n_eval_users = config.get('eval', {}).get('n_eval_users', None)
    n_eval_candidates = config.get('eval', {}).get('candidate_size', 10)
    logger.info(f"Eval users: {n_eval_users if n_eval_users else 'all'}")
    logger.info(f"Eval candidates: {n_eval_candidates}")

    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)

    try:
        parallel_workers = args.parallel_workers
        if parallel_workers != 8:
            pass
        else:
            parallel_workers = config.get('train', {}).get('parallel_workers', 8)

        if args.warmup_parallel_workers is not None:
            warmup_parallel_workers = args.warmup_parallel_workers
        else:
            warmup_parallel_workers = config.get('train', {}).get('warmup_parallel_workers', parallel_workers)

        logger.info(f"Evaluation parallel workers: {parallel_workers}")
        logger.info(f"Warmup parallel workers: {warmup_parallel_workers}")

        metrics = trainer.train_epoch(
            split='test',
            eval_feedback=config.get('train', {}).get('eval_feedback', 'gt'),
            max_interactions=args.max_interactions,
            n_eval_users=n_eval_users,
            n_eval_candidates=n_eval_candidates,
            seed=args.seed,
            parallel_workers=parallel_workers,
            warmup_config=config.get('warmup', {}),
            warmup_parallel_workers=warmup_parallel_workers
        )

        logger.info("="*60)
        logger.info("Training Completed")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        logger.info("Final Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


        output_file = output_dir / f"results_{Path(args.config).stem}.yaml"
        results = {
            'config': args.config,
            'mode': 'baseline',
            'metrics': metrics,
            'parallel': {
                'eval_workers': parallel_workers,
                'warmup_workers': warmup_parallel_workers
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True)
        logger.info(f"Results saved to: {output_file}")

        _save_summary_csv(
            output_dir=output_dir,
            config=config,
            metrics=metrics,
            args=args,
            n_eval_users=n_eval_users,
            n_eval_candidates=n_eval_candidates,
            parallel_workers=parallel_workers,
            warmup_parallel_workers=warmup_parallel_workers
        )

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
