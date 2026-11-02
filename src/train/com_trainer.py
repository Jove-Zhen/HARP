
from typing import Dict, List, Any, Optional
import sys
import os
import logging
import threading
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger('ComTrainer')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import random

_thread_local = threading.local()


class ComTrainer:

    def __init__(
        self,
        agent,
        dataset,
        profile_config: Dict[str, Any] = None
    ):
        self.agent = agent
        self.dataset = dataset
        self.profile_config = profile_config or {}
        # full config (replaced by the entry script); read for eval.eval_user_list
        self.config = {}  # replaced by the full config in the entry script

        self.interaction_count = 0
        self._facets_counts = []

    def train_epoch(
        self,
        split: str = "test",
        eval_feedback: str = "gt",
        max_interactions: Optional[int] = None,
        n_eval_users: Optional[int] = None,
        n_eval_candidates: int = 10,
        seed: int = 42,
        parallel_workers: int = 1,
        warmup_config: Optional[Dict] = None,
        warmup_parallel_workers: Optional[int] = None
    ) -> Dict[str, float]:
        import random as _random
        _random.seed(seed)
        self._candidate_seed = seed

        warmup_cfg = warmup_config or {}
        if warmup_cfg.get('enabled', False):
            warmup_rounds = warmup_cfg.get('rounds', 1)
            wp_workers = warmup_parallel_workers if warmup_parallel_workers else parallel_workers
            logger.info(f"{'='*60}")
            logger.info(f"🔥 Warmup Training ({warmup_rounds} round(s))")
            logger.info(f"{'='*60}")
            warmup_user_ids = self._get_eval_user_ids(split, n_eval_users, seed)
            # collect the warmup (hit/miss) records used by the profile reflection step
            if os.getenv("HARP_USER_PROFILE", "0") == "1":
                self._warmup_cases = []
            from src import settings
            if settings.PROFILE_FIRST and os.getenv("HARP_USER_PROFILE", "0") == "1":
                self._build_profiles_before_warmup(warmup_user_ids, wp_workers)
            self._run_warmup(warmup_user_ids, split, warmup_rounds, n_eval_candidates, seed, wp_workers)

            if os.getenv("HARP_RERANKER_PROFILE", "0") == "1":
                self._precompute_user_profiles(warmup_user_ids, wp_workers)


        test_data = self._get_test_data(split, max_interactions, n_eval_users, n_eval_candidates, seed)

        if parallel_workers > 1:
            logger.info(f"🚀 Parallel evaluation mode: {parallel_workers} workers, {len(test_data)} interactions")
            result = self._train_epoch_parallel(test_data, eval_feedback, parallel_workers, seed)
        else:
            logger.info(f"🔄 Serial evaluation mode: {len(test_data)} interactions")
            result = self._train_epoch_serial(test_data, eval_feedback, seed)

        self._save_eval_traces()

        return result

    def _get_eval_user_ids(self, split: str, n_eval_users: Optional[int], seed: int) -> List[int]:
        """Evaluation user ids, used by both warmup and evaluation. Honors `eval.eval_user_list`."""
        eval_list_path = self.config.get('eval', {}).get('eval_user_list')
        if eval_list_path:
            import os, json
            from pathlib import Path
            if not os.path.isabs(eval_list_path):
                eval_list_path = Path(__file__).parent.parent.parent / eval_list_path
            if not os.path.exists(eval_list_path):
                raise FileNotFoundError(f"eval.eval_user_list not found: {eval_list_path}")
            with open(eval_list_path, 'r') as f:
                fixed_users = json.load(f)
            if isinstance(fixed_users, dict):
                fixed_users = fixed_users['user_ids']
            split_users = set(getattr(self.dataset, f'{split}_data', {}).keys())
            kept = [u for u in fixed_users if u in split_users]
            if len(kept) < len(fixed_users):
                logger.warning(f"{len(fixed_users) - len(kept)} listed users are not in the {split} split")
            logger.info(f"Loaded {len(kept)} fixed evaluation users from {eval_list_path}")
            if n_eval_users and n_eval_users < len(kept):
                return kept[:n_eval_users]
            return kept

        import random as _random
        _random.seed(seed)
        if hasattr(self.dataset, f'{split}_data'):
            split_data = getattr(self.dataset, f'{split}_data')
            all_user_ids = list(split_data.keys())
            if n_eval_users and n_eval_users < len(all_user_ids):
                _random.seed(seed)
                all_user_ids = _random.sample(all_user_ids, n_eval_users)
            return all_user_ids
        return []

    def _build_profiles_before_warmup(self, user_ids: List[int], parallel_workers: int = 1):
        """
        Build step (phi_u = LLM(history bands || P_profile), one call per user), then bind the
        profile manager to the agent / manager / pruner exactly as the warmup loop
        does after its first round. The warmup loop reuses this manager afterwards.
        """
        from concurrent.futures import ThreadPoolExecutor
        from src.profiles.user_profile import UserProfileManager

        profile_cfg = self.profile_config
        run_tag = getattr(self, 'run_tag', 'default')
        base_dir = getattr(self, 'output_dir', 'results')
        mgr = UserProfileManager(
            llm_client=self.agent.llm_client,
            persist_dir=str(Path(base_dir) / run_tag / "user_profiles"),
            max_rules_per_user=profile_cfg.get('max_rules_per_user', 3),
            min_cases_to_reflect=1,
            trace_dir=str(Path(base_dir) / run_tag / "user_profile_traces"),
            dataset=self.dataset,
        )
        logger.info(f"Building user profiles before warmup for {len(user_ids)} users")
        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                list(ex.map(mgr.build_base_profile, user_ids))
        else:
            for uid in user_ids:
                mgr.build_base_profile(uid)

        if not hasattr(self.agent, '_user_profile_prompt_cache'):
            self.agent._user_profile_prompt_cache = {}
        for uid in user_ids:
            prompt = mgr.get_user_profile_prompt(uid)
            if prompt:
                self.agent._user_profile_prompt_cache[uid] = prompt
        self._user_profile_mgr = mgr
        self.agent._user_profile_mgr = mgr
        if hasattr(self.agent, 'manager'):
            self.agent.manager._user_profile_mgr = mgr
        if hasattr(self.agent, 'pruner'):
            self.agent.pruner._user_profile_mgr = mgr
        mgr.save_all()

    def _run_warmup(
        self,
        eval_user_ids: List[int],
        split: str,
        warmup_rounds: int,
        n_eval_candidates: int,
        seed: int,
        parallel_workers: int = 1
    ):
        import random as _random
        _random.seed(seed)

        if parallel_workers > 1:
            logger.info(f"🚀 Parallel warmup: {parallel_workers} workers, {len(eval_user_ids)} users")

        for warmup_round in range(warmup_rounds):
            target_offset = warmup_rounds - warmup_round
            n_updates = 0
            lock = threading.Lock()

            def warmup_single(user_id: int):
                all_interactions = self.dataset.get_user_history(user_id, split)
                if len(all_interactions) <= target_offset:
                    return False

                target_idx = -(target_offset + 1)
                history = all_interactions[:target_idx]
                target_item = all_interactions[target_idx]

                if target_item is None:
                    return False

                user_history_list = self.dataset.get_user_train_items(user_id)
                if user_history_list is None:
                    user_history_list = []
                    
                all_items = set(range(getattr(self.dataset, 'n_items', 0)))
                
                user_positives = set(user_history_list + [target_item])
                
                negative_pool = list(all_items - user_positives)
                
                if len(negative_pool) < n_eval_candidates - 1:
                    negative_items = negative_pool
                else:
                    negative_items = random.sample(negative_pool, n_eval_candidates - 1)
                
                candidates = [target_item] + negative_items
                random.shuffle(candidates)

                instruction = None
                if hasattr(self.dataset, 'instructions') and self.dataset.instructions:
                    inst_data = self.dataset.instructions.get(user_id)
                    if inst_data:
                        if isinstance(inst_data, dict):
                            instruction = inst_data.get('instruction') or inst_data.get('persona')
                        elif isinstance(inst_data, str):
                            instruction = inst_data

                try:
                    recent_interactions_str = ""
                    if hasattr(self.dataset, 'train_data') and user_id in self.dataset.train_data:
                        recent_items = self.dataset.train_data[user_id][-10:]
                        items_info = []
                        for iid in recent_items:
                            info = f"Item {iid}"
                            if hasattr(self.dataset, 'item_metadata') and self.dataset.item_metadata:
                                meta = self.dataset.item_metadata.get(iid, {})
                                title = meta.get('title', 'Unknown')
                                cats = meta.get('categories', [])
                                info = f"{info}: {title} (Cat: {cats})"
                            items_info.append(info)
                        recent_interactions_str = "\n".join(items_info)

                    ranked_items, details = self.agent.rerank(
                        user_id=user_id,
                        candidates=candidates,
                        instruction=instruction,
                        recent_interactions=recent_interactions_str
                    )

                    if target_item not in ranked_items:
                        return False

                    target_pos = ranked_items.index(target_item)

                    if hasattr(self, '_warmup_cases'):
                        candidates_meta = {}
                        if hasattr(self.dataset, 'item_metadata') and self.dataset.item_metadata:
                            for cid in candidates:
                                meta = self.dataset.item_metadata.get(cid, {})
                                cm = {"title": meta.get("title", ""), "categories": meta.get("categories", [])}
                                desc = meta.get("description", "")
                                if desc:
                                    for seg in desc.strip('"').split(' | '):
                                        if ':' not in seg:
                                            continue
                                        k = seg.split(':')[0].strip().lower()
                                        v = seg.split(':', 1)[1].strip()
                                        if k in ('stars', 'review_count', 'city', 'categories'):
                                            cm[k] = v
                                candidates_meta[cid] = cm
                        facets = details.get('facets', []) if isinstance(details, dict) else []
                        user_mem = self.agent.storage.render_user_summary(user_id) if hasattr(self.agent, 'storage') else ""
                        case = {
                            'user_id': user_id,
                            'ground_truth': target_item,
                            'ranked_items': ranked_items,
                            'candidates_meta': candidates_meta,
                            'facets': facets,
                            'user_memory': user_mem,
                            'hit': (ranked_items[0] == target_item) if ranked_items else False,
                        }
                        with lock:
                            self._warmup_cases.append(case)

                    if target_pos < 10:
                        feedback = {
                            'action': 'CLICK',
                            'item_id': target_item,
                            'position': target_pos
                        }
                        self.agent.write(
                            user_id=user_id,
                            feedback=feedback,
                            recent_facets=details.get('facets', []),
                            pruned_subgraph=details.get('pruned_subgraph', None)
                        )
                        return True
                    return False

                except Exception as e:
                    return False

            if parallel_workers > 1:
                futures = {}
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    for user_id in eval_user_ids:
                        futures[executor.submit(warmup_single, user_id)] = user_id

                    from tqdm import tqdm
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Warmup Round {warmup_round+1}/{warmup_rounds}"):
                        if future.result():
                            with lock:
                                n_updates += 1
            else:
                try:
                    from tqdm import tqdm
                    iter_users = tqdm(eval_user_ids, desc=f"Warmup Round {warmup_round+1}/{warmup_rounds}")
                except ImportError:
                    iter_users = eval_user_ids

                for user_id in iter_users:
                    if warmup_single(user_id):
                        n_updates += 1

            logger.info(f"✓ Warmup Round {warmup_round+1}/{warmup_rounds}: {n_updates}/{len(eval_user_ids)} users updated memories")

            if (hasattr(self, '_warmup_cases') and self._warmup_cases
                    and os.getenv("HARP_USER_PROFILE", "0") == "1"):

                from src.profiles.user_profile import UserProfileManager

                if not hasattr(self, '_user_profile_mgr') or self._user_profile_mgr is None:
                    profile_cfg = self.profile_config
                    max_rules = profile_cfg.get('max_rules_per_user', 3)
                    run_tag = getattr(self, 'run_tag', 'default')
                    base_dir = getattr(self, 'output_dir', 'results')
                    persist_dir = str(Path(base_dir) / run_tag / "user_profiles")
                    trace_dir = str(Path(base_dir) / run_tag / "user_profile_traces")

                    self._user_profile_mgr = UserProfileManager(
                        llm_client=self.agent.llm_client,
                        persist_dir=persist_dir,
                        max_rules_per_user=max_rules,
                        min_cases_to_reflect=1,
                        trace_dir=trace_dir,
                        dataset=self.dataset,
                    )
                    if not hasattr(self.agent, '_user_profile_prompt_cache'):
                        self.agent._user_profile_prompt_cache = {}

                round_cases = list(self._warmup_cases)
                for case in round_cases:
                    self._user_profile_mgr.record_case(
                        user_id=case['user_id'],
                        ground_truth=case['ground_truth'],
                        ranked_items=case['ranked_items'],
                        candidates_meta=case.get('candidates_meta', {}),
                        user_memory=case.get('user_memory', ''),
                    )

                reflected, eligible = self._user_profile_mgr.reflect_all(parallel_workers=parallel_workers)
                stats = self._user_profile_mgr.get_stats()
                n_round_hits = sum(1 for c in round_cases if c.get('hit'))
                logger.info(f"  🧬 User profile reflection after warmup round {warmup_round+1}: "
                            f"{reflected}/{eligible} users reflected, "
                            f"round_hit_rate={n_round_hits}/{len(round_cases)}, "
                            f"avg_version={stats['avg_version']:.1f}")

                for uid in eval_user_ids:
                    profile_prompt = self._user_profile_mgr.get_user_profile_prompt(uid)
                    if profile_prompt:
                        self.agent._user_profile_prompt_cache[uid] = profile_prompt
                self.agent._user_profile_mgr = self._user_profile_mgr
                if hasattr(self.agent, 'manager'):
                    self.agent.manager._user_profile_mgr = self._user_profile_mgr
                if hasattr(self.agent, 'pruner'):
                    self.agent.pruner._user_profile_mgr = self._user_profile_mgr

                self._warmup_cases = []

        logger.info(f"{'='*60}")
        logger.info(f"🔥 Warmup Training Complete!")
        logger.info(f"{'='*60}")

    def _precompute_user_profiles(self, user_ids: List[int], parallel_workers: int = 1):
        """
        Pre-compute structured user profiles after warmup.
        This ensures all profiles are based on rich post-warmup memory,
        and avoids LLM calls during parallel evaluation (thread-safe).

        """
        if not hasattr(self.agent.manager, 'extract_reranker_profile'):
            return

        logger.info(f"{'='*60}")
        logger.info(f"📊 Pre-computing user profiles for the reranker ({len(user_ids)} users)")
        logger.info(f"{'='*60}")

        self.agent.manager._profile_cache = {}
        computed = 0
        empty = 0

        from src import settings
        mgr = getattr(self, '_user_profile_mgr', None)
        skip_profiled = (settings.SHORT_PROFILE_FALLBACK_ONLY and mgr is not None
                         and os.getenv("HARP_USER_PROFILE", "0") == "1"
                         and float(os.getenv("HARP_PROFILE_MIN_CONFIDENCE", "0.0")) <= 0)

        def compute_one(uid: int) -> bool:
            # the reranker reads the short profile only when the structured profile is missing
            if skip_profiled and mgr.get_user_profile_prompt(uid):
                return False
            try:
                mem = self.agent.storage.render_user_summary(uid)
                if not mem or not mem.strip():
                    return False
                profile = self.agent.manager.extract_reranker_profile(uid, user_memory=mem)
                return bool(profile)
            except Exception as e:
                logger.warning(f"  Profile extraction failed for user {uid}: {e}")
                return False

        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=min(parallel_workers, 4)) as executor:
                futures = {executor.submit(compute_one, uid): uid for uid in user_ids}
                for future in as_completed(futures):
                    if future.result():
                        computed += 1
                    else:
                        empty += 1
        else:
            for uid in user_ids:
                if compute_one(uid):
                    computed += 1
                else:
                    empty += 1

        logger.info(f"✓ Profile pre-computation: {computed} computed, {empty} empty/skipped")

    def _generate_feedback(
        self,
        user_id: int,
        target_item: int,
        ranked_items: List[int],
        mode: str
    ) -> Optional[Dict]:
        if mode == 'gt':
            if target_item in ranked_items[:5]:
                return {
                    'action': 'CLICK',
                    'item_id': target_item,
                    'position': ranked_items.index(target_item)
                }
            else:
                return None

        elif mode == 'random':
            selected_item = random.choice(ranked_items[:5])
            action = random.choice(['CLICK', 'SKIP'])
            return {
                'action': action,
                'item_id': selected_item,
                'position': ranked_items.index(selected_item)
            }

        return None

    def _build_candidates(self, user_id: int, ground_truth: int, n_eval_candidates: int) -> list:
        from src import settings
        if settings.CANDIDATE_SEED:
            # candidate set depends only on (seed, user_id), not on the thread that evaluates it
            rng = np.random.RandomState((getattr(self, '_candidate_seed', 42) + int(user_id)) % (2**32))
        else:
            if not hasattr(_thread_local, 'rng'):
                _thread_local.rng = np.random.RandomState(
                    seed=hash(threading.current_thread().ident) % (2**32)
                )
            rng = _thread_local.rng

        user_history_list = self.dataset.get_user_train_items(user_id)
        if user_history_list is None:
            user_history_list = []
        user_history = set(user_history_list)

        if isinstance(ground_truth, (list, tuple)):
            ground_truth = ground_truth[0]

        all_items = set(range(getattr(self.dataset, 'n_items', 0)))
        negative_pool = list(all_items - user_history - {ground_truth})

        if len(negative_pool) < n_eval_candidates - 1:
            return None

        negative_items = rng.choice(
            negative_pool, size=n_eval_candidates - 1, replace=False
        ).tolist()
        candidates = [ground_truth] + negative_items
        rng.shuffle(candidates)
        return candidates

    def _train_epoch_serial(
        self,
        test_data: List[Dict],
        eval_feedback: str,
        seed: int
    ) -> Dict[str, float]:
        all_recommendations = []
        all_ground_truths = []
        self._eval_traces = []

        for i, sample in enumerate(test_data):
            user_id = sample['user_id']
            random.seed((seed + user_id) % (2**32))
            ground_truth = sample['ground_truth']
            n_eval_candidates = sample.get('n_eval_candidates', 10)

            candidates = self._build_candidates(user_id, ground_truth, n_eval_candidates)
            if candidates is None:
                continue

            try:
                recent_interactions_str = ""
                if hasattr(self.dataset, 'train_data') and user_id in self.dataset.train_data:
                    recent_items = self.dataset.train_data[user_id][-10:]
                    items_info = []
                    for iid in recent_items:
                        info = f"Item {iid}"
                        if hasattr(self.dataset, 'item_metadata') and self.dataset.item_metadata:
                            meta = self.dataset.item_metadata.get(iid, {})
                            title = meta.get('title', 'Unknown')
                            cats = meta.get('categories', [])
                            info = f"{info}: {title} (Cat: {cats})"
                        items_info.append(info)
                    recent_interactions_str = "\n".join(items_info)

                ranked_items, details = self.agent.rerank(
                    user_id=user_id,
                    candidates=candidates,
                    instruction=sample.get('instruction', None),
                    return_details=True,
                    recent_interactions=recent_interactions_str
                )

                all_recommendations.append(ranked_items)
                all_ground_truths.append(ground_truth)
                try:
                    self._facets_counts.append(len(details.get('facets', []) if isinstance(details, dict) else []))
                except Exception:
                    pass

                gt_rank = ranked_items.index(ground_truth) if ground_truth in ranked_items else len(ranked_items)
                trace_entry = {
                    'user_id': user_id,
                    'ground_truth': ground_truth,
                    'gt_rank': gt_rank,
                    'ranked_items': ranked_items,
                    'candidates': candidates,
                    'instruction': sample.get('instruction', None),
                    'hit@1': gt_rank < 1,
                    'hit@3': gt_rank < 3,
                    'hit@5': gt_rank < 5,
                    'hit@10': gt_rank < 10,
                }
                if isinstance(details, dict):
                    trace_entry['scores'] = details.get('scores', [])
                    trace_entry['facets'] = details.get('facets', [])
                self._eval_traces.append(trace_entry)

                
                if eval_feedback != 'none':
                    feedback = self._generate_feedback(
                        user_id=user_id,
                        target_item=ground_truth,
                        ranked_items=ranked_items,
                        mode=eval_feedback
                    )
                    if feedback:
                        self.agent.write(
                            user_id=user_id,
                            feedback=feedback,
                            recent_facets=details.get('facets', []) if isinstance(details, dict) else [],
                            pruned_subgraph=details.get('pruned_subgraph', None) if isinstance(details, dict) else None
                        )

                self.interaction_count += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} interactions")

            except Exception as e:
                logger.warning(f"Failed to evaluate user {user_id}: {e}")
                continue

        return self._compute_final_metrics(all_recommendations, all_ground_truths)

    def _train_epoch_parallel(
        self,
        test_data: List[Dict],
        eval_feedback: str,
        parallel_workers: int,
        seed: int
    ) -> Dict[str, float]:

        all_recommendations = [None] * len(test_data)
        all_ground_truths = [None] * len(test_data)
        all_eval_traces = [None] * len(test_data)
        completed_count = [0]
        lock = threading.Lock()

        def eval_single(idx: int, sample: Dict):
            import threading
            import random
            import numpy as np
            user_id = sample['user_id']
            thread_seed = (seed + user_id) % (2**32)
            np.random.seed(thread_seed)
            random.seed(thread_seed)
            
            ground_truth = sample['ground_truth']
            n_eval_candidates = sample.get('n_eval_candidates', 10)

            candidates = self._build_candidates(user_id, ground_truth, n_eval_candidates)
            if candidates is None:
                return idx, None, ground_truth, 0, None, "Not enough negative samples"

            try:
                recent_interactions_str = ""
                if hasattr(self.dataset, 'train_data') and user_id in self.dataset.train_data:
                    recent_items = self.dataset.train_data[user_id][-10:]
                    items_info = []
                    for iid in recent_items:
                        info = f"Item {iid}"
                        if hasattr(self.dataset, 'item_metadata') and self.dataset.item_metadata:
                            meta = self.dataset.item_metadata.get(iid, {})
                            title = meta.get('title', 'Unknown')
                            cats = meta.get('categories', [])
                            info = f"{info}: {title} (Cat: {cats})"
                        items_info.append(info)
                    recent_interactions_str = "\n".join(items_info)

                ranked_items, details = self.agent.rerank(
                    user_id=user_id,
                    candidates=candidates,
                    instruction=sample.get('instruction', None),
                    return_details=True,
                    recent_interactions=recent_interactions_str
                )

                if eval_feedback != 'none':
                    feedback = self._generate_feedback(
                        user_id=user_id,
                        target_item=ground_truth,
                        ranked_items=ranked_items,
                        mode=eval_feedback
                    )
                    if feedback:
                        self.agent.write(
                            user_id=user_id,
                            feedback=feedback,
                            recent_facets=details.get('facets', []) if isinstance(details, dict) else [],
                            pruned_subgraph=details.get('pruned_subgraph', None) if isinstance(details, dict) else None
                        )

                facets_count = 0
                try:
                    facets_count = len(details.get('facets', []) if isinstance(details, dict) else [])
                except Exception:
                    facets_count = 0

                gt_rank = ranked_items.index(ground_truth) if ground_truth in ranked_items else len(ranked_items)
                trace_entry = {
                    'user_id': user_id,
                    'ground_truth': ground_truth,
                    'gt_rank': gt_rank,
                    'ranked_items': ranked_items,
                    'candidates': candidates,
                    'instruction': sample.get('instruction', None),
                    'hit@1': gt_rank < 1,
                    'hit@3': gt_rank < 3,
                    'hit@5': gt_rank < 5,
                    'hit@10': gt_rank < 10,
                }
                if isinstance(details, dict):
                    trace_entry['scores'] = details.get('scores', [])
                    trace_entry['facets'] = details.get('facets', [])

                return idx, ranked_items, ground_truth, facets_count, trace_entry, None

            except Exception as e:
                return idx, None, ground_truth, 0, None, str(e)

        try:
            from tqdm import tqdm
            _use_tqdm = True
        except ImportError:
            _use_tqdm = False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(eval_single, idx, sample): idx
                for idx, sample in enumerate(test_data)
            }

            if _use_tqdm:
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating")
                future_iter = pbar
            else:
                future_iter = as_completed(futures)

            for future in future_iter:
                idx, ranked_items, ground_truth, facets_count, trace_entry, error = future.result()

                if error:
                    logger.warning(f"Failed to evaluate sample {idx}: {error}")
                    continue

                all_recommendations[idx] = ranked_items
                all_ground_truths[idx] = ground_truth
                all_eval_traces[idx] = trace_entry

                with lock:
                    completed_count[0] += 1
                    self.interaction_count += 1
                    cnt = completed_count[0]
                    self._facets_counts.append(facets_count)

                if not _use_tqdm and cnt % 10 == 0:
                    logger.info(f"Processed {cnt}/{len(test_data)} interactions")

            if _use_tqdm:
                pbar.close()

        self._eval_traces = [t for t in all_eval_traces if t is not None]
        valid_pairs = [
            (rec, gt)
            for rec, gt in zip(all_recommendations, all_ground_truths)
            if rec is not None
        ]
        if not valid_pairs:
            return {}
        valid_recs, valid_gts = zip(*valid_pairs)

        return self._compute_final_metrics(list(valid_recs), list(valid_gts))

    def _get_test_data(self, split: str, max_interactions: Optional[int],
                       n_eval_users: Optional[int] = None,
                       n_eval_candidates: int = 10,
                       seed: int = 42) -> List[Dict]:
        import random as _random
        _random.seed(seed)
        test_data = []

        if hasattr(self.dataset, f'{split}_data'):
            split_data = getattr(self.dataset, f'{split}_data')
            all_user_ids = self._get_eval_user_ids(split, n_eval_users, seed)

            for user_id in all_user_ids:
                interactions = split_data[user_id]
                if interactions is None:
                    continue

                if isinstance(interactions, int):
                    ground_truth = interactions
                elif isinstance(interactions, (list, tuple)) and len(interactions) > 0:
                    last = interactions[-1]
                    if isinstance(last, (list, tuple)):
                        ground_truth = last[0]
                    else:
                        ground_truth = last
                else:
                    continue

                instruction = None
                if hasattr(self.dataset, 'instructions') and self.dataset.instructions:
                    inst_data = self.dataset.instructions.get(user_id)
                    if inst_data:
                        if isinstance(inst_data, dict):
                            instruction = inst_data.get('instruction') or inst_data.get('persona')
                        elif isinstance(inst_data, str):
                            instruction = inst_data

                test_data.append({
                    'user_id': user_id,
                    'ground_truth': ground_truth,
                    'instruction': instruction,
                    'n_eval_candidates': n_eval_candidates
                })

                if max_interactions and len(test_data) >= max_interactions:
                    break

        return test_data

    def _save_eval_traces(self):
        """Save per-case eval traces to results/<run_tag>/eval_traces.json"""
        traces = getattr(self, '_eval_traces', [])
        if not traces:
            return

        run_tag = getattr(self, 'run_tag', None)
        base_dir = getattr(self, 'output_dir', 'results')
        if not run_tag:
            return

        trace_dir = Path(base_dir) / run_tag
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "eval_traces.json"

        n_total = len(traces)
        n_hit1 = sum(1 for t in traces if t.get('hit@1'))
        n_hit3 = sum(1 for t in traces if t.get('hit@3'))
        n_hit5 = sum(1 for t in traces if t.get('hit@5'))
        n_hit10 = sum(1 for t in traces if t.get('hit@10'))

        output = {
            'summary': {
                'n_total': n_total,
                'hit@1': n_hit1 / max(1, n_total),
                'hit@3': n_hit3 / max(1, n_total),
                'hit@5': n_hit5 / max(1, n_total),
                'hit@10': n_hit10 / max(1, n_total),
                'n_hit1': n_hit1,
                'n_miss1': n_total - n_hit1,
            },
            'cases': traces
        }

        try:
            with open(trace_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"📊 Eval traces saved: {trace_path} ({n_total} cases, Hit@1={n_hit1}/{n_total})")
        except Exception as e:
            logger.warning(f"Failed to save eval traces: {e}")

    def _compute_final_metrics(
        self,
        recommendations: List[List[int]],
        ground_truths: List[int]
    ) -> Dict[str, float]:
        if not recommendations:
            return {}

        import numpy as np

        positions = []
        for rec, gt in zip(recommendations, ground_truths):
            try:
                pos = rec.index(gt)
            except ValueError:
                pos = len(rec)
            positions.append(pos)

        positions = np.array(positions)
        metrics = {}

        for k in [1, 3, 5, 10]:
            metrics[f'Hit@{k}'] = float(np.mean(positions < k))

            ndcg_scores = np.where(
                positions < k,
                1.0 / np.log2(positions + 2),
                0.0
            )
            metrics[f'NDCG@{k}'] = float(np.mean(ndcg_scores))

        debug_dir = os.getenv("HARP_DEBUG_DIR", "")
        if os.getenv("HARP_DEBUG_PROMPTS", "") == "1" and debug_dir:
            try:
                xs = list(self._facets_counts)
                xs_sorted = sorted(xs)
                def _p(pct: float) -> float:
                    if not xs_sorted:
                        return 0.0
                    idx = int(round((len(xs_sorted) - 1) * pct))
                    return float(xs_sorted[idx])
                run_stats = {
                    "facets_count": {
                        "n": len(xs),
                        "mean": float(np.mean(xs)) if xs else 0.0,
                        "p0": _p(0.0),
                        "p50": _p(0.5),
                        "p90": _p(0.9),
                        "p100": _p(1.0)
                    }
                }
                out_path = Path(debug_dir) / "run_stats.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(run_stats, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return metrics
