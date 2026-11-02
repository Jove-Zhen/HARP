"""
Agent: Stage-R (retrieval), Stage-ReRank and Stage-W (write)
Three-stage architecture: Stage-R (Retrieval) + Stage-ReRank + Stage-W (Write)
"""
from typing import List, Dict, Tuple, Optional
import os
import time

from src.memory.graph import UserItemGraph
from src.memory.packer import SnippetPacker
from src.memory.storage import MemoryStorage
from src.memory.pruner_llm_rules import LLMRulePruner
from src.memory.temporal_memory import TemporalMemory
from src.models.reranker_llm import LLMReranker


class MemRecAgent:
    """
    MemRec Agent: Three-stage collaborative memory
    
    Workflow (per interaction):
    1. Pruning: Select top-k neighbors (mixing constraint)
    2. Packing: Pack snippets within token budget
    3. Stage-R: Generate facets, vector_profile, support_edges (no candidate scoring)
    4. Stage-ReRank: Score candidates using reranker (LLM or vector)
    5. (Optional) Stage-W: Generate user_patches + neighbor_patches after feedback
    """
    
    def __init__(
        self,
        dataset,
        llm_client,
        k: int = 16,
        tau: int = 1800,
        n_facets: int = 7,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        mix_min_users: int = 4,
        mix_min_items: int = 6,
        fanout_cap: int = 6,
        reranker_mode: str = "llm",
        pruner_mode: str = "llm_rules",
        enable_stage_r: bool = True,
        reranker_llm_client = None,
        enable_temporal_weighting: bool = False,
        debug: bool = False
    ):
        """
        Initialize MemRec Agent
        
        Args:
            dataset: RecDataset instance
            llm_client: LLMClient instance (with cache)
            k: Number of neighbors selected by pruner
            tau: Token budget
            n_facets: Number of facets requested
            temperature: LM sampling temperature
            max_tokens: Maximum tokens generated (Stage-R)
            mix_min_users: Minimum user neighbors
            mix_min_items: Minimum item neighbors
            fanout_cap: Stage-W propagation fanout cap
            reranker_mode: Reranker mode (only "llm")
            pruner_mode: Pruner mode (only "llm_rules")
            debug: Whether to print debug information
        """
        self.dataset = dataset
        self.llm_client = llm_client
        self.k = k
        self.tau = tau
        self.n_facets = n_facets
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        self.fanout_cap = fanout_cap
        self.reranker_mode = reranker_mode
        self.pruner_mode = pruner_mode
        self.enable_stage_r = enable_stage_r
        self.enable_temporal_weighting = enable_temporal_weighting
        self.debug = debug

        self.temporal_memory = TemporalMemory(lambda_param=0.1)

        print("Building user-item graph...")
        self.graph = UserItemGraph(dataset)
        print(self.graph)
        
        if pruner_mode == "llm_rules":
            from src import settings
            self.pruner = LLMRulePruner(
                dataset_name=settings.resolve_pruner_rules_dataset(dataset),
                k=k,
                dataset=dataset,
                mix_min_users=mix_min_users,
                mix_min_items=mix_min_items,
            )
            print(f"  Using LLM Rule-Based Pruner for domain-specific neighbor selection")
        else:
            raise ValueError(f"Unsupported pruner_mode: {pruner_mode!r} (only 'llm_rules')")
        self.packer = SnippetPacker(tau=tau)
        self.manager = None  # set by the subclass (needs the prompt registry)
        self.storage = MemoryStorage()
        
        if dataset.item_metadata:
            self.storage.initialize_item_descriptions(dataset.item_metadata)
        
        import os
        if os.getenv("HARP_HOMOPHILY", "0") == "1" or os.getenv("HARP_ITEM_PROFILE", "0") == "1":
            reviews = getattr(dataset, 'reviews', None)
            if dataset.item_metadata:
                self.storage.initialize_item_profiles(dataset.item_metadata, reviews=reviews)

        from src import settings
        if settings.ITEM_CHANNELS and dataset.item_metadata:
            self.storage.initialize_item_channel_profiles(
                dataset.item_metadata, reviews=getattr(dataset, 'reviews', None),
                dataset_name=getattr(dataset, 'name', ''))
            # pruner needs the item profiles for the user-item homophily gate
            self.pruner._storage = self.storage
        
        if reranker_mode == "llm":
            print("  Reranker: LLM-based")
            # If separate reranker_llm_client is provided, use it; otherwise use default llm_client
            reranker_client = reranker_llm_client if reranker_llm_client is not None else llm_client
            if reranker_llm_client is not None:
                print(f"    Using separate LLMClient: {reranker_client.model} @ {reranker_client.api_endpoint}")
            self.reranker = LLMReranker(reranker_client)
        else:
            raise ValueError(f"Unsupported reranker_mode: {reranker_mode!r} (only 'llm')")
        
        self.n_stage_r_calls = 0
        self.n_stage_rr_calls = 0
        self.n_stage_w_calls = 0
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        instruction: str = None,
        recent_interactions: str = None,
        return_details: bool = False,
        debug_logger = None
    ) -> Tuple[List[int], Dict]:
        """
        Rerank candidate items for user (three stages: Pruning → Stage-R → Stage-ReRank)
        
        Args:
            user_id: User ID
            candidates: List of candidate item IDs (typically 10)
            instruction: User instruction/intent (optional, from dataset)
            return_details: Whether to return detailed information
            debug_logger: Optional debug logger
            
        Returns:
            (ranked_items, details)
            - ranked_items: Reranked list of item IDs (descending order)
            - details: Contains facets, scores, retrieval_bundle, etc.
        """
        t0 = time.time()
        pruned_subgraph = self.pruner.prune(user_id, self.graph, candidates)
        pruning_time = time.time() - t0
        
        user_memory_summary = self.storage.render_user_summary(user_id)
        packed_context = self.packer.pack(
            pruned_subgraph=pruned_subgraph,
            dataset=self.dataset,
            candidates=candidates,
            user_memory_summary=user_memory_summary
        )
        
        if self.debug:
            print(f"\n[MemRec] User {user_id}")
            print(f"  Pruned: {pruned_subgraph['n_items']} items, {pruned_subgraph['n_users']} users")
            print(f"  Packed: {packed_context['n_neighbors']} neighbors, ~{packed_context['estimated_tokens']} tokens")
        
        if self.enable_stage_r:
            t0 = time.time()
            retrieval_bundle = self.manager.run_stage_r(
                user_id=user_id,
                packed_context=packed_context,
                instruction=instruction,
                n_facets=self.n_facets,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                debug_logger=debug_logger
            )
            stage_r_time = time.time() - t0
            self.n_stage_r_calls += 1

            # Apply temporal weighting to facets (weight recent interactions higher)
            if retrieval_bundle.get('facets') and getattr(self, 'enable_temporal_weighting', False):
                facets = retrieval_bundle['facets']
                weighted_facets = self.temporal_memory.apply_temporal_decay(
                    facets,
                    use_exponential=True
                )
                retrieval_bundle['facets'] = weighted_facets

            if self.debug:
                print(f"  Stage-R: {len(retrieval_bundle.get('facets', []))} facets, {stage_r_time*1000:.1f}ms")
        else:
            retrieval_bundle = {
                'facets': [],
                'vector_profile': None,
                'support_edges': []
            }
            stage_r_time = 0.0
            
            if self.debug:
                print(f"  Stage-R: SKIPPED (No-Mem mode)")

        t0 = time.time()
        candidate_list = self._prepare_candidate_list(candidates)
        item_mems = self._get_item_mems(candidates)

        user_profile = None

        if self.reranker_mode == "llm" and os.getenv("HARP_USER_PROFILE", "0") == "1":
            per_user_prompt = None
            min_confidence = float(os.getenv("HARP_PROFILE_MIN_CONFIDENCE", "0.0"))
            
            if hasattr(self, '_user_profile_prompt_cache') and user_id in self._user_profile_prompt_cache:
                cached_prompt = self._user_profile_prompt_cache[user_id]
                if min_confidence > 0 and hasattr(self, '_user_profile_mgr') and self._user_profile_mgr:
                    user_prof = self._user_profile_mgr._profiles.get(user_id)
                    if user_prof and user_prof.confidence >= min_confidence:
                        per_user_prompt = cached_prompt
                else:
                    per_user_prompt = cached_prompt
            elif hasattr(self, '_user_profile_mgr'):
                user_prof = self._user_profile_mgr._profiles.get(user_id)
                if user_prof and (min_confidence <= 0 or user_prof.confidence >= min_confidence):
                    per_user_prompt = self._user_profile_mgr.get_user_profile_prompt(user_id)

            if per_user_prompt:
                user_profile = per_user_prompt

        if not user_profile:
            if self.reranker_mode == "llm" and os.getenv("HARP_RERANKER_PROFILE", "0") == "1":
                if hasattr(self, 'manager') and hasattr(self.manager, 'extract_reranker_profile'):
                    user_profile = self.manager.extract_reranker_profile(user_id, user_memory=user_memory_summary)

        if self.reranker_mode == "llm" and os.getenv("HARP_ENRICH_CANDIDATES", "0") == "1":
            seq_signal = self._extract_recent_sequence(user_id)
            if seq_signal:
                if user_profile:
                    user_profile = f"{user_profile}\n{seq_signal}"
                else:
                    user_profile = seq_signal
        
        rerank_scores = self.reranker.rerank(
            user_id=user_id,
            retrieval_bundle=retrieval_bundle,
            candidates=candidate_list,
            item_mems=item_mems,
            instruction=instruction,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            debug_logger=debug_logger,
            user_profile=user_profile
        )
        stage_rr_time = time.time() - t0
        self.n_stage_rr_calls += 1
        
        if self.debug:
            print(f"  Stage-ReRank ({self.reranker_mode}): {len(rerank_scores)} scores, {stage_rr_time*1000:.1f}ms")
        
        rerank_scores_sorted = sorted(rerank_scores, key=lambda x: x.get('score', 0), reverse=True)

        ranked_items = [s['item_id'] for s in rerank_scores_sorted]
        scores = [s['score'] for s in rerank_scores_sorted]
        
        details = {
            'facets': retrieval_bundle.get('facets', []),
            'support_edges': retrieval_bundle.get('support_edges', []),
            'rerank_scores': rerank_scores,
            'scores': scores,
            'retrieval_bundle': retrieval_bundle,
            'pruned_subgraph': pruned_subgraph,
            'packed_context': packed_context,
            'timings': {
                'pruning': pruning_time,
                'stage_r': stage_r_time,
                'stage_rr': stage_rr_time
            }
        }
        
        if return_details:
            return ranked_items, details
        else:
            return ranked_items, {}
    
    def write(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        pruned_subgraph: Dict = None,
        debug_logger = None
    ) -> Dict:
        """
        Execute Stage-W (Write)

        Args:
            user_id: User ID
            feedback: Feedback information, e.g., {'action': 'CLICK', 'item_id': 123}
            recent_facets: Recent facets (from Stage-R)
            pruned_subgraph: Pruned subgraph (optional, for obtaining neighbor IDs)
            debug_logger: Optional debug logger

        Returns:
            {
                'user_patches': [...],
                'neighbor_patches': [...],
                'propagation_policy': {...},
                'stats': {...}
            }
        """
        current_profile = self.storage.get_user_memory(user_id)
        user_mem_keys = [current_profile] if current_profile else []
        
        item_mem_keys = {}
        clicked_item_info = None
        if 'item_id' in feedback:
            item_id = feedback['item_id']
            current_item_mem = self.storage.get_item_memory(item_id)
            item_mem_keys[item_id] = [current_item_mem] if current_item_mem else []
            
            # Get item metadata (title + semantic attributes) for generating semantic memory
            if self.dataset.item_metadata:
                item_text = self.dataset.item_metadata.get(item_id, {}).get('title', f'Item-{item_id}')
                # Append semantic description (categories, stars, etc.) from storage
                item_sem = self.storage.get_item_memory(item_id)
                if item_sem and item_text in item_sem:
                    clicked_item_info = f"[Item-{item_id}] {item_sem[:200]}"
                elif item_sem:
                    clicked_item_info = f"[Item-{item_id}] {item_text}. {item_sem[:150]}"
                else:
                    clicked_item_info = f"[Item-{item_id}] {item_text}"
            else:
                clicked_item_info = f"[Item-{item_id}]"
        
        neighbor_ids = []
        neighbor_map = {}
        neighbor_details = []
        
        used_neighbors = set()
        for facet in recent_facets:
            supporting = facet.get('supporting_neighbors', [])
            used_neighbors.update(supporting)
        
        if pruned_subgraph:
            for nb in pruned_subgraph.get('neighbors', []):
                nb_id = nb['id']
                nb_type = nb['type']
                id_str = f"{nb_type.capitalize()}-{nb_id}"
                neighbor_ids.append(id_str)
                neighbor_map[id_str] = (nb_id, nb_type)
                
                nb_memory = ""
                if nb_type == 'user':
                    nb_memory = self.storage.get_user_memory(nb_id)
                else:
                    nb_memory = self.storage.get_item_memory(nb_id)
                
                if nb_memory:
                    if isinstance(nb_memory, list):
                        nb_memory = str(nb_memory)
                    if nb_memory.lower() in ['nan', '[nan]', "['']", '']:
                        nb_memory = "(No memory)"
                else:
                    nb_memory = "(No memory)"
                
                neighbor_details.append({
                    'id_str': id_str,
                    'score': nb.get('score', 0.0),
                    'memory': nb_memory,
                    'used_in_retrieval': id_str in used_neighbors
                })
        
        stage_w_response = self.manager.run_stage_w(
            user_id=user_id,
            feedback=feedback,
            recent_facets=recent_facets,
            user_mem_keys=user_mem_keys,
            item_mem_keys=item_mem_keys,
            neighbor_ids=neighbor_ids,
            neighbor_details=neighbor_details,
            fanout_cap=self.fanout_cap,
            clicked_item_info=clicked_item_info,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            debug_logger=debug_logger
        )
        self.n_stage_w_calls += 1
        
        new_user_mem = stage_w_response.get('user_memory', '')
        new_item_mem = stage_w_response.get('item_memory', '')
        neighbor_updates = stage_w_response.get('neighbor_updates', [])
        
        # Output comparison of original and updated memories (for debugging)
        if debug_logger:
            debug_logger("\n>>> MEMORY UPDATE COMPARISON <<<")
            
            old_user_mem = user_mem_keys[0] if user_mem_keys else "(No memory yet)"
            debug_logger(f"\n[User Memory]")
            debug_logger(f"  OLD: {old_user_mem}")
            debug_logger(f"  NEW: {new_user_mem}")
            
            old_item_mem = None
            if item_mem_keys:
                for item_id, mem_list in item_mem_keys.items():
                    if mem_list:
                        old_item_mem = mem_list[0]
                        break
            old_item_mem = old_item_mem if old_item_mem else "(No memory yet)"
            debug_logger(f"\n[Item Memory]")
            debug_logger(f"  OLD: {old_item_mem}")
            debug_logger(f"  NEW: {new_item_mem}")
            
            if neighbor_updates:
                debug_logger(f"\n[Neighbor Memory Updates]")
                for update in neighbor_updates:
                    neighbor_id_str = update.get('neighbor_id', '')
                    memory_update = update.get('memory_update', '')
                    rationale = update.get('rationale', '')
                    
                    old_neighbor_mem = "(No memory)"
                    if neighbor_details:
                        nb_detail = next((d for d in neighbor_details if d.get('id_str') == neighbor_id_str), None)
                        if nb_detail:
                            old_neighbor_mem = nb_detail.get('memory', '(No memory)')
                    
                    debug_logger(f"\n  [{neighbor_id_str}]")
                    debug_logger(f"    OLD: {old_neighbor_mem}")
                    debug_logger(f"    NEW: {memory_update}")
                    debug_logger(f"    Rationale: {rationale}")
            else:
                debug_logger(f"\n[Neighbor Memory Updates] (None)")
            
            debug_logger("")
        
        if new_user_mem:
            self.storage.update_user_memory(user_id, new_user_mem)
            user_updated = 1
        else:
            user_updated = 0

        clicked_item_id = feedback.get('item_id')
        if new_item_mem and clicked_item_id:
            self.storage.update_item_memory(clicked_item_id, new_item_mem)
            self.storage.record_interaction_time(user_id, clicked_item_id)
            item_updated = 1
        elif clicked_item_id:
            self.storage.record_interaction_time(user_id, clicked_item_id)
            item_updated = 0
        else:
            item_updated = 0
        
        neighbor_updated = 0
        user_neighbors_propagated = []
        item_neighbors_propagated = []
        for update in neighbor_updates:
            neighbor_id_str = update.get('neighbor_id', '')
            memory_update = update.get('memory_update', '')
            rationale = update.get('rationale', '')
            
            if not neighbor_id_str or not memory_update:
                continue
            
            if neighbor_id_str not in neighbor_map:
                if debug_logger:
                    debug_logger(f"  ⚠️ Unknown neighbor: {neighbor_id_str}")
                continue
            
            nb_id, nb_type = neighbor_map[neighbor_id_str]
            
            if nb_type == 'user':
                self.storage.update_user_memory(nb_id, memory_update)
                user_neighbors_propagated.append(nb_id)
                if debug_logger:
                    debug_logger(f"  ✓ Updated User-{nb_id}: {rationale}")
            else:
                self.storage.update_item_memory(nb_id, memory_update)
                item_neighbors_propagated.append(nb_id)
                if debug_logger:
                    debug_logger(f"  ✓ Updated Item-{nb_id}: {rationale}")
            
            neighbor_updated += 1

        import os as _os
        if _os.getenv("HARP_HOMOPHILY", "0") == "1":
            user_profile_update = stage_w_response.get('user_skill_update', {})
            if user_profile_update and user_profile_update.get('should_update', False):
                user_profile_mgr = getattr(self, '_user_profile_mgr', None)
                if user_profile_mgr:
                    user_profile_mgr.update_user_profile(user_id, user_profile_update)

            item_profile_update = stage_w_response.get('item_skill_update', {})
            if item_profile_update and item_profile_update.get('should_update', False) and clicked_item_id:
                fans_delta = item_profile_update.get('typical_fans_delta', '')
                if fans_delta:
                    existing = self.storage.get_item_profile(clicked_item_id)
                    if existing:
                        old_fans = existing.get('typical_fans', '')
                        new_fans = f"{old_fans}; {fans_delta}" if old_fans else fans_delta
                        self.storage.update_item_profile(clicked_item_id, {'typical_fans': new_fans})

        combined_stats = {
            'user_applied': user_updated,
            'item_applied': item_updated,
            'neighbor_applied': neighbor_updated,
            'user_neighbors_propagated': user_neighbors_propagated,
            'item_neighbors_propagated': item_neighbors_propagated,
            'total_applied': user_updated + item_updated + neighbor_updated
        }

        if debug_logger:
            if neighbor_updated > 0:
                debug_logger(f"\n🔄 Collaborative Propagation (LLM-decided):")
                debug_logger(f"  User neighbors: {len(user_neighbors_propagated)} updated")
                debug_logger(f"  Item neighbors: {len(item_neighbors_propagated)} updated")
                debug_logger(f"  Total propagated: {neighbor_updated}")
            else:
                debug_logger(f"\n⚠️ No neighbor propagation (LLM chose not to propagate)")

        return {
            'user_memory': new_user_mem,
            'item_memory': new_item_mem,
            'stats': combined_stats,
            'raw_response': stage_w_response
        }
    
    def _prepare_candidate_list(self, candidates: List[int]) -> List[Dict]:
        candidate_list = []
        for cid in candidates:
            item_meta = self.dataset.item_metadata.get(cid, {})
            entry = {
                'id': cid,
                'title': item_meta.get('title', 'N/A'),
                'tags': item_meta.get('tags', [])
            }

            if os.getenv("HARP_ENRICH_CANDIDATES", "0") == "1":
                desc = item_meta.get('description', '')
                if desc:
                    enriched = self._parse_item_attributes(desc)
                    entry.update(enriched)

            candidate_list.append(entry)
        return candidate_list

    @staticmethod
    def _parse_item_attributes(description: str) -> Dict:
        attrs = {}
        try:
            parts = description.strip('"').split(' | ')
            for part in parts:
                if ':' not in part:
                    continue
                key, val = part.split(':', 1)
                key = key.strip().lower()
                val = val.strip()

                if key == 'stars':
                    attrs['stars'] = val
                elif key == 'review_count':
                    attrs['review_count'] = val
                elif key == 'categories':
                    attrs['categories'] = val
                elif key == 'city':
                    attrs['city'] = val
                elif key == 'attributes':
                    if "'RestaurantsPriceRange2'" in val:
                        import re
                        m = re.search(r"'RestaurantsPriceRange2':\s*'(\d)'", val)
                        if m:
                            price_map = {'1': '$', '2': '$$', '3': '$$$', '4': '$$$$'}
                            attrs['price'] = price_map.get(m.group(1), m.group(1))
        except Exception:
            pass
        return attrs

    def _extract_recent_sequence(self, user_id: int, n_recent: int = 5) -> str:
        try:
            items = self.graph.get_user_items(user_id)
            if not items or len(items) < 2:
                return ""
            recent = items[-n_recent:]

            from collections import Counter
            cat_counter = Counter()
            for item_id in recent:
                meta = self.dataset.item_metadata.get(item_id, {})
                desc = meta.get('description', '')
                for part in desc.strip('"').split(' | '):
                    if part.strip().lower().startswith('categories:'):
                        cats = part.split(':', 1)[1].strip()
                        for c in cats.split(','):
                            c = c.strip()
                            if c:
                                cat_counter[c] += 1

            if not cat_counter:
                return ""

            top_cats = cat_counter.most_common(3)
            cat_str = ", ".join(f"{c}({n})" for c, n in top_cats)
            return f"**Recent Activity Pattern** (last {len(recent)} visits): {cat_str}"
        except Exception:
            return ""
    
    def _get_item_mems(self, candidates: List[int]) -> Dict[int, Dict]:
        """
        Get memories of candidate items
        
        Args:
            candidates: List of candidate item IDs
            
        Returns:
            {item_id: ItemMem, ...}
        """
        item_mems = {}
        for cid in candidates:
            item_mem = self.storage.get_item_memory(cid)
            if item_mem:
                item_mems[cid] = item_mem
        return item_mems
    
