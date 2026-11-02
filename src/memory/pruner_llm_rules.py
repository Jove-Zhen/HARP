"""
LLM-Generated Rule-Based Neighbor Pruner
Domain-specific pruning using interpretable, zero-shot rules
"""
import os
from typing import Dict, List, Optional
from .domain_rules import get_domain_rules


class LLMRulePruner:
    """
    LLM-generated rule-based pruner for domain-specific neighbor selection
    
    Key features:
    - Domain-adaptive: Different rules for Books, Movies, Restaurants, etc.
    - Interpretable: Clear semantic rules with explicit thresholds
    - Zero-shot: No training data required
    - Efficient: Simple rule evaluation at inference time
    """
    
    def __init__(
        self,
        dataset_name: str,
        k: int = 16,
        dataset=None,
        mix_min_users: int = 4,
        mix_min_items: int = 6,
    ):
        """
        Initialize LLM rule pruner
        
        Args:
            dataset_name: Name of dataset (e.g., 'instructrec-books')
            k: Total number of neighbors to select
            dataset: Dataset instance (for metadata access)
            mix_min_users: Minimum number of user neighbors
            mix_min_items: Minimum number of item neighbors
        """
        self.dataset_name = dataset_name
        self.k = k
        self.dataset = dataset
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        
        self.rules_class = get_domain_rules(dataset_name)
        print(f"  LLMRulePruner: Using {self.rules_class.__name__} for {dataset_name}")
        print(f"  Rules: {self.rules_class.get_description()}")
    
    def extract_features(
        self,
        neighbor: Dict,
        user_memory: Optional[str] = None,
        neighbor_memory: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Extract features for a single neighbor
        
        Args:
            neighbor: Neighbor dict with keys:
                - type: 'user' or 'item'
                - id: neighbor ID
                - score: base CF score (edge_weight)
                - timestamp: last interaction time (optional)
                - metadata: item metadata (optional)
            user_memory: User's memory string
            neighbor_memory: Neighbor's memory string
        
        Returns:
            features: Dict with standardized feature names
        """
        features = {}
        
        features['edge_weight'] = neighbor.get('score', 0.0)
        
        features['neighbor_type'] = neighbor.get('type', 'item')
        
        features['co_interaction_count'] = neighbor.get('co_count', 0)
        
        timestamp = neighbor.get('timestamp', 0)
        if timestamp > 0:
            # Assume timestamp is days ago (or convert from actual timestamp)
            features['recency_days'] = timestamp
        else:
            features['recency_days'] = 0
        
        # This should be computed externally based on domain-specific metadata
        features['metadata_overlap_score'] = neighbor.get('metadata_overlap', 0.0)
        
        if user_memory and neighbor_memory:
            features['memory_similarity_score'] = neighbor.get('memory_sim', 0.0)
        else:
            features['memory_similarity_score'] = 0.0
        
        return features
    
    def score_neighbor(
        self,
        neighbor: Dict,
        user_memory: Optional[str] = None,
        neighbor_memory: Optional[str] = None,
    ) -> float:
        """
        Score a single neighbor using domain-specific rules
        
        Args:
            neighbor: Neighbor dict
            user_memory: User's memory string
            neighbor_memory: Neighbor's memory string
        
        Returns:
            score: Final weighted score
        """
        features = self.extract_features(neighbor, user_memory, neighbor_memory)
        
        score = self.rules_class.apply_rules(features)
        
        return score
    
    def _item_homophily_gate(self, user_id, item_id, user_profile_mgr) -> float:
        """
        g(H(phi_u, phi_i)) = alpha_ui + (1 - alpha_ui) * H, H over the four channels
        (preference, quality, context, audience). Neutral (1.0) when the user profile or the
        item's channel information is unavailable, so the ranking is unchanged in that case.
        """
        from src import settings
        from src.homophily import profile_homophily, gate, has_channel_info

        storage = getattr(self, '_storage', None)
        if user_profile_mgr is None or storage is None:
            return 1.0
        user_prof = user_profile_mgr.get_user_profile(user_id)
        item_profile = storage.get_item_channel_profile(item_id) if hasattr(storage, 'get_item_channel_profile') else None
        if not user_prof or not user_prof.profile or not item_profile or not has_channel_info(item_profile):
            return 1.0
        return gate(profile_homophily(user_prof.profile, item_profile), settings.ALPHA_UI)

    def prune(
        self,
        user_id: int = None,
        graph = None,
        candidates: List[int] = None,
        user_memory: Optional[str] = None,
        neighbor_memories: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Select top-k neighbors using domain-specific rules
        (Compatible with NeighborPruner interface)
        
        Args:
            user_id: User ID
            graph: UserItemGraph instance
            candidates: Optional list of candidate item IDs
            user_memory: User's memory string (unused in basic version)
            neighbor_memories: Dict mapping neighbor_id -> memory string (unused)
        
        Returns:
            pruned_result: Dict with 'neighbors' key containing selected neighbors
        """
        if graph is None:
            return {'neighbors': []}
        
        neighbor_data = graph.get_user_neighbors(user_id, max_items=100, max_users=50)
        
        candidate_neighbors = []
        
        for item_id, recency in neighbor_data['item_neighbors']:
            metadata_overlap = 0.0
            if self.dataset and hasattr(self.dataset, 'item_metadata'):
                metadata_overlap = 0.5
            
            edge_weight = recency if recency else 0.5
            
            # Convert recency to days (approximate: assume higher recency = more recent = lower days)
            days_ago = (1.0 - recency) * 365 if recency else 180
            
            candidate_neighbors.append({
                'type': 'item',
                'id': item_id,
                'score': edge_weight,
                'metadata_overlap': metadata_overlap,
                'co_count': 0,
                'memory_sim': 0.0,
                'timestamp': days_ago,
            })
        
        for neighbor_user_id, overlap_count in neighbor_data['user_neighbors']:
            user_degree = graph.get_user_degree(user_id)
            edge_weight = overlap_count / user_degree if user_degree > 0 else 0.5
            
            candidate_neighbors.append({
                'type': 'user',
                'id': neighbor_user_id,
                'score': edge_weight,
                'metadata_overlap': 0.0,
                'co_count': overlap_count,
                'memory_sim': 0.0,
                'timestamp': 0,
            })
        
        if not candidate_neighbors:
            return {'neighbors': []}
        
        scored_neighbors = []
        _homophily_on = os.getenv("HARP_HOMOPHILY", "0") == "1"
        _user_profile_mgr = getattr(self, '_user_profile_mgr', None) if _homophily_on else None
        from src import settings
        _use_ui_gate = settings.UI_GATE
        _user_profile_mgr_any = getattr(self, '_user_profile_mgr', None)

        for neighbor in candidate_neighbors:
            score = self.score_neighbor(neighbor, user_memory, None)

            if _user_profile_mgr and neighbor['type'] == 'user':
                uu_compat = _user_profile_mgr.compute_user_user_homophily(user_id, neighbor['id'])
                score *= (settings.ALPHA_UU + (1 - settings.ALPHA_UU) * uu_compat)

            if _use_ui_gate and neighbor['type'] == 'item':
                score *= self._item_homophily_gate(user_id, neighbor['id'], _user_profile_mgr_any)

            scored_neighbors.append((score, neighbor))
        
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        user_neighbors = [(s, n) for s, n in scored_neighbors if n['type'] == 'user']
        item_neighbors = [(s, n) for s, n in scored_neighbors if n['type'] == 'item']
        
        selected = []
        
        selected.extend([n for _, n in user_neighbors[:self.mix_min_users]])
        
        selected.extend([n for _, n in item_neighbors[:self.mix_min_items]])
        
        remaining_slots = self.k - len(selected)
        if remaining_slots > 0:
            selected_ids = {(n['type'], n['id']) for n in selected}
            for score, neighbor in scored_neighbors:
                if (neighbor['type'], neighbor['id']) not in selected_ids:
                    selected.append(neighbor)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
        
        selected = selected[:self.k]
        
        n_items = sum(1 for n in selected if n['type'] == 'item')
        n_users = sum(1 for n in selected if n['type'] == 'user')
        
        return {
            'user_id': user_id,
            'neighbors': selected,
            'n_items': n_items,
            'n_users': n_users
        }
