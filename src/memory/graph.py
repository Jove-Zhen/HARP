"""
User-Item Graph Builder
Build lightweight collaborative graph from training set, supporting neighbor queries and temporal decay
"""
from collections import defaultdict
from typing import Dict, List, Tuple
from .temporal_memory import TemporalMemory


class UserItemGraph:
    """User-item bipartite graph"""
    
    def __init__(self, dataset):
        """
        Build graph from dataset

        Args:
            dataset: RecDataset instance containing train_data
        """
        self.dataset = dataset

        self.items_by_user = defaultdict(list)
        self.users_by_item = defaultdict(list)
        self.recency = defaultdict(dict)

        self.temporal_memory = TemporalMemory(lambda_param=0.1)

        self._build_from_train()

        self.user_degrees = {u: len(items) for u, items in self.items_by_user.items()}
        self.item_degrees = {i: len(users) for i, users in self.users_by_item.items()}
        
    def _build_from_train(self):
        """Build graph from training data"""
        for user_id, item_list in self.dataset.train_data.items():
            n_items = len(item_list)
            for pos, item_id in enumerate(item_list):
                self.items_by_user[user_id].append(item_id)
                self.users_by_item[item_id].append(user_id)
                
                recency_score = (pos + 1) / n_items if n_items > 0 else 1.0
                self.recency[user_id][item_id] = recency_score
    
    def get_user_items(self, user_id: int) -> List[int]:
        """Get items the user has interacted with"""
        return self.items_by_user.get(user_id, [])
    
    def get_item_users(self, item_id: int) -> List[int]:
        """Get users who have interacted with the item"""
        return self.users_by_item.get(item_id, [])
    
    def get_item_recency(self, user_id: int, item_id: int) -> float:
        """Get temporal decay score for user-item interaction"""
        return self.recency.get(user_id, {}).get(item_id, 0.0)
    
    def get_user_degree(self, user_id: int) -> int:
        """Get user degree (number of interacted items)"""
        return self.user_degrees.get(user_id, 0)
    
    def get_user_neighbors(
        self,
        user_id: int,
        max_items: int = 20,
        max_users: int = 10,
        temporal_decay_lambda: float = None,
        use_exponential: bool = False
    ) -> Dict:
        """
        Get neighborhood subgraph for user

        Args:
            user_id: Target user
            max_items: Maximum number of item neighbors to return (sorted by temporal decay)
            max_users: Maximum number of user neighbors to return (discovered through common items)
            temporal_decay_lambda: Optional lambda parameter for exponential decay (overrides default)
            use_exponential: If True, use exponential decay; if False, use linear recency

        Returns:
            {
                'item_neighbors': [(item_id, recency_score), ...],
                'user_neighbors': [(neighbor_user_id, overlap_count), ...]
            }
        """
        user_items = self.get_user_items(user_id)
        item_neighbors = []

        if use_exponential and temporal_decay_lambda is not None:
            old_lambda = self.temporal_memory.lambda_param
            self.temporal_memory.set_lambda(temporal_decay_lambda)
            weighted_items = self.temporal_memory.get_temporal_weighted_items(user_items, use_exponential=True)
            self.temporal_memory.set_lambda(old_lambda)
            item_neighbors = weighted_items
        elif use_exponential:
            weighted_items = self.temporal_memory.get_temporal_weighted_items(user_items, use_exponential=True)
            item_neighbors = weighted_items
        else:
            for item_id in user_items:
                recency = self.get_item_recency(user_id, item_id)
                item_neighbors.append((item_id, recency))

        item_neighbors.sort(key=lambda x: x[1], reverse=True)
        item_neighbors = item_neighbors[:max_items]

        user_item_set = set(user_items)
        neighbor_overlap = defaultdict(int)

        for item_id in user_items:
            co_users = self.get_item_users(item_id)
            for co_user in co_users:
                if co_user != user_id:
                    neighbor_overlap[co_user] += 1

        user_neighbors = sorted(
            neighbor_overlap.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_users]

        return {
            'item_neighbors': item_neighbors,
            'user_neighbors': user_neighbors
        }
    
    def get_temporal_weighted_items(
        self,
        user_id: int,
        lambda_param: float = None,
        use_exponential: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get user's items with temporal decay weighting

        Args:
            user_id: Target user
            lambda_param: Optional lambda parameter for exponential decay
            use_exponential: If True, use exponential decay; if False, use linear recency

        Returns:
            List of (item_id, temporal_weight) tuples sorted by weight (descending)
        """
        user_items = self.get_user_items(user_id)

        if not user_items:
            return []

        if use_exponential:
            if lambda_param is not None:
                old_lambda = self.temporal_memory.lambda_param
                self.temporal_memory.set_lambda(lambda_param)
                weighted_items = self.temporal_memory.get_temporal_weighted_items(user_items, use_exponential=True)
                self.temporal_memory.set_lambda(old_lambda)
            else:
                weighted_items = self.temporal_memory.get_temporal_weighted_items(user_items, use_exponential=True)
        else:
            weighted_items = []
            for item_id in user_items:
                recency = self.get_item_recency(user_id, item_id)
                weighted_items.append((item_id, recency))

        weighted_items.sort(key=lambda x: x[1], reverse=True)

        return weighted_items

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        n_users = len(self.items_by_user)
        n_items = len(self.users_by_item)
        n_edges = sum(len(items) for items in self.items_by_user.values())
        
        avg_user_degree = n_edges / n_users if n_users > 0 else 0
        avg_item_degree = n_edges / n_items if n_items > 0 else 0
        
        return {
            'n_users': n_users,
            'n_items': n_items,
            'n_edges': n_edges,
            'avg_user_degree': avg_user_degree,
            'avg_item_degree': avg_item_degree
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"UserItemGraph(users={stats['n_users']}, "
                f"items={stats['n_items']}, "
                f"edges={stats['n_edges']}, "
                f"avg_user_deg={stats['avg_user_degree']:.2f})")
