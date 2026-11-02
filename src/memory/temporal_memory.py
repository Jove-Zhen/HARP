"""
Temporal Memory Module with Ebbinghaus Forgetting Curve
Implements exponential decay weighting to distinguish long-term vs recent memory
"""
import numpy as np
from typing import List, Dict, Tuple


class TemporalMemory:
    """Temporal memory weighting based on Ebbinghaus forgetting curve"""

    def __init__(self, lambda_param: float = 0.1):
        """
        Initialize temporal memory module

        Args:
            lambda_param: Decay rate for exponential function (higher = faster decay)
                         Default 0.1 provides reasonable decay over ~20 interactions
        """
        self.lambda_param = lambda_param

        self.RECENT_WINDOW = 5
        self.MEDIUM_WINDOW = 20

        self.RECENT_WEIGHT = 1.5
        self.MEDIUM_WEIGHT = 1.0
        self.LONGTERM_WEIGHT = 0.7

    def compute_temporal_weight(
        self,
        interaction_index: int,
        total_interactions: int,
        use_exponential: bool = True
    ) -> float:
        """
        Compute temporal weight for a single interaction

        Args:
            interaction_index: Position in interaction history (0 = earliest, n-1 = most recent)
            total_interactions: Total number of interactions
            use_exponential: If True, use exponential decay; if False, use window-based weights

        Returns:
            Temporal weight (higher = more recent/important)
        """
        if total_interactions == 0:
            return 1.0

        # Calculate recency: how far back this interaction is (0 = oldest, 1 = newest)
        recency = (interaction_index + 1) / total_interactions

        if use_exponential:
            # time_distance = how far back from present (0 = most recent, 1 = oldest)
            time_distance = 1.0 - recency
            weight = np.exp(-self.lambda_param * time_distance * total_interactions)

            weight = 0.5 + weight
            return float(weight)
        else:
            position_from_end = total_interactions - interaction_index

            if position_from_end <= self.RECENT_WINDOW:
                return self.RECENT_WEIGHT
            elif position_from_end <= self.MEDIUM_WINDOW:
                return self.MEDIUM_WEIGHT
            else:
                return self.LONGTERM_WEIGHT

    def apply_temporal_decay(
        self,
        memories: List[Dict],
        timestamps: List[int] = None,
        use_exponential: bool = True
    ) -> List[Dict]:
        """
        Apply temporal decay to a list of memories

        Args:
            memories: List of memory dicts (must have 'index' or position info)
            timestamps: Optional list of timestamps (if None, use list position)
            use_exponential: If True, use exponential decay; if False, use window-based

        Returns:
            List of memories with added 'temporal_weight' field
        """
        if not memories:
            return []

        total = len(memories)
        weighted_memories = []

        for i, memory in enumerate(memories):
            index = memory.get('index', i)

            weight = self.compute_temporal_weight(index, total, use_exponential)

            weighted_memory = memory.copy()
            weighted_memory['temporal_weight'] = weight

            # If memory has a confidence score, multiply by temporal weight
            if 'confidence' in weighted_memory:
                weighted_memory['weighted_confidence'] = weighted_memory['confidence'] * weight

            weighted_memories.append(weighted_memory)

        return weighted_memories

    def get_temporal_weighted_items(
        self,
        item_list: List[int],
        use_exponential: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get items with temporal weights

        Args:
            item_list: List of item IDs in chronological order (earliest to latest)
            use_exponential: If True, use exponential decay; if False, use window-based

        Returns:
            List of (item_id, temporal_weight) tuples
        """
        if not item_list:
            return []

        total = len(item_list)
        weighted_items = []

        for i, item_id in enumerate(item_list):
            weight = self.compute_temporal_weight(i, total, use_exponential)
            weighted_items.append((item_id, weight))

        return weighted_items

    def set_lambda(self, lambda_param: float):
        """Update decay rate parameter"""
        self.lambda_param = lambda_param

    def __repr__(self):
        return f"TemporalMemory(lambda={self.lambda_param}, windows=[{self.RECENT_WINDOW}, {self.MEDIUM_WINDOW}])"
