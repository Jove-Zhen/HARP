"""
MovieTV Domain Pruning Rules
Sparse graph with recency-critical volatile preferences
"""
import numpy as np
from typing import Dict


class MovieTVRules:
    """
    InstructRec-MovieTV pruning rules
    
    Key characteristics:
    - Sparse graph (14.1 items/user, 2.8 users/item)
    - Recency matters (trending content, mood-based viewing)
    - Volatile preferences (tastes change with trends)
    """
    
    @staticmethod
    def apply_rules(features: Dict[str, float]) -> float:
        """
        Apply MovieTV-specific pruning rules
        
        Args:
            features: Dict with keys:
                - edge_weight: Base CF score
                - metadata_overlap_score: Content similarity (0-1)
                - co_interaction_count: Number of co-rated movies/shows
                - memory_similarity_score: Memory similarity (0-1)
                - recency_days: Days since last interaction
                - neighbor_type: 'user' or 'item'
        
        Returns:
            final_score: Weighted score for neighbor selection
        """
        edge_weight = features.get('edge_weight', 0.0)
        metadata_overlap = features.get('metadata_overlap_score', 0.0)
        co_count = features.get('co_interaction_count', 0)
        memory_sim = features.get('memory_similarity_score', 0.0)
        recency_days = features.get('recency_days', 0)
        
        if recency_days <= 60:
            recency_decay = 1.0
        elif recency_days <= 180:
            recency_decay = np.exp(-0.018 * recency_days)
        else:
            recency_decay = np.exp(-0.025 * recency_days)
        
        metadata_boost = 1.0
        if co_count < 3:
            metadata_boost = 2.8 * metadata_overlap
        
        cf_boost = 1.0
        memory_boost = 1.0 + 1.5 * memory_sim
        if co_count >= 3:
            cf_boost = 2.5
            memory_boost = 1.0 + 1.8 * memory_sim
        
        if metadata_overlap > 0.6:
            memory_boost *= 1.5
        
        recency_penalty = 1.0
        if recency_days > 365:
            recency_penalty = 0.3
        
        final_score = (edge_weight * cf_boost * recency_decay * recency_penalty + 
                      metadata_boost) * memory_boost
        
        return final_score
    
    @staticmethod
    def get_description() -> str:
        return "MovieTV: Recency-heavy (e^-0.018), metadata compensation (2.8×), CF threshold >= 3"

