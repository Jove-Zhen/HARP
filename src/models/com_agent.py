"""
ComMemRecAgent: MemRecAgent with template-driven Stage-R / Stage-W prompts (ComMemRecManager)
and an LLM reranker. rerank, write and the candidate helpers are inherited.
"""
from typing import List, Dict, Tuple
from src.models.memrec_agent import MemRecAgent
from src.memory.com_manager import ComMemRecManager
from src.prompts.prompt_registry import PromptRegistry
from src.models.reranker_llm import LLMReranker


class ComMemRecAgent(MemRecAgent):
    """
    Communication agent: user/item profiles, homophily-gated neighbour selection, profile update.

    Holds the PromptRegistry and swaps in ComMemRecManager and the LLM reranker.
    """

    def __init__(
        self,
        dataset,
        llm_client,
        prompt_registry: PromptRegistry,
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
        reranker_llm_client=None,
        enable_temporal_weighting: bool = False,
        debug: bool = False
    ):
        super().__init__(
            dataset=dataset, llm_client=llm_client,
            k=k, tau=tau, n_facets=n_facets,
            temperature=temperature, max_tokens=max_tokens,
            mix_min_users=mix_min_users, mix_min_items=mix_min_items,
            fanout_cap=fanout_cap, reranker_mode=reranker_mode,
            pruner_mode=pruner_mode,
            enable_stage_r=enable_stage_r,
            reranker_llm_client=reranker_llm_client,
            enable_temporal_weighting=enable_temporal_weighting,
            debug=debug
        )

        self.prompt_registry = prompt_registry
        self.manager = ComMemRecManager(llm_client, prompt_registry)
        self.manager._storage = self.storage

        if reranker_mode == "llm":
            reranker_client = reranker_llm_client if reranker_llm_client is not None else llm_client
            self.reranker = LLMReranker(reranker_client)
            print("  ComMemRecAgent: LLM reranker")

        print(f"  ComMemRecAgent: prompts from {prompt_registry.prompts_dir}")

    def rerank(
        self,
        user_id: int,
        candidates: List[int] = None,
        instruction: str = None,
        return_details: bool = False,
        debug_logger=None,
        recent_interactions: str = None
    ) -> Tuple[List[int], Dict]:
        """Rerank the candidate items for the user."""
        ranked_items, details = super().rerank(
            user_id=user_id,
            candidates=candidates,
            instruction=instruction,
            recent_interactions=recent_interactions,
            return_details=True,
            debug_logger=debug_logger
        )
        return (ranked_items, details) if return_details else (ranked_items, {})
