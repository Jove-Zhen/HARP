"""
LLM-based Reranker for Stage-ReRank
Listwise ranking: LLM directly sorts items from most to least relevant.
Synthetic scores from rank position ensure downstream compatibility.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List


class LLMReranker:
    """Use LLM to rerank candidate items — listwise ranking, single LLM call"""

    def __init__(self, llm_client, prompt_registry=None):
        self.llm = llm_client
        self.prompt_registry = prompt_registry
        self._debug_enabled = os.getenv("HARP_DEBUG_PROMPTS", "") == "1"
        self._debug_dir = os.getenv("HARP_DEBUG_DIR", "")
        try:
            self._debug_user_limit = int(os.getenv("HARP_DEBUG_PROMPTS_N", "0"))
        except Exception:
            self._debug_user_limit = 0
        self._debug_seen_users = set()
        self._instruction_priority = os.getenv("HARP_INSTRUCTION_PRIORITY", "0") == "1"

    def _maybe_trace_prompt(self, kind: str, user_id: int, prompt: str, meta: Dict) -> None:
        if not self._debug_enabled or not self._debug_dir or self._debug_user_limit <= 0:
            return
        if user_id not in self._debug_seen_users and len(self._debug_seen_users) >= self._debug_user_limit:
            return
        self._debug_seen_users.add(user_id)

        out_dir = Path(self._debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{kind}.jsonl"
        record = {
            "ts": time.time(),
            "user_id": user_id,
            "kind": kind,
            "prompt_chars": len(prompt),
            "prompt_tokens_approx": self._approx_tokens(prompt),
            **(meta or {}),
            "prompt": prompt
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_item_desc(self, c: Dict, item_mems: Dict = None) -> str:
        """Build a concise item description for the prompt."""
        cid = c['id']
        title = c.get('title', f'Item {cid}')

        attr_parts = []
        if c.get('stars'):
            attr_parts.append(f"★{c['stars']}")
        if c.get('review_count'):
            attr_parts.append(f"{c['review_count']} reviews")
        if c.get('price'):
            attr_parts.append(c['price'])
        if c.get('categories'):
            attr_parts.append(c['categories'])
        if c.get('city'):
            attr_parts.append(c['city'])
        attr_str = f" [{', '.join(attr_parts)}]" if attr_parts else ""

        memory = ""
        if item_mems and cid in item_mems:
            memory = item_mems[cid]
            if len(memory) > 250:
                memory = memory[:250] + "..."
        else:
            if attr_parts:
                memory = f"{title}: {', '.join(attr_parts)}"
            else:
                memory = f"{title}"
        return f"Item {cid} ({title}){attr_str}: {memory}"

    def build_rerank_prompt(
        self,
        user_id: int,
        facets: List[Dict],
        candidates: List[Dict],
        item_mems: Dict[int, Dict] = None,
        instruction: str = None,
        vanilla_mode: bool = False,
        user_profile: str = None
    ) -> List[Dict[str, str]]:
        """
        Build listwise ranking prompt — LLM sorts items, no scoring.
        """
        if vanilla_mode:
            prompt_parts = [
                f"You are an intelligent recommendation ranking system. Your task is to rank candidate items from most to least relevant for the target user.",
                f"\n**Target User:** User {user_id}"
            ]

            if instruction:
                prompt_parts.append(f"\n**User Profile:**\n{instruction}")
            else:
                prompt_parts.append(f"\n**User Profile:**\nNo specific user profile provided.")

            prompt_parts.append("\n**Candidate Items:**")
            for c in candidates:
                cid = c['id']
                title = c.get('title', f'Item {cid}')
                description = c.get('description', '')
                tags = c.get('tags', [])
                if description:
                    if len(description) > 200:
                        description = description[:200] + "..."
                    prompt_parts.append(f"  • Item {cid}: {title}. {description}")
                elif tags:
                    tags_str = ", ".join(tags[:5])
                    prompt_parts.append(f"  • Item {cid}: {title} (Tags: {tags_str})")
                else:
                    prompt_parts.append(f"  • Item {cid}: {title}")

        else:
            prompt_parts = [
                f"You are an intelligent recommendation ranking system. Your task is to rank candidate items from most to least relevant for the target user based on their personal memory and collaborative signals.",
                f"\n**Target User:** User {user_id}"
            ]

            if instruction:
                if self._instruction_priority:
                    prompt_parts.append(f"\n**User's Current Request (PRIMARY ranking criterion):**\n{instruction}")
                    prompt_parts.append("\n⚠️ **CRITICAL**: The user's current request describes EXACTLY what they want. Items matching the requested type/category MUST be ranked above items that only match general preference patterns.")
                else:
                    prompt_parts.append(f"\n**User's Current Request:**\n{instruction}")

            if user_profile:
                prompt_parts.append(f"\n**Ranking Guidelines (MUST follow):**\n{user_profile}")

            prompt_parts.append("\n**User Preferences (Extracted from Collaborative Memories):**")
            if self._instruction_priority:
                prompt_parts.append("These are SECONDARY signals. Use them to differentiate items that equally match the user's current request:")
            else:
                prompt_parts.append("Based on collaborative signals from neighboring users and items, we have identified the following preference patterns:")
            if facets:
                for i, f in enumerate(facets[:10], 1):
                    facet_text = f.get('facet', f.get('text', 'N/A'))
                    conf = f.get('confidence', 0)
                    prompt_parts.append(f"  {i}. {facet_text} (confidence: {conf:.2f})")
            else:
                prompt_parts.append("  (No facets extracted)")

            prompt_parts.append("\n**Candidate Items:**")
            for c in candidates:
                prompt_parts.append(f"  • {self._build_item_desc(c, item_mems)}")

        # Listwise ranking task — same for both vanilla and non-vanilla
        prompt_parts.append("""
**Your Task:**
Rank ALL candidate items from MOST relevant to LEAST relevant for this user.

Ranking criteria (in priority order):
1. How well the item matches the user's specific request/profile
2. Item quality (stars, reviews)
3. Alignment with user's preference patterns

You MUST include every candidate item exactly once. The first item in your ranking is the best recommendation.

**Expected Output Format:**
JSON object with a "ranking" array, ordered from best to worst match. Each element:
  - "item_id": The item's ID (integer)
  - "rationale": Brief explanation of why this item is ranked at this position (string)
""")

        prompt = "".join(prompt_parts)
        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages

    def get_rerank_schema(self) -> Dict:
        """Get listwise ranking JSON schema"""
        return {
            "ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "integer"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["item_id", "rationale"],
                    "additionalProperties": False
                }
            }
        }

    def rerank(
        self,
        user_id: int,
        retrieval_bundle: Dict,
        candidates: List[Dict],
        item_mems: Dict[int, Dict] = None,
        instruction: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger=None,
        vanilla_mode: bool = False,
        user_profile: str = None,
        **kwargs
    ) -> List[Dict]:
        """
        Rerank candidate items — listwise ranking with synthetic scores.

        Returns:
            [{'item_id': int, 'score': float, 'rationale': str}, ...]
        """
        messages = self.build_rerank_prompt(
            user_id=user_id,
            facets=retrieval_bundle.get('facets', []) if not vanilla_mode else [],
            candidates=candidates,
            item_mems=item_mems if not vanilla_mode else {},
            instruction=instruction,
            vanilla_mode=vanilla_mode,
            user_profile=user_profile
        )

        try:
            prompt_text = messages[0].get("content", "")
        except Exception:
            prompt_text = ""
        self._maybe_trace_prompt(
            kind="stage_rerank",
            user_id=user_id,
            prompt=prompt_text,
            meta={
                "facets_count": len(retrieval_bundle.get("facets", []) or []) if not vanilla_mode else 0,
                "candidates_count": len(candidates or [])
            }
        )

        properties = self.get_rerank_schema()

        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )

            ranking = response.get('ranking', [])
            N = max(len(ranking), 1)

            scores = []
            seen_ids = set()
            for i, entry in enumerate(ranking):
                item_id = entry.get('item_id')
                rationale = entry.get('rationale', '')
                synthetic_score = (N - i) / N
                scores.append({
                    'item_id': item_id,
                    'score': synthetic_score,
                    'rationale': rationale
                })
                seen_ids.add(item_id)

            # Safety: if LLM missed some candidates, append them with low scores
            for c in candidates:
                if c['id'] not in seen_ids:
                    scores.append({
                        'item_id': c['id'],
                        'score': 0.0,
                        'rationale': 'Not ranked by LLM'
                    })

            if debug_logger and scores:
                debug_logger(f"\n📊 Listwise Ranking (Stage-ReRank):")
                for i, score_entry in enumerate(scores[:10], 1):
                    item_id = score_entry.get('item_id')
                    score = score_entry.get('score', 0)
                    rationale = score_entry.get('rationale', 'N/A')[:80]
                    debug_logger(f"  [{i}] Item {item_id}: {score:.3f} - {rationale}")
                debug_logger("")

            return scores
        except Exception as e:
            print(f"Error in LLM Reranker for user {user_id}: {e}")
            return [{'item_id': c['id'], 'score': 0.5, 'rationale': 'Error'} for c in candidates]
