"""
ComMemRecManager: two-stage LM calls (Stage-R / Stage-W).
Reads its prompts from markdown template files
instead of hardcoded Python strings.

Key difference from MemRecManager:
- build_stage_r_prompt() reads from prompts/user_understanding.md
- build_stage_w_prompt() reads from prompts/memory_update.md
- JSON schemas remain IDENTICAL (backward compatible)
"""
from typing import Dict, List
import json
import os
import time
from pathlib import Path
from src.prompts.prompt_registry import PromptRegistry
from src.prompts.template_engine import TemplateEngine


_EMPTY_MEMORY = {
    "No specific memory recorded for this user yet.",
    "(No personal memory recorded yet for this user)",
}

class ComMemRecManager:
    """
    Two-stage manager (Stage-R / Stage-W) driven by markdown prompt templates.
    Drop-in replacement for MemRecManager with identical interface.
    """

    def __init__(self, llm_client, prompt_registry: PromptRegistry):
        """
        Args:
            llm_client: LLMClient instance (same as MemRecManager)
            prompt_registry: PromptRegistry with the loaded template files
        """
        self.llm = llm_client
        self.registry = prompt_registry
        self._storage = None
        self._user_profile_mgr = None
        import threading
        self._profile_cache = {}
        self._profile_lock = threading.Lock()
        self._debug_enabled = os.getenv("HARP_DEBUG_PROMPTS", "") == "1"
        self._debug_dir = os.getenv("HARP_DEBUG_DIR", "")
        try:
            self._debug_user_limit = int(os.getenv("HARP_DEBUG_PROMPTS_N", "0"))
        except Exception:
            self._debug_user_limit = 0
        self._debug_seen_users = set()

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

    def extract_reranker_profile(self, user_id: int, user_memory: str = None) -> str:
        """
        Extract a structured user profile for Reranker injection.

        Uses a single constrained LLM call to extract 3 structured fields
        from user_memory. The extraction prompt is loaded from a template file if available.

        Results are cached per user_id per session (no redundant LLM calls).
        Thread-safe with threading.Lock().

        Args:
            user_id: User ID
            user_memory: User memory text (if None, will be fetched from storage)

        Returns:
            Concise profile string (~30 tokens), or empty string.
        """
        # Thread-safe cache check
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        try:
            if not user_memory:
                return ""
            if not user_memory.strip():
                return ""

            from src import settings
            if (settings.SHORT_PROFILE_EMPTY_MODE == "empty"
                    and user_memory.strip() in _EMPTY_MEMORY):
                # An empty profile formats to "", so no profile block is added to the
                # reranker prompt. Cache even this empty result.
                result = self._format_reranker_profile({"likes": [], "dislikes": [], "style": ""})
                with self._profile_lock:
                    self._profile_cache[user_id] = result
                return result

            if user_memory == "(No personal memory recorded yet for this user)":
                return ""

            if settings.SHORT_PROFILE_TEMPLATE and user_memory.strip() in _EMPTY_MEMORY:
                # nothing to summarise yet: use the fixed template instead of an LLM call
                result = settings.SHORT_PROFILE_TEMPLATE_TEXT
                with self._profile_lock:
                    self._profile_cache[user_id] = result
                return result

            prompt = self._build_extraction_prompt(user_memory)

            schema = {
                "likes": {"type": "array", "items": {"type": "string"}, "description": "2-3 preference themes, max 5 words each"},
                "dislikes": {"type": "array", "items": {"type": "string"}, "description": "0-1 avoidance patterns"},
                "style": {"type": "string", "description": "One phrase, max 5 words"}
            }

            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                properties=schema,
                temperature=0.0,
                max_tokens=200
            )

            result = self._format_reranker_profile(response)
            with self._profile_lock:
                self._profile_cache[user_id] = result
            return result

        except Exception:
            with self._profile_lock:
                self._profile_cache[user_id] = ""
            return ""

    @staticmethod
    def _format_reranker_profile(response: Dict) -> str:
        """Format the three short-profile fields as likes / avoids / style."""
        likes = response.get("likes", [])[:3]
        dislikes = response.get("dislikes", [])[:1]
        style = response.get("style", "")
        parts = []
        if likes:
            parts.append("Likes: " + ", ".join(likes))
        if dislikes:
            parts.append("Avoids: " + ", ".join(dislikes))
        if style:
            parts.append("Style: " + style)
        return " | ".join(parts) if parts else ""

    def _build_extraction_prompt(self, user_memory: str) -> str:
        """
        Build extraction prompt. Tries to load from the template file first,
        falls back to default hardcoded prompt.

        """
        try:
            prompt_path = self.registry.prompts_dir / "profile_extraction_prompt.md"
            if prompt_path.exists():
                template = prompt_path.read_text(encoding='utf-8').strip()
                if template and "{{user_memory}}" in template:
                    return template.replace("{{user_memory}}", user_memory[:500])
        except Exception:
            pass

        return f"""Based on this user's memory, extract their core preferences in exactly this format.
Be very concise (max 5 words per item). Only include patterns clearly supported by the memory.

User Memory:
{user_memory[:500]}

Extract:
1. "likes": Top 2-3 preference themes (e.g., "Mexican food", "casual dining")
2. "dislikes": 0-1 things they clearly avoid (leave empty if unsure)
3. "style": One phrase describing their taste (e.g., "adventurous foodie", "budget-friendly casual")"""

    def build_stage_r_prompt(
        self,
        user_id: int,
        user_mem_bullets: str,
        neighbor_table_json: str,
        candidates_json: str = "",
        n_facets: int = 7,
        instruction: str = None
    ) -> List[Dict[str, str]]:
        """
        Build Stage-R prompt by rendering the user_understanding.md template.

        SAME SIGNATURE as MemRecManager.build_stage_r_prompt() (lines 25-97).

        Variables injected into user.md:
          {{user_id}} -> str(user_id)
          {{user_mem_bullets}} -> user_mem_bullets or "(No personal memory...)"
          {{neighbor_table_json}} -> neighbor_table_json
          {{candidates_json}} -> candidates_json (controls {{#if candidates_json}})
          {{n_facets}} -> str(n_facets)

        Returns:
            List of messages (same format as MemRecManager)
        """
        user_profile = self.registry.get("user_understanding")
        variables = {
            "user_id": str(user_id),
            "user_mem_bullets": user_mem_bullets if user_mem_bullets else "(No personal memory recorded yet for this user)",
            "neighbor_table_json": neighbor_table_json,
            "candidates_json": candidates_json,
            "n_facets": str(n_facets),
            "instruction": instruction or "",
        }
        rendered = TemplateEngine.render(user_profile.raw_text, variables)

        messages = [{"role": "user", "content": rendered}]
        return messages

    def get_stage_r_schema(self) -> Dict:
        """
        IDENTICAL to MemRecManager.get_stage_r_schema() (lines 99-128).
        Returns the same JSON schema for facets + support_edges.
        No changes needed -- the schema is structural, not prompt-dependent.
        """
        return {
            "facets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "facet": {"type": "string"},
                        "confidence": {"type": "number"},
                        "supporting_neighbors": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["facet", "confidence", "supporting_neighbors"],
                    "additionalProperties": False
                }
            },
            "support_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "w": {"type": "number"}
                    },
                    "required": ["from", "to", "w"],
                    "additionalProperties": False
                }
            }
        }

    def run_stage_r(
        self,
        user_id: int,
        packed_context: Dict,
        instruction: str = None,
        n_facets: int = 7,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger=None
    ) -> Dict:
        """
        Run Stage-R. SAME SIGNATURE as MemRecManager.run_stage_r() (lines 130-184).

        Only difference: build_stage_r_prompt() now reads from user.md.
        Return format is identical: {'facets': [...], 'support_edges': [...]}
        """
        messages = self.build_stage_r_prompt(
            user_id=user_id,
            user_mem_bullets=packed_context.get('memory_text', ''),
            neighbor_table_json=packed_context.get('neighbors_text', ''),
            candidates_json=packed_context.get('candidates_text', ''),
            n_facets=n_facets,
            instruction=instruction
        )
        try:
            self._maybe_trace_prompt(
                kind="stage_r",
                user_id=user_id,
                prompt=messages[0].get("content", ""),
                meta={
                    "n_facets": n_facets,
                    "memory_text_chars": len(packed_context.get("memory_text", "") or ""),
                    "neighbors_text_chars": len(packed_context.get("neighbors_text", "") or ""),
                    "candidates_text_chars": len(packed_context.get("candidates_text", "") or "")
                }
            )
        except Exception:
            pass
        properties = self.get_stage_r_schema()

        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )

            # Apply facet quality control (can be disabled via env var for ablation)
            if 'facets' in response and os.getenv("HARP_FACET_FILTER", "1") == "1":
                raw_facets = response.get('facets') or []
                filtered = self.filter_high_quality_facets(raw_facets)
                if not filtered and raw_facets:
                    raw_facets_sorted = sorted(raw_facets, key=lambda x: x.get('confidence', 0.0), reverse=True)
                    filtered = raw_facets_sorted[: min(5, len(raw_facets_sorted))]
                response['facets'] = filtered

            return response
        except Exception as e:
            print(f"Error in Stage-R for user {user_id}: {e}")
            return {"facets": [], "support_edges": []}

    def filter_high_quality_facets(self, facets: List[Dict], min_confidence: float = 0.4, min_support: int = 2) -> List[Dict]:
        """
        Filter facets by quality criteria:
        1. Confidence threshold (default: 0.4)
        2. Minimum supporting neighbors (default: 2)
        3. Deduplication by semantic similarity

        Args:
            facets: List of facet dicts with 'facet', 'confidence', 'supporting_neighbors'
            min_confidence: Minimum confidence score
            min_support: Minimum number of supporting neighbors

        Returns:
            Filtered and sorted facets (by confidence descending)
        """
        if not facets:
            return []

        high_quality = []
        for f in facets:
            conf = f.get('confidence', 0.0)
            support = len(f.get('supporting_neighbors', []))

            if conf >= min_confidence and support >= min_support:
                high_quality.append(f)

        # More sophisticated: use embedding similarity, but that requires extra dependencies
        seen_texts = set()
        deduplicated = []
        for f in high_quality:
            facet_text = f.get('facet', '').strip().lower()
            if facet_text and facet_text not in seen_texts:
                seen_texts.add(facet_text)
                deduplicated.append(f)

        deduplicated.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

        return deduplicated

    def build_stage_w_prompt(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        user_mem_keys: List[str],
        item_mem_keys: Dict[int, List[str]],
        neighbor_ids: List[str],
        neighbor_details: List[Dict] = None,
        fanout_cap: int = 6,
        clicked_item_info: str = None
    ) -> List[Dict[str, str]]:
        """
        Build Stage-W prompt by rendering the memory_update.md template.

        SAME SIGNATURE as MemRecManager.build_stage_w_prompt() (lines 190-360).

        Variable mapping to discovery.md placeholders:
          {{user_id}} -> str(user_id)
          {{item_id}} -> str(feedback.get('item_id', 'unknown'))
          {{clicked_item_info}} -> clicked_item_info or "Unknown Item"
          {{facets_text}} -> formatted recent_facets
          {{current_user_memory}} -> user_mem_keys[0] or "(No memory...)"
          {{current_item_memory}} -> item_mem_keys first value or "(No memory...)"
          {{neighbor_details_text}} -> formatted neighbor list
        """
        current_user_memory = user_mem_keys[0] if user_mem_keys else "(No memory recorded yet)"
        if len(current_user_memory) > 300:
            current_user_memory = current_user_memory[:300] + "..."

        current_item_memory = "(No memory recorded yet)"
        if item_mem_keys:
            for item_id, mem_list in item_mem_keys.items():
                if mem_list:
                    current_item_memory = mem_list[0]
                    if len(current_item_memory) > 300:
                        current_item_memory = current_item_memory[:300] + "..."
                    break

        facets_lines = []
        if recent_facets:
            for i, f in enumerate(recent_facets[:5], 1):
                facet_text = f.get('facet', f.get('text', 'N/A'))
                conf = f.get('confidence', 1.0)
                facets_lines.append(f"  {i}. {facet_text} (confidence: {conf:.2f})")
        else:
            facets_lines.append("  (No preference patterns extracted)")
        facets_text = "\n".join(facets_lines)

        # Format neighbor details (same logic as original lines 262-284)
        neighbor_lines = []
        if neighbor_ids and len(neighbor_ids) > 0:
            n_neighbors = min(len(neighbor_ids), fanout_cap)
            neighbor_lines.append(f"The following {n_neighbors} neighbors are available:")
            for i, nb_id in enumerate(neighbor_ids[:fanout_cap], 1):
                nb_detail = None
                if neighbor_details:
                    nb_detail = next((d for d in neighbor_details if d.get('id_str') == nb_id), None)
                if nb_detail:
                    score = nb_detail.get('score', 0.0)
                    memory_preview = nb_detail.get('memory', 'N/A')
                    if memory_preview and len(memory_preview) > 150:
                        memory_preview = memory_preview[:150] + "..."
                    neighbor_lines.append(f"  {i}. {nb_id} (strength: {score:.3f})")
                    neighbor_lines.append(f"     Current memory: {memory_preview or '(empty)'}")
                else:
                    neighbor_lines.append(f"  {i}. {nb_id}")
        else:
            neighbor_lines.append("  (No neighbors available)")
        neighbor_details_text = "\n".join(neighbor_lines)

        memory_update_template = self.registry.get("memory_update")
        variables = {
            "user_id": str(user_id),
            "item_id": str(feedback.get('item_id', 'unknown')),
            "clicked_item_info": clicked_item_info or "Unknown Item",
            "facets_text": facets_text,
            "current_user_memory": current_user_memory,
            "current_item_memory": current_item_memory,
            "neighbor_details_text": neighbor_details_text,
        }

        import os
        if os.getenv("HARP_HOMOPHILY", "0") == "1":
            variables["profile_update_enabled"] = "true"
            user_profile_text = "(No user skill profile yet)"
            item_profile_text = "(No item skill profile yet)"

            if hasattr(self, '_storage') and self._storage:
                item_id = feedback.get('item_id')
                if item_id:
                    item_profile = self._storage.get_item_profile(item_id)
                    if item_profile:
                        item_profile_text = str(item_profile)

            if hasattr(self, '_user_profile_mgr') and self._user_profile_mgr:
                user_prof = self._user_profile_mgr.get_user_profile(user_id)
                if user_prof and user_prof.profile:
                    import json as _json
                    user_profile_text = _json.dumps(user_prof.profile, indent=2)
                elif user_prof and user_prof.rules:
                    user_profile_text = "\n".join(user_prof.rules)

            variables["current_user_profile"] = user_profile_text
            variables["current_item_profile"] = item_profile_text
        else:
            variables["profile_update_enabled"] = ""

        rendered = TemplateEngine.render(memory_update_template.raw_text, variables)

        messages = [{"role": "user", "content": rendered}]
        return messages

    def get_stage_w_schema(self) -> Dict:
        """Stage-W schema, plus optional profile-update fields."""
        schema = {
            "user_memory": {"type": "string", "description": "Updated user memory"},
            "item_memory": {"type": "string", "description": "Updated item memory"},
            "neighbor_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "neighbor_id": {"type": "string"},
                        "memory_update": {"type": "string"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["neighbor_id", "memory_update", "rationale"],
                    "additionalProperties": False
                }
            }
        }

        import os
        if os.getenv("HARP_HOMOPHILY", "0") == "1":
            schema["user_skill_update"] = {
                "type": "object",
                "properties": {
                    "should_update": {"type": "boolean", "description": "Whether to update user skill"},
                    "primary_interests_delta": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New interests to add"
                    },
                    "reason": {"type": "string", "description": "Why update is needed"}
                },
                "required": ["should_update", "primary_interests_delta", "reason"],
                "additionalProperties": False
            }
            schema["item_skill_update"] = {
                "type": "object",
                "properties": {
                    "should_update": {"type": "boolean", "description": "Whether to update item skill"},
                    "typical_fans_delta": {"type": "string", "description": "New audience type"},
                    "reason": {"type": "string", "description": "Why update is needed"}
                },
                "required": ["should_update", "typical_fans_delta", "reason"],
                "additionalProperties": False
            }

        return schema

    def run_stage_w(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        user_mem_keys: List[str],
        item_mem_keys: Dict[int, List[str]],
        neighbor_ids: List[str],
        neighbor_details: List[Dict] = None,
        fanout_cap: int = 6,
        clicked_item_info: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger=None
    ) -> Dict:
        """
        Run Stage-W. SAME SIGNATURE as MemRecManager.run_stage_w() (lines 398-474).
        Return format identical: {'user_memory': str, 'item_memory': str, 'neighbor_updates': [...]}
        """
        messages = self.build_stage_w_prompt(
            user_id=user_id, feedback=feedback,
            recent_facets=recent_facets, user_mem_keys=user_mem_keys,
            item_mem_keys=item_mem_keys, neighbor_ids=neighbor_ids,
            neighbor_details=neighbor_details, fanout_cap=fanout_cap,
            clicked_item_info=clicked_item_info
        )
        properties = self.get_stage_w_schema()

        try:
            response = self.llm.generate_json(
                messages=messages, properties=properties,
                temperature=temperature, max_tokens=max_tokens,
                debug_logger=debug_logger
            )
            return response
        except Exception as e:
            print(f"Error in Stage-W for user {user_id}: {e}")
            return {"user_memory": "", "item_memory": "", "neighbor_updates": []}
