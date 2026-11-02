"""
UserProfileManager: per-user structured profiles, persisted on disk and updated by reflection.

  - every user has an independent profile (JSON file on disk)
  - the profile is built from the user's interaction history
  - after warmup it is reflected on from the user's own hit/miss cases (optionally gated by a critic)
  - the (frozen) profile is injected into the reranker prompt and used by the homophily gates

Lifecycle:
  1. Build   - profile from the history bands
  2. Warmup  - rerank under the profile, collect each user's hit/miss cases
  3. Sync    - reflect on the cases, accept or reject the rewrite
  4. Eval    - use the frozen profile
  5. Persist - saved to disk, reusable across runs
"""

import json
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('UserProfile')


class UserProfile:

    PROFILE_FIELDS = [
        'primary_interests', 'quality_bar', 'behavior_pattern',
        'secondary_interests', 'unlikely_interests', 'city_preference', 'summary'
    ]

    def __init__(self, user_id: int, version: int = 0, rules: List[str] = None,
                 confidence: float = 0.0, profile: Dict = None):
        self.user_id = user_id
        self.version = version
        self.rules = rules or []
        self.confidence = confidence
        self.profile = profile or {}
        self.history: List[Dict] = []
        self.created_at = time.time()
        self.updated_at = time.time()

    def format_for_prompt(self) -> str:
        if self.profile:
            return self._format_profile_prompt()
        if not self.rules:
            return ""
        lines = [f"**Personal Decision Rules (v{self.version}, confidence={self.confidence:.0%}):**"]
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"  {i}. {rule}")
        return "\n".join(lines)

    def _format_profile_prompt(self) -> str:
        p = self.profile
        lines = [f"**User Profile (v{self.version}):**"]
        if p.get('primary_interests'):
            interests = p['primary_interests']
            if isinstance(interests, list):
                interests = ', '.join(interests)
            lines.append(f"  Core interests: {interests}")
        if p.get('city_preference'):
            lines.append(f"  Location: {p['city_preference']}")
        if p.get('quality_bar'):
            lines.append(f"  Quality: {p['quality_bar']}")
        if p.get('behavior_pattern'):
            lines.append(f"  Pattern: {p['behavior_pattern']}")
        return "\n".join(lines)

    def get_primary_interests_set(self) -> set:
        if not self.profile or not self.profile.get('primary_interests'):
            if self.rules:
                import re
                words = set()
                for rule in self.rules:
                    words.update(w.lower() for w in re.findall(r'[A-Za-z]+', rule) if len(w) > 2)
                return words
            return set()

        interests = self.profile['primary_interests']
        if isinstance(interests, str):
            interests = [s.strip() for s in interests.split(',')]
        phrases = set()
        for interest in interests:
            normed = interest.strip().lower()
            if normed and len(normed) > 2:
                phrases.add(normed)
        return phrases

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "version": self.version,
            "rules": self.rules,
            "profile": self.profile,
            "confidence": self.confidence,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        user_prof = cls(
            user_id=data["user_id"],
            version=data.get("version", 0),
            rules=data.get("rules", []),
            confidence=data.get("confidence", 0.0),
            profile=data.get("profile", {}),
        )
        user_prof.history = data.get("history", [])
        user_prof.created_at = data.get("created_at", time.time())
        user_prof.updated_at = data.get("updated_at", time.time())
        return user_prof


class UserCaseRecord:

    def __init__(self, user_id: int, ground_truth: int, ranked_items: List[int],
                 gt_rank: int, hit: bool, candidates_meta: Dict = None):
        self.user_id = user_id
        self.ground_truth = ground_truth
        self.ranked_items = ranked_items
        self.gt_rank = gt_rank
        self.hit = hit
        self.candidates_meta = candidates_meta or {}

    def to_summary(self) -> str:
        gt_meta = self.candidates_meta.get(self.ground_truth, {})
        top1 = self.ranked_items[0] if self.ranked_items else None
        top1_meta = self.candidates_meta.get(top1, {}) if top1 else {}

        if self.hit:
            return (f"✓ Correctly ranked '{gt_meta.get('title', '?')}' "
                    f"[{gt_meta.get('categories', '?')}, ★{gt_meta.get('stars', '?')}] as #1")
        else:
            return (f"✗ Expected '{gt_meta.get('title', '?')}' "
                    f"[{gt_meta.get('categories', '?')}, ★{gt_meta.get('stars', '?')}] "
                    f"but got '{top1_meta.get('title', '?')}' "
                    f"[{top1_meta.get('categories', '?')}, ★{top1_meta.get('stars', '?')}] "
                    f"(GT at rank {self.gt_rank})")


class UserProfileManager:

    def __init__(
        self,
        llm_client,
        persist_dir: str,
        max_rules_per_user: int = 3,
        min_cases_to_reflect: int = 2,
        trace_dir: str = None,
        dataset=None,
    ):
        self.llm = llm_client
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.max_rules = max_rules_per_user
        self.min_cases = min_cases_to_reflect
        self.dataset = dataset

        self._profiles: Dict[int, UserProfile] = {}
        self._cases: Dict[int, List[UserCaseRecord]] = defaultdict(list)
        self._lock = threading.Lock()

        self._trace_dir = None
        if trace_dir:
            self._trace_dir = Path(trace_dir)
            self._trace_dir.mkdir(parents=True, exist_ok=True)

        self.n_reflections = 0
        self.n_users_with_profiles = 0

        self._load_all_from_disk()

    def _load_all_from_disk(self):
        loaded = 0
        for prompt_file in self.persist_dir.glob("user_*.json"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                user_prof = UserProfile.from_dict(data)
                self._profiles[user_prof.user_id] = user_prof
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load profile from {prompt_file}: {e}")

        if loaded > 0:
            logger.info(f"  📂 Loaded {loaded} user profiles from {self.persist_dir}")
        self.n_users_with_profiles = len(self._profiles)

    def _build_user_history_context(self, user_id: int) -> str:
        if not self.dataset or not hasattr(self.dataset, 'train_data'):
            return ""

        items = self.dataset.train_data.get(user_id, [])
        if not items:
            return ""

        meta = getattr(self.dataset, 'item_metadata', {}) or {}
        reviews = getattr(self.dataset, 'reviews', {}) or {}
        user_reviews = reviews.get(user_id, {})

        def _item_info(iid):
            m = meta.get(iid, {})
            title = m.get('title', f'Item-{iid}')[:60]
            desc = m.get('description', '')
            cats = ''
            stars = ''
            city = ''
            for seg in desc.strip('"').split(' | '):
                if ':' not in seg:
                    continue
                k, v = seg.split(':', 1)
                k = k.strip().lower()
                if k == 'categories':
                    cats = v.strip()
                elif k == 'stars':
                    stars = v.strip()
                elif k == 'city':
                    city = v.strip()
            return title, cats, stars, city

        def _review_first_sentence(iid):
            review = user_reviews.get(iid, '')
            if not review:
                return ''
            import re
            sentences = re.split(r'[.!?]', review)
            first = sentences[0].strip() if sentences else ''
            if len(first) > 80:
                first = first[:77] + '...'
            return first

        lines = []

        recent = items[-5:] if len(items) >= 5 else items
        recent_descs = []
        for iid in reversed(recent):
            title, cats, stars, city = _item_info(iid)
            parts = [title]
            if cats:
                parts.append(f"[{cats}]")
            if stars:
                parts.append(f"★{stars}")
            if city:
                parts.append(f"@{city}")
            review_sent = _review_first_sentence(iid)
            if review_sent:
                parts.append(f'— "{review_sent}"')
            recent_descs.append(" ".join(parts))
        lines.append("Recent activity (most recent first):")
        for d in recent_descs:
            lines.append(f"  - {d}")

        if len(items) > 5:
            mid_range = items[max(0, len(items)-15):len(items)-5]
            if mid_range:
                from collections import Counter
                cat_counter = Counter()
                city_counter = Counter()
                for iid in mid_range:
                    _, cats, _, city = _item_info(iid)
                    if cats:
                        for c in cats.split(','):
                            c = c.strip()
                            if c and c not in ('Restaurants',):
                                cat_counter[c] += 1
                    if city:
                        city_counter[city] += 1
                if cat_counter:
                    top_cats = cat_counter.most_common(5)
                    cat_str = ", ".join(f"{c}({n})" for c, n in top_cats)
                    lines.append(f"Stable preferences (category distribution): {cat_str}")
                if city_counter:
                    top_cities = city_counter.most_common(3)
                    city_str = ", ".join(f"{c}({n})" for c, n in top_cities)
                    lines.append(f"Active cities: {city_str}")

        if len(items) > 15:
            early = items[:len(items)-15]
            early_domains = set()
            for iid in early:
                _, cats, _, _ = _item_info(iid)
                if cats:
                    for c in cats.split(','):
                        c = c.strip()
                        if c and c not in ('Restaurants',):
                            early_domains.add(c)
            if early_domains:
                lines.append(f"Historical domains: {', '.join(sorted(early_domains)[:8])}")

        return "\n".join(lines)

    def get_user_profile_prompt(self, user_id: int) -> str:
        user_prof = self._profiles.get(user_id)
        if user_prof and (user_prof.rules or user_prof.profile):
            return user_prof.format_for_prompt()
        return ""

    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        return self._profiles.get(user_id)

    def compute_user_user_homophily(self, user_id_a: int, user_id_b: int) -> float:
        profile_a = self._profiles.get(user_id_a)
        profile_b = self._profiles.get(user_id_b)
        if not profile_a or not profile_b:
            return 0.5

        phrases_a = profile_a.get_primary_interests_set()
        phrases_b = profile_b.get_primary_interests_set()

        if not phrases_a or not phrases_b:
            return 0.5

        # Soft Jaccard: phrase a matches phrase b if either contains the other
        def _soft_match_count(set_x, set_y):
            count = 0
            for px in set_x:
                for py in set_y:
                    if px in py or py in px:
                        count += 1
                        break
            return count

        matches_ab = _soft_match_count(phrases_a, phrases_b)
        matches_ba = _soft_match_count(phrases_b, phrases_a)
        total = len(phrases_a) + len(phrases_b)
        return (matches_ab + matches_ba) / total if total > 0 else 0.0

    def update_user_profile(self, user_id: int, profile_delta: Dict) -> bool:
        user_prof = self._profiles.get(user_id)
        if not user_prof or not user_prof.profile:
            return False

        changed = False

        if 'primary_interests_delta' in profile_delta:
            delta = profile_delta['primary_interests_delta']
            if isinstance(delta, list) and delta:
                current = user_prof.profile.get('primary_interests', [])
                if isinstance(current, str):
                    current = [s.strip() for s in current.split(',')]
                remove_set = set(profile_delta.get('remove_interests', []))
                if remove_set:
                    current = [c for c in current if c not in remove_set]
                    changed = True
                new_items = [d for d in delta if d not in current]
                if new_items:
                    current = current + new_items
                    changed = True
                if len(current) > 7:
                    current = current[-7:]
                user_prof.profile['primary_interests'] = current

        if changed:
            user_prof.updated_at = time.time()
            self._save_user_profile(user_id)

        return changed

    def record_case(
        self,
        user_id: int,
        ground_truth: int,
        ranked_items: List[int],
        candidates_meta: Dict = None,
        user_memory: str = "",
    ):
        gt_rank = ranked_items.index(ground_truth) if ground_truth in ranked_items else len(ranked_items)
        hit = (gt_rank == 0)
        case = UserCaseRecord(
            user_id=user_id,
            ground_truth=ground_truth,
            ranked_items=ranked_items,
            gt_rank=gt_rank,
            hit=hit,
            candidates_meta=candidates_meta or {},
        )
        case.user_memory = user_memory
        with self._lock:
            self._cases[user_id].append(case)

    def reflect_user(self, user_id: int) -> bool:
        with self._lock:
            cases = list(self._cases.get(user_id, []))
            n_cases_snapshot = len(cases)

        if len(cases) < self.min_cases:
            return False

        hits = [c for c in cases if c.hit]
        misses = [c for c in cases if not c.hit]
        hit_rate = len(hits) / len(cases)

        miss_summaries = [c.to_summary() for c in misses[:8]]
        hit_summaries = [c.to_summary() for c in hits[:4]]

        current_profile = self._profiles.get(user_id)
        from src import settings
        # the critic compares the rewrite with an existing profile; without one the profile is
        # generated in this single call (no extra call is made to build a base profile)
        _use_critic = settings.SYNC_CRITIC and current_profile is not None and bool(current_profile.profile)
        current_profile_str = ""
        if current_profile and current_profile.profile:
            import json as _json
            current_profile_str = _json.dumps(current_profile.profile, indent=2)
        elif current_profile and current_profile.rules:
            current_profile_str = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(current_profile.rules))
        else:
            current_profile_str = "  (No profile yet — first evolution for this user)"

        history_context = self._build_user_history_context(user_id)

        memory_summary = ""
        for c in cases:
            mem = getattr(c, 'user_memory', '')
            if mem and len(mem) > len(memory_summary):
                memory_summary = mem
        if memory_summary and len(memory_summary) > 150:
            memory_summary = memory_summary[:150] + "..."

        instruction_context = ""
        if self.dataset and hasattr(self.dataset, 'instructions') and self.dataset.instructions:
            inst = self.dataset.instructions.get(user_id)
            if inst:
                if isinstance(inst, dict):
                    instruction_context = inst.get('instruction', '') or inst.get('persona', '')
                elif isinstance(inst, str):
                    instruction_context = inst

        prompt = f"""You are building a STRUCTURED USER PROFILE for a recommendation system. Analyze this user's complete behavior history to create an accurate, personalized profile.

## Current Profile (v{current_profile.version if current_profile else 0}):
{current_profile_str}

## User's Complete Behavior History (time-weighted):
{history_context if history_context else "  (No history available)"}

{f'## Memory Summary: {memory_summary}' if memory_summary else ''}
{f'## User Intent: {instruction_context[:200]}' if instruction_context else ''}

## Latest Recommendation Results ({len(cases)} cases, {hit_rate:.0%} accuracy):

### Mistakes (wrong item ranked #1):
{chr(10).join(miss_summaries) if miss_summaries else "  (No mistakes)"}

### Correct Predictions:
{chr(10).join(hit_summaries) if hit_summaries else "  (None yet)"}

## Your Task:
Create a structured user profile based on the COMPLETE behavior history above. The profile must capture:
1. **primary_interests**: List of 3-5 specific category interests from history (e.g., ["American breakfast", "Thai food", "Coffee"])
2. **quality_bar**: Rating preference pattern (e.g., "prefers 4.0+ stars")
3. **behavior_pattern**: Behavioral tendency (e.g., "category-loyal, repeat visitor" or "explorer, tries new categories")
4. **secondary_interests**: 2-3 occasional interests
5. **unlikely_interests**: Categories this user would NOT visit (inferred from gaps in history)
6. **city_preference**: Primary city/cities from history
7. **summary**: One-sentence user portrait

Also provide {self.max_rules} concrete ranking rules derived from the profile.

Output as JSON with "profile" (object) and "rules" (array of strings)."""

        if settings.USER_CONTEXT:
            prompt += ("\n\nAdditionally, the profile MUST contain a **context** field: short phrases for "
                       "the situations/places/occasions of this user's visits (e.g. [\"weekend brunch\", \"downtown\"]).")
        if _use_critic:
            prompt += ("\n\nAlso set \"accept_update\" to true ONLY if the new profile is more specifically "
                       "supported by the correct predictions AND concretely repairs at least one mistake; "
                       "otherwise false.")

        try:
            schema = {
                "profile": {
                    "type": "object",
                    "properties": {
                        "primary_interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 core category interests from behavior history"
                        },
                        "quality_bar": {
                            "type": "string",
                            "description": "Rating/quality preference pattern"
                        },
                        "behavior_pattern": {
                            "type": "string",
                            "description": "Behavioral tendency (loyal/explorer/etc)"
                        },
                        "secondary_interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-3 occasional interests"
                        },
                        "unlikely_interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories this user would NOT visit"
                        },
                        "city_preference": {
                            "type": "string",
                            "description": "Primary active cities"
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-sentence user portrait"
                        },
                    },
                    "required": ["primary_interests", "quality_bar", "behavior_pattern",
                                 "secondary_interests", "unlikely_interests", "city_preference", "summary"],
                    "additionalProperties": False,
                    "description": "Structured user profile"
                },
                "rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"{self.max_rules} personalized ranking rules"
                },
            }
            if settings.USER_CONTEXT:
                _pp = schema["profile"]
                _pp["properties"]["context"] = {
                    "type": "array", "items": {"type": "string"},
                    "description": "Situations/places/occasions of visits (context channel)"}
                _pp["required"] = list(_pp["required"]) + ["context"]
            if _use_critic:
                schema["accept_update"] = {"type": "boolean",
                                           "description": "Critic marker: accept the rewrite?"}

            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                properties=schema,
                temperature=0.3,
                max_tokens=600
            )

            new_profile = response.get("profile", {})
            new_rules = response.get("rules", [])[:self.max_rules]

            if not new_profile and not new_rules:
                logger.warning(f"  User {user_id}: LLM returned empty profile and rules")
                return False

            critic_decision = None
            rejected = False
            if _use_critic and current_profile is not None and current_profile.profile:
                critic_decision = self._critic_decide(
                    current_profile.profile, new_profile, cases, bool(response.get("accept_update", False)))
                if not critic_decision["accept"]:
                    new_profile = current_profile.profile
                    new_rules = current_profile.rules
                    rejected = True

            old_rules = []
            old_profile = {}
            if current_profile:
                old_rules = current_profile.rules
                old_profile = current_profile.profile
                current_profile.history.append({
                    "from_version": current_profile.version,
                    "to_version": current_profile.version if rejected else current_profile.version + 1,
                    "accepted": not rejected,
                    "old_rules": old_rules,
                    "new_rules": new_rules,
                    "old_profile": old_profile,
                    "new_profile": new_profile,
                    "hit_rate": hit_rate,
                    "n_cases": len(cases),
                })
                if not rejected:
                    current_profile.version += 1
                current_profile.rules = new_rules
                current_profile.profile = new_profile
                current_profile.confidence = hit_rate
                current_profile.updated_at = time.time()
            else:
                current_profile = UserProfile(
                    user_id=user_id,
                    version=1,
                    rules=new_rules,
                    confidence=hit_rate,
                    profile=new_profile,
                )
                current_profile.history.append({
                    "from_version": 0,
                    "to_version": 1,
                    "old_rules": [],
                    "new_rules": new_rules,
                    "new_profile": new_profile,
                    "hit_rate": hit_rate,
                    "n_cases": len(cases),
                })
                self._profiles[user_id] = current_profile

            with self._lock:
                current_cases = self._cases.get(user_id, [])
                self._cases[user_id] = current_cases[n_cases_snapshot:]
                self.n_reflections += 1
                self.n_users_with_profiles = len([s for s in self._profiles.values() if s.rules or s.profile])

            self._save_user_profile(user_id)

            self._trace("evolve", {
                "user_id": user_id,
                "version": current_profile.version,
                "profile": new_profile,
                "rules": new_rules,
                "hit_rate": hit_rate,
                "n_cases": len(cases),
                "critic": critic_decision,
            })

            return True

        except Exception as e:
            logger.warning(f"  User {user_id} profile reflection failed: {e}")
            return False

    def _item_channel_profile(self, item_id: int) -> Dict:
        """Four-channel item profile of `item_id` straight from dataset metadata (zero LLM)."""
        from src.profiles import extract_item_profile
        meta = {}
        if self.dataset is not None and getattr(self.dataset, "item_metadata", None):
            meta = self.dataset.item_metadata.get(item_id, {}) or {}
        return extract_item_profile(getattr(self.dataset, "name", ""), meta)

    def build_base_profile(self, user_id: int) -> Optional["UserProfile"]:
        """phi_u = LLM(history bands || P_profile). One call, history only."""
        history_context = self._build_user_history_context(user_id)
        if not history_context:
            return None
        from src import settings
        ctx = settings.USER_CONTEXT
        prompt = (
            "You are building a STRUCTURED USER PROFILE for a recommendation system from this user's "
            "behavior history (recent items in detail, older ones summarised).\n\n"
            f"## User's Behavior History:\n{history_context}\n\n"
            "Fields: primary_interests (3-5 specific categories), quality_bar (rating preference), "
            "behavior_pattern (loyal/explorer/...), secondary_interests (2-3), unlikely_interests, "
            "city_preference, summary (one sentence)"
            + (", context (short phrases for situations/places/occasions of visits)" if ctx else "")
            + ".\nOutput JSON with a \"profile\" object.")
        props = {
            "primary_interests": {"type": "array", "items": {"type": "string"}},
            "quality_bar": {"type": "string"},
            "behavior_pattern": {"type": "string"},
            "secondary_interests": {"type": "array", "items": {"type": "string"}},
            "unlikely_interests": {"type": "array", "items": {"type": "string"}},
            "city_preference": {"type": "string"},
            "summary": {"type": "string"},
        }
        if ctx:
            props["context"] = {"type": "array", "items": {"type": "string"}}
        schema = {"profile": {"type": "object", "properties": props,
                              "required": list(props), "additionalProperties": False}}
        try:
            resp = self.llm.generate_json(messages=[{"role": "user", "content": prompt}],
                                          properties=schema, temperature=0.3, max_tokens=600)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  User {user_id}: base profile build failed: {e}")
            return None
        profile = resp.get("profile", {})
        if not profile:
            return None
        user_prof = UserProfile(user_id=user_id, version=0, rules=[], confidence=0.0, profile=profile)
        self._profiles[user_id] = user_prof
        return user_prof

    @staticmethod
    def _channel_coverage(profile: Dict, item_channel_profiles: List[Dict]) -> int:
        from src.homophily import channel_hits
        return sum(len(channel_hits(profile, ip)) for ip in item_channel_profiles)

    def _critic_decide(self, old_profile: Dict, new_profile: Dict,
                          cases: List["UserCaseRecord"], marker: bool) -> Dict:
        """
        Zero-extra-LLM critic. The proposed profile is accepted unless it covers the ground-truth
        items of the warmup cases on fewer channels than the old profile. Stricter modes
        (settings.CRITIC_MIN_GAIN, settings.CRITIC_REQUIRE_MARKER) also require a coverage gain
        and/or the reflection's own accept marker; ties and uncertainty then keep the old profile.
        """
        gt_profiles = [self._item_channel_profile(c.ground_truth) for c in cases]
        old_cov = self._channel_coverage(old_profile, gt_profiles)
        new_cov = self._channel_coverage(new_profile, gt_profiles)
        from src import settings
        accept = (bool(marker) or not settings.CRITIC_REQUIRE_MARKER) and (new_cov - old_cov) >= settings.CRITIC_MIN_GAIN
        return {"accept": accept, "marker": bool(marker), "old_coverage": old_cov,
                "new_coverage": new_cov, "n_cases": len(cases)}

    def reflect_all(self, parallel_workers: int = 1) -> Tuple[int, int]:
        eligible_users = []
        with self._lock:
            for uid, cases in self._cases.items():
                if len(cases) >= self.min_cases:
                    eligible_users.append(uid)

        if not eligible_users:
            return 0, 0

        reflected = 0

        if parallel_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(self.reflect_user, uid): uid for uid in eligible_users}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="User profile reflection"):
                    if future.result():
                        reflected += 1
        else:
            for uid in eligible_users:
                if self.reflect_user(uid):
                    reflected += 1

        return reflected, len(eligible_users)

    def _save_user_profile(self, user_id: int):
        user_prof = self._profiles.get(user_id)
        if not user_prof:
            return
        path = self.persist_dir / f"user_{user_id}.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(user_prof.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save profile for user {user_id}: {e}")

    def save_all(self):
        saved = 0
        for uid in self._profiles:
            self._save_user_profile(uid)
            saved += 1
        stats_path = self.persist_dir / "_stats.json"
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_stats(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        logger.info(f"  💾 Saved {saved} user profiles to {self.persist_dir}")

    def get_stats(self) -> Dict:
        profiles_with_rules = [s for s in self._profiles.values() if s.rules]
        versions = [s.version for s in profiles_with_rules] if profiles_with_rules else [0]
        confidences = [s.confidence for s in profiles_with_rules] if profiles_with_rules else [0.0]

        pending_cases = sum(len(cases) for cases in self._cases.values())

        return {
            "n_users_with_profiles": len(profiles_with_rules),
            "n_total_users": len(self._profiles),
            "n_reflections": self.n_reflections,
            "avg_version": sum(versions) / max(1, len(versions)),
            "max_version": max(versions) if versions else 0,
            "avg_confidence": sum(confidences) / max(1, len(confidences)),
            "pending_cases": pending_cases,
            "persist_dir": str(self.persist_dir),
        }

    def _trace(self, event_type: str, data: Dict):
        if not self._trace_dir:
            return
        try:
            record = {"timestamp": time.time(), "event": event_type, **data}
            trace_file = self._trace_dir / f"trace_{event_type}.jsonl"
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with self._lock:
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass
