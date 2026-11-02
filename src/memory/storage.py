"""
Simplified memory storage - single text mode
User Memory: single profile field (natural language description)
Item Memory: single description field (natural language description)
Temporal Tracking: timestamps for interactions
"""
import time
from typing import Dict, Optional


class MemoryStorage:
    """Simplified memory storage: each entity stores only one piece of text"""
    
    def __init__(self, **kwargs):
        """
        Initialize storage
        user_profiles: {user_id: str}  # User profiles (natural language)
        item_descriptions: {item_id: str}  # Item descriptions (natural language)
        interaction_timestamps: {user_id: [(item_id, timestamp), ...]}  # Temporal tracking
        """
        self.user_profiles = {}  # {user_id: "A book enthusiast interested in..."}
        self.item_descriptions = {}
        self.item_profiles = {}  # {item_id: {"serves": ..., "best_for": ..., "typical_fans": ..., "quality_signal": ..., "summary": ...}}
        self.interaction_timestamps = {}
        self.item_channel_profiles = {}
        self.n_updates = 0
    
    def get_user_memory(self, user_id: int) -> Optional[str]:
        """Get user memory (natural language profile)"""
        return self.user_profiles.get(user_id, None)
    
    def get_item_memory(self, item_id: int) -> Optional[str]:
        """Get item memory (natural language description)"""
        return self.item_descriptions.get(item_id, None)
    
    def update_user_memory(self, user_id: int, new_profile: str):
        """Update user memory (complete overwrite)"""
        if new_profile and new_profile.strip():
            self.user_profiles[user_id] = new_profile.strip()
            self.n_updates += 1
    
    def update_item_memory(self, item_id: int, new_description: str):
        """Update item memory (complete overwrite)"""
        if new_description and new_description.strip():
            self.item_descriptions[item_id] = new_description.strip()
            self.n_updates += 1

    def record_interaction_time(self, user_id: int, item_id: int, timestamp: float = None):
        """
        Record interaction timestamp for temporal tracking

        Args:
            user_id: User ID
            item_id: Item ID
            timestamp: Unix timestamp (if None, use current time)
        """
        if timestamp is None:
            timestamp = time.time()

        if user_id not in self.interaction_timestamps:
            self.interaction_timestamps[user_id] = []

        self.interaction_timestamps[user_id].append((item_id, timestamp))

    @staticmethod
    def _extract_semantic_description(raw_desc: str) -> str:
        """
        Extract semantically useful information from Yelp-style raw description.
        Prioritizes categories, stars, and review_count over address/coordinates.
        
        Input format: "address: ... | city: ... | stars: 4.0 | ... | categories: Pizza, Italian"
        Output: "Categories: Pizza, Italian. Stars: 4.0. 58 reviews. City: Las Vegas"
        """
        if not raw_desc:
            return ""
        
        parts = raw_desc.strip('"').split(' | ')
        parsed = {}
        for part in parts:
            if ':' not in part:
                continue
            key, val = part.split(':', 1)
            parsed[key.strip().lower()] = val.strip()
        
        # Build description prioritizing useful info (categories first!)
        desc_parts = []
        if 'categories' in parsed and parsed['categories']:
            desc_parts.append(f"Categories: {parsed['categories']}")
        if 'stars' in parsed:
            desc_parts.append(f"Stars: {parsed['stars']}")
        if 'review_count' in parsed:
            desc_parts.append(f"{parsed['review_count']} reviews")
        if 'city' in parsed:
            desc_parts.append(f"City: {parsed['city']}")
        
        if 'attributes' in parsed:
            attr_str = parsed['attributes']
            if "'RestaurantsPriceRange2'" in attr_str:
                import re
                m = re.search(r"'RestaurantsPriceRange2':\s*'(\d)'", attr_str)
                if m:
                    price_map = {'1': '$', '2': '$$', '3': '$$$', '4': '$$$$'}
                    desc_parts.append(f"Price: {price_map.get(m.group(1), m.group(1))}")
            if "'GoodForMeal'" in attr_str:
                import re
                meals = re.findall(r"'(\w+)':\s*True", attr_str.split("'GoodForMeal'")[1].split('}')[0])
                if meals:
                    desc_parts.append(f"Good for: {', '.join(meals)}")
        
        return ". ".join(desc_parts)

    def initialize_item_descriptions(self, item_metadata: Dict[int, Dict]):
        """
        Initialize item descriptions from metadata.
        
        For Yelp data: extracts semantic info (categories, stars, reviews) 
        instead of raw address/coordinates that waste token budget.
        
        Args:
            item_metadata: {item_id: {'title': ..., 'description': ...}}
        """
        print("Initializing item descriptions from metadata...")
        for item_id, meta in item_metadata.items():
            if item_id in self.item_descriptions:
                continue
            
            title = meta.get('title', f'Item-{item_id}')
            desc = meta.get('description', '')
            
            if desc:
                if isinstance(desc, list):
                    non_empty = [d.strip() for d in desc if d and d.strip() and d.strip().lower() not in ['nan', 'n/a']]
                    raw_text = non_empty[0] if non_empty else ""
                else:
                    raw_text = str(desc)
                
                # Try semantic extraction first (works for Yelp-style structured data)
                semantic_desc = self._extract_semantic_description(raw_text)
                if semantic_desc:
                    description = f"{title}. {semantic_desc}"
                elif raw_text:
                    description = f"{title}. {raw_text[:300]}"
                else:
                    description = title
            else:
                description = title
            
            self.item_descriptions[item_id] = description
        
        print(f"  Initialized descriptions for {len(self.item_descriptions)} items")

    def get_item_profile(self, item_id: int) -> Optional[Dict]:
        """Get the item profile."""
        return self.item_profiles.get(item_id, None)

    def update_item_profile(self, item_id: int, fields: Dict):
        """Update the item profile (incremental merge)."""
        if not fields:
            return
        existing = self.item_profiles.get(item_id, {})
        existing.update(fields)
        self.item_profiles[item_id] = existing

    def initialize_item_profiles(self, item_metadata: Dict[int, Dict], reviews: Dict = None):
        import re

        item_reviews = {}
        if reviews:
            for uid, user_reviews in reviews.items():
                if isinstance(user_reviews, dict):
                    for iid, text in user_reviews.items():
                        if text and isinstance(text, str):
                            if iid not in item_reviews:
                                item_reviews[iid] = []
                            item_reviews[iid].append(text)

        print("Initializing item profiles from metadata...")
        initialized = 0
        for item_id, meta in item_metadata.items():
            if item_id in self.item_profiles:
                continue

            title = meta.get('title', f'Item-{item_id}')
            desc = meta.get('description', '')

            parsed = {}
            if desc:
                raw = str(desc).strip('"')
                for seg in raw.split(' | '):
                    if ':' not in seg:
                        continue
                    k, v = seg.split(':', 1)
                    parsed[k.strip().lower()] = v.strip()

            categories = parsed.get('categories', '')
            stars = parsed.get('stars', '')
            review_count = parsed.get('review_count', '')
            city = parsed.get('city', '')

            serves = categories if categories else title

            best_for_parts = []
            attr_str = parsed.get('attributes', '')
            if attr_str:
                meal_match = re.findall(r"'(\w+)':\s*True", attr_str)
                if meal_match:
                    best_for_parts.extend(meal_match[:3])
                price_match = re.search(r"'RestaurantsPriceRange2':\s*'(\d)'", attr_str)
                if price_match:
                    price_map = {'1': 'budget', '2': 'moderate', '3': 'upscale', '4': 'fine-dining'}
                    best_for_parts.append(price_map.get(price_match.group(1), ''))
            best_for = ', '.join(best_for_parts) if best_for_parts else ''

            typical_fans = ''
            item_revs = item_reviews.get(item_id, [])
            if item_revs:
                sorted_revs = sorted(item_revs, key=len)
                mid_idx = len(sorted_revs) // 2
                sample_revs = sorted_revs[max(0, mid_idx-1):mid_idx+2]
                first_sentences = []
                for rev in sample_revs[:3]:
                    sents = re.split(r'[.!?]', rev)
                    if sents and sents[0].strip():
                        first_sentences.append(sents[0].strip()[:60])
                if first_sentences:
                    typical_fans = '; '.join(first_sentences)

            quality_parts = []
            if stars:
                quality_parts.append(f"★{stars}")
            if review_count:
                quality_parts.append(f"{review_count} reviews")
            quality_signal = ', '.join(quality_parts) if quality_parts else ''

            summary_parts = [title]
            if categories:
                summary_parts.append(categories)
            if stars:
                summary_parts.append(f"★{stars}")
            if city:
                summary_parts.append(city)
            summary = '. '.join(summary_parts)

            self.item_profiles[item_id] = {
                'serves': serves,
                'best_for': best_for,
                'typical_fans': typical_fans,
                'quality_signal': quality_signal,
                'city': city,
                'summary': summary,
            }
            initialized += 1

        print(f"  Initialized profiles for {initialized} items (total: {len(self.item_profiles)})")
    
    def get_item_channel_profile(self, item_id: int) -> Optional[Dict]:
        """Four-channel item profile (None if not initialised)."""
        return self.item_channel_profiles.get(item_id)

    def initialize_item_channel_profiles(self, item_metadata: Dict[int, Dict], reviews: Dict = None,
                                 dataset_name: str = ""):
        """
        Zero-LLM RuleExtract(meta_i, reviews_i) into the four-channel item profile. Kept apart
        from `item_profiles`, so the prompts that use `item_profiles` are unchanged.
        """
        from src.profiles import extract_item_profile

        item_reviews = {}
        if reviews:
            for uid, user_reviews in reviews.items():
                if isinstance(user_reviews, dict):
                    for iid, text in user_reviews.items():
                        if text and isinstance(text, str):
                            item_reviews.setdefault(iid, []).append(text)

        n = 0
        for item_id, meta in item_metadata.items():
            if item_id in self.item_channel_profiles:
                continue
            self.item_channel_profiles[item_id] = extract_item_profile(dataset_name, meta, item_reviews.get(item_id))
            n += 1
        print(f"  Initialized four-channel item profiles for {n} items (total: {len(self.item_channel_profiles)})")

    def render_user_summary(self, user_id: int) -> str:
        """
        Render user memory summary (for Stage-R prompt)
        
        Returns:
            User profile text, or default prompt if none exists
        """
        profile = self.get_user_memory(user_id)
        if profile:
            return profile
        else:
            return "No specific memory recorded for this user yet."
    