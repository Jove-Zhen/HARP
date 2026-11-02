<!-- Generation: 0 -->
<!-- Version: v0.1.0 -->
<!-- Description: Evolvable profile extraction prompt for Skill-Guided Reranker -->
<!-- EVOLVABLE_START -->
Based on this user's memory, extract their core preferences in exactly this format.
Be very concise (max 5 words per item). Only include patterns clearly supported by the memory.

User Memory:
{{user_memory}}

Extract:
1. "likes": Top 2-3 preference themes (e.g., "Mexican food", "casual dining")
2. "dislikes": 0-1 things they clearly avoid (leave empty if unsure)
3. "style": One phrase describing their taste (e.g., "adventurous foodie", "budget-friendly casual")
<!-- EVOLVABLE_END -->
