"""
Parameters of the profile, homophily and profile-update components.

Configurable parameters can be overridden by optional `homophily:` / `profile:` sections of the config.
The structural components below are always part of the pipeline; the tests set them to False to run
the plain pipeline for comparison.
"""

# homophily gates: g(H) = alpha + (1 - alpha) * H   (alpha = 1 gives an identity gate)
ALPHA_UI = 0.85
ALPHA_UU = 0.3
# channel weights of the profile homophily H = sum_c w_c * H_c (non-negative, sum to 1); the earlier
# user-user score used the preference channel only, so the preference channel keeps most of the mass
CHANNEL_WEIGHTS = {"preference": 0.70, "quality": 0.10, "context": 0.10, "audience": 0.10}

# profile update: a rewrite replaces the old profile unless it covers more than |CRITIC_MIN_GAIN|
# fewer of the target item's channels (-1 tolerates losing one channel); raise CRITIC_MIN_GAIN to 1 and
# set CRITIC_REQUIRE_MARKER for a strict, reject-by-default update
CRITIC_MIN_GAIN = -1
CRITIC_REQUIRE_MARKER = False

ITEM_CHANNELS = True
UI_GATE = True
USER_CONTEXT = False      # ask the LLM for a separate `context` field; off: the context channel uses `city_preference`
SYNC_CRITIC = False       # critic-gated update when a user already has a profile (no extra LLM call)
PROFILE_FIRST = False     # build the user profile before the warmup rerank (one extra LLM call per user)
CANDIDATE_SEED = False    # True: evaluation candidates derived from (seed, user_id); False: per-thread RNG as before

# short reranker profile ("Likes / Avoids / Style"), the fallback when a user has no structured profile
SHORT_PROFILE_TEMPLATE = True        # empty user memory -> fixed template line, no LLM call
SHORT_PROFILE_FALLBACK_ONLY = True   # after warmup, extract only for users without a structured profile
SHORT_PROFILE_TEMPLATE_TEXT = "Likes: not known yet | Style: no interaction memory recorded yet"
# Empty-memory shortcut: a fixed template line, or an empty profile (no profile block in the
# reranker prompt). Neither calls the LLM.
SHORT_PROFILE_EMPTY_MODE = "empty"  # "empty" | "template"

# Pruning rules used for every dataset.
# This controls pruning only; dataset.name remains the real data domain.
PRUNER_RULES_DATASET = "instructrec-books"


def configure(config: dict) -> None:
    """Override homophily and profile parameters from the optional config sections."""
    global ALPHA_UI, ALPHA_UU, CHANNEL_WEIGHTS, CRITIC_MIN_GAIN, CRITIC_REQUIRE_MARKER
    global SHORT_PROFILE_EMPTY_MODE
    global PRUNER_RULES_DATASET
    h = (config or {}).get("homophily") or {}
    p = (config or {}).get("profile") or {}
    pruning = (config or {}).get("pruner") or {}
    ALPHA_UI = float(h.get("alpha_ui", ALPHA_UI))
    ALPHA_UU = float(h.get("alpha_uu", ALPHA_UU))
    if "channel_weights" in h:
        w = dict(h["channel_weights"])
        assert set(w) == set(CHANNEL_WEIGHTS) and abs(sum(w.values()) - 1.0) < 1e-6, "channel weights must sum to 1"
        CHANNEL_WEIGHTS = w
    CRITIC_MIN_GAIN = int(p.get("critic_min_gain", CRITIC_MIN_GAIN))
    CRITIC_REQUIRE_MARKER = bool(p.get("critic_require_marker", CRITIC_REQUIRE_MARKER))
    empty_mode = p.get("short_profile_empty_mode", SHORT_PROFILE_EMPTY_MODE)
    if not isinstance(empty_mode, str) or empty_mode not in {"template", "empty"}:
        raise ValueError("profile.short_profile_empty_mode must be 'template' or 'empty'")
    SHORT_PROFILE_EMPTY_MODE = empty_mode
    rules_dataset = pruning.get("rules_dataset", PRUNER_RULES_DATASET)
    if not isinstance(rules_dataset, str) or not rules_dataset.strip():
        raise ValueError("pruner.rules_dataset must be a non-empty dataset name or 'dataset'")
    PRUNER_RULES_DATASET = rules_dataset.strip()


def resolve_pruner_rules_dataset(dataset) -> str:
    """Choose pruning rules independently of metadata/profile domain handling."""
    if PRUNER_RULES_DATASET == "dataset":
        return getattr(dataset, "name", "instructrec-books")
    return PRUNER_RULES_DATASET
