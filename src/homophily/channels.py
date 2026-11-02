"""
Channel-aligned profile homophily.

Both user and item carry a four-channel profile over
    C = {preference, quality, context, audience}
with the following field correspondence:

    channel      user field            item field
    preference   primary_interests     serves
    quality      quality_bar           quality_signal
    context      context               context
    audience     behavior_pattern      audience

Only these four channel fields enter H. Other profile fields (secondary_interests,
unlikely_interests, summary, ...) are never read here: the gate consumes positive
channels only.
"""

import re
from typing import Dict, Iterable, List, Optional, Set

CHANNELS = ("preference", "quality", "context", "audience")

USER_CHANNEL_FIELDS = {
    "preference": "primary_interests",
    "quality": "quality_bar",
    "context": "context",
    "audience": "behavior_pattern",
}
ITEM_CHANNEL_FIELDS = {
    "preference": "serves",
    "quality": "quality_signal",
    "context": "context",
    "audience": "audience",
}


# Neutral score when a channel cannot be compared (either side empty / unparsable).
NEUTRAL = 0.5

QUALITY_LEVELS = {"inconsistent": 0, "average": 1, "high": 2}

_SPLIT_RE = re.compile(r"[,;|/\n]+")
_STARS_RE = re.compile(r"(?<![\d.])([1-5](?:\.\d+)?)(?![\d])")


def to_phrases(value) -> Set[str]:
    """Normalise a channel value (list or delimited string) into a lower-cased phrase set."""
    if value is None:
        return set()
    if isinstance(value, str):
        parts: Iterable = _SPLIT_RE.split(value)
    elif isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = [str(value)]
    phrases = set()
    for p in parts:
        p = str(p).strip().lower()
        if len(p) > 2:
            phrases.add(p)
    return phrases


def _substr_match(a: str, b: str) -> bool:
    return a in b or b in a


def sem_phrase(A: Set[str], B: Set[str]) -> float:
    """
    Fraction of the query-side phrases A that are echoed by some phrase in B
    (substring in either direction). Denominator is |A| (query side, i.e. the user).
    Returns NEUTRAL when either side is empty (formula undefined for |A| = 0; an item without
    information is neither rewarded nor penalised).
    """
    if not A or not B:
        return NEUTRAL
    hit = sum(1 for a in A if any(_substr_match(a, b) for b in B))
    return hit / len(A)


def sem_context(A: Set[str], B: Set[str]) -> float:
    """Bidirectional substring hit: 1 on hit, 0 on miss, NEUTRAL when either side is empty."""
    if not A or not B:
        return NEUTRAL
    return 1.0 if any(_substr_match(a, b) for a in A for b in B) else 0.0


def quality_level(value) -> Optional[int]:
    """Map a quality field to an ordinal bin (0 inconsistent, 1 average, 2 high); None if unknown."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(v) for v in value)
    text = str(value).strip().lower()
    if not text:
        return None
    for name in ("inconsistent", "average", "high"):
        if text == name or text.startswith(name):
            return QUALITY_LEVELS[name]
    # free text such as "prefers 4.0+ stars": the first star rating (1-5) is the expected quality
    m = _STARS_RE.search(text)
    if m:
        stars = float(m.group(1))
        if 1.0 <= stars <= 5.0:
            return QUALITY_LEVELS["high"] if stars >= 4.0 else QUALITY_LEVELS["average"] if stars >= 3.0 else QUALITY_LEVELS["inconsistent"]
    return None


def sem_quality(user_value, item_value) -> float:
    """
    Item quality >= user-preferred threshold -> 1, otherwise linear interpolation into [0, 1)
    over the ordinal distance (one bin below -> 0.5, two bins below -> 0).
    NEUTRAL if either side is not a recognised bin.
    """
    u, i = quality_level(user_value), quality_level(item_value)
    if u is None or i is None:
        return NEUTRAL
    if i >= u:
        return 1.0
    return max(0.0, 1.0 - (u - i) / 2.0)


_PLACEHOLDERS = {"", "n/a", "na", "none", "unknown", "not specified", "unspecified"}


def user_channel_value(profile: Dict, channel: str):
    """Value of the user's channel field. The context channel falls back to `city_preference`."""
    value = (profile or {}).get(USER_CHANNEL_FIELDS[channel])
    if channel == "context" and not value:
        value = (profile or {}).get("city_preference")
        if isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
            value = None
    return value


def has_channel_info(item_profile: Dict) -> bool:
    """True if at least one channel of the item profile is non-empty."""
    return any(to_phrases((item_profile or {}).get(ITEM_CHANNEL_FIELDS[c])) for c in CHANNELS)


def channel_similarity(channel: str, user_profile: Dict, item_profile: Dict) -> float:
    """H_c(phi_u, phi_i) = sem(phi_u^(c), phi_i^(c)) for one channel."""
    u = user_channel_value(user_profile, channel)
    i = (item_profile or {}).get(ITEM_CHANNEL_FIELDS[channel])
    if channel == "quality":
        return sem_quality(u, i)
    if channel == "context":
        return sem_context(to_phrases(u), to_phrases(i))
    return sem_phrase(to_phrases(u), to_phrases(i))


def profile_homophily(user_profile: Dict, item_profile: Dict,
                      weights: Optional[Dict[str, float]] = None) -> float:
    """H = sum_c w_c * H_c, with non-negative channel weights summing to 1."""
    from src import settings
    w = weights or settings.CHANNEL_WEIGHTS
    return sum(w[c] * channel_similarity(c, user_profile, item_profile) for c in CHANNELS)


def channel_hits(user_profile: Dict, item_profile: Dict) -> List[str]:
    """
    Channels on which the user profile genuinely covers the item profile (strict: empty sides
    never count). Used by the Sync critic.
    """
    hits = []
    for c in CHANNELS:
        u = user_channel_value(user_profile, c)
        i = (item_profile or {}).get(ITEM_CHANNEL_FIELDS[c])
        if c == "quality":
            ul, il = quality_level(u), quality_level(i)
            if ul is not None and il is not None and il >= ul:
                hits.append(c)
            continue
        A, B = to_phrases(u), to_phrases(i)
        if A and B and any(_substr_match(a, b) for a in A for b in B):
            hits.append(c)
    return hits


def user_user_compat(profile_a: Dict, profile_b: Dict) -> float:
    """
    User-user edge score: symmetric phrase overlap on the preference channel only
    NEUTRAL when either profile has no phrases.
    """
    A = to_phrases((profile_a or {}).get(USER_CHANNEL_FIELDS["preference"]))
    B = to_phrases((profile_b or {}).get(USER_CHANNEL_FIELDS["preference"]))
    if not A or not B:
        return NEUTRAL
    m_ab = sum(1 for a in A if any(_substr_match(a, b) for b in B))
    m_ba = sum(1 for b in B if any(_substr_match(a, b) for a in A))
    return (m_ab + m_ba) / (len(A) + len(B))


def gate(h: float, alpha: float) -> float:
    """Monotone homophily gate g(H) = alpha + (1 - alpha) * H."""
    return alpha + (1.0 - alpha) * h
