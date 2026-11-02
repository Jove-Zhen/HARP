"""
Item profile: deterministic, zero-LLM RuleExtract(meta_i, reviews_i), rules R1-R4.

    phi_i = (serves, quality_signal, context, audience)   # + summary for prompts/traces

    R1  ontology projection  : metadata fields -> preference / context / audience (map S)
    R2  lexicon encoding     : optional, scan reviews / structured text with phrase lexicon D
                               (longest match), merged into the channel declared in D
    R3  quality induction    : (mean rating, #reviews) -> {high, average, inconsistent}
    R4  channel exclusivity  : a phrase routed to several channels is kept only in the channel
                               of highest priority  preference > quality > context > audience

The dataset-level ontology map S and lexicon D are defined for Yelp only, because only the Yelp
metadata (.meta) carries structured fields (categories, stars, review_count, city, attributes).
The Books / Goodreads / MovieTV .meta files contain only a title and a free-text description.
For those datasets `extract_item_profile` returns channel-empty profiles (every channel score is
then the neutral 0.5) and logs a warning once. Register an extractor in EXTRACTORS to enable them.
"""

import ast
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("ItemProfile")

QUALITY_HIGH_STARS = 4.0       # mean rating at/above which the signal is "high"
QUALITY_MIN_REVIEWS = 10       # below this the rating is too noisy -> "inconsistent"

YELP_LEXICON = [
    ("date night", "audience"),
    ("family friendly", "audience"),
    ("kid friendly", "audience"),
    ("business lunch", "context"),
    ("happy hour", "context"),
    ("late night", "context"),
    ("outdoor seating", "context"),
]
_LEXICON_SORTED = sorted(YELP_LEXICON, key=lambda pc: -len(pc[0]))

GENERIC_CATEGORIES = {"restaurants", "food", "nightlife", "arts & entertainment", "event planning & services"}

_PRICE = {"1": "budget", "2": "moderate price", "3": "upscale price", "4": "fine dining price"}
_BOOL_ATTR_AUDIENCE = {
    "GoodForKids": "kids",
    "RestaurantsGoodForGroups": "groups",
}
_BOOL_ATTR_CONTEXT = {
    "OutdoorSeating": "outdoor seating",
    "RestaurantsDelivery": "delivery",
    "RestaurantsTakeOut": "takeout",
    "RestaurantsReservations": "reservations",
}
_STRUCT_TRUE_KEYS_CONTEXT = ("GoodForMeal",)     # {'breakfast': True, ...} -> keys that are True
_STRUCT_TRUE_KEYS_AUDIENCE = ("Ambience",)       # {'romantic': True, ...}  -> keys that are True


def _parse_description(desc) -> Dict[str, str]:
    """Parse the pipe-separated `key: value | key: value` description into a lower-case-key dict."""
    parsed: Dict[str, str] = {}
    if not desc:
        return parsed
    raw = str(desc).strip().strip('"')
    for seg in raw.split(" | "):
        if ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        parsed[k.strip().lower()] = v.strip()
    return parsed


def _literal_dict(text) -> Dict:
    """Best-effort literal_eval of a (possibly doubly quoted) dict string; {} on failure."""
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    t = str(text).strip()
    for _ in range(2):  # attribute values are sometimes a quoted dict inside a dict
        try:
            v = ast.literal_eval(t)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            t = v
            continue
        return {}
    return {}


def _true_keys(d: Dict) -> List[str]:
    return [str(k) for k, v in d.items() if v is True or str(v) == "True"]


def _quality_bin(stars: Optional[float], n_reviews: Optional[int]) -> str:
    """R3: monotone rule over (mean rating, #reviews) with three bins."""
    if stars is None or n_reviews is None or n_reviews < QUALITY_MIN_REVIEWS:
        return "inconsistent"
    return "high" if stars >= QUALITY_HIGH_STARS else "average"


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _lexicon_scan(text: str) -> List[tuple]:
    """R2: longest-match scan of `text` against the lexicon; returns [(phrase, channel), ...]."""
    if not text:
        return []
    low = text.lower()
    found = []
    for phrase, channel in _LEXICON_SORTED:
        if phrase in low:
            found.append((phrase, channel))
    return found


def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def _exclusive(serves: List[str], context: List[str], audience: List[str]):
    """R4: priority preference > quality > context > audience on normalised phrases."""
    taken = {s.lower() for s in serves}
    context = [c for c in context if c.lower() not in taken]
    taken |= {c.lower() for c in context}
    audience = [a for a in audience if a.lower() not in taken]
    return serves, context, audience


def extract_yelp(meta: Dict, review_texts: Optional[List[str]] = None) -> Dict:
    title = meta.get("title", "")
    parsed = _parse_description(meta.get("description", ""))

    categories = [c.strip() for c in parsed.get("categories", "").split(",") if c.strip()]
    # Drop catch-all categories: under substring matching they would echo any "... food" phrase
    # of the user and produce spurious homophily. Kept only if nothing more specific exists.
    specific = [c for c in categories if c.lower() not in GENERIC_CATEGORIES]
    categories = specific or categories
    serves = list(categories) if categories else ([title] if title else [])

    context: List[str] = []
    audience: List[str] = []
    city = parsed.get("city", "").strip()
    if city:
        context.append(city)

    attrs = _literal_dict(parsed.get("attributes", ""))
    price = str(attrs.get("RestaurantsPriceRange2", "")).strip("'\"")
    if price in _PRICE:
        context.append(_PRICE[price])
    for key, phrase in _BOOL_ATTR_CONTEXT.items():
        if str(attrs.get(key, "")) == "True":
            context.append(phrase)
    for key, phrase in _BOOL_ATTR_AUDIENCE.items():
        if str(attrs.get(key, "")) == "True":
            audience.append(phrase)
    for key in _STRUCT_TRUE_KEYS_CONTEXT:
        context.extend(_true_keys(_literal_dict(attrs.get(key))))
    for key in _STRUCT_TRUE_KEYS_AUDIENCE:
        audience.extend(_true_keys(_literal_dict(attrs.get(key))))

    for text in (review_texts or [])[:20]:
        for phrase, channel in _lexicon_scan(text):
            (context if channel == "context" else audience).append(phrase)

    stars = _to_float(parsed.get("stars"))
    n_reviews = _to_int(parsed.get("review_count"))
    quality = _quality_bin(stars, n_reviews)

    serves, context, audience = _exclusive(_dedupe(serves), _dedupe(context), _dedupe(audience))

    summary_parts = [title]
    if serves:
        summary_parts.append(", ".join(serves[:3]))
    if stars is not None:
        summary_parts.append(f"★{stars}")
    if city:
        summary_parts.append(city)
    return {
        "serves": serves,
        "quality_signal": quality,
        "context": context,
        "audience": audience,
        "summary": ". ".join(p for p in summary_parts if p),
        # raw numeric evidence kept for traces / candidate enrichment; never read by the gate
        "stars": stars,
        "review_count": n_reviews,
        "city": city,
    }


def _empty_profile(meta: Dict) -> Dict:
    title = meta.get("title", "")
    return {"serves": [], "quality_signal": "", "context": [], "audience": [], "summary": title}


# dataset key (substring of dataset.name, e.g. "instructrec-yelp") -> extractor
EXTRACTORS: Dict[str, Callable[[Dict, Optional[List[str]]], Dict]] = {
    "yelp": extract_yelp,
}

_warned = set()


def extract_item_profile(dataset_name: str, meta: Dict, review_texts: Optional[List[str]] = None) -> Dict:
    name = (dataset_name or "").lower()
    for key, fn in EXTRACTORS.items():
        if key in name:
            return fn(meta, review_texts)
    if name not in _warned:
        _warned.add(name)
        logger.warning(
            "No item-profile extractor registered for dataset '%s' (only %s). Item channels are "
            "left empty; profile homophily is then neutral (0.5) on the item side.",
            dataset_name, sorted(EXTRACTORS),
        )
    return _empty_profile(meta)
