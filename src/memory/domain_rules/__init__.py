"""
Domain-specific pruning rules for different datasets
"""
from .books_rules import BooksRules
from .goodreads_rules import GoodReadsRules
from .movietv_rules import MovieTVRules
from .yelp_rules import YelpRules

DOMAIN_RULES = {
    'instructrec-books': BooksRules,
    'instructrec-goodreads': GoodReadsRules,
    'instructrec-movietv': MovieTVRules,
    'instructrec-yelp': YelpRules,
}

PKL_NAME_RULES = {
    'booksall_recagent': BooksRules,
    'goodreadsall_recagent': GoodReadsRules,
    'movietvall_recagent': MovieTVRules,
    'yelpall_recagent': YelpRules,
}

KEYWORD_RULES = {
    'yelp': YelpRules,
    'movie': MovieTVRules,
    'goodread': GoodReadsRules,
    'book': BooksRules,
}

def get_domain_rules(dataset_name: str):
    """Get the appropriate rules class for a dataset"""
    base_name = dataset_name.lower().strip()

    if base_name in DOMAIN_RULES:
        return DOMAIN_RULES[base_name]

    pkl_base = base_name.replace('.pkl', '').replace('_1ads', '')
    if pkl_base in PKL_NAME_RULES:
        return PKL_NAME_RULES[pkl_base]

    for key in DOMAIN_RULES:
        if key in base_name or base_name in key:
            return DOMAIN_RULES[key]

    for keyword, rules_class in KEYWORD_RULES.items():
        if keyword in base_name:
            return rules_class

    print(f"Warning: No rules found for dataset '{dataset_name}', using Books rules as default")
    return BooksRules

__all__ = [
    'BooksRules',
    'GoodReadsRules',
    'MovieTVRules',
    'YelpRules',
    'DOMAIN_RULES',
    'get_domain_rules',
]

