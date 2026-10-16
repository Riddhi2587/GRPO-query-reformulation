"""Expansion prompts. Every style must answer inside exactly one <query>...</query> block."""

STYLES = ("keywords", "pseudo_doc", "clarified")

_COMMON = (
    "You help a keyword (BM25) search engine find relevant documents. "
    "Do not explain. Reply with exactly one <query>...</query> block and nothing else."
)

SYSTEMS = {
    # Extra terms only: the original query is appended by the caller, so do not repeat it.
    "keywords": (
        _COMMON + " Inside the block, write 5 to 10 additional keywords and synonyms, specific terms "
        "likely to appear in relevant documents, separated by spaces. Do not repeat the query."
    ),
    # query2doc style
    "pseudo_doc": (
        _COMMON + " Inside the block, write a short passage of 2 to 3 sentences that a relevant "
        "document might contain, using precise terminology."
    ),
    "clarified": (
        _COMMON + " Inside the block, rewrite the query as one clearer, more specific question "
        "that uses precise terminology."
    ),
}


def build_messages(style: str, query: str) -> list[dict]:
    if style not in SYSTEMS:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    return [
        {"role": "system", "content": SYSTEMS[style]},
        {"role": "user", "content": f"Query: {query}"},
    ]
