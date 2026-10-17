"""Expansion prompts. Every style must answer inside exactly one <query>...</query> block."""
import hashlib
import json

STYLES = ("keywords", "pseudo_doc", "clarified")
FEWSHOT_STYLE = "keywords_fs"
ALL_STYLES = STYLES + (FEWSHOT_STYLE,)

# (query, expansion) examples for the few-shot style; none of these queries is in any NFCorpus split.
FEWSHOT_EXAMPLES = (
    ("tea and bone strength",
     "catechins polyphenols camellia sinensis bone mineral density osteoporosis fracture"),
    ("beets and exercise performance",
     "nitrate nitric oxide beetroot juice oxygen uptake endurance vo2 athletes"),
    ("does spicy food cause stomach ulcers",
     "capsaicin chili peppers peptic dyspepsia helicobacter pylori gastric mucosa gastritis"),
)

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


SYSTEMS[FEWSHOT_STYLE] = SYSTEMS["keywords"]  # same instructions; the examples are added as chat turns


def build_messages(style: str, query: str) -> list[dict]:
    if style not in SYSTEMS:
        raise ValueError(f"unknown style {style!r}; expected one of {ALL_STYLES}")
    msgs = [{"role": "system", "content": SYSTEMS[style]}]
    if style == FEWSHOT_STYLE:
        for q, e in FEWSHOT_EXAMPLES:
            msgs += [{"role": "user", "content": f"Query: {q}"},
                     {"role": "assistant", "content": f"<query>{e}</query>"}]
    msgs.append({"role": "user", "content": f"Query: {query}"})
    return msgs


def prompt_fingerprint(style: str) -> str:
    """Short hash of everything the model sees for `style` (system prompt and few-shot turns, query left as a placeholder)."""
    return hashlib.sha256(json.dumps(build_messages(style, "{query}"), sort_keys=True).encode()).hexdigest()[:16]
