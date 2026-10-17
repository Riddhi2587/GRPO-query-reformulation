"""Cross-encoder reranking of a BM25 candidate pool (sentence-transformers MiniLM by default)."""
from functools import lru_cache

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def require_gpu(allow_local: bool = False) -> None:
    """A real run is ~100k cross-encoder pairs: run it on Kaggle/JarvisLabs, not on a laptop's MPS/CPU."""
    import torch
    if not (torch.cuda.is_available() or allow_local):
        raise SystemExit("No CUDA GPU found. Reranking is too heavy for local MPS/CPU: run it on Kaggle "
                         "(e.g. kaggle/rerank_kaggle.ipynb), or pass allow_local / --allow-local for a deliberate small run.")


@lru_cache(maxsize=None)
def load_doc_text(dataset: str) -> dict[str, str]:
    """{docno: "title text"} from the cached BEIR corpus (ir_datasets; no network once downloaded)."""
    import ir_datasets
    return {d.doc_id: f"{d.title} {d.text}".strip() for d in ir_datasets.load(f"beir/{dataset}").docs_iter()}


class CrossEncoderReranker:
    """Scores (query, doc) pairs once each and reorders pools by score (ties keep the BM25 order).

    `scorer` maps a list of (query, doc_text) pairs to a list of floats; the default is the MiniLM cross-encoder.
    Scores are cached by (query, docno), so the same pair is never scored twice across systems.
    Building the real model needs a CUDA GPU unless `allow_local` is set (see `require_gpu`).
    """

    def __init__(self, scorer=None, model_name: str = MODEL, batch_size: int = 64, max_length: int = 512,
                 allow_local: bool = False):
        if scorer is None:
            require_gpu(allow_local)
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, max_length=max_length)
            scorer = lambda pairs: model.predict(pairs, batch_size=batch_size, show_progress_bar=True).tolist()  # noqa: E731
        self.scorer = scorer
        self.cache: dict[tuple[str, str], float] = {}
        self.pairs_scored = 0

    def rerank_many(self, queries: list[str], pools: list[list[str]], doc_text: dict[str, str]) -> list[list[str]]:
        """Reorder each pool by cross-encoder score for its query; empty pools stay empty."""
        todo = list(dict.fromkeys((q, d) for q, pool in zip(queries, pools) for d in pool if (q, d) not in self.cache))
        if todo:
            scores = self.scorer([(q, doc_text[d]) for q, d in todo])
            self.cache.update(zip(todo, scores))
            self.pairs_scored += len(todo)
        return [sorted(pool, key=lambda d: -self.cache[(q, d)]) for q, pool in zip(queries, pools)]  # sorted() is stable
