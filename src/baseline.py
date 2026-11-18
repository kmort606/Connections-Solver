"""Embedding-baseline GroupScorers.

score(words, group) = mean pairwise cosine similarity of the 4 words' vectors (6 pairs).
Scorers differ only in HOW a word's vector is made:

  SentenceScorer    sentence-transformer on each word ALONE  (no context)
                    template="the word {w}" gives variation (d)
  ContextualScorer  all 16 words fed as ONE string "w1, w2, ..., w16";
                    word vector = mean of its subword-token vectors (prof's idea)
  StaticScorer      GloVe static vectors (optional, not needed for the TA meeting)

Vectors are cached per puzzle, so scoring all 1,820 subsets = one encoder pass.

Quick check on real data:   python -m src.baseline --scorer st_single contextual
"""
from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable, Optional

import numpy as np

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GLOVE_MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


class EmbeddingScorer:
    """Base class. Subclasses implement _embed_puzzle(words) -> (len(words), d) array."""

    name = "embedding"

    def __init__(self):
        self._cache: dict[tuple[str, ...], dict[str, np.ndarray]] = {}

    def _embed_puzzle(self, words: tuple[str, ...]) -> np.ndarray:
        raise NotImplementedError

    def vectors(self, words) -> dict[str, np.ndarray]:
        """Unit-length vector per word, cached per puzzle (unit length -> cosine = dot product)."""
        words = tuple(words)
        if words not in self._cache:
            E = _unit(np.asarray(self._embed_puzzle(words), dtype=np.float32))
            self._cache[words] = dict(zip(words, E))
        return self._cache[words]

    def similarity_matrix(self, words) -> np.ndarray:
        """16x16 cosine matrix -- handy for the notebook heatmap."""
        V = self.vectors(words)
        E = np.stack([V[w] for w in words])
        return E @ E.T

    def score(self, words, group) -> float:
        V = self.vectors(words)
        sims = [float(V[a] @ V[b]) for a, b in itertools.combinations(group, 2)]
        return sum(sims) / len(sims)


# --------------------------------------------------------------------------- #
# Sentence-transformer, each word alone
# --------------------------------------------------------------------------- #
class SentenceScorer(EmbeddingScorer):
    def __init__(self, model_name: str = DEFAULT_ST_MODEL, template: str = "{w}",
                 device: str | None = None, model=None):
        super().__init__()
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
        self.model, self.template = model, template
        self.name = "st_single" if template == "{w}" else "st_template"
        self._word_cache: dict[str, np.ndarray] = {}  # no context -> a word's vector never changes

    def _embed_puzzle(self, words):
        todo = [w for w in words if w not in self._word_cache]
        if todo:
            texts = [self.template.format(w=w.lower()) for w in todo]
            E = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            self._word_cache.update(zip(todo, E))
        return np.stack([self._word_cache[w] for w in words])


# --------------------------------------------------------------------------- #
# Contextual: all 16 words in one input
# --------------------------------------------------------------------------- #
class ContextualScorer(EmbeddingScorer):
    """Feed "w1, w2, ..., w16" to a transformer; word vector = mean of its token vectors.

    Self-attention lets each word's vector depend on the other 15 words, so BASS
    next to PIKE/TROUT can lean "fish" and next to DRUM/TUBA lean "music".
    n_orders > 1 averages over shuffled word orders to wash out position effects.
    """

    name = "contextual"

    def __init__(self, model_name: str = DEFAULT_ST_MODEL, n_orders: int = 1, layer: int = -1,
                 device: str | None = None, seed: int = 0, tokenizer=None, model=None):
        super().__init__()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModel.from_pretrained(model_name)).to(self.device).eval()
        self.n_orders, self.layer = n_orders, layer
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def join_words(words) -> tuple[str, list[tuple[int, int]]]:
        """'a, b, c' plus each word's (start, end) character span in that string."""
        text, spans = "", []
        for i, w in enumerate(words):
            if i:
                text += ", "
            spans.append((len(text), len(text) + len(w)))
            text += w
        return text, spans

    def token_states(self, words):
        """Run the model once. Returns (hidden states (n_tokens, d), token offsets, word spans)."""
        text, spans = self.join_words([w.lower() for w in words])
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        H = out.hidden_states[self.layer][0].cpu().numpy()
        return H, offsets, spans

    def _embed_order(self, words) -> np.ndarray:
        H, offsets, spans = self.token_states(words)
        vecs = []
        for (s, e), w in zip(spans, words):
            # tokens whose characters lie inside this word's span (special tokens have empty offsets)
            idx = [i for i, (ts, te) in enumerate(offsets) if te > ts and ts >= s and te <= e]
            if not idx:
                raise ValueError(f"No tokens found for {w!r} (input truncated?)")
            vecs.append(H[idx].mean(axis=0))
        return np.stack(vecs)

    def _embed_puzzle(self, words):
        words = list(words)
        total = self._embed_order(words)
        for _ in range(self.n_orders - 1):
            perm = self.rng.permutation(len(words))
            total = total + self._embed_order([words[i] for i in perm])[np.argsort(perm)]
        return total / self.n_orders


# --------------------------------------------------------------------------- #
# Static vectors (optional)
# --------------------------------------------------------------------------- #
class StaticScorer(EmbeddingScorer):
    """Static word vectors via lookup(token) -> vector | None.
    Multi-word entries are averaged; a word with no known tokens is OOV -> zero vector."""

    name = "glove"

    def __init__(self, lookup: Callable[[str], Optional[np.ndarray]], dim: int):
        super().__init__()
        self.lookup, self.dim = lookup, dim
        self._seen, self._oov = set(), set()

    def _word_vec(self, w):
        vecs = [v for v in (self.lookup(t) for t in w.lower().replace("-", " ").split()) if v is not None]
        self._seen.add(w)
        if not vecs:
            self._oov.add(w)
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _embed_puzzle(self, words):
        return np.stack([self._word_vec(w) for w in words])

    @property
    def oov_rate(self):
        return len(self._oov) / max(1, len(self._seen))

    @property
    def oov_words(self):
        return sorted(self._oov)


def load_glove() -> StaticScorer:
    from sentence_transformers import SentenceTransformer
    we = SentenceTransformer(DEFAULT_GLOVE_MODEL, device="cpu")[0]
    W = we.emb_layer.weight.detach().cpu().numpy()
    vocab = {w: i for i, w in enumerate(we.tokenizer.vocab)}
    return StaticScorer(lambda t: W[vocab[t]] if t in vocab else None, dim=W.shape[1])


def make_scorer(kind: str, **kw) -> EmbeddingScorer:
    """kind: "st_single" | "st_template" | "contextual" | "glove"."""
    if kind == "st_single":
        return SentenceScorer(kw.get("model_name", DEFAULT_ST_MODEL))
    if kind == "st_template":
        return SentenceScorer(kw.get("model_name", DEFAULT_ST_MODEL), template="the word {w}")
    if kind == "contextual":
        return ContextualScorer(kw.get("model_name", DEFAULT_ST_MODEL), n_orders=kw.get("n_orders", 1))
    if kind == "glove":
        return load_glove()
    raise ValueError(f"unknown scorer {kind!r}")


# --------------------------------------------------------------------------- #
# Quick sanity check (no search yet): do true groups outscore random groups?
# --------------------------------------------------------------------------- #
def sanity(scorer, puzzles, n_random: int = 20, seed: int = 0) -> dict:
    """Per puzzle: compare true-group scores against random 4-word groups,
    and check whether the single best of all 1,820 subsets is a true group."""
    rng = random.Random(seed)
    true_s, rand_s, top_is_true = [], [], 0
    by_level = [[] for _ in range(4)]
    for p in puzzles:
        truth = {frozenset(g.words) for g in p.groups}
        for g in p.groups:
            s = scorer.score(p.words, g.words)
            true_s.append(s)
            by_level[g.level].append(s)
        for _ in range(n_random):
            g = tuple(rng.sample(p.words, 4))
            if frozenset(g) not in truth:
                rand_s.append(scorer.score(p.words, g))
        best = max(itertools.combinations(p.words, 4), key=lambda g: scorer.score(p.words, g))
        top_is_true += frozenset(best) in truth
    return {
        "mean_true": float(np.mean(true_s)),
        "mean_random": float(np.mean(rand_s)),
        "mean_true_by_level": [float(np.mean(x)) for x in by_level],
        "top1_is_true_group": top_is_true / len(puzzles),
    }


def main():
    from src.data import load_split
    from src.puzzle import LEVELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", nargs="+", default=["st_single", "contextual"])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    puzzles = load_split(args.split)
    print(f"{len(puzzles)} {args.split} puzzles\n")
    print(f"{'scorer':<12}{'true':>7}{'random':>8}  " + "".join(f"{l:>8}" for l in LEVELS) + "   top1=true")
    for kind in args.scorer:
        r = sanity(make_scorer(kind), puzzles)
        lv = "".join(f"{x:>8.3f}" for x in r["mean_true_by_level"])
        print(f"{kind:<12}{r['mean_true']:>7.3f}{r['mean_random']:>8.3f}  {lv}   {r['top1_is_true_group']:>8.0%}")


if __name__ == "__main__":
    main()
