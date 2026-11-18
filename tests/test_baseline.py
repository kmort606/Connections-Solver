"""Step 3 checks -- no model downloads. Run from repo root:  python -m pytest -q"""
import itertools

import numpy as np
import pytest

from src.baseline import ContextualScorer, EmbeddingScorer, SentenceScorer, StaticScorer

WORDS = ("BASS", "PIKE", "TROUT", "SOLE", "DRUM", "HARP", "OBOE", "TUBA")


class FixedScorer(EmbeddingScorer):
    def __init__(self, vecs):
        super().__init__(); self.vecs = vecs
    def _embed_puzzle(self, words):
        return np.stack([self.vecs[w] for w in words])


def test_score_is_mean_pairwise_cosine():
    rng = np.random.default_rng(0)
    vecs = {w: rng.normal(size=5) for w in WORDS}
    g = WORDS[:4]
    cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    expected = np.mean([cos(vecs[a], vecs[b]) for a, b in itertools.combinations(g, 2)])
    assert FixedScorer(vecs).score(WORDS, g) == pytest.approx(expected, abs=1e-5)


def test_true_group_beats_mixed_group():
    vecs = {w: np.array([1.0, 0.1]) if i < 4 else np.array([0.1, 1.0]) for i, w in enumerate(WORDS)}
    s = FixedScorer(vecs)
    assert s.score(WORDS, WORDS[:4]) > s.score(WORDS, WORDS[:3] + WORDS[4:5])


def test_sentence_scorer_caches_per_word():
    calls = []
    class FakeModel:
        def encode(self, texts, **kw):
            calls.append(list(texts))
            return np.stack([np.ones(3) * len(t) for t in texts])
    s = SentenceScorer(model=FakeModel(), template="the word {w}")
    s.score(WORDS, WORDS[:4])
    s.score(WORDS[::-1], WORDS[:4])          # new puzzle, same words -> no new encode
    assert len(calls) == 1 and calls[0][0] == "the word bass"


def test_static_oov():
    s = StaticScorer(lambda t: None if t == "pike" else np.ones(2), dim=2)
    s.score(WORDS, WORDS[:4])
    assert s.oov_words == ["PIKE"] and s.oov_rate == pytest.approx(1 / 8)


def test_contextual_pools_subword_tokens():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    # Tiny WordPiece tokenizer built directly (works the same across transformers versions)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", "bass", "pike", "ice", "##cream"]
    tk = Tokenizer(models.WordPiece(vocab={w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=16)
    torch.manual_seed(0)
    sc = ContextualScorer(tokenizer=tok, model=transformers.BertModel(cfg), device="cpu")

    words = ("BASS", "ICECREAM", "PIKE")       # ICECREAM -> "ice" + "##cream"
    H, offsets, _ = sc.token_states(words)
    E = sc._embed_order(words)
    toks = tok.convert_ids_to_tokens(tok("bass, icecream, pike")["input_ids"])
    i_ice, i_cream = toks.index("ice"), toks.index("##cream")
    assert np.allclose(E[1], H[[i_ice, i_cream]].mean(axis=0), atol=1e-5)
    assert np.allclose(E[0], H[toks.index("bass")], atol=1e-5)

    # n_orders averaging keeps the output aligned with the input word order
    sc2 = ContextualScorer(tokenizer=tok, model=sc.model, device="cpu", n_orders=3)
    assert sc2._embed_puzzle(words).shape == (3, 8)
